/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#if defined(__linux__)
#include <linux/fs.h>  // needed for BLKSSZGET
#endif
#include <stdint.h>
#include <stdio.h>
#if defined(__linux__)
#include <sys/ioctl.h>
#endif
#include <sys/mman.h>

#if defined(__linux__)
#include <sys/sendfile.h>
#endif
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "tensorflow/core/platform/default/posix_file_system.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/error.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {

// 128KB of copy buffer
constexpr size_t kPosixCopyFileBufferSize = 128 * 1024;

// The environment variable to configure the throttle (format: <int64>)
constexpr char kThrottleRate[] = "POSIX_THROTTLE_TOKEN_RATE";
// The environment variable to configure the token bucket size (format: <int64>)
constexpr char kThrottleBucket[] = "POSIX_THROTTLE_BUCKET_SIZE";
// The environment variable that controls the number of tokens per request.
// (format: <int64>)
constexpr char kTokensPerRequest[] = "POSIX_TOKENS_PER_REQUEST";
// The environment variable to configure the initial tokens (format: <int64>)
constexpr char kInitialTokens[] = "POSIX_INITIAL_TOKENS";

// Helper function to extract an environment variable and convert it into a
// value of type T.
template <typename T>
bool GetEnvVar(const char* varname, bool (*convert)(StringPiece, T*),
               T* value) {
  const char* env_value = std::getenv(varname);
  if (env_value == nullptr) {
    return false;
  }
  return convert(env_value, value);
}

PosixFileSystem::PosixFileSystem():
  throttle_(std::make_shared<PosixThrottle>()) {
  int64 token_value;
  if (GetEnvVar(kThrottleRate, strings::safe_strto64, &token_value)) {
    PosixThrottleConfig config;
    config.enabled = true;
    config.token_rate = token_value;

    if (GetEnvVar(kThrottleBucket, strings::safe_strto64, &token_value)) {
      config.bucket_size = token_value;
    }

    if (GetEnvVar(kTokensPerRequest, strings::safe_strto64, &token_value)) {
      config.tokens_per_request = token_value;
    } else {
      // NOTE(mkuchnik): no per-read limit by default
      config.tokens_per_request = 0;
    }

    if (GetEnvVar(kInitialTokens, strings::safe_strto64, &token_value)) {
      config.initial_tokens = token_value;
    }
    throttle_->SetConfig(config);
    LOG(ERROR) << "Throttling is enabled with BW limit of "
               << throttle_->available_max_bandwidth_kibytes_sec()
               << "kiB/s";
  }
}

// pread() based random-access
class PosixRandomAccessFile : public RandomAccessFile {
 private:
  string filename_;
  int fd_;
  const std::shared_ptr<PosixThrottle> throttle_;

  void maybe_wait_on_throttle() const {
    if (throttle_) {
      bool is_admitted = throttle_->AdmitRequest();
      while (!is_admitted) {
        Env::Default()->SleepForMicroseconds(50000); // Sleep for 50ms
        is_admitted = throttle_->AdmitRequest();
      }
    }
  }

  void maybe_record_throttle_bytes(size_t bytes) const {
    if (throttle_) {
      throttle_->RecordResponse(bytes);
    }
  }

 public:
  PosixRandomAccessFile(const string& fname, int fd)
      : filename_(fname), fd_(fd) {}

  PosixRandomAccessFile(const string& fname, int fd,
      std::shared_ptr<PosixThrottle> throttle)
      : filename_(fname), fd_(fd), throttle_(throttle) {
  }

  ~PosixRandomAccessFile() override {
    if (close(fd_) < 0) {
      LOG(ERROR) << "close() failed: " << strerror(errno);
    }
  }

  Status Name(StringPiece* result) const override {
    *result = filename_;
    return Status::OK();
  }

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    maybe_wait_on_throttle();
    Status s;
    char* dst = scratch;
    while (n > 0 && s.ok()) {
      // Some platforms, notably macs, throw EINVAL if pread is asked to read
      // more than fits in a 32-bit integer.
      size_t requested_read_length;
      if (n > INT32_MAX) {
        requested_read_length = INT32_MAX;
      } else {
        requested_read_length = n;
      }
      ssize_t r =
          pread(fd_, dst, requested_read_length, static_cast<off_t>(offset));
      if (r > 0) {
        dst += r;
        n -= r;
        offset += r;
      } else if (r == 0) {
        s = Status(error::OUT_OF_RANGE, "Read less bytes than requested");
      } else if (errno == EINTR || errno == EAGAIN) {
        // Retry
      } else {
        s = IOError(filename_, errno);
      }
    }
    maybe_record_throttle_bytes(dst - scratch);
    *result = StringPiece(scratch, dst - scratch);
    return s;
  }

#if defined(TF_CORD_SUPPORT)
  Status Read(uint64 offset, size_t n, absl::Cord* cord) const override {
    if (n == 0) {
      return Status::OK();
    }
    if (n < 0) {
      return errors::InvalidArgument(
          "Attempting to read ", n,
          " bytes. You cannot read a negative number of bytes.");
    }

    char* scratch = new char[n];
    if (scratch == nullptr) {
      return errors::ResourceExhausted("Unable to allocate ", n,
                                       " bytes for file reading.");
    }

    StringPiece tmp;
    Status s = Read(offset, n, &tmp, scratch);

    absl::Cord tmp_cord = absl::MakeCordFromExternal(
        absl::string_view(static_cast<char*>(scratch), tmp.size()),
        [scratch](absl::string_view) { delete[] scratch; });
    cord->Append(tmp_cord);
    return s;
  }
#endif
};

// modified from https://stackoverflow.com/questions/1898153/how-to-determine-if-memory-is-aligned
static inline bool is_aligned(const void* pointer, size_t byte_count) {
  return reinterpret_cast<std::uintptr_t>(pointer) % byte_count == 0;
}

// pread() based random-access and aligned buffers (e.g., for DIRECT_IO)
class AlignedPosixRandomAccessFile : public RandomAccessFile {
 private:
  string filename_;
  int fd_;
  uint64 alignment_;
  const std::shared_ptr<PosixThrottle> throttle_;

  void maybe_wait_on_throttle() const {
    if (throttle_) {
      bool is_admitted = throttle_->AdmitRequest();
      while (!is_admitted) {
        Env::Default()->SleepForMicroseconds(50000); // Sleep for 50ms
        is_admitted = throttle_->AdmitRequest();
      }
    }
  }

  void maybe_record_throttle_bytes(size_t bytes) const {
    if (throttle_) {
      throttle_->RecordResponse(bytes);
    }
  }

 public:
  AlignedPosixRandomAccessFile(const string& fname, int fd, uint64 alignment)
      : filename_(fname), fd_(fd), alignment_(alignment) {}

  AlignedPosixRandomAccessFile(const string& fname, int fd, uint64 alignment,
      std::shared_ptr<PosixThrottle> throttle)
      : filename_(fname), fd_(fd), alignment_(alignment), throttle_(throttle) {
  }

  ~AlignedPosixRandomAccessFile() override {
    // We flush the entire file to get any stragglers.
    int ret = posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED);
    if (ret) {
      LOG(ERROR) << "POSIX fadvise failed with errno: " << strerror(errno);
    }
    if (close(fd_) < 0) {
      LOG(ERROR) << "close() failed: " << strerror(errno);
    }
  }

  Status Name(StringPiece* result) const override {
    *result = filename_;
    return Status::OK();
  }

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    maybe_wait_on_throttle();
    Status s;
    if (alignment_ && !is_aligned(scratch, alignment_)) {
      // NOTE(mkuchnik): when alignment_ is 0, assume alignment not needed
        s = Status(error::FAILED_PRECONDITION,
                   "Alignment of scratch does not match file alignment");
        return s;
    }
    char* dst = scratch;
    while (n > 0 && s.ok()) {
      // Some platforms, notably macs, throw EINVAL if pread is asked to read
      // more than fits in a 32-bit integer.
      size_t requested_read_length;
      if (n > INT32_MAX) {
        requested_read_length = INT32_MAX;
      } else {
        requested_read_length = n;
      }
      ssize_t r =
          pread(fd_, dst, requested_read_length, static_cast<off_t>(offset));
      if (r > 0) {
        dst += r;
        n -= r;
        offset += r;
        // NOTE(mkuchnik): We tell kernel to let go of these (full) pages.
        // Unaligned data is not cleared and will need to be purged manually.
        int ret = posix_fadvise(fd_, offset, r, POSIX_FADV_DONTNEED);
        if (ret) {
          LOG(ERROR) << "POSIX fadvise failed with errno: " << strerror(errno);
        }
      } else if (r == 0) {
        s = Status(error::OUT_OF_RANGE, "Read less bytes than requested");
      } else if (errno == EINTR || errno == EAGAIN) {
        // Retry
      } else {
        s = IOError(filename_, errno);
      }
    }
    maybe_record_throttle_bytes(dst - scratch);
    *result = StringPiece(scratch, dst - scratch);
    return s;
  }
};

class PosixWritableFile : public WritableFile {
 private:
  string filename_;
  FILE* file_;

 public:
  PosixWritableFile(const string& fname, FILE* f)
      : filename_(fname), file_(f) {}

  ~PosixWritableFile() override {
    if (file_ != nullptr) {
      // Ignoring any potential errors
      fclose(file_);
    }
  }

  Status Append(StringPiece data) override {
    size_t r = fwrite(data.data(), 1, data.size(), file_);
    if (r != data.size()) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

#if defined(TF_CORD_SUPPORT)
  // \brief Append 'cord' to the file.
  Status Append(const absl::Cord& cord) override {
    for (const auto& chunk : cord.Chunks()) {
      size_t r = fwrite(chunk.data(), 1, chunk.size(), file_);
      if (r != chunk.size()) {
        return IOError(filename_, errno);
      }
    }
    return Status::OK();
  }
#endif

  Status Close() override {
    if (file_ == nullptr) {
      return IOError(filename_, EBADF);
    }
    Status result;
    if (fclose(file_) != 0) {
      result = IOError(filename_, errno);
    }
    file_ = nullptr;
    return result;
  }

  Status Flush() override {
    if (fflush(file_) != 0) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

  Status Name(StringPiece* result) const override {
    *result = filename_;
    return Status::OK();
  }

  Status Sync() override {
    Status s;
    if (fflush(file_) != 0) {
      s = IOError(filename_, errno);
    }
    return s;
  }

  Status Tell(int64* position) override {
    Status s;
    *position = ftell(file_);

    if (*position == -1) {
      s = IOError(filename_, errno);
    }

    return s;
  }
};

class PosixReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  PosixReadOnlyMemoryRegion(const void* address, uint64 length)
      : address_(address), length_(length) {}
  ~PosixReadOnlyMemoryRegion() override {
    munmap(const_cast<void*>(address_), length_);
  }
  const void* data() override { return address_; }
  uint64 length() override { return length_; }

 private:
  const void* const address_;
  const uint64 length_;
};

// Use ioctl with BLKSSZGET to get this alignment
// uint64 sector_size = 0;
// ioctl(fd, BLKSSZGET, &sector_size);
// Often this is 512 or 4096
Status GetLogicalBlockSize(const string& filename, int fd, uint64* sector_size) {
#ifdef BLKSSZGET
  Status s;
  // TODO(mkuchnik): This is bound to fail as fd must refer to the device
  // holding the file. Use fstat to get that device or just rely on reasonable
  // defaults. It may also be possible to just try until no errors appear.
  int32 ret = ioctl(fd, BLKSSZGET, sector_size);
  if (ret < 0) {
    s = IOError(filename, errno);
  } else {
    s = Status::OK();
  }
  return s;
#else
  // Non-linux devices may not be compatible, so give up.
  return errors::Unavailable("Logical block size not implemented")
#endif
}

Status PosixFileSystem::NewRandomAccessFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<RandomAccessFile>* result) {
  string translated_fname = TranslateName(fname);
  Status s;
  int fd = open(translated_fname.c_str(), O_RDONLY);
  if (fd < 0) {
    s = IOError(fname, errno);
  } else {
    if (throttle_ && throttle_->is_enabled()) {
      result->reset(new PosixRandomAccessFile(translated_fname, fd, throttle_));
    } else {
      result->reset(new PosixRandomAccessFile(translated_fname, fd));
    }
  }
  return s;
}

Status PosixFileSystem::NewAlignedRandomAccessFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<RandomAccessFile>* result) {
  // Since linux kernel 2.6.0, alignment of user buffer and transfer sizes
  // must be multiple of underlying block device (usually 512 bytes).
  string translated_fname = TranslateName(fname);
  Status s;
  int fd = open(translated_fname.c_str(), O_RDONLY);
  if (fd < 0) {
    s = IOError(fname, errno);
  } else {
    // TODO(mkuchnik): For now, we just use a default alignment as O_DIRECT is
    // not implemented. A call to GetLogicalBlockSize can be used to find the
    // alignment, if needed.
    uint64 alignment = 0;
    result->reset(
        new AlignedPosixRandomAccessFile(translated_fname, fd, alignment));
  }
  return s;
}

Status PosixFileSystem::NewRandomAccessFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<RandomAccessFile>* result,
    const FileOptions& options) {
  if (!options.hint_no_cache) {
    return NewRandomAccessFile(fname, token, result);
  } else {
    return NewAlignedRandomAccessFile(fname, token, result);
  }
}

Status PosixFileSystem::NewWritableFile(const string& fname,
                                        TransactionToken* token,
                                        std::unique_ptr<WritableFile>* result) {
  string translated_fname = TranslateName(fname);
  Status s;
  FILE* f = fopen(translated_fname.c_str(), "w");
  if (f == nullptr) {
    s = IOError(fname, errno);
  } else {
    result->reset(new PosixWritableFile(translated_fname, f));
  }
  return s;
}

Status PosixFileSystem::NewAppendableFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
  string translated_fname = TranslateName(fname);
  Status s;
  FILE* f = fopen(translated_fname.c_str(), "a");
  if (f == nullptr) {
    s = IOError(fname, errno);
  } else {
    result->reset(new PosixWritableFile(translated_fname, f));
  }
  return s;
}

Status PosixFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  string translated_fname = TranslateName(fname);
  Status s = Status::OK();
  int fd = open(translated_fname.c_str(), O_RDONLY);
  if (fd < 0) {
    s = IOError(fname, errno);
  } else {
    struct stat st;
    ::fstat(fd, &st);
    const void* address =
        mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (address == MAP_FAILED) {
      s = IOError(fname, errno);
    } else {
      result->reset(new PosixReadOnlyMemoryRegion(address, st.st_size));
    }
    if (close(fd) < 0) {
      s = IOError(fname, errno);
    }
  }
  return s;
}

Status PosixFileSystem::FileExists(const string& fname,
                                   TransactionToken* token) {
  if (access(TranslateName(fname).c_str(), F_OK) == 0) {
    return Status::OK();
  }
  return errors::NotFound(fname, " not found");
}

Status PosixFileSystem::GetChildren(const string& dir, TransactionToken* token,
                                    std::vector<string>* result) {
  string translated_dir = TranslateName(dir);
  result->clear();
  DIR* d = opendir(translated_dir.c_str());
  if (d == nullptr) {
    return IOError(dir, errno);
  }
  struct dirent* entry;
  while ((entry = readdir(d)) != nullptr) {
    StringPiece basename = entry->d_name;
    if ((basename != ".") && (basename != "..")) {
      result->push_back(entry->d_name);
    }
  }
  if (closedir(d) < 0) {
    return IOError(dir, errno);
  }
  return Status::OK();
}

Status PosixFileSystem::GetMatchingPaths(const string& pattern,
                                         TransactionToken* token,
                                         std::vector<string>* results) {
  return internal::GetMatchingPaths(this, Env::Default(), pattern, results);
}

Status PosixFileSystem::DeleteFile(const string& fname,
                                   TransactionToken* token) {
  Status result;
  if (unlink(TranslateName(fname).c_str()) != 0) {
    result = IOError(fname, errno);
  }
  return result;
}

Status PosixFileSystem::CreateDir(const string& name, TransactionToken* token) {
  string translated = TranslateName(name);
  if (translated.empty()) {
    return errors::AlreadyExists(name);
  }
  if (mkdir(translated.c_str(), 0755) != 0) {
    return IOError(name, errno);
  }
  return Status::OK();
}

Status PosixFileSystem::DeleteDir(const string& name, TransactionToken* token) {
  Status result;
  if (rmdir(TranslateName(name).c_str()) != 0) {
    result = IOError(name, errno);
  }
  return result;
}

Status PosixFileSystem::GetFileSize(const string& fname,
                                    TransactionToken* token, uint64* size) {
  Status s;
  struct stat sbuf;
  if (stat(TranslateName(fname).c_str(), &sbuf) != 0) {
    *size = 0;
    s = IOError(fname, errno);
  } else {
    *size = sbuf.st_size;
  }
  return s;
}

Status PosixFileSystem::Stat(const string& fname, TransactionToken* token,
                             FileStatistics* stats) {
  Status s;
  struct stat sbuf;
  if (stat(TranslateName(fname).c_str(), &sbuf) != 0) {
    s = IOError(fname, errno);
  } else {
    stats->length = sbuf.st_size;
    stats->mtime_nsec = sbuf.st_mtime * 1e9;
    stats->is_directory = S_ISDIR(sbuf.st_mode);
  }
  return s;
}

Status PosixFileSystem::RenameFile(const string& src, const string& target,
                                   TransactionToken* token) {
  Status result;
  if (rename(TranslateName(src).c_str(), TranslateName(target).c_str()) != 0) {
    result = IOError(src, errno);
  }
  return result;
}

Status PosixFileSystem::CopyFile(const string& src, const string& target,
                                 TransactionToken* token) {
  string translated_src = TranslateName(src);
  struct stat sbuf;
  if (stat(translated_src.c_str(), &sbuf) != 0) {
    return IOError(src, errno);
  }
  int src_fd = open(translated_src.c_str(), O_RDONLY);
  if (src_fd < 0) {
    return IOError(src, errno);
  }
  string translated_target = TranslateName(target);
  // O_WRONLY | O_CREAT | O_TRUNC:
  //   Open file for write and if file does not exist, create the file.
  //   If file exists, truncate its size to 0.
  // When creating file, use the same permissions as original
  mode_t mode = sbuf.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO);
  int target_fd =
      open(translated_target.c_str(), O_WRONLY | O_CREAT | O_TRUNC, mode);
  if (target_fd < 0) {
    close(src_fd);
    return IOError(target, errno);
  }
  int rc = 0;
  off_t offset = 0;
  std::unique_ptr<char[]> buffer(new char[kPosixCopyFileBufferSize]);
  while (offset < sbuf.st_size) {
    // Use uint64 for safe compare SSIZE_MAX
    uint64 chunk = sbuf.st_size - offset;
    if (chunk > SSIZE_MAX) {
      chunk = SSIZE_MAX;
    }
#if defined(__linux__) && !defined(__ANDROID__)
    rc = sendfile(target_fd, src_fd, &offset, static_cast<size_t>(chunk));
#else
    if (chunk > kPosixCopyFileBufferSize) {
      chunk = kPosixCopyFileBufferSize;
    }
    rc = read(src_fd, buffer.get(), static_cast<size_t>(chunk));
    if (rc <= 0) {
      break;
    }
    rc = write(target_fd, buffer.get(), static_cast<size_t>(chunk));
    offset += chunk;
#endif
    if (rc <= 0) {
      break;
    }
  }

  Status result = Status::OK();
  if (rc < 0) {
    result = IOError(target, errno);
  }

  // Keep the error code
  rc = close(target_fd);
  if (rc < 0 && result == Status::OK()) {
    result = IOError(target, errno);
  }
  rc = close(src_fd);
  if (rc < 0 && result == Status::OK()) {
    result = IOError(target, errno);
  }

  return result;
}

}  // namespace tensorflow
