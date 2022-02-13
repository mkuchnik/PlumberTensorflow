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

#include <sys/time.h>
#include <time.h>
#include <iostream>

#include "tensorflow/core/platform/env_time.h"

namespace tensorflow {

constexpr int32 kCalibrationInterval = 100;

/* static */
uint64 EnvTime::NowNanos() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (static_cast<uint64>(ts.tv_sec) * kSecondsToNanos +
          static_cast<uint64>(ts.tv_nsec));
}

/* static */
int64 ThreadExecutionTime::ClockId() {
  clockid_t cid;
  int s = pthread_getcpuclockid(pthread_self(), &cid);
  if (s != 0) {
    std::cerr << "Error getting ClockId " << s << std::endl;
    return s;
  }
  return static_cast<int64>(cid);
}

/* static */
uint64 ThreadExecutionTime::NowNanos() {
  struct timespec ts;
  // Note(mkuchnik): Thread time can be subject to inaccuracies due to e.g.,
  // rounding or coarse timers. For instance, process time may be used instead.
  int32 ret = clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  if (!ret) {
    return (static_cast<uint64>(ts.tv_sec) * kSecondsToNanos +
            static_cast<uint64>(ts.tv_nsec));
  } else {
    // Failure
    std::cerr << "Error getting ThreadExecutionTime: " << ret << std::endl;
    return 0;
  }
}

/* static */
uint64 ThreadExecutionTime::ResolutionNanos() {
  struct timespec ts;
  int32 ret = clock_getres(CLOCK_THREAD_CPUTIME_ID, &ts);
  if (!ret) {
    return (static_cast<uint64>(ts.tv_sec) * kSecondsToNanos +
            static_cast<uint64>(ts.tv_nsec));
  } else {
    // Failure
    std::cerr << "Error getting ProcessExecutionTime: " << ret << std::endl;
    return 0;
  }
}

/* static */
uint64 FastThreadExecutionTime::NowNanos() {
  thread_local uint64 last_wallclock_time_ = 0;
  thread_local uint64 last_thread_time_ = 0;
  thread_local uint64 current_wallclock_time_ = 0;
  thread_local uint64 current_thread_time_ = 0;
  thread_local float ratio_ = 0.;
  thread_local int64 steps_since_calibration_ = 0;
  struct timespec ts;
  if (!ratio_ || (steps_since_calibration_ > kCalibrationInterval)) {
    // TODO(mkuchnik): Check return values
    clock_gettime(CLOCK_MONOTONIC, &ts);
    last_wallclock_time_ = current_wallclock_time_; // Swap
    current_wallclock_time_ = static_cast<uint64>(ts.tv_sec) * kSecondsToNanos +
                              static_cast<uint64>(ts.tv_nsec);
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    last_thread_time_ = current_thread_time_; // Swap
    current_thread_time_ = static_cast<uint64>(ts.tv_sec) * kSecondsToNanos +
                           static_cast<uint64>(ts.tv_nsec);
    const auto wallclock_diff = current_wallclock_time_ - last_wallclock_time_;
    const auto thread_diff = current_thread_time_ - last_thread_time_;
    ratio_ = static_cast<float>(thread_diff)
             / static_cast<float>(wallclock_diff);
    if (ratio_ >= 0.0) {
      steps_since_calibration_ = 0;
      return current_wallclock_time_ * ratio_;
    } else {
      // Reject
      ratio_ = 0.;
      return current_wallclock_time_;
    }
  } else {
    clock_gettime(CLOCK_MONOTONIC, &ts);
    auto wallclock_time = static_cast<uint64>(ts.tv_sec) * kSecondsToNanos +
                          static_cast<uint64>(ts.tv_nsec);
    steps_since_calibration_++;
    return wallclock_time * ratio_;
  }
}

/* static */
uint64 ProcessExecutionTime::NowNanos() {
  struct timespec ts;
  // Note(mkuchnik): Thread time can be subject to inaccuracies due to e.g.,
  // rounding or coarse timers. For instance, process time may be used instead.
  int32 ret = clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  if (!ret) {
    return (static_cast<uint64>(ts.tv_sec) * kSecondsToNanos +
            static_cast<uint64>(ts.tv_nsec));
  } else {
    // Failure
    return 0;
  }
}

/* static */
uint64 ProcessExecutionTime::ResolutionNanos() {
  struct timespec ts;
  int32 ret = clock_getres(CLOCK_PROCESS_CPUTIME_ID, &ts);
  if (!ret) {
    return (static_cast<uint64>(ts.tv_sec) * kSecondsToNanos +
            static_cast<uint64>(ts.tv_nsec));
  } else {
    // Failure
    return 0;
  }
}

}  // namespace tensorflow
