/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_FRAMEWORK_MODEL_H_
#define TENSORFLOW_CORE_FRAMEWORK_MODEL_H_

#include <list>
#include <memory>
#include <string>
// TODO(b/114492873): Move this include into core/platform.
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/model.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringprintf.h"

#include "tensorflow/core/lib/hash/hash.h"

// #define LIGHTWEIGHT_METRICS
#define CPU_BASED_METRICS 1 // 1 for on, 2 for dynamic

#if defined(COMPILER_GCC3)
#define TF_ADD_OVERFLOW(lhs, rhs, sum) (__builtin_add_overflow(lhs, rhs, &sum))
#else
#define TF_ADD_OVERFLOW(lhs, rhs, sum) (sum = lhs + rhs)
#endif

#if defined(__PRETTY_FUNCTION__)
#define TF_FUNCTION_NAME __PRETTY_FUNCTION__
#else
  #if defined(__FUNCTION__)
  #define TF_FUNCTION_NAME __FUNCTION__
  #else
  #define TF_FUNCTION_NAME __func__
  #endif
#endif

#define RECORD_AUTOTUNE_STATS

namespace tensorflow {
namespace data {
namespace model {

// A constant that can be used to enable auto-tuning.
constexpr int64_t kAutotune = -1;
constexpr char kParallelism[] = "parallelism";
constexpr char kBufferSize[] = "buffer_size";

// A key used to identify the input time of the model.
constexpr char kModelInputTimeKey[] = "model_input_time";

#ifndef LIGHTWEIGHT_METRICS
constexpr bool default_is_wallclock = true;
constexpr bool lightweight_metrics = false;
#else
constexpr bool lightweight_metrics = true;
constexpr bool default_is_wallclock = false;
#endif

constexpr bool CPU_based_metrics(CPU_BASED_METRICS);
constexpr bool dynamic_tuning(CPU_BASED_METRICS == 2);
constexpr bool default_is_CPU(CPU_based_metrics);

inline int64 ThreadNowNanos() {
  if (CPU_based_metrics) {
    return ThreadExecutionTime::NowNanos();
  } else {
    return EnvTime::NowNanos();
  }
}

enum class TraversalOrder {
  BFS = 0,
  REVERSE_BFS = 1,
};

// Represents thread-safe state that can be shared between an input pipeline and
// the performance model.
struct SharedState {
 public:
  SharedState(int64_t value, std::shared_ptr<mutex> mu,
              std::shared_ptr<condition_variable> cond_var)
      : value(value),
        mu(std::move(mu)),
        cond_var(std::move(cond_var)),
        tunable(value == kAutotune) {}

  double value;
  const std::shared_ptr<mutex> mu;
  const std::shared_ptr<condition_variable> cond_var;
  const bool tunable;
};

// Represents a parameter.
struct Parameter {
  Parameter(const string& name, std::shared_ptr<SharedState> state, double min,
            double max)
      : name(name),
        // Sometimes non-autotune nodes (with `autotune_=false`) may contain
        // parameters (for example inputs of parallel interleave dataset which
        // are not in the current cycle). To avoid unrealistic situation
        // (say `buffer_size=-1` or `parallelism=-1`) in the optimization
        // computation, if the state value is `kAutotune=-1` (just to indicate
        // the `SharedState` is tunable), we initialize the parameter value to
        // be the minimal value of the state.
        value(state->value == kAutotune ? min : state->value),
        min(min),
        max(max),
        state(std::move(state)) {}

  // Human-readable name of the parameter.
  const string name;

  // Identifies the model value of the parameter. This can be different from
  // the actual value (e.g. during optimization search).
  double value;

  // Identifies the minimum value of the parameter.
  const double min;

  // Identifies the maximum value of the parameter.
  const double max;

  // Shared state of the parameter.
  std::shared_ptr<SharedState> state;
};

// num_elements / total_time
typedef std::pair<int64, int64> Rate;

// Used by model to collect runtime stats for analysis
struct Node_Stats {
  int64 elements_produced;
  int64 wallclock_time;
  int64 processing_time;
  int64 parallelism;
  double ratio;
  int64 count;
  int64 bytes_produced;
  int64 bytes_consumed;
  int64 processing_time_clock;
#ifndef LIGHTWEIGHT_METRICS
  int64 scheduling_delay_time; // TODO(mkuchnik): is this used?
#endif
  int64 cardinality;
};

class AggregateSystemMetric {
#ifndef LIGHTWEIGHT_METRICS
    using key_type = std::string;
#else
    using key_type = uint64;
#endif
  private:
    mutex mu_;
    absl::flat_hash_map<key_type, int64> filename_to_filesize_;
#ifdef RECORD_AUTOTUNE_STATS
    double output_time_;
#endif

    // Return true if file can't be found or the filesize is smaller
    bool filename_may_not_exist(const key_type& filename, int64 size)
          TF_LOCKS_EXCLUDED(mu_) {
        tf_shared_lock l(mu_);
        const auto it = filename_to_filesize_.find(filename);
        const bool file_found = (it != filename_to_filesize_.end());
        return !file_found || (size < it->second);
    }

  public:
#ifdef RECORD_AUTOTUNE_STATS
    explicit AggregateSystemMetric(): output_time_(-1.0) {}
#else
    explicit AggregateSystemMetric(): {}
#endif

#ifndef LIGHTWEIGHT_METRICS
    void add_filename_size(const std::string& filename, int64 size) {
      DCHECK_GE(size, 0);
      if (filename_may_not_exist(filename, size)) {
        mutex_lock l(mu_);
        auto insert_pair = filename_to_filesize_.insert({filename, size});
        // Try to insert one more time, then give up (holding lock).
        if (!insert_pair.second) {
           // Insert failed
           VLOG(0) << "AddFilenameSize Insert (Failed): " << filename
                   << ": " << size;
           if (size > insert_pair.first->second) {
             insert_pair = filename_to_filesize_.insert({filename, size});
           }
        }
      }
    }

    // Make a copy
    absl::flat_hash_map<std::string, int64> filename_sizes()
        TF_LOCKS_EXCLUDED(mu_) {
      tf_shared_lock l(mu_);
      return absl::flat_hash_map<std::string, int64>(filename_to_filesize_);
    }

#else
    void add_filename_size(const std::string& filename, int64 size) {
      DCHECK_GE(size, 0);
      const uint64 hash = Hash64(filename);
      if (filename_may_not_exist(hash, size)) {
        mutex_lock l(mu_);
        auto insert_pair = filename_to_filesize_.insert({hash, size});
        // Try to insert one more time, then give up (holding lock).
        if (!insert_pair.second) {
           // Insert failed
           VLOG(0) << "AddFilenameSize Insert (Failed): " << filename
                   << ": " << size;
           if (size > insert_pair.first->second) {
             insert_pair = filename_to_filesize_.insert({hash, size});
           }
        }
      }
    }

    // Make a copy
    absl::flat_hash_map<std::string, int64> filename_sizes()
        TF_LOCKS_EXCLUDED(mu_) {
      absl::flat_hash_map<std::string, int64> return_mapping;
      return_mapping.reserve(filename_to_filesize_.size());
      tf_shared_lock l(mu_);
      for (auto const& pair: filename_to_filesize_) {
        std::string hash_string = std::to_string(pair.first);
        return_mapping.insert({hash_string, pair.second});
      }
      return return_mapping;
    }
#endif

    void record_output_time(double output_time) {
#ifdef RECORD_AUTOTUNE_STATS
      output_time_ = output_time;
#endif
    }

    double output_time() {
#ifdef RECORD_AUTOTUNE_STATS
      return output_time_;
#else
      return -1.0;
#endif
    }
};

// Used for aggregating ephemeral node statistics. Can be passed to nodes to
// atomically increment.
class AggregateDatasetMetric {
  private:
#ifndef LIGHTWEIGHT_METRICS
    const std::string name_; // for debug
#endif
    std::atomic<int64> elements_produced_;
    std::atomic<int64> processing_time_;
    std::atomic<int64> processing_time_clock_;
    std::atomic<int64> bytes_produced_;
    std::atomic<int64> bytes_consumed_;
#ifndef LIGHTWEIGHT_METRICS
    std::atomic<int64> udf_processing_time_;
    std::atomic<int64> udf_processing_time_clock_;
    std::atomic<int64> scheduling_delay_time_;
    std::atomic<int64> num_active_threads_;
    std::atomic<int64> num_active_threads_obs_count_;
    std::atomic<int64> wait_time_;
#endif
    std::atomic<int64> disk_bytes_read_;
    std::atomic<int64> elements_consumed_;
    std::atomic<int64> parallelism_;
#ifndef LIGHTWEIGHT_METRICS
    std::atomic<int64> active_elements_;
    std::atomic<int64> max_buffer_size_;
    std::atomic<int64> max_bytes_per_element_;
    std::atomic<int64> misc_buffer_size_;
    std::atomic<double> ratio_;
    std::atomic<bool> inter_op_parallelism_;
#endif

    // A check that any input is positive. Can be removed for production.
#ifndef LIGHTWEIGHT_METRICS
    inline void check_positive(int64 delta, const std::string& dbg_str) {
      if (TF_PREDICT_FALSE(delta < 0)) {
        VLOG(0) << "AggregateDatasetMetric saw negative delta: "
          << name_  << " " << dbg_str;
      }
    }
#endif

#ifndef LIGHTWEIGHT_METRICS
    // thread-safe max update
    // https://stackoverflow.com/questions/16190078/how-to-atomically-update-a-maximum-value
    template<typename T>
    void update_maximum(std::atomic<T>& maximum_value, T const& value) noexcept
    {
        T prev_value = maximum_value;
        while(prev_value < value &&
                !maximum_value.compare_exchange_weak(prev_value, value))
            {}
    }
#endif

  public:
    explicit AggregateDatasetMetric(const std::string& name) :
#ifndef LIGHTWEIGHT_METRICS
      name_(name),
#endif
      elements_produced_(0),
      processing_time_(0),
      processing_time_clock_(0),
      bytes_produced_(0),
      bytes_consumed_(0),
#ifndef LIGHTWEIGHT_METRICS
      udf_processing_time_(0),
      udf_processing_time_clock_(0),
      scheduling_delay_time_(0),
      num_active_threads_(0),
      num_active_threads_obs_count_(0),
      wait_time_(0),
#endif
      disk_bytes_read_(0),
      elements_consumed_(0),
#ifndef LIGHTWEIGHT_METRICS
      parallelism_(0),
      active_elements_(0),
      max_buffer_size_(0),
      max_bytes_per_element_(0),
      misc_buffer_size_(0),
      ratio_(-1.),
      inter_op_parallelism_(false)
#else
      parallelism_(0)
#endif
      {
#ifndef LIGHTWEIGHT_METRICS
        if (name_.empty()) {
          VLOG(0) << "Empty name created";
        }
#endif
    }

    void add_elements_produced(int64 delta) {
      DCHECK_GE(delta, 0);
#ifndef LIGHTWEIGHT_METRICS
      check_positive(delta, TF_FUNCTION_NAME);
#endif
      elements_produced_ += delta;
    }

    void increment_elements_produced() {
      ++elements_produced_;
    }

    int64 elements_produced() const {
      return elements_produced_;
    }

    void add_elements_consumed(int64 delta) {
      DCHECK_GE(delta, 0);
#ifndef LIGHTWEIGHT_METRICS
      check_positive(delta, TF_FUNCTION_NAME);
#endif
      elements_consumed_ += delta;
    }

    void increment_elements_consumed() {
      ++elements_consumed_;
    }

    int64 elements_consumed() const {
      return elements_consumed_;
    }

    inline void add_processing_time(int64 delta) {
      DCHECK_GE(delta, 0);
#ifndef LIGHTWEIGHT_METRICS
      check_positive(delta, TF_FUNCTION_NAME);
#endif
      processing_time_ += delta;
    }

    int64 processing_time() const {
      return processing_time_;
    }

    void add_udf_processing_time(int64 delta) {
#ifndef LIGHTWEIGHT_METRICS
      DCHECK_GE(delta, 0);
      check_positive(delta, TF_FUNCTION_NAME);
      udf_processing_time_ += delta;
#endif
    }

    int64 udf_processing_time() const {
#ifndef LIGHTWEIGHT_METRICS
      return udf_processing_time_;
#else
      return 0;
#endif
    }

    inline void add_processing_time_clock(int64 delta) {
      DCHECK_GE(delta, 0);
#ifndef LIGHTWEIGHT_METRICS
      check_positive(delta, TF_FUNCTION_NAME);
#endif
      processing_time_clock_ += delta;
    }

    int64 processing_time_clock() const {
      return processing_time_clock_;
    }

    void add_udf_processing_time_clock(int64 delta) {
#ifndef LIGHTWEIGHT_METRICS
      DCHECK_GE(delta, 0);
      check_positive(delta, TF_FUNCTION_NAME);
      udf_processing_time_clock_ += delta;
#endif
    }

    int64 udf_processing_time_clock() const {
#ifndef LIGHTWEIGHT_METRICS
      return udf_processing_time_clock_;
#else
      return 0;
#endif
    }

    void add_bytes_produced(int64 delta) {
      DCHECK_GE(delta, 0);
#ifndef LIGHTWEIGHT_METRICS
      check_positive(delta, TF_FUNCTION_NAME);
#endif
      bytes_produced_ += delta;
    }

    int64 bytes_produced() const {
      return bytes_produced_;
    }

    void add_disk_bytes_read(int64 delta) {
      DCHECK_GE(delta, 0);
#ifndef LIGHTWEIGHT_METRICS
      check_positive(delta, TF_FUNCTION_NAME);
#endif
      disk_bytes_read_ += delta;
    }

    int64 disk_bytes_read() const {
      return disk_bytes_read_;
    }

    void add_bytes_consumed(int64 delta) {
      DCHECK_GE(delta, 0);
#ifndef LIGHTWEIGHT_METRICS
      check_positive(delta, TF_FUNCTION_NAME);
#endif
      bytes_consumed_ += delta;
    }

    int64 bytes_consumed() const {
      return bytes_consumed_;
    }

    void add_scheduling_delay_time(int64 delta) {
#ifndef LIGHTWEIGHT_METRICS
      DCHECK_GE(delta, 0);
      check_positive(delta, TF_FUNCTION_NAME);
      scheduling_delay_time_ += delta;
#endif
    }

    int64 scheduling_delay_time() const {
#ifndef LIGHTWEIGHT_METRICS
      return scheduling_delay_time_;
#else
      return 0;
#endif
    }

    void add_num_active_threads_observation(int64 num_threads) {
#ifndef LIGHTWEIGHT_METRICS
      check_positive(num_threads, TF_FUNCTION_NAME);
      num_active_threads_ += num_threads;
      num_active_threads_obs_count_ += 1;
#endif
    }

    double avg_num_active_threads() {
#ifndef LIGHTWEIGHT_METRICS
      if (!num_active_threads_obs_count_) {
        // Division by 0
        return 1.0;
      } else {
        return static_cast<double>(num_active_threads_)
               / static_cast<double>(num_active_threads_obs_count_);
      }
#else
      return 0;
#endif
    }

    void add_wait_time(int64 delta) {
#ifndef LIGHTWEIGHT_METRICS
      DCHECK_GE(delta, 0);
      check_positive(delta, TF_FUNCTION_NAME);
      wait_time_ += delta;
#endif
    }

    int64 wait_time() const {
#ifndef LIGHTWEIGHT_METRICS
      return wait_time_;
#else
      return 0;
#endif
    }

    void set_inter_op_parallelism() {
#ifndef LIGHTWEIGHT_METRICS
      inter_op_parallelism_ = true;
#endif
    }

    bool inter_op_parallelism() {
#ifndef LIGHTWEIGHT_METRICS
      return inter_op_parallelism_;
#else
      return false;
#endif
    }

    void set_ratio(double ratio) {
#ifndef LIGHTWEIGHT_METRICS
      ratio_ = ratio;
#endif
    }

    double ratio() {
#ifndef LIGHTWEIGHT_METRICS
      return ratio_;
#else
      return 0;
#endif
    }

    void set_parallelism(int64 parallelism) {
      parallelism_ = parallelism;
    }

    bool parallelism() {
      return parallelism_;
    }

    void increment_active_elements() {
#ifndef LIGHTWEIGHT_METRICS
      active_elements_++;
#endif
    }

    void decrement_active_elements() {
#ifndef LIGHTWEIGHT_METRICS
      active_elements_--;
#endif
    }

    int64 active_elements() {
#ifndef LIGHTWEIGHT_METRICS
      return active_elements_;
#else
      return 0;
#endif
    }

    void record_buffer_size(int64 buffer_size) {
#ifndef LIGHTWEIGHT_METRICS
      update_maximum(max_buffer_size_, buffer_size);
#endif
    }

    int64 max_buffer_size() {
#ifndef LIGHTWEIGHT_METRICS
      return max_buffer_size_;
#else
      return 0;
#endif
    }

    void record_element_size(int64 bytes) {
#ifndef LIGHTWEIGHT_METRICS
      update_maximum(max_bytes_per_element_, bytes);
#endif
    }

    int64 max_bytes_per_element() {
#ifndef LIGHTWEIGHT_METRICS
      return max_bytes_per_element_;
#else
      return 0;
#endif
    }

    void record_misc_buffer_size(int64 buffer_size) {
#ifndef LIGHTWEIGHT_METRICS
      update_maximum(misc_buffer_size_, buffer_size);
#endif
    }

    int64 misc_buffer_size() {
#ifndef LIGHTWEIGHT_METRICS
      return misc_buffer_size_;
#else
      return 0;
#endif
    }

};


std::shared_ptr<Parameter> MakeParameter(const string& name,
                                         std::shared_ptr<SharedState> state,
                                         double min, double max);

class Model;  // Forward declaration for Node pointer

#ifndef LIGHTWEIGHT_METRICS
// Helper function to add only positive time
template <typename T>
inline void safe_processing_time_increment(T& processing_time, int64 delta,
    const std::string& tag) {
    if (TF_PREDICT_FALSE(delta < 0)) {
        VLOG(0) << "Record stop saw negative processing time: " << tag;
    } else {
      processing_time += delta;
    }
}
#endif


// Abstract representation of a TensorFlow input pipeline node. It collects
// information about inputs to this node, processing time spent executing the
// node logic, number of elements produced by the node, various other
// information (e.g. batch size or execution parallelism).
//
// Developers of tf.data transformations are not expected to interact with
// this class directly. Boiler plate code for creating the abstract
// representation of the input pipeline and collecting common information has
// been added to the implementation of `DatasetBase` and `DatasetBaseIterator`
// respectively.
//
// In addition, `DatasetBaseIterator` provides wrappers that can be used for
// transformation-specific information collection. The `SetMetadata` wrapper
// can be used to pass arbitrary metadata to the modeling framework, while the
// `StartWork` and `StopWork` wrappers should be used to correctly account for
// processing time of multi-threaded transformation that yield the CPU; such
// transformations should invoke `StartWork()` when a transformation thread
// starts executing (e.g. when created or woken up) and `StopWork()` when a
// transformation thread stops executing (e.g. when returning or waiting).
class Node {
 public:
  // Arguments for `Node` constructor.
  struct Args {
    int64 id;
    string name;
    std::shared_ptr<Node> output;
    Model* model;  // Model owns Node
    string dataset_name;
    int64 parallelism; // -1 for autotune, we use dynamic value if so
    int64 cardinality;
    bool is_copy;
    AggregateDatasetMetric* aggregate_dataset_metric;  // Owned by Model
  };

  using Factory = std::function<std::shared_ptr<Node>(Args)>;
  using NodeVector = std::vector<std::shared_ptr<Node>>;
  using NodePairList =
      std::list<std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>>>;
  using ModelParameters =
      std::vector<std::pair<string, std::shared_ptr<Parameter>>>;
  using NodeValues = absl::flat_hash_map<string, double>;
  using ParameterGradients =
      absl::flat_hash_map<std::pair<string, string>, double>;

  explicit Node(Args args)
      : id_(args.id),
        name_(std::move(args.name)),
        autotune_(true),
        buffered_bytes_(0),
        buffered_elements_(0),
        bytes_consumed_(0),
        bytes_produced_(0),
        num_elements_(0),
        processing_time_(0),
#ifndef LIGHTWEIGHT_METRICS
        processing_time_wallclock_(0),
#endif
        processing_time_clock_(0),
#ifndef LIGHTWEIGHT_METRICS
        scheduling_delay_time_(0),
#endif
        start_time_(0),
        record_metrics_(true),
        parallelism_(args.parallelism),
        cardinality_(args.cardinality),
        metrics_(name_),
        output_(args.output.get()),
        model_(args.model),
        dataset_name_(std::move(args.dataset_name)),
        is_copy_(args.is_copy),
        aggregate_dataset_metric_(args.aggregate_dataset_metric) {}

  virtual ~Node() {
    // Clear the sub-nodes instead of relying on implicit shared pointer
    // destructor to avoid potential stack overflow when the tree is deep.
    std::deque<std::shared_ptr<Node>> queue;
    {
      mutex_lock l(mu_);
      while (inputs_.size() > 0) {
        queue.push_back(inputs_.front());
        inputs_.pop_front();
      }
    }
    while (!queue.empty()) {
      auto node = queue.back();
      queue.pop_back();
      {
        mutex_lock l(node->mu_);
        while (node->inputs_.size() > 0) {
          queue.push_back(node->inputs_.front());
          node->inputs_.pop_front();
        }
      }
    }

    FlushMetrics();
  }

  // Adds an input.
  void add_input(std::shared_ptr<Node> node) TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    inputs_.push_back(node);
  }

  // Increments the aggregate processing time by the given delta.
  void add_processing_time(int64_t delta) TF_LOCKS_EXCLUDED(mu_) {
    processing_time_ += delta;
#ifndef LIGHTWEIGHT_METRICS
    processing_time_wallclock_ += delta;
#endif
    aggregate_dataset_metric_->add_processing_time(delta);
#ifndef LIGHTWEIGHT_METRICS
    aggregate_dataset_metric_->add_udf_processing_time(delta);
#endif
  }

  // Increments the aggregate processing time by the given delta.
  void add_CPU_processing_time(int64 delta) TF_LOCKS_EXCLUDED(mu_) {
    processing_time_clock_ += delta;
    aggregate_dataset_metric_->add_processing_time_clock(delta);
#ifndef LIGHTWEIGHT_METRICS
    aggregate_dataset_metric_->add_udf_processing_time_clock(delta);
#endif
  }

  // Increments the aggregate scheduling delay time by the given delta.
  void add_scheduling_delay_time(int64 delta) TF_LOCKS_EXCLUDED(mu_) {
#ifndef LIGHTWEIGHT_METRICS
    scheduling_delay_time_ += delta;
    aggregate_dataset_metric_->add_scheduling_delay_time(delta);
#endif
  }

  // Returns an indication whether autotuning is enabled for this node.
  bool autotune() const TF_LOCKS_EXCLUDED(mu_) {
    return autotune_;
  }

  // Returns an indication of whether this node is cheap enough to warrant
  // bypassing some metrics
  inline bool is_cheap() const {
    if (!dynamic_tuning) {
      return false;
    }
    constexpr int64 min_elements = 256; // arbitrary power of two (unnecessary)
    // 16 us is about as fast as a cache can go
    constexpr int64 min_ns_per_element = 65536; // power of two
    bool cheapness = ((num_elements_ > min_elements)
         && (processing_time_clock_ < (min_ns_per_element * num_elements_)));
    return cheapness;
  }

  // Returns the number of bytes stored in this node's buffer.
  int64 buffered_bytes() const TF_LOCKS_EXCLUDED(mu_) {
    return buffered_bytes_;
  }

  // Returns the number of elements stored in this node's buffer.
  int64 buffered_elements() const TF_LOCKS_EXCLUDED(mu_) {
    return buffered_elements_;
  }

  // Returns the number of bytes consumed by the node.
  int64 bytes_consumed() const TF_LOCKS_EXCLUDED(mu_) {
    return bytes_consumed_;
  }

  // Returns the number of bytes produced by the node.
  int64 bytes_produced() const TF_LOCKS_EXCLUDED(mu_) {
    return bytes_produced_;
  }

  int64 parallelism() const TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return Parallelism();
  }

  // Indicates whether the node has tunable parameters.
  bool has_tunable_parameters() const TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    for (const auto& pair : parameters_) {
      if (pair.second->state->tunable) return true;
    }
    return false;
  }

  // Returns the unique node ID.
  int64 id() const TF_LOCKS_EXCLUDED(mu_) { return id_; }

  // Returns the node inputs.
  std::list<std::shared_ptr<Node>> inputs() const TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return inputs_;
  }

  // Returns a longer node name that is guaranteed to be unique.
  string long_name() const { return strings::StrCat(name_, "(id:", id_, ")"); }

  // Returns the node name.
  const string& name() const { return name_; }

  // Returns the number of elements produced by the node.
  int64 num_elements() const TF_LOCKS_EXCLUDED(mu_) {
    return num_elements_;
  }

  // Returns the node output.
  Node* output() const { return output_; }

  // Returns the parameter value.
  double parameter_value(const string& name) const TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return parameters_.at(name)->state->value;
  }

  // Returns the aggregate processing time.
  int64 processing_time() const TF_LOCKS_EXCLUDED(mu_) {
    return processing_time_;
  }

  // Returns the elapsed time.
  int64 elapsed_time(int64 time_nanos) const TF_LOCKS_EXCLUDED(mu_) {
    if (start_time_) {
      int64 elapsed_time = time_nanos - start_time_;
      return elapsed_time;
    } else {
      return 0;
    }
  }

  const string& dataset_name() const { return dataset_name_; }

  // Records that the node consumed the given number of bytes.
  void record_bytes_consumed(int64_t num_bytes) {
    bytes_consumed_ += num_bytes;
    aggregate_dataset_metric_->add_bytes_consumed(num_bytes);
  }

  // Records that the node produced the given number of bytes.
  void record_bytes_produced(int64_t num_bytes) {
    bytes_produced_ += num_bytes;
    aggregate_dataset_metric_->add_bytes_produced(num_bytes);
  }

  // Records the change in this node's buffer.
  void record_buffer_event(int64_t bytes_delta, int64_t elements_delta) {
    buffered_bytes_ += bytes_delta;
    buffered_elements_ += elements_delta;
#ifndef LIGHTWEIGHT_METRICS
    int64 average_size = 0;
    if (elements_delta != 0) {
      average_size = bytes_delta / elements_delta;
    }
    record_buffer_size(buffered_elements_);
    record_element_size(average_size);
#endif
  }

  // Records that the node produced an element.
  void record_element() TF_LOCKS_EXCLUDED(mu_) {
    num_elements_++;
    aggregate_dataset_metric_->increment_elements_produced();
  }

  // Records that the node consumed an element.
  // Mainly focused on dataset sources
  void record_element_consumed() TF_LOCKS_EXCLUDED(mu_) {
    aggregate_dataset_metric_->increment_elements_consumed();
  }

  // Records that an node was either produced or passed into this node.
  void record_element_start() TF_LOCKS_EXCLUDED(mu_) {
#ifndef LIGHTWEIGHT_METRICS
    aggregate_dataset_metric_->increment_active_elements();
#endif
  }

  // Records that an element was either thrown away or passed into next node.
  void record_element_exit() TF_LOCKS_EXCLUDED(mu_) {
#ifndef LIGHTWEIGHT_METRICS
    aggregate_dataset_metric_->decrement_active_elements();
#endif
  }

  // Records the number of active threads.
  void record_num_active_threads(int64 num_threads) {
#ifndef LIGHTWEIGHT_METRICS
    aggregate_dataset_metric_->add_num_active_threads_observation(num_threads);
#endif
  }

  // Records that inter-op parallelism is used.
  void record_inter_op_parallelism() {
#ifndef LIGHTWEIGHT_METRICS
    aggregate_dataset_metric_->set_inter_op_parallelism();
#endif
  }

  // Records that parallelism is used.
  void record_parallelism() {
    record_parallelism(Parallelism());
  }

  // Records that parallelism is used.
  void record_parallelism(int64 parallelism) {
    aggregate_dataset_metric_->set_parallelism(parallelism);
  }

  // Records buffer_size used.
  void record_buffer_size(int64 buffer_size) {
    aggregate_dataset_metric_->record_buffer_size(buffer_size);
  }

  // Records misc buffer_size used.
  void record_misc_buffer_size(int64 buffer_size) {
#ifndef LIGHTWEIGHT_METRICS
    aggregate_dataset_metric_->record_misc_buffer_size(buffer_size);
#endif
  }

  void record_element_size(int64 element_size) {
#ifndef LIGHTWEIGHT_METRICS
    aggregate_dataset_metric_->record_element_size(element_size);
#endif
  }

  // Records element ratio is used.
  void record_ratio() {
#ifndef LIGHTWEIGHT_METRICS
    aggregate_dataset_metric_->set_ratio(ElementRatio());
#endif
  }

  // Records the number of bytes read from disk
  void record_disk_bytes_read(int64 bytes_delta) {
    aggregate_dataset_metric_->add_disk_bytes_read(bytes_delta);
  }

 // Records that a node thread has started executing.
  template<bool is_wallclock=default_is_wallclock, bool is_CPU=default_is_CPU>
  void record_start(int64_t time_nanos) TF_LOCKS_EXCLUDED(mu_) {
    record_start_processing(time_nanos);
    if (is_wallclock) {
      record_start_wallclock(time_nanos);
    }
    if (is_CPU && !is_cheap()) {
      record_start_CPU();
    }
  }

  template<bool is_wallclock=default_is_wallclock, bool is_CPU=default_is_CPU>
  void record_start(int64_t time_nanos, int64_t thread_time_nanos) TF_LOCKS_EXCLUDED(mu_) {
    record_start_processing(time_nanos);
    if (is_wallclock) {
      record_start_wallclock(time_nanos);
    }
    if (is_CPU && !is_cheap()) {
      record_start_CPU(thread_time_nanos);
    }
  }

  template<bool is_wallclock=default_is_wallclock, bool is_CPU=default_is_CPU>
  void record_start_and_wait(int64_t time_nanos, int64_t thread_time_nanos) TF_LOCKS_EXCLUDED(mu_) {
    record_start<is_wallclock,is_CPU>(time_nanos, thread_time_nanos);
#ifndef LIGHTWEIGHT_METRICS
    record_stop_wait(time_nanos);
#endif
  }

  // Records that a node thread has started executing (wallclock time).
  void record_start_processing(int64_t time_nanos) TF_LOCKS_EXCLUDED(mu_) {
    DCHECK_EQ(work_start_, 0);
    work_start_ = time_nanos;
    if (TF_PREDICT_FALSE(!start_time_)) {
      start_time_ = time_nanos;
    }
  }

  // Records that a node thread has started executing (wallclock time).
  void record_start_wallclock(int64_t time_nanos) TF_LOCKS_EXCLUDED(mu_) {
#ifndef LIGHTWEIGHT_METRICS
    DCHECK_EQ(work_start_wallclock_, 0);
    work_start_wallclock_ = time_nanos;
#endif
  }

  // Records that a node thread has started executing (CPU time).
  void record_start_CPU() TF_LOCKS_EXCLUDED(mu_) {
    record_start_CPU(ThreadNowNanos());
  }

  // Records that a node thread has started executing (CPU time).
  void record_start_CPU(int64_t thread_time_nanos) TF_LOCKS_EXCLUDED(mu_) {
    DCHECK_EQ(work_start_clock_, 0);
    work_start_clock_ = thread_time_nanos;
  }

  // Records that a node thread has stopped executing.
  template<bool is_wallclock=default_is_wallclock, bool is_CPU=default_is_CPU>
  void record_stop(int64_t time_nanos) TF_LOCKS_EXCLUDED(mu_) {
    record_stop_processing(time_nanos);
    if (is_wallclock) {
      record_stop_wallclock(time_nanos);
    }
    if (is_CPU && !is_cheap()) {
      record_stop_CPU();
    }
  }

  template<bool is_wallclock=default_is_wallclock, bool is_CPU=default_is_CPU>
  void record_stop(int64_t time_nanos, int64_t thread_time_nanos) TF_LOCKS_EXCLUDED(mu_) {
    record_stop_processing(time_nanos);
    if (is_wallclock) {
      record_stop_wallclock(time_nanos);
    }
    if (is_CPU && !is_cheap()) {
      record_stop_CPU(thread_time_nanos);
    }
  }

  template<bool is_wallclock=default_is_wallclock, bool is_CPU=default_is_CPU>
  void record_stop_and_wait(int64_t time_nanos, int64_t thread_time_nanos) TF_LOCKS_EXCLUDED(mu_) {
    record_stop<is_wallclock,is_CPU>(time_nanos, thread_time_nanos);
#ifndef LIGHTWEIGHT_METRICS
    record_start_wait(time_nanos);
#endif
  }

  // Records that a node thread has stopped executing (processing time).
  void record_stop_processing(int64_t time_nanos) TF_LOCKS_EXCLUDED(mu_) {
    // TODO(jsimsa): Use DCHECK_NE(work_start_, 0) here.
    if (TF_PREDICT_TRUE(work_start_ != 0)) {
      auto delta = time_nanos - work_start_;
#ifndef LIGHTWEIGHT_METRICS
      safe_processing_time_increment(processing_time_, delta, TF_FUNCTION_NAME);
#else
      processing_time_ += delta;
#endif
      aggregate_dataset_metric_->add_processing_time(delta);
      work_start_ = 0;
    } else {
      VLOG(1) << "Encountered a stop event without a matching start event." <<
        TF_FUNCTION_NAME << ":" << dataset_name();
    }
  }

  // Records that a node thread has stopped executing (wallclock time).
  void record_stop_wallclock(int64_t time_nanos) TF_LOCKS_EXCLUDED(mu_) {
#ifndef LIGHTWEIGHT_METRICS
    // TODO(jsimsa): Use DCHECK_NE(work_start_, 0) here.
    if (TF_PREDICT_TRUE(work_start_wallclock_ != 0)) {
      auto delta = time_nanos - work_start_wallclock_;
      safe_processing_time_increment(processing_time_wallclock_, delta,
          TF_FUNCTION_NAME);
      work_start_wallclock_ = 0;
    } else {
      VLOG(1) << "Encountered a stop event without a matching start event." <<
        TF_FUNCTION_NAME << ":" << dataset_name();
    }
#endif
  }

  // Records that a node thread has stopped executing (CPU time).
  void record_stop_CPU() TF_LOCKS_EXCLUDED(mu_) {
    record_stop_CPU(ThreadNowNanos());
  }

  // Records that a node thread has stopped executing (CPU time).
  inline void record_stop_CPU(int64_t thread_time_nanos) TF_LOCKS_EXCLUDED(mu_) {
    const int64_t work_end_clock_ = thread_time_nanos;
    if (TF_PREDICT_TRUE(work_start_clock_ != 0)) {
      auto delta = work_end_clock_ - work_start_clock_;
#ifndef LIGHTWEIGHT_METRICS
      safe_processing_time_increment(processing_time_clock_, delta, TF_FUNCTION_NAME);
#else
      processing_time_clock_ += delta;
#endif
      aggregate_dataset_metric_->add_processing_time_clock(delta);
      work_start_clock_ = 0;
    } else {
      VLOG(1) << "Encountered a stop event without a matching start event." <<
        TF_FUNCTION_NAME << ":" << dataset_name();
    }
  }

  void record_start_wait(int64_t time_nanos) TF_LOCKS_EXCLUDED(mu_) {
#ifndef LIGHTWEIGHT_METRICS
    DCHECK_EQ(work_start_wait_, 0);
    work_start_wait_ = time_nanos;
#endif
  }

  void record_stop_wait(int64_t time_nanos) TF_LOCKS_EXCLUDED(mu_) {
#ifndef LIGHTWEIGHT_METRICS
    // NOTE(mkuchnik): Wait is very likely to have a missing work_start_wait_.
    if (work_start_wait_ != 0) {
      auto delta = time_nanos - work_start_wait_;
      aggregate_dataset_metric_->add_wait_time(delta);
      work_start_wait_ = 0;
    } else {
      VLOG(1) << "Encountered a stop event without a matching start event." <<
        TF_FUNCTION_NAME;
    }
#endif
  }

  // Returns whether work is currently being recorded, i.e. whether we are
  // currently between a `record_start` and a `record_stop`.
  bool is_recording() TF_LOCKS_EXCLUDED(mu_) { return work_start_ > 0; }

  // Removes an input.
  void remove_input(std::shared_ptr<Node> input) TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    inputs_.remove(input);
  }

  // Sets the value that determines whether autotuning is enabled for this node.
  void set_autotune(bool autotune) TF_LOCKS_EXCLUDED(mu_) {
    autotune_.store(autotune);
  }

  // Given the average time between events when the elements in the buffer are
  // produced (`producer_time`), the average time between events when elements
  // in the buffer are consumed (`consumer_time`) and the buffer size, the
  // method computes the expected time an consumer event will have to wait.
  //
  // The wait time is approximated as the product of the probability the buffer
  // will be empty and the time it takes to produce an element into the buffer.
  //
  // The formula used for computing the probability is derived by modeling the
  // problem as an M/M/1/K queue
  // (https://en.wikipedia.org/wiki/Birth%E2%80%93death_process#M/M/1/K_queue).
  //
  // Collects derivatives of `ComputeWaitTime` w.r.t `producer_time`,
  // `consumer_time' and `buffer_size` if the corresponding pointers are not
  // `nullptr`.
  static double ComputeWaitTime(const double& producer_time,
                                const double& consumer_time,
                                const double& buffer_size,
                                double* producer_time_derivative,
                                double* consumer_time_derivative,
                                double* buffer_size_derivative);

  // Collects tunable parameters in the subtree rooted in this node.
  ModelParameters CollectTunableParameters() const TF_LOCKS_EXCLUDED(mu_);

  // Returns a human-readable representation of this node.
  string DebugString() const TF_LOCKS_EXCLUDED(mu_);

  // Flushes the metrics recorded by this node.
  void FlushMetrics() TF_LOCKS_EXCLUDED(mu_);

  // Returns the per-element output time for this node and if `gradients` is not
  // `nullptr`, collects the output time gradient w.r.t. tunable parameters of
  // the subtree rooted in this node.
  double OutputTime(NodeValues* input_times,
                    ParameterGradients* gradients) const TF_LOCKS_EXCLUDED(mu_);

  // Returns a copy of this node, making a deep copy of its inputs and a
  // shallow copy of its tunable parameters.
  //
  // The purpose for this method is to allow the model optimization logic to
  // operate over immutable state while allowing concurrent model updates.
  std::shared_ptr<Node> Snapshot() const TF_LOCKS_EXCLUDED(mu_);

  // Returns the per-element processing time spent in this node.
  double SelfProcessingTime() const TF_LOCKS_EXCLUDED(mu_);

  // Returns the total number of bytes buffered in all nodes in the subtree for
  // which autotuning is enabled.
  double TotalBufferedBytes() const TF_LOCKS_EXCLUDED(mu_);

  // Collects the total buffer limit of all nodes in the subtree for which
  // autotuning is enabled. This number represents the amount of memory that
  // would be used by the subtree nodes if all of their buffers were full.
  double TotalMaximumBufferedBytes() const TF_LOCKS_EXCLUDED(mu_);

  // Returns the per-element CPU time spent in the subtree rooted in this node.
  // If `processing_times` is not `nullptr`, collects the per-element CPU time
  // spent in each node of the subtree.
  double TotalProcessingTime(NodeValues* processing_times)
      TF_LOCKS_EXCLUDED(mu_);

  // Returns the production rates and other statistics of each element in
  // subtree
  void CollectProductionStats(
      absl::flat_hash_map<string, std::shared_ptr<Node_Stats>>&
          production_stats,
      int64 time_nanos) TF_LOCKS_EXCLUDED(mu_);

  // Produces a proto for this node. Does not produce a proto for input nodes.
  virtual Status ToProto(ModelProto::Node* node_proto) const;

  // Restores a node from the proto. Does not restore input nodes.
  static Status FromProto(ModelProto::Node node_proto,
                          std::shared_ptr<Node> output,
                          std::shared_ptr<Node>* node);

 protected:
  // Used for (incrementally) recording metrics. The class is thread-safe.
  class Metrics {
   public:
    explicit Metrics(const string& name)
        : bytes_consumed_counter_(metrics::GetTFDataBytesConsumedCounter(name)),
          bytes_produced_counter_(metrics::GetTFDataBytesProducedCounter(name)),
          num_elements_counter_(metrics::GetTFDataElementsCounter(name)),
          recorded_bytes_consumed_(0),
          recorded_bytes_produced_(0),
          recorded_num_elements_(0) {}

    // Expects the total number of bytes consumed and records the delta since
    // last invocation.
    void record_bytes_consumed(int64_t total_bytes) {
      int64_t delta =
          total_bytes - recorded_bytes_consumed_.exchange(total_bytes);
      bytes_consumed_counter_->IncrementBy(delta);
    }

    // Expects the total number of bytes produced and records the delta since
    // last invocation.
    void record_bytes_produced(int64_t total_bytes) {
      int64_t delta =
          total_bytes - recorded_bytes_produced_.exchange(total_bytes);
      bytes_produced_counter_->IncrementBy(delta);
    }

    // Expects the total number of elements produced and records the delta since
    // last invocation.
    void record_num_elements(int64_t total_elements) {
      int64_t delta =
          total_elements - recorded_num_elements_.exchange(total_elements);
      num_elements_counter_->IncrementBy(delta);
    }

   private:
    monitoring::CounterCell* const bytes_consumed_counter_;
    monitoring::CounterCell* const bytes_produced_counter_;
    monitoring::CounterCell* const num_elements_counter_;
    std::atomic<int64> recorded_bytes_consumed_;
    std::atomic<int64> recorded_bytes_produced_;
    std::atomic<int64> recorded_num_elements_;
  };

  // Returns the number of inputs.
  int64 num_inputs() const TF_SHARED_LOCKS_REQUIRED(mu_) {
    int64_t num_inputs = 0;
    for (auto& input : inputs_) {
      // Inputs for which autotuning is disabled are excluded.
      if (input->autotune()) {
        ++num_inputs;
      }
    }
    return num_inputs;
  }

  // Creates a clone of this node.
  virtual std::shared_ptr<Node> Clone(std::shared_ptr<Node> output) const
      TF_SHARED_LOCKS_REQUIRED(mu_) = 0;

  // Returns the average size of an element buffered in this node.
  double AverageBufferedElementSize() const TF_SHARED_LOCKS_REQUIRED(mu_);

  // Returns the sum of per-element output time for the tunable inputs of this
  // node.
  double OutputTimeForInputs(const NodeValues& output_times) const
      TF_SHARED_LOCKS_REQUIRED(mu_);

  // Returns the sum of output time gradient w.r.t. input time for the tunable
  // inputs of this node.
  double OutputTimeGradientsForInputs(const NodeValues& output_time_gradients)
      const TF_SHARED_LOCKS_REQUIRED(mu_);

  // Computes the input time for this node and stores it in `input_times`.
  virtual void InputTimeLocked(NodeValues* input_times) const
      TF_SHARED_LOCKS_REQUIRED(mu_) = 0;

  // Computes the per-element output time for this node and stores it in
  // `output_times`. If `gradients` is not `nullptr`, computes the output time
  // gradient w.r.t. tunable parameters of the subtree rooted in this node and
  // stores it in `gradients`, also computes the output time gradient w.r.t.
  // input time and stores it in `output_time_gradients`.
  virtual void OutputTimeLocked(const NodeValues& input_times,
                                ParameterGradients* gradients,
                                NodeValues* output_times,
                                NodeValues* output_time_gradients) const
      TF_SHARED_LOCKS_REQUIRED(mu_) = 0;

  // Returns the sum of per-element processing time for the inputs of this node
  // by adding values for input nodes in `total_processing_times`. Processing
  // time for a given input is a weighted combination of a statistic based on
  // history of input processing time and the actual time. This is done to
  // improve accuracy of processing time estimation for newly created inputs.
  //
  // Uniform distribution of per-element processing times across different
  // inputs is assumed.
  double TotalProcessingTimeForInputs(const NodeValues& total_processing_times)
      TF_SHARED_LOCKS_REQUIRED(mu_);

  // Returns the per-element processing time spent in this node.
  double SelfProcessingTimeLocked() const TF_SHARED_LOCKS_REQUIRED(mu_);

  Rate production_rate_locked(int64 time_nanos) const
      TF_SHARED_LOCKS_REQUIRED(mu_) {
    if (start_time_) {
      if (time_nanos > 0) {
        int64 elapsed_time = time_nanos - start_time_;
        return Rate(num_elements_, elapsed_time);
      } else if (time_nanos == 0) {
        return Rate(num_elements_, processing_time_);
      } else {  // -1 is clock
        return Rate(num_elements_, processing_time_clock_);
      }
    } else {
      return Rate(0, 0);
    }
  }

  // Computes the per-element CPU time spent in the subtree rooted in this node
  // and stores it in `total_processing_times`. If `processing_times` is not
  // `nullptr`, collects the per-element CPU time spent in each node of the
  // subtree.
  virtual void TotalProcessingTimeLocked(NodeValues* processing_times,
                                         NodeValues* total_processing_times)
      TF_SHARED_LOCKS_REQUIRED(mu_) = 0;

  virtual double ElementRatio() const TF_SHARED_LOCKS_REQUIRED(mu_) {
    // TODO(jsimsa): The current implementation assumes that the number of input
    // elements consumed per output is the same across all inputs.
    if (inputs_.empty() || inputs_.front()->num_elements() == 0) {
      return -1.0;
    }
    std::shared_ptr<Node> input = inputs_.front();
    double ratio = static_cast<double>(input->num_elements()) /
                   static_cast<double>(num_elements_);
    return ratio;
  }

  virtual Rate ElementRate() const TF_SHARED_LOCKS_REQUIRED(mu_) {
    // TODO(jsimsa): The current implementation assumes that the number of input
    // elements consumed per output is the same across all inputs.
    std::shared_ptr<Node> input = inputs_.front();
    return Rate(input->num_elements(), num_elements_);
  }

  int64 Parallelism() const TF_SHARED_LOCKS_REQUIRED(mu_) {
    int64 parallelism = parallelism_;
    if (autotune_) {
      auto* parallelism_parameter = gtl::FindOrNull(parameters_, kParallelism);
      if (parallelism_parameter) {
        parallelism = (*parallelism_parameter)->value;
        if (!lightweight_metrics) {
          VLOG(3) << "Found value autotune for " << DebugString()
                  << " " << parallelism << std::endl;
          if (!autotune_) {
           std::cerr << "Parallelism exists but no autotune" << std::endl;
          }
        }
      }
    }
    return parallelism;
  }

  int64 Cardinality() const TF_SHARED_LOCKS_REQUIRED(mu_) {
    return cardinality_;
  }

  int64 BytesProduced() const TF_SHARED_LOCKS_REQUIRED(mu_) {
    return bytes_produced_;
  }

  int64 BytesConsumed() const TF_SHARED_LOCKS_REQUIRED(mu_) {
    return bytes_consumed_;
  }

  Node_Stats Stats(int64 time_nanos) const TF_SHARED_LOCKS_REQUIRED(mu_) {
    Rate abs_rate = production_rate_locked(time_nanos);
    Rate rate = production_rate_locked(0);
    Rate clock_rate = production_rate_locked(-1);
    double ratio = ElementRatio();
    int64 parallelism = Parallelism();
    Node_Stats stats;
    stats.elements_produced = rate.first;
    stats.wallclock_time = abs_rate.second;
    stats.processing_time = rate.second;
    stats.parallelism = parallelism;
    stats.ratio = ratio;
    stats.count = 1;
    stats.bytes_produced = BytesProduced();
    stats.bytes_consumed = BytesConsumed();
    stats.processing_time_clock = clock_rate.second;
    stats.cardinality = Cardinality();
    return stats;
  }

  // Returns a vector of nodes of the subtree rooted in this node. The nodes are
  // either in breadth-first search or reverse breadth-first search order
  // depending on the `order` argument. The nodes are collected based on the
  // results of the `collect_node` predicate: if the predicate returns `false`
  // for a given node, then the subtree rooted in this node is excluded. The
  // root node itself is not collected.
  NodeVector CollectNodes(TraversalOrder order,
                          bool collect_node(const std::shared_ptr<Node>)) const
      TF_SHARED_LOCKS_REQUIRED(mu_);

  // Collects tunable parameters in the subtree rooted in this node assuming
  // mutex locked.
  ModelParameters CollectTunableParametersLocked() const
      TF_SHARED_LOCKS_REQUIRED(mu_);

  // Collect tunable parameters on the nodes which have recorded elements.
  void CollectTunableParametersHelper(ModelParameters* parameters) const
      TF_SHARED_LOCKS_REQUIRED(mu_);

  // Build up debug string for the node and store in the debug strings map.
  void DebugStringHelper(absl::flat_hash_map<string, string>* debug_strings)
      const TF_SHARED_LOCKS_REQUIRED(mu_);

  // Copy the node and add the (input, copy) pairs to the NodePairList.
  std::shared_ptr<Node> SnapshotHelper(std::shared_ptr<Node> cloned_output,
                                       NodePairList* node_pairs) const;

  // Compute total buffered bytes for the node and store in the total bytes map.
  void TotalBufferedBytesHelper(NodeValues* total_bytes) const
      TF_SHARED_LOCKS_REQUIRED(mu_);

  // Compute total maximum buffered bytes for the node and store in the total
  // bytes map.
  void TotalMaximumBufferedBytesHelper(NodeValues* total_bytes) const
      TF_SHARED_LOCKS_REQUIRED(mu_);

  // Compute and return the maximum buffered bytes on the node itself. By
  // default non-tunable nodes are assumed not to buffer any bytes, so the
  // tunable nodes as subclasses are expected to override this method to ensure
  // that the optimization algorithm respects the memory budget.
  virtual double MaximumBufferedBytes() const TF_SHARED_LOCKS_REQUIRED(mu_);

  // Restores node from the proto. Note that this is not done recursively, i.e.
  // input nodes are not restored.
  static Status FromProtoHelper(ModelProto::Node node_proto,
                                std::shared_ptr<Node> node);

  // Stores the time passed to the last call to `Node::record_start()` on the
  // current thread.
  //
  // NOTE: This thread-local variable is shared between all instances of `Node`
  // on which the same thread calls `record_start()` or `record_stop()`. It
  // relies on the invariant that at most one `Node` can be "active" on a
  // particular thread at any time. Therefore if `n->record_start()` is called
  // on thread `t`, then `n->record_stop()` must be called before another call
  // to `Node::record_start()` (for any node).
  static thread_local int64_t work_start_;  // Will be initialized to zero.
  static thread_local int64_t work_start_wallclock_;  // Will be initialized to zero.
  static thread_local int64_t work_start_clock_;
  static thread_local int64_t work_start_wait_;

  mutable mutex mu_;
  const int64 id_;
  const string name_;

  // Indicates whether the subtree rooted in this node should be included in
  // autotuning. In particular, if this is `false`, then the subtree is excluded
  // from computation of output time and processing time.
  std::atomic<bool> autotune_;
  std::atomic<int64> buffered_bytes_;
  std::atomic<int64> buffered_elements_;
  std::atomic<int64> bytes_consumed_;
  std::atomic<int64> bytes_produced_;
  std::atomic<int64> num_elements_;
  std::atomic<int64> processing_time_;
  std::atomic<int64> processing_time_wallclock_;
  std::atomic<int64> processing_time_clock_;
#ifndef LIGHTWEIGHT_METRICS
  std::atomic<int64> scheduling_delay_time_;
#endif
  std::atomic<int64> start_time_;
  std::atomic<bool> record_metrics_;
  std::atomic<int64> parallelism_;
  std::atomic<int64> cardinality_;
  std::atomic<bool> is_copy_;
  Metrics metrics_;
  absl::flat_hash_map<string, std::shared_ptr<Parameter>> parameters_
      TF_GUARDED_BY(mu_);

  // Statistic of inputs processing time history.
  double input_processing_time_sum_ = 0.0L;
  int64 input_processing_time_count_ = 0;

  // Inputs of this node. These can represent an iterator created from the input
  // dataset but also other input iterators (e.g. created by the user-defined
  // functions of `flat_map` or `interleave`).
  std::list<std::shared_ptr<Node>> inputs_ TF_GUARDED_BY(mu_);

  // The reference to the output node is not owned so that deletion of a
  // node results in recursive deletion of the subtree rooted in the node.
  Node* const output_;  // NOTE(mkuchnik): This can be used to trace parents
  Model* const model_;
  const string dataset_name_;
  AggregateDatasetMetric* aggregate_dataset_metric_;
};

// InterleaveMany is used to model datasets whose inputs are used to create
// datasets whose elements are then interleaved.
std::shared_ptr<Node> MakeInterleaveManyNode(Node::Args args);

// AsyncInterleaveMany nodes are the asynchronous version of InterleaveMany
// nodes.
std::shared_ptr<Node> MakeAsyncInterleaveManyNode(
    Node::Args args, std::vector<std::shared_ptr<Parameter>> parameters);

// KnownMany nodes model datasets that synchronously consume known number of
// input element per output element.
std::shared_ptr<Node> MakeKnownRatioNode(Node::Args args, double ratio);

// AsyncKnownRatio nodes are the asynchronous version of KnownRate nodes.
std::shared_ptr<Node> MakeAsyncKnownRatioNode(
    Node::Args args, double ratio, double memory_ratio,
    std::vector<std::shared_ptr<Parameter>> parameters);

std::shared_ptr<Node> MakeAsyncKnownRatioNode(
    Node::Args args, double ratio,
    std::vector<std::shared_ptr<Parameter>> parameters);

// Source nodes represent data sources.
std::shared_ptr<Node> MakeSourceNode(Node::Args args);

// UnknownMany nodes represent datasets that synchronously consume an
// unknown number of input elements per output.
//
// Unlike KnownRatio nodes which expect the ratio between inputs and outputs is
// specified as a parameter, UnknownRatio estimates the ratio empirically.
std::shared_ptr<Node> MakeUnknownRatioNode(Node::Args args);

// Unknown nodes represent datasets for which we do not have a model. It acts
// as pass-through between inputs and output.
std::shared_ptr<Node> MakeUnknownNode(Node::Args args);

// Abstract representation of a TensorFlow input pipeline that can be used
// for collecting runtime information and optimizing performance. It collects
// runtime information about execution of the input pipeline that is used to
// create a performance model, which is in turn used to identify optimal values
// of tunable parameters.
//
// Developers of tf.data transformations are not expected to interact with this
// class directly. Boiler plate code for creating the abstract representation of
// the input pipeline and collecting runtime information has been added to the
// implementation of `DatasetBase` and `DatasetBaseIterator` respectively.
class Model {
 public:
  using OptimizationParams = ModelProto::OptimizationParams;
  using ModelParameters = Node::ModelParameters;
  using NodeValues = Node::NodeValues;
  using ParameterGradients = Node::ParameterGradients;

  // Creates a new model.
  Model();

  // Creates a new model.
  Model(bool collect_resource_usage);

  ~Model();

  // Indicates whether to collect resource usage.
  bool collect_resource_usage() const { return collect_resource_usage_; }

  // Indicates whether to collect (expensive to monitor) resource usage.
  bool collect_heavy_resource_usage() const {
    return collect_heavy_resource_usage_;
  }

  // Returns a pointer to the model's output node.
  const std::shared_ptr<Node> output() {
    mutex_lock l(mu_);
    return output_;
  }

  // Adds a node with the given name and given parent.
  // dataset_name is from DatasetBase
  void AddNode(Node::Factory factory, const string& name,
               std::shared_ptr<Node> parent, std::shared_ptr<Node>* out_node,
               const string& dataset_name, const int64 parallelism,
               const int64 cardinality)
      TF_LOCKS_EXCLUDED(mu_);

  // Returns a human-readable string representation of the model. This method
  // can be invoked automatically by monitoring gauges and to avoid frequent
  // recomputation, the implementation caches the result.
  std::string DebugString();

  // Uses the given algorithm and resource budgets to periodically perform the
  // autotuning optimization.
  //
  // To terminate the execution of the optimization loop, the caller needs to
  // invoke `cancellation_mgr->StartCancel()`.
  Status OptimizeLoop(AutotuneAlgorithm algorithm, int64_t cpu_budget,
                      int64_t ram_budget,
                      CancellationManager* cancellation_manager);

  // Uses the given algorithm and resource budgets to perform the autotuning
  // optimization.
  void Optimize(AutotuneAlgorithm algorithm, int64_t cpu_budget,
                int64_t ram_budget, double model_input_time,
                CancellationManager* cancellation_manager);

  // Collects the output time and if `gradients` is not `nullptr`, the output
  // time gradient w.r.t. tunable parameters of the subtree rooted in the given
  // node.
  double OutputTime(std::shared_ptr<Node> node, double model_input_time,
                    ParameterGradients* gradients);

  // Removes the given node.
  void RemoveNode(std::shared_ptr<Node> node) TF_LOCKS_EXCLUDED(mu_);

  // Collects statistics on the data pipeline.
  absl::flat_hash_map<string, std::shared_ptr<Node_Stats>>
  CollectProductionStats(int64 time_nanos) TF_LOCKS_EXCLUDED(mu_);

  absl::flat_hash_map<string, std::shared_ptr<AggregateDatasetMetric>>
  CollectDatasetProductionStats() TF_LOCKS_EXCLUDED(mu_);

  // TODO(mkuchnik): clean up. This can be moved to IteratorContext.
  GraphDef graph_def_;

  // Produces a proto for this model.
  Status ToProto(ModelProto* model_proto);

  // Restores a model from the proto.
  static Status FromProto(ModelProto model_proto,
                          std::unique_ptr<Model>* model);

  // Flushes metrics recorded by the model.
  void FlushMetrics() TF_LOCKS_EXCLUDED(mu_);

  // Records that the file produced this many bytes total.
  void RecordFileRead(const std::string& filename, int64 size) {
    aggregate_system_metrics_->add_filename_size(filename, size);
  }

  void RecordOutputTime() {
    std::shared_ptr<Node> snapshot;
    {
      tf_shared_lock l(mu_);
      snapshot = output_->Snapshot();
    }
    // TODO(mkuchnik): Get input time from options
    double model_input_time = 0.0;
    double output_time = OutputTime(snapshot, model_input_time,
                                    /*gradients=*/nullptr);
    aggregate_system_metrics_->record_output_time(output_time);
  }

  absl::flat_hash_map<std::string, int64> CollectFileReads() {
    return aggregate_system_metrics_->filename_sizes();
  };

  double CollectOutputTime() {
    RecordOutputTime();
    return aggregate_system_metrics_->output_time();
  };

  // Saves this model with a given snapshot and its optimization parameters to a
  // file. Note that the file directory must already exist.
  Status Save(const string& fname, std::shared_ptr<Node> snapshot,
              const OptimizationParams& optimization_params);

  // Loads a model and its optimization parameters from a file with the given
  // name.
  static Status Load(const string& fname, std::unique_ptr<Model>* model,
                     OptimizationParams* optimization_params);

 private:
  static constexpr int64_t kOptimizationPeriodMinMs = 10;
  static constexpr int64_t kOptimizationPeriodMaxMs =
      60 * EnvTime::kSecondsToMillis;

  // Collects tunable parameters in the tree rooted in the given node, returning
  // a vector which contains pairs of node names and tunable parameters.
  ModelParameters CollectTunableParameters(std::shared_ptr<Node> node);

  // Collects production rates rooted in the given node.
  absl::flat_hash_map<string, std::shared_ptr<Node_Stats>>
  CollectProductionStatsInternal(std::shared_ptr<Node> node, int64 time_nanos);

  // This optimization algorithm starts by setting all tunable parallelism
  // parameters to the minimum value. It then repeatedly identifies the
  // parameter whose increase in parallelism decreases the output time the most.
  // This process is repeated until all parameters reach their maximum values or
  // the projected output time is less than or equal to the processing time
  // needed to produce an element divided by CPU budget.
  void OptimizeHillClimb(std::shared_ptr<Node> snapshot,
                         const OptimizationParams& optimization_params,
                         CancellationManager* cancellation_manager);

  // This optimization algorithm starts by setting all tunable parallelism
  // parameters to the minimum value. It then improves current parameters by
  // making a step in the direction opposite to the gradient of `OutputTime` and
  // projecting resulting values on the feasible intervals. Improvement step is
  // repeated until either the output time improvement is smaller than threshold
  // value or the output time is less than the processing time needed to produce
  // an element divided by CPU budget.
  void OptimizeGradientDescent(std::shared_ptr<Node> snapshot,
                               const OptimizationParams& optimization_params,
                               CancellationManager* cancellation_manager);

  // Determines if we should stop the gradient descent optimization iterations
  // based on number of increasable parameters, CPU budget, RAM budget and
  // current resource usage.
  bool ShouldStop(int64_t cpu_budget, int64_t ram_budget,
                  const ModelParameters& parameters,
                  const ModelParameters& parallelism_parameters,
                  const ModelParameters& buffer_size_parameters,
                  std::shared_ptr<Node> snapshot, bool* cpu_budget_reached);

  // Collects the processing time for the given node.
  double TotalProcessingTime(std::shared_ptr<Node> node);

  // Collects the total number of bytes buffered in all nodes in the subtree
  // rooted in the given node for which autotuning is enabled.
  double TotalBufferedBytes(std::shared_ptr<Node> node);

  // Collects the total buffer limit of all nodes in the subtree rooted in the
  // given node for which autotuning is enabled. This number represents the
  // amount of memory that would be used by the subtree nodes if all of their
  // buffers were full.
  double TotalMaximumBufferedBytes(std::shared_ptr<Node> node);

  // Return reference to dataset_name's metric.
  AggregateDatasetMetric* get_dataset_aggregate_metric(
      const string& dataset_name);

  // Used for coordination between different input pipeline threads. Exclusive
  // access is required only when adding or removing nodes. Concurrent access to
  // existing nodes is protected by a node mutex.
  mutex mu_;
  // Used for coordinating the optimization loop and model modifications.
  condition_variable optimize_cond_var_;
  int64 id_counter_ TF_GUARDED_BY(mu_) = 1;
  std::shared_ptr<Node> output_ TF_GUARDED_BY(mu_) = nullptr;

  // Indicates whether the modeling framework should collect resource usage
  // (e.g. CPU, memory). The logic for collecting this information assumes that
  // the collection is not repeatedly disabled and enabled. As a consequence,
  // the implementation starts collecting resource usage when it encounters a
  // tunable parameter (because the information is used for tuning the value of
  // the parameter) and never stops.
  std::atomic<bool> collect_resource_usage_;
  std::atomic<bool> collect_heavy_resource_usage_; // enables all metric collection

  // Used for aggregating ephemeral node's statistics for analysis
  // NOTE(mkuchnik): References to values are safe after rehash, though
  // iterators are unsafe.
  std::unordered_map<string, std::shared_ptr<AggregateDatasetMetric> >
    aggregate_metrics_ TF_GUARDED_BY(mu_);

  std::shared_ptr<AggregateSystemMetric>
    aggregate_system_metrics_ TF_GUARDED_BY(mu);

  // Determines the time the optimization loop should wait between
  // running optimizations.
  int64 optimization_period_ms_ TF_GUARDED_BY(mu_);

  // Gauge cell that can be used to collect the state of the model.
  monitoring::GaugeCell<std::function<std::string()>>* model_gauge_cell_ =
      nullptr;
  // Time use for rate limitting the recomputation of human-readable string
  // represention of the model.
  absl::Time cache_until_ = absl::InfinitePast();
  // Cached result of the `DebugString()` invocation used to implement rate
  // limitting of the computation.
  std::string cached_debug_string_ = "";
};

}  // namespace model
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_MODEL_H_
