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
#include "tensorflow/core/kernels/data/model_dataset_op.h"

#include "tensorflow/core/framework/cancellation.h"

// On mobile we do not provide model dataset op because not all of its
// dependencies are available there. The op is replaced with a no-op.
#if !defined(IS_MOBILE_PLATFORM)
#include "absl/memory/memory.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/util/ptr_util.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/data/analysis_utils.h"

// NOTE(mkuchnik): Enable to turn on default behavior
// #define MODEL_DATASET_COMPAT

namespace tensorflow {
namespace data {
namespace {

// Default share of available RAM that can be used by model's internal buffers.
constexpr double kRamBudgetShare = 0.5;
constexpr int64 kMaxSpanCollectionSize = 20; // 20 GetNext() calls
constexpr int64 kDefaultDumpPeriodMs = 10000; // 10 seconds

}  // namespace

/* static */ constexpr const char* const ModelDatasetOp::kDatasetType;
/* static */ constexpr const char* const ModelDatasetOp::kDatasetOp;
/* static */ constexpr const char* const ModelDatasetOp::kAlgorithm;
/* static */ constexpr const char* const ModelDatasetOp::kCpuBudget;
/* static */ constexpr const char* const ModelDatasetOp::kRamBudget;
/* static */ constexpr const char* const ModelDatasetOp::kStatsFilename;
/* static */ constexpr const char* const ModelDatasetOp::kStatsDumpPeriod;
/* static */ constexpr const char* const ModelDatasetOp::kSpanCollectionInterval;

class ModelDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          model::AutotuneAlgorithm algorithm, int64 cpu_budget,
          int64 ram_budget,
          const std::string& stats_filename,
          int64 stats_dump_period,
          int64 span_collection_interval)
      : Dataset(DatasetContext(ctx), input, algorithm, cpu_budget, ram_budget,
                stats_filename, stats_dump_period, span_collection_interval) {
        if (!stats_filename.empty()) {
          VLOG(0) << "Stats filename being used: '" << stats_filename_ << "'";
          VLOG(0) << "Lightweight tracing: " << model::lightweight_metrics
                  << ", CPU tracing: " << model::CPU_based_metrics;
        }
  }

  Dataset(DatasetContext&& ctx, const DatasetBase* input,
          model::AutotuneAlgorithm algorithm, int64_t cpu_budget,
          int64_t ram_budget,
          const std::string& stats_filename,
          int64 stats_dump_period,
          int64 span_collection_interval)
      : DatasetBase(std::move(ctx)),
        input_(input),
        algorithm_(algorithm),
        cpu_budget_(cpu_budget),
        ram_budget_(ram_budget),
        stats_filename_(std::move(stats_filename)),
        stats_dump_period_(std::move(stats_dump_period)),
        span_collection_interval_(span_collection_interval),
        traceme_metadata_(
            {{"algorithm", algorithm == model::AutotuneAlgorithm::HILL_CLIMB
                               ? "hill climb"
                               : "gradient descent"},
             {"cpu_budget",
              strings::Printf("%lld", static_cast<long long>(cpu_budget))},
             {"ram_budget",
              strings::Printf("%lldB", static_cast<long long>(ram_budget))}}) {
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); VLOG(0) << "Del ModelDataset"; }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(
        Iterator::Params{this, strings::StrCat(prefix, "::Model")});
  }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }
  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }

  string DebugString() const override { return "ModelDatasetOp::Dataset"; }

  int64 Cardinality() const override { return input_->Cardinality(); }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node}, output));
    AttrValue algorithm_attr;
    b->BuildAttrValue(static_cast<int64>(algorithm_), &algorithm_attr);
    AttrValue cpu_budget_attr;
    b->BuildAttrValue(cpu_budget_, &cpu_budget_attr);
    AttrValue ram_budget_attr;
    b->BuildAttrValue(ram_budget_, &ram_budget_attr);
    AttrValue stats_filename_attr;
    b->BuildAttrValue(stats_filename_, &stats_filename_attr);
    AttrValue stats_dump_period_attr;
    b->BuildAttrValue(stats_dump_period_,
                      &stats_dump_period_attr);
    AttrValue span_collection_interval_attr;
    b->BuildAttrValue(span_collection_interval_,
                      &span_collection_interval_attr);

    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {input_graph_node},
                      {std::make_pair(kAlgorithm, algorithm_attr),
                       std::make_pair(kCpuBudget, cpu_budget_attr),
                       std::make_pair(kRamBudget, ram_budget_attr),
                       std::make_pair(kStatsFilename, stats_filename_attr),
                       std::make_pair(kStatsDumpPeriod,
                           stats_dump_period_attr),
                       std::make_pair(kSpanCollectionInterval,
                           span_collection_interval_attr),
                       },
                      output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          cpu_budget_(dataset()->cpu_budget_ == 0 ? port::NumSchedulableCPUs()
                                                  : dataset()->cpu_budget_),
          ram_budget_(dataset()->ram_budget_ == 0
                          ? kRamBudgetShare * port::AvailableRam()
                          : dataset()->ram_budget_),
          stats_filename_(dataset()->stats_filename_),
          stats_dump_period_(dataset()->stats_dump_period_ == 0
                             ? kDefaultDumpPeriodMs
                             : dataset()->stats_dump_period_),
          span_collection_interval_(dataset()->span_collection_interval_) {
      const bool force_modeling = !stats_filename_.empty();
      cancellation_manager_ = absl::make_unique<CancellationManager>();
      dump_cancellation_manager_ = absl::make_unique<CancellationManager>();
      model_ = std::make_shared<model::Model>(force_modeling);
    }

    ~Iterator() override {
      // TODO(mkuchnik): This may race
      cancellation_manager_->StartCancel();
      dump_cancellation_manager_->StartCancel();
    }

    Status Initialize(IteratorContext* ctx) override {
      return dataset()->input_->MakeIterator(IteratorContext(CreateParams(ctx)),
                                             this, prefix(), &input_impl_);
    }


    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      if (!ctx->model()) {
        mutex_lock l(mu_);
        // TODO(mkuchnik): Start thread if AUTOTUNE on only
        TF_RETURN_IF_ERROR(EnsureOptimizationLoopThreadStarted(ctx));
        if (!stats_filename_.empty()) {
          TF_RETURN_IF_ERROR(EnsureDumpThreadStarted(ctx));
        }
      }
      // TODO(mkuchnik): Move RecordInput/Output to mainline implementation
      RecordInput();
      const auto s = input_impl_->GetNext(IteratorContext(CreateParams(ctx)),
                                          out_tensors, end_of_sequence);
      if (!model::lightweight_metrics) {
        mutex_lock l(mu_);
        RecordOutput();
      }
      return s;
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       /*ratio=*/1);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      return SaveInput(ctx, writer, input_impl_);
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      return RestoreInput(IteratorContext(CreateParams(ctx)), reader,
                          input_impl_);
    }

    TraceMeMetadata GetTraceMeMetadata() const override {
      return dataset()->traceme_metadata_;
    }

   private:
    Status dump_stats(int64 time_nanos,
                      const std::shared_ptr<IteratorContext> ctx) {
      RootDatasetStats root_stats;
      {
        mutex_lock l(mu_);
        root_stats.average_duration = AverageDuration();
        root_stats.variance_duration = VarianceDuration();
        root_stats.average_wallclock_duration = AverageWallclockDuration();
      }
      root_stats.start_time = start_time_;
      root_stats.process_start_time = process_start_time_;
      return DumpModelStats(model_.get(), stats_filename_, root_stats,
          time_nanos, ctx);
    }

    Status EnsureDumpThreadStarted(IteratorContext* ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (!dump_thread_) {
        std::shared_ptr<IteratorContext> new_ctx =
            std::make_shared<IteratorContext>(*ctx);
        dump_thread_ = ctx->StartThread(
            "tf_data_dump_stats", [this, new_ctx]() {
            Status status = DumpThread(new_ctx,
                                       dump_cancellation_manager_.get());
            if (!status.ok()) {
              LOG(WARNING) << "Dump Stats loop failed: " << status.ToString();
            }
            VLOG(0) << "Dump exit";
            cond_var_.notify_all(); // TODO(mkuchnik): necessary?
        });
      }
      return Status::OK();
    }

    void DumpStats(const std::shared_ptr<IteratorContext> ctx) {
       int64 time_nanos = EnvTime::NowNanos();
       dump_stats(time_nanos, ctx);
    }

    void RecordInput() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (!model::lightweight_metrics) {
        int64 time_nanos = EnvTime::NowNanos();
        if (last_output_time_ != 0) {
          DCHECK_LE(last_output_time_, time_nanos);
          const int64 duration = time_nanos - last_output_time_;
          const int64 duration_us = duration / 1000;
          double m_k1 = AverageDuration();
          input_time_ += duration;
          num_input_events_++;
          double m_k = AverageDuration();
          var_input_time_v_ += (duration_us - m_k1) * (duration_us - m_k);
        }
        if (start_time_ == 0) {
          start_time_ = time_nanos;
        }
      } else {
        if (start_time_ == 0) {
          start_time_ = EnvTime::NowNanos();
        }
      }
      if (process_start_time_ == 0) {
        process_start_time_ = ProcessExecutionTime::NowNanos();
      }
    }

    double AverageWallclockDuration() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      auto elapsed_time = (last_output_time_ - start_time_) / 1000.;
      return static_cast<double>(elapsed_time) / num_input_events_;
    }

    double AverageDuration() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (num_input_events_ <= 0) {
        return 0.0;
      }
      auto elapsed_time = input_time_ / 1000.;
      return static_cast<double>(elapsed_time) / num_input_events_;
    }

    double VarianceDuration() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (num_input_events_ <= 1) {
        return 0.0;
      }
      return static_cast<double>(var_input_time_v_) / (num_input_events_ - 1);
    }

    IteratorContext::Params CreateParams(IteratorContext* ctx) {
      IteratorContext::Params params(ctx);
      if (!ctx->model()) {
        params.model = model_;
      }
      return params;
    }

    Status EnsureOptimizationLoopThreadStarted(IteratorContext* ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (!model_thread_) {
        model_thread_ = ctx->StartThread("tf_data_model", [this]() {
          VLOG(0) << "Optimize start";
          Status status =
              model_->OptimizeLoop(dataset()->algorithm_, cpu_budget_,
                                   ram_budget_, cancellation_manager_.get());
          if (!status.ok()) {
            LOG(WARNING) << "Optimization loop failed: " << status.ToString();
          }
          VLOG(0) << "Optimize exit";
          cond_var_.notify_all(); // TODO(mkuchnik): necessary?
        });
      }
      return Status::OK();
    }

    Status DumpThread(const std::shared_ptr<IteratorContext> ctx,
                      CancellationManager* cancellation_manager) {
      std::function<void()> unused;
      TF_RETURN_IF_ERROR(RegisterCancellationCallback(
          cancellation_manager,
          [this]() {
            mutex_lock l(mu_);
            cond_var_.notify_all();
          },
          /*deregister_fn=*/&unused));
      int64 last_dump_ms = 0;
      int64 current_time_ms = EnvTime::NowMicros() / EnvTime::kMillisToMicros;
      while (true) {
        {
          mutex_lock l(mu_);
          while (!cancellation_manager->IsCancelled() &&
                 (last_dump_ms + stats_dump_period_) > current_time_ms) {
            auto wait_ms =
                last_dump_ms + stats_dump_period_ - current_time_ms;
            cond_var_.wait_for(l, std::chrono::milliseconds(wait_ms));
            current_time_ms = EnvTime::NowMicros() / EnvTime::kMillisToMicros;
          }
          if (cancellation_manager->IsCancelled()) {
            return Status::OK();
          }
        }
        // NOTE(mkuchnik): The stats are only loosely in sync at dump time.
        model_->FlushMetrics();
        DumpStats(ctx);
        last_dump_ms = EnvTime::NowMicros() / EnvTime::kMillisToMicros;
      }
    }

    void RecordOutput() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (!model::lightweight_metrics) {
        last_output_time_ = EnvTime::NowNanos();
      }
    }

    mutex mu_;
    std::shared_ptr<model::Model> model_;
    condition_variable cond_var_; // TODO(mkuchnik): remove
    // Controls cancellation of `model_thread_`. Must be ordered before
    // `model_thread_` so that `model_thread_` is destroyed first.
    std::unique_ptr<CancellationManager> cancellation_manager_;
    std::unique_ptr<CancellationManager> dump_cancellation_manager_;
    std::unique_ptr<Thread> model_thread_ TF_GUARDED_BY(mu_);
    std::unique_ptr<Thread> dump_thread_ TF_GUARDED_BY(mu_);
    std::unique_ptr<IteratorBase> input_impl_;
    const int64 cpu_budget_;
    const int64 ram_budget_;
    const std::string stats_filename_;
    const int64 stats_dump_period_;
    const int64 span_collection_interval_;
    int64 input_time_ TF_GUARDED_BY(mu_) = 0; // TODO(mkuchnik): Remove
    int64 var_input_time_v_ TF_GUARDED_BY(mu_) = 0;
    int64 start_time_ TF_GUARDED_BY(mu_) = 0;
    int64 process_start_time_ TF_GUARDED_BY(mu_) = 0;
    int64 last_output_time_ TF_GUARDED_BY(mu_) = 0; // TODO(mkuchnik): Remove
    int64 num_input_events_ TF_GUARDED_BY(mu_) = 0; // TODO(mkuchnik): Remove
    int64 bytes_read_ TF_GUARDED_BY(mu_) = 0;
  };

  const DatasetBase* input_;
  const model::AutotuneAlgorithm algorithm_;
  const int64 cpu_budget_;
  const int64 ram_budget_;
  std::string stats_filename_;
  const int64 stats_dump_period_;
  const int64 span_collection_interval_;
  const TraceMeMetadata traceme_metadata_;
};

// static
void ModelDatasetOp::MakeDatasetFromOptions(OpKernelContext* ctx,
                                            DatasetBase* input,
                                            model::AutotuneAlgorithm algorithm,
                                            bool cpu_budget, bool ram_budget,
                                            const std::string& stats_filename,
                                            int64 stats_dump_period,
                                            int64 span_collection_interval,
                                            DatasetBase** output) {
  *output = new ModelDatasetOp::Dataset(
      DatasetContext(DatasetContext::Params(
          {ModelDatasetOp::kDatasetType, ModelDatasetOp::kDatasetOp})),
      input, algorithm, cpu_budget, ram_budget, stats_filename,
      stats_dump_period, span_collection_interval);
}

ModelDatasetOp::ModelDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  if (ctx->HasAttr(kAlgorithm)) {
    int64_t algorithm;
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kAlgorithm, &algorithm));
    algorithm_ = model::AutotuneAlgorithm(algorithm);
  } else {
    algorithm_ = model::AutotuneAlgorithm::HILL_CLIMB;
  }
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kCpuBudget, &cpu_budget_));
  OP_REQUIRES(ctx, cpu_budget_ >= 0,
              errors::InvalidArgument("CPU budget must be positive but is ",
                                      cpu_budget_, "."));
  if (ctx->HasAttr(kRamBudget)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kRamBudget, &ram_budget_));
  } else {
    ram_budget_ = 0;
  }
  OP_REQUIRES(ctx, ram_budget_ >= 0,
              errors::InvalidArgument("RAM budget must be positive but is ",
                                      ram_budget_, "."));
  if (ctx->HasAttr(kStatsFilename)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kStatsFilename, &stats_filename_));
  }
  #ifdef MODEL_DATASET_COMPAT
  else {
    // NOTE(mkuchnik): We disable new attrs for backward compatibility
    // so `stats_filename` is likely always false
    // It may be useful to use secret values for other attrs to enable/disable
    stats_filename_ = "stats.pb";
  }
  #endif

  if (ctx->HasAttr(kStatsDumpPeriod)) {
    OP_REQUIRES_OK(ctx,
        ctx->GetAttr(kStatsDumpPeriod, &stats_dump_period_));
  } else {
    stats_dump_period_ = 0;
  }
  #ifdef MODEL_DATASET_COMPAT
  else {
    stats_dump_period_ = 0;
  }
  #endif

  if (ctx->HasAttr(kSpanCollectionInterval)) {
    OP_REQUIRES_OK(ctx,
        ctx->GetAttr(kSpanCollectionInterval, &span_collection_interval_));
  }
  #ifdef MODEL_DATASET_COMPAT
  else {
    span_collection_interval_ = 0;
  }
  #endif
}

void ModelDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                 DatasetBase** output) {
  *output = new ModelDatasetOp::Dataset(ctx, input, algorithm_, cpu_budget_,
                                        ram_budget_, stats_filename_,
                                        stats_dump_period_,
                                        span_collection_interval_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ModelDataset").Device(DEVICE_CPU),
                        ModelDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
#else   // !IS_MOBILE_PLATFORM
namespace tensorflow {
namespace data {
// static
void ModelDatasetOp::MakeDatasetFromOptions(OpKernelContext* ctx,
                                            DatasetBase* input,
                                            model::AutotuneAlgorithm algorithm,
                                            bool cpu_budget, bool ram_budget,
                                            const std::string& stats_filename,
                                            int64 stats_dump_period,
                                            int64 span_collection_interval,
                                            DatasetBase** output) {
  input->Ref();
  *output = input;
}

ModelDatasetOp::ModelDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {}

void ModelDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                 DatasetBase** output) {
  input->Ref();
  *output = input;
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ModelDataset").Device(DEVICE_CPU),
                        ModelDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
#endif  // !IS_MOBILE_PLATFORM
