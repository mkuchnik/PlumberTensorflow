/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/data/root_dataset.h"

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/rewrite_utils.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/data/analysis_utils.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kDatasetType[] = "Root";
constexpr char kAlgorithm[] = "algorithm";
constexpr char kCpuBudget[] = "cpu_budget";
constexpr char kRamBudget[] = "ram_budget_bytes";
constexpr char kHillClimb[] = "hill_climb";
constexpr char kGradientDescent[] = "gradient_descent";
constexpr char kIntraOpParallelism[] = "intra_op_parallelism";
constexpr char kPrivateThreadpoolSize[] = "threadpool_size";

// Default share of available RAM that can be used by model's internal buffers.
constexpr double kRamBudgetShare = 0.5;
constexpr int64 kMaxSpanCollectionSize = 20; // 20 GetNext() calls
constexpr int64 kDefaultDumpPeriodMs = 10000; // 10 seconds

// If value `x` matches `y`, returns default value `z`. Otherwise, return `x`.
inline int64 value_or_default(int64_t x, int64_t y, int64_t z) {
  return x == y ? z : x;
}

}  // namespace

// static
Status RootDataset::FromOptions(DatasetBase* input, DatasetBase** output) {
  auto graphdef = input->graph_def_;
  const Options& options = input->options();
  VLOG(0) << "Starting from root dataset. Stats filename: "
          << options.optimization_options().autotune_stats_filename();
  Params params;
  params.graphdef = graphdef;
  if (ShouldConfigureMaxIntraOpParallelism(options)) {
    params.max_intra_op_parallelism =
        options.threading_options().max_intra_op_parallelism();
  }
  if (ShouldUsePrivateThreadPool(options)) {
    params.private_threadpool_size =
        options.threading_options().private_threadpool_size();
  }
  params.autotune = ShouldUseAutotuning(options);
  // TODO(mkuchnik): Using both force_modeling and stats_dumping may be
  // redundant
  params.force_modeling = ShouldUseModeling(options);
  params.stats_dumping = ShouldUseAnalysisTracing(options);
  if (!params.autotune && params.force_modeling) {
    // NOTE(mkuchnk): We add modeling if using analysis
    VLOG(0) << "Adding modeling for analysis";
    params.autotune = true;
  }
  if (params.force_modeling) {
    params.stats_filename = options.optimization_options().autotune_stats_filename();
    params.stats_dump_period = value_or_default(
        options.optimization_options().autotune_stats_dump_period(),
        0, kDefaultDumpPeriodMs);
  }
  if (params.autotune) {
    params.autotune_algorithm = model::AutotuneAlgorithm::HILL_CLIMB;
    if (options.optimization_options().autotune_buffers()) {
      params.autotune_algorithm = model::AutotuneAlgorithm::GRADIENT_DESCENT;
    }
    params.autotune_cpu_budget =
        value_or_default(options.optimization_options().autotune_cpu_budget(),
                         0, port::NumSchedulableCPUs());
    params.autotune_ram_budget =
        value_or_default(options.optimization_options().autotune_ram_budget(),
                         0, kRamBudgetShare * port::AvailableRam());
  }
  *output = new RootDataset(input, params);
  return Status::OK();
}

class RootDataset::Iterator : public DatasetIterator<RootDataset> {
 public:
  explicit Iterator(const Params& params)
      : DatasetIterator<RootDataset>(params) {
    if (dataset()->params_.autotune) {
      model_ = std::make_shared<model::Model>(dataset()->params_.force_modeling);
      model_->graph_def_ = dataset()->params_.graphdef;
    }
    if (dataset()->params_.max_intra_op_parallelism >= 0) {
      max_intra_op_parallelism_ =
          value_or_default(dataset()->params_.max_intra_op_parallelism, 0,
                           port::MaxParallelism());
    }
    if (dataset()->params_.private_threadpool_size >= 0) {
      threadpool_size_ =
          value_or_default(dataset()->params_.private_threadpool_size, 0,
                           port::MaxParallelism());
      thread_pool_ = absl::make_unique<thread::ThreadPool>(
          Env::Default(), ThreadOptions{}, "data_private_threadpool",
          threadpool_size_);
    }
    cancellation_manager_ = absl::make_unique<CancellationManager>();
  }

  ~Iterator() override { cancellation_manager_->StartCancel(); }

  Status Initialize(IteratorContext* ctx) override {
    return dataset()->input_->MakeIterator(IteratorContext(CreateParams(ctx)),
                                           this, prefix(), &input_impl_);
  }

  Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) override {
    if (dataset()->params_.autotune) {
      TF_RETURN_IF_ERROR(EnsureModelThreadStarted(ctx));
    }
    if (dataset()->params_.stats_dumping) {
      TF_RETURN_IF_ERROR(EnsureDumpThreadStarted(ctx));
    }
    return input_impl_->GetNext(IteratorContext(CreateParams(ctx)), out_tensors,
                                end_of_sequence);
  }

 protected:
  std::shared_ptr<model::Node> CreateNode(
      IteratorContext* ctx, model::Node::Args args) const override {
    return model::MakeKnownRatioNode(std::move(args), /*ratio=*/1);
  }

  Status SaveInternal(SerializationContext* ctx,
                      IteratorStateWriter* writer) override {
    TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
    return Status::OK();
  }

  Status RestoreInternal(IteratorContext* ctx,
                         IteratorStateReader* reader) override {
    TF_RETURN_IF_ERROR(
        RestoreInput(IteratorContext(CreateParams(ctx)), reader, input_impl_));
    return Status::OK();
  }

  TraceMeMetadata GetTraceMeMetadata() const override {
    return dataset()->traceme_metadata_;
  }

 private:
  IteratorContext::Params CreateParams(IteratorContext* ctx) {
    IteratorContext::Params params(ctx);
    if (dataset()->params_.autotune) {
      params.model = model_;
    }
    if (dataset()->params_.private_threadpool_size >= 0) {
      params.runner = [pool = thread_pool_.get()](std::function<void()> c) {
        pool->Schedule(std::move(c));
      };
      params.runner_threadpool_size = threadpool_size_;
    }
    if (dataset()->params_.max_intra_op_parallelism >= 0) {
      params.runner =
          RunnerWithMaxParallelism(params.runner, max_intra_op_parallelism_);
    }
    return params;
  }

  Status EnsureModelThreadStarted(IteratorContext* ctx) {
    mutex_lock l(mu_);
    if (!model_thread_) {
      model_thread_ = ctx->StartThread("tf_data_model", [this]() {
        Status status =
            model_->OptimizeLoop(dataset()->params_.autotune_algorithm,
                                 dataset()->params_.autotune_cpu_budget,
                                 dataset()->params_.autotune_ram_budget,
                                 cancellation_manager_.get());
        if (!status.ok()) {
          LOG(WARNING) << "Optimization loop failed: " << status.ToString();
        }
      });
    }
    return Status::OK();
  }

  Status dump_stats(int64 time_nanos,
                    const std::shared_ptr<IteratorContext> ctx) {
    RootDatasetStats root_stats;
    root_stats.average_duration = -1.0;
    root_stats.variance_duration = -1.0;
    root_stats.average_wallclock_duration = -1.0;
    root_stats.start_time = start_time_;
    root_stats.process_start_time = process_start_time_;
    return DumpModelStats(model_.get(), dataset()->params_.stats_filename,
                          root_stats, time_nanos, ctx);
  }

  void DumpStats(const std::shared_ptr<IteratorContext> ctx) {
     int64 time_nanos = EnvTime::NowNanos();
     dump_stats(time_nanos, ctx);
  }

  Status DumpThread(const std::shared_ptr<IteratorContext> ctx,
                    CancellationManager* cancellation_manager) {
    VLOG(0) << "Starting dump thread";
    auto stats_dump_period = dataset()->params_.stats_dump_period;
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
               (last_dump_ms + stats_dump_period) > current_time_ms) {
          auto wait_ms =
              last_dump_ms + stats_dump_period - current_time_ms;
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

  Status EnsureDumpThreadStarted(IteratorContext* ctx) {
    mutex_lock l(mu_);
    if (!dump_thread_) {
      start_time_ = EnvTime::NowNanos();
      process_start_time_ = ProcessExecutionTime::NowNanos();
      std::shared_ptr<IteratorContext> new_ctx =
          std::make_shared<IteratorContext>(*ctx);
      dump_thread_ = ctx->StartThread("tf_data_dump_stats", [this, new_ctx]() {
        Status status = DumpThread(new_ctx,
                                   cancellation_manager_.get());
        if (!status.ok()) {
          LOG(WARNING) << "Dump Stats loop failed: " << status.ToString();
        }
        VLOG(0) << "Dump exit";
      });
    }
    return Status::OK();
  }

  std::shared_ptr<model::Model> model_ = nullptr;
  // Controls cancellation of `model_thread_`. Must be ordered before
  // `model_thread_` so that `model_thread_` is destroyed first.
  condition_variable cond_var_; // TODO(mkuchnik): remove
  std::unique_ptr<CancellationManager> cancellation_manager_;
  mutex mu_;
  std::unique_ptr<Thread> model_thread_ TF_GUARDED_BY(mu_);
  std::unique_ptr<Thread> dump_thread_ TF_GUARDED_BY(mu_);
  int64 max_intra_op_parallelism_;
  int64 threadpool_size_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  int64 start_time_ TF_GUARDED_BY(mu_) = 0;
  int64 process_start_time_ TF_GUARDED_BY(mu_) = 0;

  // Must be ordered last as its execution may depend on other members.
  std::unique_ptr<IteratorBase> input_impl_;
};

RootDataset::RootDataset(const DatasetBase* input, Params params)
    : DatasetBase(DatasetContext({name_utils::OpName(kDatasetType),
                                  name_utils::OpName(kDatasetType)})),
      input_(input),
      params_(std::move(params)) {
  if (params_.autotune) {
    traceme_metadata_.push_back(std::make_pair(
        kAlgorithm,
        params_.autotune_algorithm == model::AutotuneAlgorithm::HILL_CLIMB
            ? kHillClimb
            : kGradientDescent));
    traceme_metadata_.push_back(std::make_pair(
        kCpuBudget, strings::Printf("%lld", static_cast<long long>(
                                                params_.autotune_cpu_budget))));
    traceme_metadata_.push_back(std::make_pair(
        kRamBudget, strings::Printf("%lld", static_cast<long long>(
                                                params_.autotune_ram_budget))));
  }
  if (params_.max_intra_op_parallelism >= 0) {
    traceme_metadata_.push_back(std::make_pair(
        kIntraOpParallelism,
        strings::Printf("%lld", static_cast<long long>(value_or_default(
                                    params_.max_intra_op_parallelism, 0,
                                    port::MaxParallelism())))));
  }
  if (params_.private_threadpool_size >= 0) {
    traceme_metadata_.push_back(std::make_pair(
        kPrivateThreadpoolSize,
        strings::Printf("%lld", static_cast<long long>(value_or_default(
                                    params_.private_threadpool_size, 0,
                                    port::MaxParallelism())))));
  }
  input_->Ref();
}

RootDataset::~RootDataset() { input_->Unref(); }

std::unique_ptr<IteratorBase> RootDataset::MakeIteratorInternal(
    const string& prefix) const {
  return absl::make_unique<Iterator>(
      Iterator::Params{this, name_utils::IteratorPrefix(kDatasetType, prefix)});
}

const DataTypeVector& RootDataset::output_dtypes() const {
  return input_->output_dtypes();
}

const std::vector<PartialTensorShape>& RootDataset::output_shapes() const {
  return input_->output_shapes();
}

string RootDataset::DebugString() const {
  return name_utils::DatasetDebugString(kDatasetType);
}

int64 RootDataset::Cardinality() const { return input_->Cardinality(); }

Status RootDataset::InputDatasets(
    std::vector<const DatasetBase*>* inputs) const {
  inputs->push_back(input_);
  return Status::OK();
}

Status RootDataset::CheckExternalState() const {
  return input_->CheckExternalState();
}

Status RootDataset::AsGraphDefInternal(SerializationContext* ctx,
                                       DatasetGraphDefBuilder* b,
                                       Node** output) const {
  return errors::Unimplemented("RootDataset does not support serialization.");
}

#if !defined(IS_MOBILE_PLATFORM)
Status FinalizeDataset(OpKernelContext* ctx, DatasetBase* input,
                       DatasetBase** output) {
  const Options& options = input->options();
  absl::flat_hash_set<tstring> optimizations_enabled;
  absl::flat_hash_set<tstring> optimizations_disabled;
  absl::flat_hash_set<tstring> optimizations_default;
  GetOptimizations(options, &optimizations_enabled, &optimizations_disabled,
                   &optimizations_default);
  // Disable `enable_gradient_descent` as it assumes presence of ModelDatasetOp.
  optimizations_disabled.insert("enable_gradient_descent");

  auto experiments = GetExperiments();
  LogAndRecordExperiments(experiments);
  auto optimizations =
      SelectOptimizations(experiments, optimizations_enabled,
                          optimizations_disabled, optimizations_default);
  if (optimizations.empty()) {
    return RootDataset::FromOptions(input, output);
  }

  auto optimization_configs = CreateGraphRewriteConfigs(options);
  auto config_factory = [&optimizations, &optimization_configs]() {
    return CreateRewriterConfig(optimizations, optimization_configs);
  };

  Status s;
  if (ShouldUseAnalysisTracing(options)) {
    GraphDef output_graph_def;
    s = RewriteDataset(ctx, input, std::move(config_factory),
                       /*record_fingerprint=*/true, output,
                       &output_graph_def);
    if (s.ok()) {
      (*output)->propagate_graphdef_update(output_graph_def);
    }
  } else {
    s = RewriteDataset(ctx, input, std::move(config_factory),
                       /*record_fingerprint=*/true, output);
  }
  if (errors::IsDeadlineExceeded(s)) {
    // Ignore DeadlineExceeded as it implies that the attempted rewrite took too
    // long which should not prevent further computation.
    LOG(WARNING) << s.ToString();
    return RootDataset::FromOptions(input, output);
  }
  if (!s.ok()) {
    return s;
  }
  input = *output;
  TF_RETURN_IF_ERROR(RootDataset::FromOptions(input, output));
  input->Unref();
  return Status::OK();
}
#else   // !IS_MOBILE_PLATFORM
Status FinalizeDataset(OpKernelContext* ctx, DatasetBase* input,
                       DatasetBase** output) {
  return RootDataset::FromOptions(input, output);
}
#endif  // !IS_MOBILE_PLATFORM

}  // namespace data
}  // namespace tensorflow
