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

#include "tensorflow/core/data/analysis_utils.h"
#include "tensorflow/core/framework/input_pipeline_analysis.pb.h"

namespace tensorflow {
namespace data {
namespace {

Status write_payload_to_file(const std::string& filename,
                             const std::string& payload,
                             const std::shared_ptr<IteratorContext> ctx) {
  // Writes payload to the specified file
  // Taken from dump_mlir_util.cc
  std::unique_ptr<WritableFile> file;
  std::string temp_filename(strings::StrCat(".", filename));
  Status status = ctx->env()->NewWritableFile(temp_filename, &file);
  if (status.ok()) {
    status = file->Append(payload);
    if (status.ok()) {
      // Pseudo-atomic write
      status = file->Close();
      if (status.ok()) {
        status = ctx->env()->RenameFile(temp_filename, filename);
        VLOG(0) << "Wrote file out to " << filename;
      }
    }
  }
  return status;
}

std::string maybe_add_timestamp_to_filename(const std::string& filename,
                                            int64 now_nanos) {
  // For filenames containing %, we substitute datetime
  std::size_t found = filename.find("%");
  if (found != std::string::npos) {
    return std::string(filename).replace(found, 1, std::to_string(now_nanos));
  }
  return filename;
}

PipelineSnapshot production_stats_to_proto(
    const absl::flat_hash_map<string, std::shared_ptr<model::Node_Stats>>& stats,
    const absl::flat_hash_map<string, std::shared_ptr<model::AggregateDatasetMetric>>& global_stats) {
  PipelineSnapshot snapshot;
  for (const auto& aggregate_stats_it : global_stats) {
    PipelineSnapshot::OpStats* op_stats = snapshot.add_stats();
    const auto& aggregate_stats = aggregate_stats_it.second;
    const auto stats_it = stats.find(aggregate_stats_it.first);
    op_stats->set_name(aggregate_stats_it.first); // 6
    if (stats_it != stats.end()) {
      op_stats->set_elements_produced(stats_it->second->elements_produced); // 1
      op_stats->set_wallclock_time(stats_it->second->wallclock_time);  // 2
      op_stats->set_processing_time(stats_it->second->processing_time); // 3
      op_stats->set_parallelism(stats_it->second->parallelism); // 4
      op_stats->set_element_ratio(stats_it->second->ratio); // 5
      op_stats->set_count(stats_it->second->count); // 7
      op_stats->set_bytes_produced(stats_it->second->bytes_produced); // 8
      op_stats->set_bytes_consumed(stats_it->second->bytes_consumed); // 9
      op_stats->set_processing_time_clock(
          stats_it->second->processing_time_clock); // 10
#ifndef LIGHTWEIGHT_METRICS
          op_stats->set_scheduling_delay_time(
              stats_it->second->scheduling_delay_time); // 15
#endif
      op_stats->set_cardinality(stats_it->second->cardinality); // 26
    } else {
      VLOG(1) << "Failed to find stats for " << aggregate_stats_it.first;
      op_stats->set_parallelism(aggregate_stats->parallelism()); // 4
      op_stats->set_element_ratio(aggregate_stats->ratio()); // 5
      op_stats->set_cardinality(-2); // 26
    }
    // 11
    // estimated_dataset_size is deprecated
    // 12
    op_stats->set_aggregate_elements_produced(
        aggregate_stats->elements_produced());
    // 13
    op_stats->set_aggregate_processing_time(
        aggregate_stats->processing_time());
    // 14
    op_stats->set_aggregate_processing_time_clock(
        aggregate_stats->processing_time_clock());
    // 16
    op_stats->set_aggregate_bytes_produced(
        aggregate_stats->bytes_produced());
    // 17
    op_stats->set_aggregate_bytes_consumed(
        aggregate_stats->bytes_consumed());
    // 18
    op_stats->set_aggregate_udf_processing_time(
        aggregate_stats->udf_processing_time());
    // 19
    op_stats->set_aggregate_udf_processing_time_clock(
        aggregate_stats->udf_processing_time_clock());
    // 20
    op_stats->set_aggregate_scheduling_delay_time(
        aggregate_stats->scheduling_delay_time());
    // 21
    op_stats->set_aggregate_avg_number_active_threads(
        aggregate_stats->avg_num_active_threads());
    // 22
    op_stats->set_aggregate_inter_op_parallelism(
        aggregate_stats->inter_op_parallelism());
    // 23
    op_stats->set_aggregate_wait_time(
        aggregate_stats->wait_time());
    // 24
    op_stats->set_aggregate_disk_bytes_read(
        aggregate_stats->disk_bytes_read());
    // 25
    op_stats->set_aggregate_elements_consumed(
        aggregate_stats->elements_consumed());
    // 27
    op_stats->set_aggregate_ratio(
        aggregate_stats->ratio());
    // 28
    op_stats->set_aggregate_parallelism(
        aggregate_stats->parallelism());
    // 29
    op_stats->set_aggregate_max_buffer_size(
        aggregate_stats->max_buffer_size());
    // 30
    op_stats->set_aggregate_max_bytes_per_element(
        aggregate_stats->max_bytes_per_element());
    // 31
    op_stats->set_aggregate_misc_buffer_size(
        aggregate_stats->misc_buffer_size());
  }
  return snapshot;
}

}  // namespace

Status DumpModelStats(model::Model* curr_model,
                      const std::string& stats_filename,
                      const RootDatasetStats& root_stats,
                      int64 time_nanos,
                      const std::shared_ptr<IteratorContext> ctx) {
  // Fills in stats protobuf and writes them out to file.
  VLOG(0) << "Starting dump";
  GraphDef& graphdef = curr_model->graph_def_;
  if (!graphdef.IsInitialized() || !graphdef.node_size()) {
    // If the model does not have a graph_def, default to ctx graph_def.
    // Note(mkuchnik): Optimizations and distributed computation may
    // cause the graph to be re-written
    VLOG(0) << "Model does not have a graphdef!";
  }
  if (stats_filename.empty()) {
    VLOG(0) << "Found empty stats filename";
    return Status::OK();
  }
  if (graphdef.IsInitialized()) {
    auto stats = curr_model->CollectProductionStats(time_nanos);
    auto global_stats = curr_model->CollectDatasetProductionStats();
    PipelineSnapshot snapshot = production_stats_to_proto(
        stats, global_stats);
    *snapshot.mutable_graph() = graphdef;
    snapshot.mutable_machine_info()->set_num_cores(
        port::NumSchedulableCPUs());
    snapshot.mutable_machine_info()->set_num_hyperthreads_per_core(
        port::NumHyperthreadsPerCore());
    snapshot.mutable_machine_info()->set_nominal_cpu_frequency(
        port::NominalCPUFrequency() * 1e-6);
    snapshot.mutable_machine_info()->set_model(
        strings::StrCat((port::CPUFamily() << 4), port::CPUModelNum()));
    port::MemoryInfo mem_info = port::GetMemoryInfo();
    snapshot.mutable_machine_info()->set_memory_free(mem_info.free);
    snapshot.mutable_machine_info()->set_memory_total(mem_info.total);
    snapshot.mutable_machine_info()->set_estimated_disk_bandwidth(0);
    snapshot.mutable_ctx_info()->set_shared_threadpool_size(
        ctx->runner_threadpool_size());
    auto file_sizes = curr_model->CollectFileReads();
    for (auto it = file_sizes.begin(); it != file_sizes.end(); ++it) {
      auto* file_size_proto = snapshot.mutable_ctx_info()->add_file_sizes();
      file_size_proto->set_name(it->first);
      file_size_proto->set_size(it->second);
    }
    auto output_time = curr_model->CollectOutputTime();
    snapshot.mutable_iter_stats()->set_autotune_output_time(output_time);
    snapshot.mutable_iter_stats()->set_avg_duration(root_stats.average_duration);
    snapshot.mutable_iter_stats()->set_var_duration(
        root_stats.variance_duration);
    snapshot.mutable_iter_stats()->set_avg_wallclock_duration(
        root_stats.average_wallclock_duration);
    snapshot.mutable_snapshot_info()->set_start_time(
        root_stats.start_time);
    snapshot.mutable_snapshot_info()->set_current_time(
        time_nanos);
    snapshot.mutable_snapshot_info()->set_process_start_time(
        root_stats.process_start_time);
    snapshot.mutable_snapshot_info()->set_process_current_time(
        ProcessExecutionTime::NowNanos());
    auto payload = snapshot.SerializeAsString();

    const auto current_filename = maybe_add_timestamp_to_filename(
        stats_filename, time_nanos);
    Status ret = write_payload_to_file(current_filename, payload, ctx);
    if (!ret.ok()) {
      VLOG(0) << "Failed to write to stats file: " <<
        current_filename << std::endl;
    }
  } else {
    VLOG(0) << "Missing graphdef in model" << std::endl;
  }
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow