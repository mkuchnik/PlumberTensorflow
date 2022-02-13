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
#ifndef TENSORFLOW_CORE_DATA_ANALYSIS_UTILS_H_
#define TENSORFLOW_CORE_DATA_ANALYSIS_UTILS_H_

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace data {

struct RootDatasetStats {
  double average_duration = -1.0;
  double variance_duration = -1.0;
  double average_wallclock_duration = -1.0;
  int64 start_time = -1;
  int64 process_start_time = -1;
};

Status DumpModelStats(model::Model* curr_model,
                      const std::string& stats_filename,
                      const RootDatasetStats& root_stats,
                      int64 time_nanos,
                      const std::shared_ptr<IteratorContext> ctx);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_ANALYSIS_UTILS_H_
