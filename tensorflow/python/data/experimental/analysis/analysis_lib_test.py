# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for tf.data analysis."""

from __future__ import absolute_import  # Not necessary in a Python 3-only module
from __future__ import division  # Not necessary in a Python 3-only module
from __future__ import print_function  # Not necessary in a Python 3-only module

import tempfile

from tensorflow.python.data.experimental.analysis import analysis_lib
from tensorflow.core.framework import input_pipeline_analysis_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.data.experimental.ops import get_single_element
import time
import os

from tensorflow.python.platform import test

def create_fake_snapshot():
    snapshot = input_pipeline_analysis_pb2.PipelineSnapshot()
    return snapshot

META_MODEL_OP = "OptionsDataset"

class AnalysisLibTest(test.TestCase):

  def testEmptyAnalysis(self):
    with tempfile.NamedTemporaryFile() as f:
        with self.assertRaisesRegex(
            RuntimeError, "Graphdef cannot be empty"):
            plumber = analysis_lib.PlumberPerformanceModel(f.name)  # pylint: disable=unused-variable
            self.assertEqual(set(plumber.dataset_node_ops()), set([]))

  def testEmptyAnalysisSnapshot(self):
    plumber_data = create_fake_snapshot()
    with self.assertRaisesRegex(
        RuntimeError, "Graphdef cannot be empty"):
        snapshot = analysis_lib.PerformanceSnapshot(plumber_data)

  def testStatsFilename(self):
    dataset = dataset_ops.Dataset.range(10000).skip(
        10).take(1000)
    path = "stats1.pb"
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    for x in dataset:
        pass
    self.assertEqual(os.path.isfile(path), False)
    options = options_lib.Options()
    options.experimental_optimization.autotune_stats_filename = path
    options.experimental_optimization.map_parallelization = False
    dataset = dataset.with_options(options)
    for x in dataset:
        pass
    self.assertEqual(os.path.isfile(path), True)

  def testStatsFilename_span(self):
    dataset = dataset_ops.Dataset.range(10000).skip(
        10).take(1000)
    path = "stats1_span.pb"
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    for x in dataset:
        pass
    self.assertEqual(os.path.isfile(path), False)
    options = options_lib.Options()
    options.experimental_optimization.autotune_stats_filename = path
    options.experimental_optimization.autotune_span_collection_interval = 1
    options.experimental_optimization.map_parallelization = False
    dataset = dataset.with_options(options)
    for x in dataset:
        pass
    self.assertEqual(os.path.isfile(path), True)

  def testDAG(self):
    dataset = dataset_ops.Dataset.range(10000).take(10000)
    path = "stats2.pb"
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    options = options_lib.Options()
    options.experimental_optimization.autotune_stats_filename = path
    options.experimental_optimization.map_parallelization = False
    dataset = dataset.with_options(options)
    for x in dataset:
        pass
    plumber = analysis_lib.PlumberPerformanceModel(path)
    snapshot = plumber.snapshot()
    model = plumber.model()
    root = model.root()
    self.assertEqual(str(root.name), "dataset")
    self.assertEqual(str(root.op), "_Retval")
    dataset_graph = snapshot.dataset_graph()
    DAG_dict = dataset_graph.DAG_op_dict_repr()
    DAG_dict_expected = {'_Retval': [META_MODEL_OP],
                         META_MODEL_OP: ['TakeDataset'],
                         'TakeDataset': ['RangeDataset'],
                         'RangeDataset': []}
    self.assertEqual(DAG_dict, DAG_dict_expected)

  def testNodeStats(self):
    dataset = dataset_ops.Dataset.range(10000).take(10000).batch(10)
    path = "stats3.pb"
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    options = options_lib.Options()
    options.experimental_optimization.autotune_stats_filename = path
    options.experimental_optimization.map_parallelization = False
    dataset = dataset.with_options(options)
    for x in dataset:
        pass
    plumber = analysis_lib.PlumberPerformanceModel(path)
    snapshot = plumber.snapshot()
    dataset_graph = snapshot.model().dataset_graph()
    DAG_dict = dataset_graph.DAG_op_dict_repr()
    DAG_dict_expected = {'_Retval': [META_MODEL_OP],
                         META_MODEL_OP: ['BatchDatasetV2'],
                         'BatchDatasetV2': ['TakeDataset'],
                         'TakeDataset': ['RangeDataset'],
                         'RangeDataset': []}
    self.assertEqual(DAG_dict, DAG_dict_expected)
    nodes = dataset_graph.nodes()
    batch_nodes = list(filter(lambda x: x.op == "BatchDatasetV2", nodes))
    self.assertEqual(len(batch_nodes), 1)
    self.assertGreater(batch_nodes[0].state.elements_produced, 0.)
    self.assertGreater(batch_nodes[0].state.wallclock_time, 0.)
    self.assertGreater(batch_nodes[0].state.processing_time, 0.)
    self.assertEqual(batch_nodes[0].state.parallelism, 1)
    self.assertEqual(batch_nodes[0].state.element_ratio, 10.)
    self.assertTrue("BatchDatasetV2" in batch_nodes[0].state.name)
    self.assertEqual(batch_nodes[0].state.count, 1)
    # TODO(mkuchnik): Test precisely what these values have to be (they vary
    # between run to run). For example, bytes_produced should equal
    # bytes_consumed. However, that is currently flaky.
    self.assertGreaterEqual(batch_nodes[0].state.bytes_produced, 0.)
    self.assertGreaterEqual(batch_nodes[0].state.bytes_consumed, 0.)
    self.assertGreaterEqual(batch_nodes[0].state.processing_time_clock, 0.)
    self.assertGreater(batch_nodes[0].state.cardinality, 0)
    self.assertGreater(batch_nodes[0].state.aggregate_elements_produced, 0.)
    self.assertGreaterEqual(batch_nodes[0].state.aggregate_processing_time, 0.)
    self.assertGreaterEqual(
        batch_nodes[0].state.aggregate_processing_time_clock, 0.)
    self.assertGreaterEqual(batch_nodes[0].state.scheduling_delay_time, 0.)
    self.assertGreaterEqual(batch_nodes[0].state.aggregate_bytes_produced, 0.)
    self.assertGreaterEqual(batch_nodes[0].state.aggregate_bytes_consumed, 0.)
    self.assertEqual(
        batch_nodes[0].state.aggregate_udf_processing_time, 0.)
    self.assertEqual(
        batch_nodes[0].state.aggregate_udf_processing_time_clock, 0.)
    self.assertEqual(
        batch_nodes[0].state.aggregate_scheduling_delay_time, 0.)
    self.assertGreaterEqual(
        batch_nodes[0].state.aggregate_avg_number_active_threads, 0.)
    self.assertEqual(
        batch_nodes[0].state.aggregate_inter_op_parallelism, 0.)
    self.assertGreaterEqual(
        batch_nodes[0].state.aggregate_wait_time, 0.)
    self.assertEqual(
        batch_nodes[0].state.aggregate_disk_bytes_read, 0.)
    self.assertEqual(
        batch_nodes[0].state.aggregate_elements_consumed, 0.)
    other_nodes = list(filter(lambda x: x.op != "BatchDatasetV2", nodes))
    for node in other_nodes:
        if node.state and node.state.elements_produced:
            self.assertGreater(node.state.elements_produced, 0.)
            if node.op == "RangeDataset":
                self.assertEqual(node.state.element_ratio, 0.)
            else:
                self.assertEqual(node.state.element_ratio, 1.)
            self.assertEqual(node.state.parallelism, 1)
            self.assertEqual(node.state.count, 1)


  def testEndToEnd(self):
    dataset = (dataset_ops.Dataset.range(10000)
               .skip(10)
               .map(lambda x: (x * x, x + x), num_parallel_calls=None)
               .take(10000))
    path = "stats4.pb"
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    options = options_lib.Options()
    options.experimental_optimization.autotune_stats_filename = path
    options.experimental_optimization.map_parallelization = False
    dataset = dataset.with_options(options)
    for x in dataset:
        pass
    plumber = analysis_lib.PlumberPerformanceModel(path)
    snapshot = plumber.snapshot()
    model = plumber.model()
    root = model.root()
    self.assertEqual(str(root.name), "dataset")
    self.assertEqual(str(root.op), "_Retval")
    dataset_graph = snapshot.dataset_graph()
    DAG_dict = dataset_graph.DAG_op_dict_repr()
    DAG_dict_expected = {'_Retval': [META_MODEL_OP],
                         META_MODEL_OP: ['TakeDataset'],
                         'TakeDataset': ['MapDataset'],
                         'MapDataset': ['SkipDataset'],
                         'SkipDataset': ['RangeDataset'],
                         'RangeDataset': []}
    self.assertEqual(DAG_dict, DAG_dict_expected)
    recommendation = model.recommendation()
    self.assertGreaterEqual(recommendation.current_rate(), 0.0)
    # NOTE(mkuchnik): Will fail if disabled
    self.assertGreaterEqual(recommendation.iterator_autotune_output_time(), 0.0)
    self.assertTrue(recommendation.bottleneck_node() in dataset_graph.nodes())
    self.assertTrue(set(recommendation.ranked_list_bottleneck_nodes()) <=
                    set(dataset_graph.nodes()))
    self.assertGreater(recommendation.bottleneck_node_analysis().element_ratio,
                       0.0)
    # No batching or filter; all nodes have 1 ratio except for source
    self.assertTrue(all(
        map(lambda x: not x.has_analysis()
                      or x.element_ratio == 1.0
                      or (x.is_src_node() and x.element_ratio == 0.0),
            recommendation.nodes())))

    # Summary tests
    self.assertGreater(model.total_CPU_time(), 0)
    self.assertGreater(model.total_wallclock_time(), 0)
    self.assertGreater(model.CPU_Util(), 0)
    self.assertEqual(model.Disk_Util(), 0)
    self.assertGreaterEqual(model.Memory_Util(), 0)
    self.assertGreaterEqual(model.max_memory_usage(), 0)

    # Output tests
    try:
        self.assertGreater(len(model.to_graphviz()), 0)
    except ImportError:
        pass
    self.assertGreater(len(model.to_text()), 0)
    self.assertGreater(len(model.to_dict().keys()), 0)

  def testEndToEnd_batch(self):
    batch_size = 64
    dataset = (dataset_ops.Dataset.range(10000)
               .skip(10)
               .map(lambda x: (x * x, x + x), num_parallel_calls=None)
               .batch(batch_size).repeat().take(100))
    path = "stats5.pb"
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    options = options_lib.Options()
    options.experimental_optimization.autotune_stats_filename = path
    options.experimental_optimization.map_and_batch_fusion = False
    options.experimental_optimization.autotune_span_collection_interval = 1
    options.experimental_optimization.map_parallelization = False
    dataset = dataset.with_options(options)
    for x in dataset:
        pass
    plumber = analysis_lib.PlumberPerformanceModel(path)
    snapshot = plumber.snapshot()
    model = plumber.model()
    root = model.root()
    self.assertEqual(str(root.name), "dataset")
    self.assertEqual(str(root.op), "_Retval")
    dataset_graph = snapshot.dataset_graph()
    DAG_dict = dataset_graph.DAG_op_dict_repr()
    DAG_dict_expected = {'_Retval': [META_MODEL_OP],
                         META_MODEL_OP: ['TakeDataset'],
                         'TakeDataset': ['RepeatDataset'],
                         'RepeatDataset': ['BatchDatasetV2'],
                         'BatchDatasetV2': ['MapDataset'],
                         'MapDataset': ['SkipDataset'],
                         'SkipDataset': ['RangeDataset'],
                         'RangeDataset': []}
    self.assertEqual(DAG_dict, DAG_dict_expected)
    recommendation = model.recommendation()
    self.assertGreaterEqual(recommendation.current_rate(), 0.0)
    self.assertGreaterEqual(recommendation.iterator_autotune_output_time(), 0.0)
    self.assertTrue(recommendation.bottleneck_node() in dataset_graph.nodes())
    self.assertGreater(recommendation.bottleneck_node_analysis().element_ratio,
                       0.0)
    self.assertGreater(recommendation.bottleneck_node_analysis().N_customers,
                       0.0)
    self.assertTrue(all(
        map(lambda x: not x.has_analysis()
                      or x.element_ratio >= 1.0
                      or (x.is_src_node() and x.element_ratio == 0.0),
            recommendation.nodes())))
    self.assertTrue(all(
        map(lambda x: not x.has_analysis()
                      or (x.p_wait >= 0.0 and x.p_wait <= 1.0),
            recommendation.nodes())))
    self.assertTrue(all(
        map(lambda x: not x.has_analysis()
                      or (x.p_wait_blame >= 0.0 and x.p_wait_blame <= 1.0),
            recommendation.nodes())))
    self.assertGreaterEqual(
        min(map(lambda x: x.scheduling_delay, recommendation.nodes())),
        0.0)
    batch_input_nodes = list(filter(lambda x: x.op == "MapDataset",
                              recommendation.nodes()))
    self.assertEqual(len(batch_input_nodes), 1)
    self.assertEqual(batch_input_nodes[0].element_ratio, float(batch_size))
    self.assertGreaterEqual(recommendation.upper_bounds(), 0)
    disk_upper_bounds = recommendation.disk_upper_bounds(100) # 100MB/s
    self.assertEqual(disk_upper_bounds, None)

    try:
        # Sometimes there is not enough data to avoid negative small numbers
        self.assertGreaterEqual(recommendation.LP_upper_bounds()[0], -1)
    except ImportError:
        pass

    # Summary tests
    self.assertGreater(model.total_CPU_time(), 0)
    self.assertGreater(model.total_wallclock_time(), 0)
    self.assertGreater(model.CPU_Util(), 0)
    self.assertEqual(model.Disk_Util(), 0)
    self.assertGreaterEqual(model.Memory_Util(), 0)
    self.assertGreaterEqual(model.max_memory_usage(), 0)
    self.assertGreaterEqual(model.dataset_working_set_size(), 0)
    self.assertGreaterEqual(sum(model.dataset_file_sizes().values()), 0)

    # Output tests
    try:
        self.assertGreater(len(model.to_graphviz()), 0)
    except ImportError:
        pass
    self.assertGreater(len(model.to_text()), 0)
    self.assertGreater(len(model.to_dict().keys()), 0)

    # Restart
    graphdef = model.graphdef()
    graph_def = graphdef.SerializeToString()
    ds = analysis_lib.ResumeDataset(graph_def, dataset.element_spec)
    for x, y in zip(dataset, ds):
        self.assertAllEqual(x, y)
    with self.assertRaisesRegex(
            ValueError, "element_spec cannot be None"):
        ds = analysis_lib.ResumeDataset(graph_def, None)

  def testOuterParallelism(self):
    filter_fn = lambda x: (x % 2 == 0)
    inner_dataset_fn = lambda x: (dataset_ops.Dataset
                                  .range(10000).filter(filter_fn).take(100))
    dataset = (dataset_ops.Dataset.range(2)
               .interleave(inner_dataset_fn, 2, 1, 2))
    path = "stats6.pb"
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    options = options_lib.Options()
    options.experimental_optimization.autotune_stats_filename = path
    options.experimental_optimization.map_parallelization = False
    dataset = dataset.with_options(options)
    for x in dataset:
        pass
    plumber = analysis_lib.PlumberPerformanceModel(path)
    snapshot = plumber.snapshot()
    model = plumber.model()

    # Summary tests
    self.assertGreater(model.total_CPU_time(), 0)
    self.assertGreater(model.total_wallclock_time(), 0)
    self.assertGreater(model.CPU_Util(), 0)
    self.assertEqual(model.Disk_Util(), 0)
    self.assertGreaterEqual(model.Memory_Util(), 0)
    self.assertGreaterEqual(model.max_memory_usage(), 0)
    self.assertGreaterEqual(model.dataset_working_set_size(), 0)
    self.assertGreaterEqual(sum(model.dataset_file_sizes().values()), 0)

    root = model.root()
    self.assertEqual(str(root.name), "dataset")
    self.assertEqual(str(root.op), "_Retval")
    dataset_graph = snapshot.dataset_graph()
    DAG_dict = dataset_graph.DAG_op_dict_repr()
    DAG_dict_expected = {'_Retval': [META_MODEL_OP],
                         META_MODEL_OP: ['ParallelInterleaveDatasetV4'],
                         'ParallelInterleaveDatasetV4': ['RangeDataset'],
                         'RangeDataset': []}
    self.assertEqual(DAG_dict, DAG_dict_expected)
    recommendation = model.recommendation()
    nodes = recommendation.ranked_list_bottleneck_nodes_analysis()
    range_nodes = list(filter(lambda x: x.op == "RangeDataset", nodes))
    self.assertEqual(len(range_nodes), 1)
    for range_node in range_nodes:
        self.assertEqual(range_node.has_outer_parallelism, True)
        self.assertEqual(range_node.outer_parallelism, 2)
        self.assertEqual(range_node.outer_parallelism_parent.op,
                         "ParallelInterleaveDatasetV4")
        self.assertEqual(range_node.parallelism, 1)
    interleave_nodes = list(filter(
        lambda x: x.op == "ParallelInterleaveDatasetV4", nodes))
    self.assertEqual(len(interleave_nodes), 1)
    for interleave_node in interleave_nodes:
        self.assertEqual(interleave_node.is_interleave_node(), True)
        self.assertEqual(interleave_node.has_outer_parallelism, False)
        self.assertEqual(interleave_node.outer_parallelism, 0)
        self.assertEqual(interleave_node.outer_parallelism_parent,
                         None)
        self.assertEqual(interleave_node.parallelism, 2)



if __name__ == "__main__":
  test.main()
