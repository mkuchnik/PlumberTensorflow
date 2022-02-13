# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Plumber pipeline debugging."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.framework import input_pipeline_analysis_pb2
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec

from abc import ABCMeta, abstractmethod
import collections
import datetime
import logging
import pathlib
import pprint
import sys
import math
import time
import itertools

from enum import Enum, IntEnum

from typing import Optional, Dict, NewType, List, Iterator, Union

# Pydot imports from keras vizualization
try:
  # pydot-ng is a fork of pydot that is better maintained.
  import pydot_ng as pydot
except ImportError:
  # pydotplus is an improved version of pydot
  try:
    import pydotplus as pydot
  except ImportError:
    # Fall back on pydot if necessary.
    try:
      import pydot
    except ImportError:
      pydot = None

# CVXPY imports for LP solving
try:
  import cvxpy
except ImportError:
  cvxpy = None

def check_pydot():
  """Returns True if PyDot and Graphviz are available."""
  if pydot is None:
    return False
  try:
    # Attempt to create an image of a blank graph
    # to check the pydot/graphviz installation.
    pydot.Dot.create(pydot.Dot())
    return True
  except (OSError, pydot.InvocationException):
    return False

def check_cvxpy():
  """Returns True if cvxpy is available."""
  return cvxpy is not None

def enable_compat_logging():
    logging.basicConfig(level=logging.INFO,
                        filename='plumber.log',
                        filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')

LOGGER_NAME = "plumber_analysis_lib"
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.WARNING)

# Types
NodeName = NewType("NodeName", str)
NodeOp = NewType("NodeOp", str)


"""Holds state mirroring from graphdef."""
NodeState = collections.namedtuple("NodeState",
                                   ["elements_produced",
                                    "wallclock_time",
                                    "processing_time",
                                    "parallelism",
                                    "element_ratio",
                                    "name",
                                    "count",
                                    "bytes_produced",
                                    "bytes_consumed",
                                    "processing_time_clock",
                                    "estimated_dataset_size",
                                    "aggregate_elements_produced",
                                    "aggregate_processing_time",
                                    "aggregate_processing_time_clock",
                                    "scheduling_delay_time",
                                    "aggregate_bytes_produced",
                                    "aggregate_bytes_consumed",
                                    "aggregate_udf_processing_time",
                                    "aggregate_udf_processing_time_clock",
                                    "aggregate_scheduling_delay_time",
                                    "aggregate_avg_number_active_threads",
                                    "aggregate_inter_op_parallelism",
                                    "aggregate_wait_time",
                                    "aggregate_disk_bytes_read",
                                    "aggregate_elements_consumed",
                                    "cardinality",
                                    "aggregate_element_ratio",
                                    "aggregate_parallelism",
                                    "aggregate_max_buffer_size",
                                    "aggregate_max_bytes_per_element",
                                    "aggregate_misc_buffer_size",
                                    ])


SnapshotState = collections.namedtuple("SnapshotState",
                                       ["start_time",
                                        "current_time",
                                        "process_start_time",
                                        "process_current_time",])

IteratorStats = collections.namedtuple("IteratorStats",
                                       ["avg_duration",
                                        "var_duration",
                                        "avg_wallclock_duration",
                                        "autotune_output_time"])

CtxInfo = collections.namedtuple("CtxInfo",
                                 ["shared_threadpool_size",
                                  "udf_threadpool_size",
                                  "file_sizes"])

MachineInfo = collections.namedtuple("MachineInfo",
                                     ["num_cores",
                                      "memory_free",
                                      "memory_total",
                                      "estimated_disk_bandwidth",
                                      "num_hyperthreads_per_core",
                                      "nominal_cpu_frequency",
                                      "model"])


def stat_to_node_state(stat) -> NodeState:
    """Maps graphdef stats to NodeState stats."""
    return NodeState(
        elements_produced=stat.elements_produced,
        wallclock_time=stat.wallclock_time,
        processing_time=stat.processing_time,
        parallelism=stat.parallelism,
        element_ratio=stat.element_ratio,
        name=stat.name,
        count=stat.count,
        bytes_produced=stat.bytes_produced,
        bytes_consumed=stat.bytes_consumed,
        processing_time_clock=stat.processing_time_clock,
        estimated_dataset_size=0,
        aggregate_elements_produced=stat.aggregate_elements_produced,
        aggregate_processing_time=stat.aggregate_processing_time,
        aggregate_processing_time_clock=stat.aggregate_processing_time_clock,
        scheduling_delay_time=stat.scheduling_delay_time,
        aggregate_bytes_produced=stat.aggregate_bytes_produced,
        aggregate_bytes_consumed=stat.aggregate_bytes_consumed,
        aggregate_udf_processing_time=stat.aggregate_udf_processing_time,
        aggregate_udf_processing_time_clock=
        stat.aggregate_udf_processing_time_clock,
        aggregate_scheduling_delay_time=stat.aggregate_scheduling_delay_time,
        aggregate_avg_number_active_threads=
        stat.aggregate_avg_number_active_threads,
        aggregate_inter_op_parallelism=stat.aggregate_inter_op_parallelism,
        aggregate_wait_time=stat.aggregate_wait_time,
        aggregate_disk_bytes_read=stat.aggregate_disk_bytes_read,
        aggregate_elements_consumed=stat.aggregate_elements_consumed,
        cardinality=stat.cardinality,
        aggregate_element_ratio=stat.aggregate_ratio,
        aggregate_parallelism=stat.aggregate_parallelism,
        aggregate_max_buffer_size=stat.aggregate_max_buffer_size,
        aggregate_max_bytes_per_element=stat.aggregate_max_bytes_per_element,
        aggregate_misc_buffer_size=stat.aggregate_misc_buffer_size,
    )


def snapshot_info_to_snapshot_state(snapshot_info) -> SnapshotState:
    """Maps graphdef SnapshotInfo to SnapshotState stats."""
    # Nano to milliseconds
    start_time = float(snapshot_info.start_time) / 1e9
    current_time = float(snapshot_info.current_time) / 1e9
    process_start_time = float(snapshot_info.process_start_time) / 1e9
    process_current_time = float(snapshot_info.process_current_time) / 1e9
    return SnapshotState(
        start_time=datetime.datetime.fromtimestamp(start_time),
        current_time=datetime.datetime.fromtimestamp(current_time),
        process_start_time=process_start_time,
        process_current_time=process_current_time,
    )

def injest_iter_stats(iter_stats) -> IteratorStats:
    """Maps proto IteratorStats to named tuple variant"""
    avg_duration = float(iter_stats.avg_duration) / 1e9
    var_duration = float(iter_stats.var_duration) / 1e9
    avg_wallclock_duration = float(iter_stats.avg_wallclock_duration) / 1e9
    autotune_output_time = float(iter_stats.autotune_output_time) / 1e9
    return IteratorStats(
        avg_duration=avg_duration,
        var_duration=var_duration,
        avg_wallclock_duration=avg_wallclock_duration,
        autotune_output_time=autotune_output_time,
    )

def injest_machine_info(machine_info) -> MachineInfo:
    """Maps proto MachineInfo to tuple variant"""
    return MachineInfo(
        num_cores=machine_info.num_cores,
        memory_free=machine_info.memory_free,
        memory_total=machine_info.memory_total,
        estimated_disk_bandwidth=machine_info.estimated_disk_bandwidth,
        num_hyperthreads_per_core=machine_info.num_hyperthreads_per_core,
        nominal_cpu_frequency=machine_info.nominal_cpu_frequency,
        model=machine_info.model)


def injest_ctx_info(ctx_info) -> CtxInfo:
    """Maps proto CtxInfo to tuple variant"""
    return CtxInfo(
        shared_threadpool_size=ctx_info.shared_threadpool_size,
        udf_threadpool_size=ctx_info.udf_threadpool_size,
        file_sizes={x.name: x.size for x in ctx_info.file_sizes})

class DatasetCardinality(IntEnum):
    # TODO(mkuchnik): Would be better to use native tf.data wrappers
    INFINITE = -1
    UNKNOWN = -2

    def __str__(self):
        return self.name

class OpResourceType(Enum):
    """The disk operators"""
    CPU = 1
    disk = 2
    memory = 3
    # TODO(mkuchnik): Add `time` type for sleep

    def __str__(self):
        return self.name

    @staticmethod
    def op_to_resource_type(op_name: NodeOp) -> 'OpResourceType':
        if op_name in DISK_DATASET_OP_NODES:
          return OpResourceType.disk
        elif op_name in MEMORY_DATASET_OP_NODES:
          return OpResourceType.memory
        else:
          return OpResourceType.CPU

PARALLEL_DATASET_OP_NODES = set([
    "ParallelMapDatasetV2",
    "ParallelInterleaveDatasetV4",
    "MapAndBatchDataset",
    "MapDataset", # Technically not, but implementation switches to parallel
    "InterleaveDataset",
    "ParallelBatchDataset", # TODO(mkuchnik): Add BatchDataset
])

PARALLELIZABLE_DATASET_OP_NODES = set([
    "MapDataset",
    "InterleaveDataset",
    "BatchDatasetV2",
])

DISK_DATASET_OP_NODES = set([
    "TFRecordDataset",
    "FixedLengthRecordDataset",
    "FixedLengthRecordDatasetV2",
    "TextLineDataset",
])

CACHE_DATASET_OP_NODES = set([
    "CacheDataset",
    "CacheDatasetV2",
])

TENSOR_SLICE_DATASET_OP_NODES = set([
    "TensorSliceDataset",
])

TAKE_DATASET_OP_NODES = set([
    "TakeDataset",
])

ZIP_DATASET_OP_NODES = set([
    "ZipDataset",
])

RANGE_DATASET_OP_NODES = set([
    "RangeDataset",
])

MEMORY_DATASET_OP_NODES = set([
    *CACHE_DATASET_OP_NODES
])

SRC_DATASET_OP_NODES = set([
    *DISK_DATASET_OP_NODES,
    *MEMORY_DATASET_OP_NODES,
    *RANGE_DATASET_OP_NODES,
])

REPEAT_DATASET_OP_NODES = set([
    "RepeatDataset",
])

INTERLEAVE_DATASET_OP_NODES = set([
    "ParallelInterleaveDatasetV4",
    "InterleaveDataset",
])

SNAPSHOT_DATASET_OP_NODES = set([
    "SnapshotDatasetV2",
])

GROUP_BY_WINDOW_DATASET_OP_NODES = set([
    "GroupByWindowDataset",
])

SLOW_OP_BLACKLIST_SET = set([
    "dataset",
])

FUNCTION_ATTR_NAMES = ["f", "key_func", "reduce_func", "window_size_func"]

# TODO(mkuchnik): Need to have special treatment for certain ops like
# SleepDataset, which use Wallclock time vs. other ops, which use compute time.
# TODO(mkuchnik): Cache datasets have strange behavior. They often have count of
# 2 and only produce 2 elements.

@tf_export("data.experimental.analysis.PlumberPerformanceModel", v1=[])
class PlumberPerformanceModel(object):
  """Plumber performance model.

  Consumes a Plumber proto file.

  Link performance data to a graph structure of the pipeline. This allows
  making inferences on slow components of the pipeline.
  """

  def __init__(self, filename: str) -> None:
    """Initializes the PlumberPerformanceModel with a history.

    @param filename The plumber file to use as input
    """
    self.snapshots = PerformanceSnapshots("{}".format(filename))

  def snapshot(self) -> 'PerformanceSnapshot':
      """Returns the most recent snapshot.

      Use this to inspect what the recorded data looks like.
      """
      return self.snapshots.get_snapshot()

  def model(self) -> 'PerformanceModel':
    """Returns the most recent snapshot's model.

    Use this to find the bottlenecks associated with the model

    plumber = PlumberPerformanceModel("./data_dir")
    model = plumber.model()
    """
    return self.snapshot().model()

  def _graphdef(self):
    """Returns the most recent snapshot's graphdef."""
    return self.snapshot().graphdef()


class DatasetTree(object):
    """In memory representation of graphdef.

    All outward access should prefer using native Python types (e.g., dict)"""
    ROOT_NODE_NAME = NodeName("dataset")

    def __init__(self, graphdef: input_pipeline_analysis_pb2.PipelineSnapshot):
        self._node_lookup = import_graphdef(graphdef)
        if self.ROOT_NODE_NAME not in self._node_lookup:
            raise RuntimeError("Graphdef cannot be empty")

    def root(self) -> 'TreeNode':
        """Convenience method for root"""
        return self.lookup_name(self.ROOT_NODE_NAME)

    def DAG_name_dict_repr(self) -> Dict[NodeName, 'TreeNode']:
        """Represents the DAG connectivity with a dictionary.
        Each node is represented by its name"""
        output = dict()
        queue = [self.root()]
        while queue:
            node = queue.pop(0)
            output[node.name] = list(map(lambda x: x.name, node.input))
            queue.extend(list(node.input))
        return output

    def DAG_op_dict_repr(self) -> Dict[NodeOp, 'TreeNode']:
        """Represents the DAG connectivity with a dictionary.
        Each node is represented by its op"""
        output = dict()
        queue = [self.root()]
        while queue:
            node = queue.pop(0)
            output[node.op] = list(map(lambda x: x.op, node.input))
            queue.extend(list(node.input))
        return output

    def lookup_name(self, node_name: NodeName) -> 'TreeNode':
        """Returns the node corresponding to the node_name"""
        return self._node_lookup[node_name]

    def nodes(self) -> List['TreeNode']:
        """A list of all nodes in the tree"""
        return self._bfs()

    def _bfs(self) -> List['TreeNode']:
        """A BFS traversal of the nodes"""
        queue = [self.root()]
        output = []
        while queue:
            output.append(queue[0])
            node = queue.pop(0)
            queue.extend(list(node.input))
        return output

    def __repr__(self):
        return repr(self.root())


class PerformanceSnapshot(object):
    """Captures all snapshot state (including global)"""
    def __init__(self,
                 snapshot_proto: input_pipeline_analysis_pb2.PipelineSnapshot):
        self._snapshot_proto = snapshot_proto
        self._dataset_graph = DatasetTree(self.graphdef())

    def dataset_graph(self) -> DatasetTree:
        return self._dataset_graph

    def snapshot_proto(self) -> input_pipeline_analysis_pb2.PipelineSnapshot:
        return self._snapshot_proto

    def graphdef(self):
        return self._snapshot_proto.graph

    def model(self) -> 'PerformanceModel':
        return PerformanceModel(self)

    def __repr__(self):
        s = ("[PerformanceSnapshot]:\nsnapshot_proto:\n{}\ndataset_graph:\n{}"
             .format(
                repr(self.snapshot_proto()),
                repr(self.dataset_graph())))
        return s

    @staticmethod
    def read_snapshot(plumber_data: str) -> 'PerformanceSnapshot':
        """Binary data to a in-memory object snapshot"""
        snapshot_proto = (input_pipeline_analysis_pb2
                          .PipelineSnapshot().FromString(plumber_data))
        if not snapshot_proto.graph or not len(snapshot_proto.graph.node):
            raise RuntimeError("Graphdef cannot be empty")
        snapshot = PerformanceSnapshot(snapshot_proto)
        return snapshot


class PerformanceSnapshots(object):
    """Mostly reader functionality"""
    def __init__(self, directory_path: Optional[str] = None):
        self._snapshots = []
        if directory_path:
            self.injest_snapshots(directory_path)
        if not self._snapshots:
            raise RuntimeError("Did not find any snapshots in directory: "
                               "'{}'".format(directory_path))

    def injest_snapshots(self, directory_path: str):
        """Note: we accept filepaths to a single snapshot"""
        path = pathlib.Path(directory_path)
        if path.is_file():
            with path.open("rb") as f:
                plumber_data = f.read()
            snapshot = PerformanceSnapshot.read_snapshot(plumber_data)
            self._snapshots.append(snapshot)
        elif path.is_dir():
            snapshot_files = sorted(path.glob("**/*.pb"))
            if not snapshot_files:
                raise RuntimeError("Did not find any snapshot files in "
                                   "directory: "
                                   "'{}'".format(directory_path))
            for p in snapshot_files:
                with p.open("rb") as f:
                    plumber_data = f.read()
                snapshot = PerformanceSnapshot.read_snapshot(plumber_data)
                self._snapshots.append(snapshot)
        else:
            raise RuntimeError("Path does not exist: "
                               "'{}'".format(directory_path))
    def get_snapshot(self, t: Optional[int] = None) -> PerformanceSnapshot:
        """Ordered by time"""
        return self._snapshots[t if t is not None else -1]

    def num_snapshots(self) -> int:
        return len(self._snapshots)


class PerformanceModel(object):
    """Joins tree structure and corresponding stats with global statistics"""
    def __init__(self, snapshot: PerformanceSnapshot):
        self._dataset_graph = snapshot.dataset_graph()
        self._global_info = snapshot.snapshot_proto()
        # Add annotations
        PerformanceModel.annotate_dataset_graph_with_info(
            self._dataset_graph, self._global_info)
        # Add global state
        self.global_state = SnapshotGlobalState(self._global_info)
        # NOTE Currently implementation detail. Prefer analysis()
        self._refresh_cache()

    def _refresh_cache(self) -> None:
        self._cached_analysis = self._analysis()
        self._cached_recommendation = self._recommendation()

    def root(self) -> 'TreeNode':
        """Public API to traverse the graph."""
        return self._dataset_graph.root()

    def dataset_graph(self) -> DatasetTree:
        return self._dataset_graph

    def graphdef(self) -> str:
        return self._global_info.graph

    @staticmethod
    def annotate_dataset_graph_with_info(
        dataset_graph: DatasetTree,
        global_info: input_pipeline_analysis_pb2.PipelineSnapshot) -> None:
        root = dataset_graph.root()
        queue = [root]
        stat_lookup = {stat.name: stat for stat in global_info.stats}
        while queue:
            node = queue.pop(0)
            try:
                state = stat_lookup[node.name]
            except KeyError:
                logger.info("Failed to find stats for {}".format(node.name))
                state = None
            if state is not None:
                state = stat_to_node_state(state)
                node.add_state_annotations(state)
            queue.extend(list(node.input))
            queue.extend(list(node.function))

    def _analysis(self) -> 'PerformanceAnalysis':
        return PerformanceAnalysis(self)

    def analysis(self) -> 'PerformanceAnalysis':
        return self._cached_analysis

    def _recommendation(self) -> 'PerformanceRecommendation':
        return PerformanceRecommendationv1(self)

    def recommendation(self) -> 'PerformanceRecommendation':
        return self._cached_recommendation

    def to_graphviz(self, filename: Optional[str]=None) -> str:
        """
        @param filename Writes data out to file if not None
        """
        dot_data = DotPerformanceSummary(self).to_string()
        if filename:
            with open(filename, "w") as f:
                f.write(dot_data)
        return dot_data

    def to_text(self, filename: Optional[str]=None) -> str:
        """
        @param filename Writes data out to file if not None
        """
        text_data = TextPerformanceSummary(self).to_string()
        if filename:
            with open(filename, "w") as f:
                f.write(text_data)
        return text_data

    def to_dict(self) -> dict:
        data = {"global_info": self._global_info,
                "dataset_graph": self._dataset_graph}
        return data

    def total_CPU_time(self, calculation_mode: Optional[str]=None) -> float:
        """By default, use wallclock. Also supports CPU time and Process-wide
        CPU time."""
        def inner_CPU_time(node):
            if not node or not node.state:
                return 0.
            if calculation_mode is None:
                CPU_time = node.state.aggregate_processing_time / 1e9
            elif calculation_mode == "CPU_clock":
                CPU_time = node.state.aggregate_processing_time_clock / 1e9
            else:
                raise ValueError("Calculation mode is not recognized:"
                                 " {}".format(calculation_mode))
            for n in node.owned_nodes():
                CPU_time += inner_CPU_time(n)
            return CPU_time

        if calculation_mode == "process_CPU_clock":
            return self.total_CPU_process_time()
        else:
            return sum([inner_CPU_time(n) for n in
                        self._dataset_graph.nodes()])

    def total_wallclock_time(self) -> float:
        snapshot_state = self.global_state.snapshot_state
        # TODO(mkuchnik): Refactor out
        elapsed_time = (snapshot_state.current_time -
                        snapshot_state.start_time).total_seconds()
        return elapsed_time

    def total_CPU_process_time(self) -> float:
        """In seconds"""
        snapshot_state = self.global_state.snapshot_state
        # TODO(mkuchnik): Refactor out
        elapsed_time = (snapshot_state.process_current_time -
                        snapshot_state.process_start_time)
        return elapsed_time

    def total_CPU_time_avail(self) -> float:
        return self.num_cores() * self.total_wallclock_time()

    def CPU_Util(self, calculation_mode: Optional[str]=None) -> float:
        """By default, use wallclock. Also supports CPU time."""
        return (self.total_CPU_time(calculation_mode=calculation_mode)
                / self.total_CPU_time_avail())

    def cores_utilized(self) -> float:
        cpu_util = min(self.CPU_Util(), 1.0)
        used_cores = self.num_cores() * cpu_util
        return used_cores

    def fractional_cores_unutilized(self) -> float:
        cpu_util = min(self.CPU_Util(), 1.0)
        remaining_cores = self.num_cores() * (1. - cpu_util)
        assert remaining_cores <= self.num_cores()
        assert remaining_cores >= 0.
        return remaining_cores

    def memory_free(self) -> float:
        return self._global_info.machine_info.memory_free

    def memory_total(self) -> float:
        return self._global_info.machine_info.memory_total

    def Memory_Util(self) -> float:
        mem_total = self.memory_total()
        if mem_total <= 0:
            logger.warning("Total memory detected to be out of range: {}. "
                            "Clamping to 1."
                            .format(memory_total))
            mem_total = 1
        mem_fraction_free = self.memory_free() / mem_total
        assert mem_fraction_free >= 0.
        return 1. - mem_fraction_free

    def max_memory_usage(self) -> int:
        """Bounds the memory usage of the pipeline."""
        def inner_memory_usage(node):
            if not node or not node.state:
                return 0.
            max_memory_usage = node.max_memory_used
            for n in node.owned_nodes():
                max_memory_usage += inner_memory_usage(n)
            return max_memory_usage

        return sum([inner_memory_usage(n) for n in
                    self._cached_analysis.nodes()])

    def Disk_Util(self) -> float:
        disk_avail = self.disk_bandwidth_avail()
        if disk_avail <= 0.:
            logger.warning("Total disk bandwidth detected to be out of range:"
                            " {}. Clamping to 1."
                            .format(disk_avail))
            disk_avail = 1.
        return self.disk_throughput() / disk_avail

    def disk_bandwidth_avail(self) -> float:
        return self._global_info.machine_info.estimated_disk_bandwidth

    def disk_throughput(self) -> float:
        return self.disk_bytes_read() / self.total_wallclock_time()

    def disk_bytes_read(self) -> int:
        def inner_disk_bytes_read(node):
            if not node.state or not node.is_disk_node():
                disk_bytes = 0.
            else:
                # NOTE(mkuchnik): aggregate_bytes_produced can also work
                disk_bytes = node.state.aggregate_disk_bytes_read
                disk_bytes2 = node.state.aggregate_bytes_produced
                if not disk_bytes and disk_bytes2:
                    logger.warning(
                        "Disk_bytes={} for node {} but see {}"
                        " bytes produced".format(
                            disk_bytes, node.name, disk_bytes2))

            for n in node.owned_nodes():
                disk_bytes += inner_disk_bytes_read(n)
            return disk_bytes
        return sum([inner_disk_bytes_read(n) for n in
                    self._dataset_graph.nodes()])

    def dataset_file_sizes(self) -> Dict[str, int]:
        """The number of bytes observed for each file in training"""
        return dict(self.global_state.ctx_info.file_sizes)

    def dataset_working_set_size(self) -> int:
        """The number of bytes observed by unique files in training.

        NOTE: due to early termination, this may be smaller than full working
        set. Further, some datasets may own the file, but then be cached, so
        care must be taken in deciding how to use this number."""
        # TODO(mkuchnik): Add method to get projected dataset size from nodes
        return sum(self.global_state.ctx_info.file_sizes.values())

    def num_cores(self) -> int:
        return self._global_info.machine_info.num_cores

class PerformanceSummary(metaclass=ABCMeta):
    @abstractmethod
    def to_string(self):
        pass

class TextPerformanceSummary(PerformanceSummary):
    """Emits command-line performance"""
    def __init__(self, performance_model: PerformanceModel):
        self._performance_model = performance_model

    def to_string(self) -> str:
        data = pprint.pformat(self._performance_model.to_dict())
        return data

# TODO(mkuchnik) Tensorboard visualizer
# https://github.com/tensorflow/tensorboard/tree/javascript/tensorboard/examples/plugins

class DotPerformanceSummary(PerformanceSummary):
    """Emits dot-file performance"""
    def __init__(self, performance_model: PerformanceModel):
        self._performance_model = performance_model

    def to_string(self) -> str:
        if not check_pydot():
          message = (
              'Failed to import pydot. You must `pip install pydot` '
              'and install graphviz (https://graphviz.gitlab.io/download/), ',
              'for `pydotprint` to work.')
          if 'IPython.core.magics.namespace' in sys.modules:
            # We don't raise an exception here in order to avoid crashing
            # notebook tests where graphviz is not available.
            logger.error(message)
            return
          else:
            raise ImportError(message)

        dot = pydot.Dot(graph_type="digraph", graph_name="PlumberModel",
                        rankdir="LR")
        # TODO(mkuchnik): Constructing analysis here is expensive. Take
        # analysis/recommendation as input? Need recommendation for bottleneck
        analysis = self._performance_model.analysis()
        recommendation = self._performance_model.recommendation()
        bottleneck_node = recommendation.bottleneck_node_analysis()

        def is_bottleneck_node(node):
            return node.name == bottleneck_node.name

        def node_color(node):
            if is_bottleneck_node(node):
                return "red"
            elif node.has_analysis() and  node.is_stale:
                return "gray"
            else:
                return None

        def node_fillcolor(node):
            if node.is_src_node():
                return "gray"
            else:
                return None

        def remove_unused_kwargs(kwargs):
            return dict([x for x in kwargs.items() if x[1] is not None])

        def node_label(node):
            attrs = node.to_summary_dict()
            if attrs:
                attrs = pprint.pformat(attrs)
                label = "{name}\n{attrs}".format(name=node.name, attrs=attrs)
            else:
                label = "{name}".format(name=n.name)
            return label

        def node_kwargs(node):
            label = node_label(node)
            color = node_color(node)
            fillcolor = node_fillcolor(node)
            kwargs = {"color": color,
                      "fillcolor": fillcolor,
                      "label": label}
            if node.is_src_node():
                kwargs["shape"] = "diamond"
            kwargs = remove_unused_kwargs(kwargs)
            return kwargs

        nodes = analysis.nodes()
        extra_cores = self._performance_model.fractional_cores_unutilized()
        # NOTE(mkuchnik): label=None is incorrect for anything below
        for n in nodes:
            kwargs = node_kwargs(n)
            pd_node = pydot.Node(n.name, **kwargs)
            dot.add_node(pd_node)
            # TODO(mkuchnik): This is ugly. Prefer to unify access to function
            analysis_node = analysis.lookup_name(n.name)
            if analysis_node.function:
                cluster = pydot.Cluster("{}_function".format(
                    analysis_node.name), color="blue")
                cluster.add_node(pd_node)
                for f in analysis_node.function:
                    kwargs = node_kwargs(f)
                    pd_node = pydot.Node(f.name, **kwargs)
                    cluster.add_node(pd_node)
                    pd_edge = pydot.Edge(f.name, n.name)
                    cluster.add_edge(pd_edge)
                dot.add_subgraph(cluster)
        for n in nodes:
            for i in n.node.input:
                i = analysis.lookup_name(i.name)
                if i.has_analysis():
                    label = "Max Parallel Rate: {:2}\n" \
                            "Max Scaled Rate: {:2}".format(
                                i.expected_parallel_max_rate(),
                                i.expected_parallel_max_rate(
                                    extra_cores=extra_cores),
                            )
                    pd_edge = pydot.Edge(i.name, n.name, label=label)
                else:
                    pd_edge = pydot.Edge(i.name, n.name)
                dot.add_edge(pd_edge)
        return dot.to_string()


class PerformanceRecommendation(metaclass=ABCMeta):

    @abstractmethod
    def bottleneck_node(self) -> 'TreeNode':
        """Returns a pointer to the bottleneck node"""
        pass

    @abstractmethod
    def current_rate(self) -> float:
        """Returns the data pipeline's maximum rate"""
        pass

    @abstractmethod
    def analysis_str(self) -> str:
        """Returns a debug string for what the user should inspect"""
        pass


class SnapshotGlobalState(object):
    """Wraps and injests global state from snapshot"""
    def __init__(self,
                 global_info: input_pipeline_analysis_pb2.PipelineSnapshot):
        self.snapshot_state = \
            snapshot_info_to_snapshot_state(global_info.snapshot_info)
        self.iter_stats = injest_iter_stats(global_info.iter_stats)
        self.machine_info = injest_machine_info(global_info.machine_info)
        self.ctx_info = injest_ctx_info(global_info.ctx_info)

    def elapsed_time_seconds(self) -> float:
        """In seconds"""
        return (self.snapshot_state.current_time -
                self.snapshot_state.start_time).total_seconds()


class AnalysisTreeNode(object):
    """Adds internal analysis variables to a DatasetTree"""
    def __init__(self, tree_node: 'TreeNode'):
        """Take a tree_node, which represents local state."""
        self.node = tree_node
        self.analysis_data = dict()
        self.function = []
        self._parent = None
        self._inputs = []

    @property
    def observed_rate(self) -> float:
        return self.analysis_data["observed_rate"]

    @observed_rate.setter
    def observed_rate(self, rate: float) -> None:
        assert rate >= 0.0, "Rate must be positive ({}--{})".format(
            rate, self.name)
        self.analysis_data["observed_rate"] = rate

    def expected_parallel_max_rate(self, parallelism=None,
                                   extra_cores=None) -> float:
        """The maximum rate if all parallel cores were dedicated to the task.
        Without additional args, current parallelism is used."""
        if extra_cores:
            assert not parallelism, "Can't set both extra_cores and parallelism"
        # NOTE(mkuchnik): For outer parallelism, parallelism may be over 1
        if not self.is_parallel_node() and not self.has_outer_parallelism:
            if self.parallelism > 1.0:
                logger.warning("Node has higher parallelism ({}) than "
                                "expected: {}".format(self.parallelism,
                                                      self.name))
            parallelism = 1.0
        else:
            if not parallelism:
                if not self.is_parallel_node() and self.has_outer_parallelism:
                    # TODO(mkuchnik): Check that interleave on records works as
                    # intended. Due to prefetching, some
                    # extra parallelism may be observed.
                    if self.outer_parallelism != self.parallelism:
                        logger.info("Node has different parallelism ({}) "
                                     "than expected ({}):"
                                     " {}".format(self.parallelism,
                                              self.outer_parallelism,
                                              self.name))
                    parallelism = self.outer_parallelism
                else:
                    parallelism = self.parallelism
            if extra_cores:
                parallelism += extra_cores
        return self.expected_per_core_max_rate * parallelism

    @property
    def parallelism(self) -> int:
        if self.node.state:
            return self.node.state.parallelism
        else:
            return 1

    @property
    def N_customers(self) -> float:
        """The number of customers in the system"""
        T = self.elapsed_time
        W = self.node.state.aggregate_processing_time / 1e9
        N = W / T
        return N

    @property
    def expected_per_core_max_rate(self) -> float:
        """The maximum rate if a single core was dedicated to the task"""
        return self.analysis_data["expected_per_core_max_rate"]

    @expected_per_core_max_rate.setter
    def expected_per_core_max_rate(self, rate: float) -> None:
        assert rate >= 0.0, "Rate must be positive"
        self.analysis_data["expected_per_core_max_rate"] = rate

    @property
    def expected_service_time(self) -> float:
        """The amount of time necessary to serve a single item"""
        return self.analysis_data["expected_service_time"]

    @expected_service_time.setter
    def expected_service_time(self, time: float) -> None:
        assert time >= 0.0, "Time must be positive"
        self.analysis_data["expected_service_time"] = time

    @property
    def _expected_per_core_max_rate_naive(self) -> float:
        """The maximum rate if a single core was dedicated to the task"""
        return self.analysis_data["expected_per_core_max_rate_naive"]

    @_expected_per_core_max_rate_naive.setter
    def _expected_per_core_max_rate_naive(self, rate: float) -> None:
        assert rate >= 0.0, "Rate must be positive"
        self.analysis_data["expected_per_core_max_rate_naive"] = rate

    @property
    def dataset_record_ratio(self) -> float:
        """For source ops, the number of records per file. Else 0."""
        if self.is_src_node() and self.node.state.aggregate_elements_consumed:
            return (self.node.state.aggregate_elements_produced /
                    self.node.state.aggregate_elements_consumed)
        else:
            return 0.0

    @property
    def p_busy(self) -> float:
        """How often the op is busy"""
        return self.analysis_data["p_busy"]

    @p_busy.setter
    def p_busy(self, p: float) -> None:
        assert p >= 0.0, "p_busy must be positive"
        self.analysis_data["p_busy"] = p

    @property
    def p_wait(self) -> float:
        """How often the op is busy"""
        return self.analysis_data["p_wait"]

    @p_wait.setter
    def p_wait(self, p: float) -> None:
        assert p >= 0.0, "p_wait must be positive"
        self.analysis_data["p_wait"] = p

    @property
    def p_wait_blame(self) -> float:
        """How often the op is busy using edge detection"""
        return self.analysis_data["p_wait_blame"]

    @p_wait_blame.setter
    def p_wait_blame(self, p: float) -> None:
        assert p >= 0.0, "p_wait_blame must be positive"
        self.analysis_data["p_wait_blame"] = p

    @property
    def p_wait_blame_non_filtered(self) -> float:
        """How often the op is busy relative to max parent p_busy"""
        return self.analysis_data["p_wait_blame_non_filtered"]

    @p_wait_blame_non_filtered.setter
    def p_wait_blame_non_filtered(self, p: float) -> None:
        assert p >= 0.0, "p_wait_blame_non_filtered must be positive"
        self.analysis_data["p_wait_blame_non_filtered"] = p

    @property
    def p_scheduling(self) -> float:
        """How often the op is busy"""
        return self.analysis_data["p_scheduling"]

    @p_scheduling.setter
    def p_scheduling(self, p: float) -> None:
        assert p >= 0.0, "p_scheduling must be positive"
        self.analysis_data["p_scheduling"] = p

    @property
    def wait_time(self) -> float:
        return self.analysis_data["wait_time"]

    @wait_time.setter
    def wait_time(self, time: float) -> None:
        self.analysis_data["wait_time"] = time

    @property
    def wait_time_diff(self) -> float:
        """How often the op is busy"""
        return self.analysis_data["wait_time_diff"]

    @wait_time_diff.setter
    def wait_time_diff(self, diff: float) -> None:
        self.analysis_data["wait_time_diff"] = diff

    @property
    def p_udf_time(self):
        return (self.node.state.aggregate_udf_processing_time
                / self.node.state.aggregate_processing_time)

    @property
    def p_udf_time_clock(self):
        return (self.node.state.aggregate_udf_processing_time_clock
                / self.node.state.aggregate_processing_time_clock)

    @property
    def num_cores_used(self) -> float:
        """The amount of CPU cores that this operation is using"""
        return self.analysis_data["num_cores_used"]

    @num_cores_used.setter
    def num_cores_used(self, cores: float) -> None:
        assert cores >= 0.0, "cores must be positive"
        self.analysis_data["num_cores_used"] = cores

    @property
    def bandwidth_used(self) -> float:
        """The amount of bytes/second that this operation is using"""
        if self.is_disk_node():
            return self.state.aggregate_bytes_produced / self.elapsed_time
        else:
            return 0.0

    @property
    def elapsed_time(self) -> float:
        """The amount of seconds the operator has run"""
        return self.analysis_data["elapsed_time"]

    @elapsed_time.setter
    def elapsed_time(self, time: float) -> None:
        self.analysis_data["elapsed_time"] = time

    @property
    def max_memory_used(self) -> int:
        """The maximum amount of bytes used by the operation in megabytes.
        Note: for unbounded ops (groupby), this is an approximation"""
        buffer_size = self.state.aggregate_max_buffer_size
        if not buffer_size:
            # TODO(mkuchnik): This is a messy assumption of how misc is used.
            buffer_size = self.state.aggregate_misc_buffer_size
        if not buffer_size:
            buffer_size = self.outer_parallelism
        bytes_per_element = self.state.aggregate_max_bytes_per_element
        max_memory_used = buffer_size * bytes_per_element
        return  max_memory_used

    @property
    def _num_cores_used_naive(self) -> float:
        """The amount of CPU cores that this operation is using"""
        return self.analysis_data["num_cores_used_naive"]

    @_num_cores_used_naive.setter
    def _num_cores_used_naive(self, cores: float) -> None:
        assert cores >= 0.0, "cores must be positive"
        self.analysis_data["num_cores_used_naive"] = cores

    @property
    def scheduling_delay(self) -> float:
        """The time spent waiting per element to run on executor.
        In units of minibatch/seconds"""
        # TODO(mkuchnik): Units seem off and uniform
        if self.node.state:
            return (self.node.state.aggregate_scheduling_delay_time /
                    max(self.node.state.elements_produced, 1) / 1e9 *
                    self.element_ratio)
        else:
            return 0.0

    @property
    def analysis(self) -> Dict:
        """Contains analysis data and anything else worthy of analysis"""
        return self.analysis_data

    @property
    def element_ratio(self) -> float:
        return self.analysis_data["element_ratio"]

    @element_ratio.setter
    def element_ratio(self, ratio: float) -> None:
        assert ratio >= 0.0, "Ratio must be positive, but is {}".format(ratio)
        self.analysis_data["element_ratio"] = ratio

    @property
    def input_names(self) -> List[NodeName]:
        return [n.name for n in self.node.input]

    @property
    def function_names(self) -> List[NodeName]:
        return [n.name for n in self.node.function]

    @property
    def name(self) -> NodeName:
        return self.node.name

    @property
    def op(self) -> NodeOp:
        return self.node.op

    @property
    def resource_type(self) -> OpResourceType:
        return OpResourceType.op_to_resource_type(self.op)

    @property
    def state(self) -> NodeState:
        return self.node.state

    @property
    def parent(self) -> 'AnalysisTreeNode':
        return self._parent

    @parent.setter
    def parent(self, node: 'AnalysisTreeNode') -> None:
        """Sets the parent node"""
        self._parent = node

    @property
    def input_nodes(self) -> List['AnalysisTreeNode']:
        return self._inputs

    def add_input_node(self, node: 'AnalysisTreeNode') -> None:
        self._inputs.append(node)

    @property
    def is_stale(self) -> bool:
        return self.analysis_data["is_stale"]

    @is_stale.setter
    def is_stale(self, stale: bool) -> None:
        self.analysis_data["is_stale"] = stale

    @property
    def has_outer_parallelism(self) -> bool:
        return bool(self.analysis_data["outer_parallelism"])

    @property
    def outer_parallelism_parent(self) -> Optional['AnalysisTreeNode']:
        if "outer_parallelism_parent" in self.analysis_data:
            return self.analysis_data["outer_parallelism_parent"]
        else:
            return None

    @outer_parallelism_parent.setter
    def outer_parallelism_parent(self, parent: 'AnalysisTreeNode') -> None:
        self.analysis_data["outer_parallelism_parent"] = parent

    @property
    def outer_parallelism(self) -> int:
        return self.analysis_data["outer_parallelism"]

    @outer_parallelism.setter
    def outer_parallelism(self, parallelism: bool) -> None:
        self.analysis_data["outer_parallelism"] = parallelism

    @property
    def average_bytes_per_element_produced(self) -> float:
        elements_produced = self.state.aggregate_elements_produced
        if not elements_produced:
            return 0.
        return self.state.aggregate_bytes_produced / elements_produced

    @property
    def average_bytes_per_element_consumed(self) -> float:
        # NOTE(mkuchnik): This is not collected for any node but TFRecord, so we
        # use normal 'elements_consumed'
        elements_consumed = self.state.aggregate_elements_consumed
        if not elements_consumed:
            if self.input_nodes and self.input_nodes[0].has_state():
                # NOTE(mkuchnik): Assume same across inputs
                elements_consumed = \
                self.input_nodes[0].state.aggregate_elements_produced
            else:
                return 0.
        if not elements_consumed:
            return 0.
        return self.state.aggregate_bytes_produced / elements_consumed

    @property
    def byte_ratio(self) -> float:
        consumed = self.state.aggregate_bytes_consumed
        if consumed:
            return self.state.aggregate_bytes_produced / consumed
        else:
            return 0.0

    @property
    def cardinality(self) -> int:
        cardinality = self.state.cardinality
        if cardinality <= 0:
            cardinality = DatasetCardinality(cardinality)
        return cardinality

    @property
    def derived_cardinality(self):
        try:
            return self.analysis_data["derived_cardinality"]
        except KeyError:
            return self.cardinality

    @derived_cardinality.setter
    def derived_cardinality(self, size: float) -> None:
        self.analysis_data["derived_cardinality"] = size

    @property
    def expected_dataset_size(self) -> int:
        """How large a materialized version of the dataset would be.
        Negative sizes are infinite cardinality."""
        return self.analysis_data["expected_dataset_size"]

    @expected_dataset_size.setter
    def expected_dataset_size(self, size: int) -> None:
        self.analysis_data["expected_dataset_size"] = size

    @property
    def expected_num_dataset_files(self) -> int:
        """How many dataset files are seen at this node."""
        return self.analysis_data["expected_num_dataset_files"]

    @expected_num_dataset_files.setter
    def expected_num_dataset_files(self, size: int) -> None:
        self.analysis_data["expected_num_dataset_files"] = size

    def has_analysis(self) -> bool:
        return bool(self.analysis_data) and self.has_state()

    def has_state(self) -> bool:
        return bool(self.state)

    def is_src_node(self) -> bool:
        return self.op in SRC_DATASET_OP_NODES

    def is_disk_node(self) -> bool:
        return self.op in DISK_DATASET_OP_NODES

    def is_cache_node(self) -> bool:
        return self.op in CACHE_DATASET_OP_NODES

    def is_tensor_slice_node(self) -> bool:
        return self.op in TENSOR_SLICE_DATASET_OP_NODES

    def is_take_node(self) -> bool:
        return self.op in TAKE_DATASET_OP_NODES

    def is_range_node(self) -> bool:
        return self.op in RANGE_DATASET_OP_NODES

    def is_repeat_node(self) -> bool:
        return self.op in REPEAT_DATASET_OP_NODES

    def is_interleave_node(self) -> bool:
        return self.op in INTERLEAVE_DATASET_OP_NODES

    def is_snapshot_node(self) -> bool:
        return self.op in SNAPSHOT_DATASET_OP_NODES

    def is_zip_node(self) -> bool:
        return self.op in ZIP_DATASET_OP_NODES

    def is_group_by_window_node(self) -> bool:
        return self.op in GROUP_BY_WINDOW_DATASET_OP_NODES

    def is_parallel_node(self) -> bool:
        return self.op in PARALLEL_DATASET_OP_NODES

    def is_parallelizable_node(self) -> bool:
        return self.op in PARALLELIZABLE_DATASET_OP_NODES

    def to_summary_dict(self) -> Dict:
        # NOTE(mkuchnik): Weird API
        d = dict(self.node.content_dict())
        d.update(self.analysis)
        return d

    def owned_nodes(self) -> Iterator['AnalysisTreeNode']:
        return iter(self.function)



class AnalysisDatasetTree(object):
    """Adds internal analysis variables to a DatasetTree"""
    ROOT_NODE_NAME = NodeName("dataset")

    def __init__(self, dataset_tree: DatasetTree,
                 global_state: SnapshotGlobalState):
        nodes = dataset_tree.nodes()
        analysis_nodes = list(map(AnalysisTreeNode, nodes))
        for n in analysis_nodes:
            n.function = list(map(AnalysisTreeNode, n.node.function))
        self._node_lookup = dict()
        for n in analysis_nodes:
            self._node_lookup[NodeName(n.name)] = n
        self.global_state = global_state
        self.src_nodes = dict()
        self._init_analysis()

    def root(self) -> 'AnalysisTreeNode':
        """Convenience method for root"""
        return self.lookup_name(self.ROOT_NODE_NAME)

    def lookup_name(self, node_name: NodeName) -> 'AnalysisTreeNode':
        """Returns the node corresponding to the node_name"""
        return self._node_lookup[node_name]

    def _init_analysis(self) -> None:
        self._propagate_analysis(self.root())
        for n in self.src_nodes.values():
            # NOTE(mkuchnik): May want to conserve global state
            self._propagate_analysis_backward(n)

    def _propagate_analysis(self, node: AnalysisTreeNode,
                            internal_state: Optional[Dict]=None) -> None:
        internal_state = {} if internal_state is None else internal_state
        # NOTE(mkuchnik): On forward pass, we set parameters to None
        # to indicate unknowns
        node.expected_dataset_size = None
        node.expected_num_dataset_files = 0
        if node.has_state():
            elapsed_time = self.global_state.elapsed_time_seconds()
            node.elapsed_time = elapsed_time
            # NOTE(mkuchnik): Element ratio is off by one node, so propagate
            if "element_ratio" in internal_state:
                assert internal_state["element_ratio"] >= 0.0
                element_ratio = internal_state["element_ratio"]
            else:
                element_ratio = 1.0
            if (("is_function" in internal_state) and
                 internal_state["is_function"]):
                if not element_ratio:
                    # TODO Some element have 0.0
                    element_ratio = 1.0
            node.element_ratio = element_ratio
            def safe_division(x):
                return 1. if x == 0. else x
            if not node.element_ratio:
                logger.warning("Encounter {} element_ratio in {}".format(
                    node.element_ratio, node.name))
            node.wait_time = (node.node.state.aggregate_wait_time
                / safe_division(node.node.state.aggregate_elements_produced)
                / safe_division(node.element_ratio))
            node.p_wait = (node.node.state.aggregate_wait_time / 1e9
                           / elapsed_time
                           / safe_division(node.element_ratio))
            node.p_scheduling = (node.scheduling_delay
                                 / elapsed_time
                                 / safe_division(node.element_ratio))
            if "parent_wait_time" in internal_state:
                parent_wait_time_diff = (internal_state["parent_wait_time"]
                                         - node.wait_time)
            else:
                parent_wait_time_diff = None
            node.wait_time_diff = parent_wait_time_diff
            if "max_p_wait" in internal_state:
                p_wait_blame = max(internal_state["max_p_wait"] - node.p_wait,
                                   0)
            else:
                p_wait_blame = 0.0
            node.p_wait_blame_non_filtered = p_wait_blame
            next_element_ratio = node.state.element_ratio
            if ((next_element_ratio < 0) or
                    (not next_element_ratio and node.is_cache_node())):
                next_element_ratio = 1.0
            next_element_ratio *= element_ratio
            internal_state["element_ratio"] = next_element_ratio
            internal_state["parent_wait_time"] = node.wait_time
            if "max_p_wait" in internal_state:
                internal_state["max_p_wait"] = max(internal_state["max_p_wait"],
                                                   node.p_wait)
            else:
                internal_state["max_p_wait"] = node.p_wait
            obs_rate = (node.state.aggregate_elements_produced / elapsed_time
                        / safe_division(node.element_ratio))
            node.observed_rate = obs_rate
            processing_time = node.state.aggregate_processing_time_clock / 1e9
            # For analysis, we also try using plain wallclock
            processing_time_naive = node.state.aggregate_processing_time / 1e9
            # NOTE(mkuchnik): Using requested parallelism is a good measure of
            # p_busy as it means that overprovisioned nodes will be penalized if
            # they cannot use all parallelism
            parallelism = float(node.parallelism)
            p_busy = processing_time / parallelism / elapsed_time
            # NOTE(mkuchnik): p_busy may be low merely due to contention from
            # other threads
            def p_busy_trace(node) -> str:
                """A debug function for tracing how p_busy was constructed"""
                dbg_str = ("p_busy = "
                "(processing_time={})/(parallelism={})/(elapsed_time={})"
                .format(processing_time, parallelism, elapsed_time))
                return dbg_str
            # NOTE(mkuchnik): This is very loose check for sanity. It is
            # possible for p_busy to be higher than 1.0 if inter-op parallelism
            # is used. However, it should never be less than 0.0.
            assert p_busy <= 100.0, \
                ("p_busy={} for {} should not be much greater than 1.0. "
                 "DEBUG: {}"
                 .format(p_busy, node.name, p_busy_trace(node)))
            assert p_busy >= 0.0, \
                ("p_busy={} for {} should not be less than 0.0. "
                 "DEBUG: {}"
                 .format(p_busy, node.name, p_busy_trace(node)))
            # NOTE(mkuchnik): It's important not to clamp parallelism
            if p_busy > 1.0:
                logger.info("p_busy={} for {} should be capped at "
                            "1.0. Inter-op parallelism may be very high."
                            .format(p_busy, node.name))
            node.p_busy = p_busy
            # NOTE(mkuchnik) node.state.aggregate_avg_number_active_threads
            # has number of actual running threads
            # num_threads_used = node.state.aggregate_avg_number_active_threads
            num_cores_used = processing_time / elapsed_time
            num_cores_used_naive = processing_time_naive / elapsed_time
            node.num_cores_used = num_cores_used
            node._num_cores_used_naive = num_cores_used_naive
            if element_ratio:
                minibatches_produced = (node.state.aggregate_elements_produced /
                                        element_ratio)
            else:
                minibatches_produced = 0.
            if processing_time:
                max_proc_rate = (minibatches_produced /
                                 processing_time)
            else:
                max_proc_rate = 0.
            if processing_time_naive:
                max_proc_rate_naive = (minibatches_produced /
                                       processing_time_naive)
            else:
                max_proc_rate_naive = 0.
            node.expected_per_core_max_rate = max_proc_rate
            # TODO(mkuchnik): Maybe used naive wallclock time?
            if max_proc_rate:
                node.expected_service_time = 1. / max_proc_rate
            else:
                node.expected_service_time = 0.0
            node._expected_per_core_max_rate_naive = max_proc_rate_naive
            is_stale = node.is_cache_node() or node.is_tensor_slice_node()
            # TODO(mkuchnik): TFRecords with filenames are not captured
            if "is_stale" in internal_state:
                is_stale |= internal_state["is_stale"]
            internal_state["is_stale"] = is_stale
            if "outer_parallelism" in internal_state:
                outer_parallelism = internal_state["outer_parallelism"]
            else:
                outer_parallelism = 0
            node.is_stale = is_stale
            node.outer_parallelism = outer_parallelism
            if "outer_parallelism_parent" in internal_state:
                node.outer_parallelism_parent = \
                    internal_state["outer_parallelism_parent"]
            internal_state["is_stale"] = is_stale
            if node.is_interleave_node():
                if not outer_parallelism:
                    internal_state["outer_parallelism"] = node.parallelism
                else:
                    internal_state["outer_parallelism"] = (node.parallelism *
                                                           outer_parallelism)
                # NOTE(mkuchnik): Override parent
                internal_state["outer_parallelism_parent"] = node
            else:
                internal_state["outer_parallelism"] = (node.parallelism *
                                                       outer_parallelism)
        max_child_p_wait = 0.0
        sum_child_p_wait = 0.0
        for node_name in node.input_names:
            input_node = self.lookup_name(node_name)
            input_node.parent = node
            node.add_input_node(input_node)
            _internal_state = dict(internal_state)
            if node.has_analysis() and node.is_interleave_node():
                # NOTE(mkuchnik): Interleave propagates junk to child, so copy
                logger.info("Copying element_ratio={} into children of {}"
                            " (as opposed to {})".format(
                             node.element_ratio, node.name,
                             _internal_state["element_ratio"]))
                _internal_state["element_ratio"] = node.element_ratio
            self._propagate_analysis(input_node, _internal_state)
            if input_node.has_analysis():
                max_child_p_wait = max(max_child_p_wait, input_node.p_wait)
                sum_child_p_wait += input_node.p_wait
        for function_node in node.owned_nodes():
            _internal_state = dict(internal_state)
            _internal_state["is_function"] = True
            if node.has_analysis() and node.is_interleave_node():
                # NOTE(mkuchnik): Interleave propagates junk to child, so copy
                logger.info("Copying element_ratio={} into children of {}"
                                " (as opposed to {})".format(
                                 node.element_ratio, node.name,
                                 _internal_state["element_ratio"]))
                _internal_state["element_ratio"] = node.element_ratio
            elif node.has_analysis() and node.is_group_by_window_node():
                # NOTE(mkuchnik): group_by_window window function doesn't
                # change input-output relationship
                logger.info("Copying element_ratio={} into children of {}"
                            " (as opposed to {})".format(
                             1.0, node.name,
                             _internal_state["element_ratio"]))
                _internal_state["element_ratio"] = 1.0
            function_node.parent = node
            self._propagate_analysis(function_node,
                                     _internal_state)
        # We cannot blame root
        if node.parent and node.parent.has_analysis():
            p_wait = 0.
            if node.has_analysis():
                p_wait = node.p_wait
            if node.is_disk_node():
                # For disk nodes, we wait on file reads only (no child)
                assert not max_child_p_wait, "Src node has child"
                node.p_wait_blame = p_wait
            elif node.is_interleave_node():
                # For interleave nodes, we want to inherit the blame from
                # children, since they are blocking us, and we wrap them.
                # Interleave is asynchronous, which means that it fetches as
                # fast as it can (independent of parent)
                p_wait += sum_child_p_wait
                node.p_wait_blame = min(max(p_wait + sum_child_p_wait, 0.), 1.)
            else:
                # For the rest, we perform edge detection on waiting
                node.p_wait_blame = max(node.parent.p_wait
                                        - max_child_p_wait
                                        - p_wait, 0.)
        else:
            node.p_wait_blame = 0.
        if node.is_src_node() and node.name not in self.src_nodes:
            self.src_nodes[node.name] = node

    def _propagate_analysis_backward(self, node: AnalysisTreeNode,
                            internal_state: Optional[Dict]=None) -> None:
        internal_state = {} if internal_state is None else internal_state
        def safe_division(x):
            return 1. if x == 0. else x
        if (not node.has_state() or "is_stale" in internal_state
                and internal_state["is_stale"]):
            # Give up for state-less nodes.
            if "dataset_size" in internal_state:
                node.expected_dataset_size = internal_state["dataset_size"]
            else:
                node.expected_dataset_size = None
            if node.parent:
                self._propagate_analysis_backward(node.parent, internal_state)
            return
        if node.is_src_node() and node.is_disk_node():
            # We are at a src node, which has no children and is reading from
            # storage. By definition, caching here would lead to full read.
            dataset_size = sum(
                self.global_state.ctx_info.file_sizes.values())
            observed_dataset_files = len(self.global_state.ctx_info.file_sizes)
            def find_num_files(node):
                """Usually interleave's node's (e.g., TensorSlice)
                have num_files cardinality"""
                # TODO(mkuchnik): The number here is NOT the number of files.
                # Cardinality is 1 usually for TensorSliceDataset.
                # Rather, get from metadata if possible
                parent = node.parent
                metadata = parent.node.metadata
                if "num_files" in metadata:
                    num_files = metadata["num_files"]
                    logger.info("Found metadata for num files {} in {}".format(
                        num_files, node.name))
                    return num_files
                else:
                    num_files = None
                    logger.info("Visiting children for num_files in "
                                 "{}".format(parent.name))
                    for x in parent.owned_nodes():
                        logger.info("Visiting num_files in "
                                     "{}".format(x.name))
                        if "num_files" in x.node.metadata:
                            if num_files is None:
                                num_files = x.node.metadata["num_files"]
                            elif num_files == x.node.metadata["num_files"]:
                                logger.warning("Found multiple num_files")
                            else:
                                logger.warning("Found multiple num_files with"
                                                " different numbers")
                                num_files = max(num_files,
                                                x.node.metadata["num_files"])
                    if num_files is not None:
                        return num_files
                    logger.warning("Cannot find number of files metadata for"
                                    " {}".format(parent.name))
                    parent_cardinalities = [x.cardinality for x in
                                            parent.owned_nodes() if
                                            x.has_state()
                                            and x.cardinality >= 0]
                    if parent_cardinalities:
                        function_cardinalities = max(parent_cardinalities)
                    else:
                        function_cardinalities = None
                    return function_cardinalities
            expected_num_dataset_files = find_num_files(node)
            node.expected_num_dataset_files = expected_num_dataset_files
            if expected_num_dataset_files is not None:
                assert expected_num_dataset_files >= 0, \
                    "num_dataset_files < 0 ({},{})".format(
                        expected_num_dataset_files, node.name)
                observed_dataset_ratio = (observed_dataset_files /
                                          expected_num_dataset_files)
                if not observed_dataset_ratio:
                    logger.error("Node {} observed dataset ratio is 0."
                                  " Casting to -1.".format(node.name))
                    observed_dataset_ratio = -1.
                node.expected_dataset_size = (dataset_size
                                              / observed_dataset_ratio)
                num_records = (node.expected_num_dataset_files
                               * node.dataset_record_ratio)
                node.derived_cardinality = num_records
            else:
                node.expected_dataset_size = -1
                num_records = -1
            internal_state["dataset_size"] = node.expected_dataset_size
            internal_state["num_records"] = num_records
            internal_state["byte_ratio"] = node.byte_ratio
        elif node.is_src_node():
            # Not a disk node but a src node, probably RangeDataset
            logger.info("BACKWARD SRC: {}, {}".format(node.name,
                                                       node.cardinality))
            num_records = node.cardinality
            dataset_size = num_records * 4  # Assume int datatype
            node.derived_cardinality = num_records
            node.expected_dataset_size = dataset_size
            internal_state["dataset_size"] = node.expected_dataset_size
            internal_state["num_records"] = num_records
            internal_state["byte_ratio"] = node.byte_ratio
        elif node.is_interleave_node() and node.cardinality == DatasetCardinality.UNKNOWN:
            # NOTE(mkuchnik): we override UNKNOWN (-2) cardinality
            # TODO(mkuchnik): Check that flatmap is not doing anything crazy
            # with dataset_size or num_records (e.g., decoding in interleave).
            logger.info("BACKWARD Interleave: {}, {}".format(node.name,
                                                              node.cardinality))
            try:
                dataset_size = internal_state["dataset_size"]
                num_records = internal_state["num_records"]
            except KeyError as ex:
                logger.error(ex)
                logger.warning("Failed to resolve backward pass for "
                                "{}".format(node.name))
                dataset_size = -2
                num_records = -2
            node.derived_cardinality = num_records
            node.expected_dataset_size = dataset_size
            internal_state["dataset_size"] = node.expected_dataset_size
            internal_state["num_records"] = num_records
            internal_state["byte_ratio"] = node.byte_ratio  # Set
        elif node.is_take_node():
            # Take is always finite. Use cardinality to get num_records.
            metadata = node.node.metadata
            if "take_size" in metadata:
                num_records = metadata["take_size"]
            else:
                logger.warning("Could not find take_size in metadata for "
                                "{}".format(node.name))
                num_records = node.cardinality
            node.derived_cardinality = num_records
            bytes_per_record = node.average_bytes_per_element_produced
            # TODO(mkuchnik): Consider max bytes per element
            node.expected_dataset_size = num_records * bytes_per_record
            internal_state["dataset_size"] = node.expected_dataset_size
            internal_state["num_records"] = num_records
            internal_state["byte_ratio"] = 1 # Set
        elif node.is_repeat_node() and node.cardinality != DatasetCardinality.INFINITE:
            # Take is always finite. Use cardinality to get num_records.
            metadata = node.node.metadata
            if "repeat_size" in metadata:
                repeat_size = metadata["repeat_size"]
            else:
                logger.warning("Could not find repeat_size in metadata for "
                                "{}".format(node.name))
                repeat_size = -1
            parent_num_records = internal_state["num_records"]
            num_records = repeat_size * parent_num_records
            node.derived_cardinality = num_records
            bytes_per_record = node.average_bytes_per_element_produced
            node.expected_dataset_size = num_records * bytes_per_record
            internal_state["dataset_size"] = node.expected_dataset_size
            internal_state["num_records"] = num_records
            internal_state["byte_ratio"] = 1 # Set
        else:
            # We are not in a src_node anymore. Until we hit interleave, we are
            # guaranteed to have constant size dataset. Once we pass interleave,
            # we may start playing with the input/output byte ratios (e.g.,
            # decompression) or size may grow infinite (e.g., repeat)
            logger.info("BACKWARD OTHER: {}, cardinality {}, ratio {}".format(
                node.name,
                node.cardinality,
                node.state.element_ratio))
            try:
                child_dataset_size = internal_state["dataset_size"]
                child_num_records = internal_state["num_records"]
                child_byte_ratio = internal_state["byte_ratio"]
            except KeyError as ex:
                logger.error("Analysis propagation fail on node "
                              "{}\n{}".format(node.name, ex))
                child_dataset_size = -2
                child_num_records = -2
                child_byte_ratio = 1.0
            if node.cardinality == -1 or child_dataset_size < 0:
                # Infinite cardinality blows up dataset
                # NOTE(mkuchnik): -2 is unknown and is not infinite
                logger.info("Setting node {} dataset size to INFINITE"
                             " (-1)".format(node.name))
                dataset_size = -1
                num_records = -1
            else:
                # NOTE(mkuchnik): we use local element ratio
                # byte ratio can also work, but for simplicity use
                # NOTE(mkuchnik): Filter with 50/1000 filtering will have ratio
                # 20, thus divide
                if node.state.element_ratio:
                    num_records = child_num_records / node.state.element_ratio
                else:
                    logger.error("Encountered {} element_ratio in "
                                  "{}".format(node.state.element_ratio,
                                             node.name))
                    num_records = child_num_records
                bytes_per_record = node.average_bytes_per_element_produced
                dataset_size = num_records * bytes_per_record
                logger.info("{} bytes {}, records {}({}) -> {} ({})".format(
                    node.name,
                    bytes_per_record,
                    child_num_records,
                    child_num_records * child_dataset_size,
                    num_records,
                    dataset_size,
                ))
            node.derived_cardinality = num_records
            node.expected_dataset_size = dataset_size
            internal_state["dataset_size"] = dataset_size
            internal_state["num_records"] = num_records
            internal_state["byte_ratio"] *= node.byte_ratio

        if node.parent:
            self._propagate_analysis_backward(node.parent, internal_state)

    def node_names(self) -> Iterator[NodeName]:
        return map(NodeName, self._node_lookup.keys())

    def nodes(self) -> Iterator['AnalysisTreeNode']:
        return self._node_lookup.values()

    def extended_nodes(self) -> Iterator['AnalysisTreeNode']:
        """Includes all reference nodes too (e.g., functions)"""
        def get_internal_iter(node):
            yield node
            for n in node.owned_nodes():
                yield n
        return itertools.chain.from_iterable(
            map(get_internal_iter, self._node_lookup.values()))

    def disk_bytes_per_root_element(self) -> float:
        """Divide this by disk bandwidth to get upper bounds"""
        def bytes_per_element(node):
            # TODO replace with node.average_bytes_per_element_produced?
            disk_bytes = node.state.aggregate_disk_bytes_read
            num_elements = node.state.aggregate_elements_produced
            disk_bytes_per_element = disk_bytes / max(num_elements, 1)
            return  disk_bytes_per_element * node.element_ratio
        bytes_per_element = [(bytes_per_element(n), n.name) for n in
                             self.extended_nodes() if n.has_analysis() and
                             n.is_disk_node()]
        if not bytes_per_element:
            return None
        if len(bytes_per_element) > 1:
            logger.warning("multiple bytes per element: {}".format(
                str(bytes_per_element)))
        bytes_per_element = max([x[0] for x in bytes_per_element])
        return bytes_per_element

    def min_latency(self) -> float:
        """Returns the minimum of the latency of the pipeline"""
        nodes = filter(lambda x: x.has_state() and x.has_analysis() and not x.is_stale,
                       self.extended_nodes())
        return sum(map(lambda x: x.expected_service_time, nodes))

    def _bottleneck_rank_function(self, node,
                                  mode: Optional[str]=None) -> float:
        """Used to rank nodes in terms of bottleneck.
        Ranks go from low to high."""
        if mode is None:
            return node.expected_parallel_max_rate()
        elif mode == "thresholded_parallelism":
            # A more conservative approach is to assume parallelism is bounded
            return node.expected_parallel_max_rate(
                parallelism=min(node.parallelism,
                                self.global_state.machine_info.num_cores)
            )
        elif mode == "p_busy":
            # p_busy is often a good indicator of bottlenecks, because it is a
            # measure of resources used vs. resources allocated. If parallelism
            # is very high, it will likely not be used and thus yield lower
            # parallelism.
            return -node.p_busy
        elif mode == "contention_p_busy":
            # Nodes may get non-fair slices of the CPU
            return -node.p_busy / node.state.aggregate_avg_number_active_threads
        elif mode == "wait_time_diff":
            if node.wait_time_diff is not None:
                return -node.wait_time_diff
            else:
                return 1e18
        elif mode == "p_wait_blame":
            return -node.p_wait_blame
        else:
            raise ValueError("Mode={} is not recognized".format(mode))

    def bottleneck_node(self, mode: Optional[str]=None) -> 'AnalysisTreeNode':
        """Returns the bottleneck node"""
        # TODO(mkuchnik): Add functions to analysis
        nodes = filter(lambda x: x.has_state() and x.has_analysis() and not x.is_stale,
                       self.extended_nodes())
        return min(nodes, key=lambda x: self._bottleneck_rank_function(x, mode))

    def ranked_list_bottleneck_nodes(self,
                                     mode: Optional[str]=None,
                                     extended: bool=True) -> List['AnalysisTreeNode']:
        """Returns ascending order sort"""
        # TODO(mkuchnik): Add functions to analysis
        if extended:
            nodes = filter(lambda x: x.has_state() and x.has_analysis() and not x.is_stale,
                           self.extended_nodes())
            return sorted(nodes, key=lambda x: self._bottleneck_rank_function(x,
                                                                              mode))
        else:
            nodes = filter(lambda x: x.has_state() and x.has_analysis() and not x.is_stale,
                           self.nodes())
            return sorted(nodes, key=lambda x: self._bottleneck_rank_function(x,
                                                                              mode))

class PerformanceAnalysis(object):
    """Forms internal analysis on a copy (adding annotations)"""
    def __init__(self, performance_model: PerformanceModel):
        self._performance_model = performance_model
        nodes = [n for n in self._performance_model.dataset_graph().nodes() if
                 n.has_state()]
        if not nodes:
            raise RuntimeError("None of the nodes in the graph have valid"
                               " state. Perhaps statistics were not dumped.")
        self._analysis_tree = \
            AnalysisDatasetTree(self._performance_model.dataset_graph(),
                                self.global_state)

    @property
    def global_state(self):
        return self._performance_model.global_state

    def observed_rate(self) -> float:
        # TODO(mkuchnik): Move to model?
        nodes = filter(lambda x: x.has_state() and x.has_analysis()
                       and not x.is_stale,
                       self.nodes())
        min_rate = min(map(lambda x: x.observed_rate, nodes))
        return min_rate

    def min_latency(self) -> float:
        return self._analysis_tree.min_latency()

    def iterator_duration(self) -> float:
        return self.global_state.iter_stats.avg_duration

    def iterator_wallclock_duration(self) -> float:
        return self.global_state.iter_stats.avg_wallclock_duration

    def iterator_variance(self) -> float:
        return self.global_state.iter_stats.var_duration

    def iterator_autotune_output_time(self) -> float:
        return self.global_state.iter_stats.autotune_output_time

    def bottleneck_node(self,
                        mode: Optional[str]=None) -> 'AnalysisTreeNode':
        return self._analysis_tree.bottleneck_node(mode)

    def ranked_list_bottleneck_nodes(self,
            mode: Optional[str]=None, extended=True) -> List['AnalysisTreeNode']:
        return self._analysis_tree.ranked_list_bottleneck_nodes(
            mode, extended=extended)

    def node_names(self) -> Iterator[NodeName]:
        return self._analysis_tree.node_names()

    def nodes(self) -> Iterator['AnalysisTreeNode']:
        return self._analysis_tree.nodes()

    def extended_nodes(self) -> Iterator['AnalysisTreeNode']:
        return self._analysis_tree.extended_nodes()

    def lookup_name(self, node_name: NodeName) -> 'AnalysisTreeNode':
        """Returns the node corresponding to the node_name"""
        return self._analysis_tree.lookup_name(node_name)

    def disk_bytes_per_root_element(self) -> float:
        """The amount of disk bytes necessary to compute a single minibatch"""
        return self._analysis_tree.disk_bytes_per_root_element()

    def span_contexts(self):
        return iter([])

    def sink_nodes(self) -> Iterator['AnalysisTreeNode']:
        def is_sink_nodes(node):
            return node.is_src_node() and not node.input_nodes
        return iter(filter(is_sink_nodes, self._analysis_tree.extended_nodes()))

    def projected_dataset_working_set_size(self) -> float:
        def is_possible_source(node) -> bool:
            if node.has_analysis():
                is_valid = (node.is_range_node() and node.parent
                            and node.parent.is_zip_node()
                            and node.parent.parent
                            and node.parent.parent.is_snapshot_node())
                # TODO(mkuchnik): Hacky way to check for snapshot
                # Make the check node position in Zip
                if not is_valid:
                    # NOTE(mkuchnik): We assume snapshots have RangeDataset
                    # in this case, we ignore these virtual nodes
                    return True
                logger.info("Ignoring {} for dataset working"
                             " set".format(node.name))
            return False

        projected_sizes = [(node.expected_dataset_size, node.name)
                           for node in self.sink_nodes() if
                           is_possible_source(node)]
        if projected_sizes:
            size = projected_sizes[0][0]
            assert all(map(lambda x: x[0] == size, projected_sizes)), \
                "{} not equal".format(projected_sizes)
            return size
        else:
            raise RuntimeError("Didn't find any sink nodes")


class PerformanceRecommendationv1(PerformanceRecommendation):
    """Analyzes state to form recommendations on what nodes are bottlenecks"""
    def __init__(self, performance_model: Union[PerformanceModel,
                                                PerformanceAnalysis]):
        if isinstance(performance_model, PerformanceModel):
            self._analysis = PerformanceAnalysis(performance_model)
        elif isinstance(performance_model, PerformanceAnalysis):
            self._analysis = performance_model
        else:
            raise ValueError("Unknown parameter {}".format(performance_model))

    def bottleneck_node(self, mode: Optional[str]=None) -> 'TreeNode':
        """Returns a pointer to the bottleneck node"""
        return self.bottleneck_node_analysis(mode).node

    def ranked_list_bottleneck_nodes(self,
            mode: Optional[str]=None, extended: bool=True) -> List['TreeNode']:
        return list(map(lambda x: x.node,
                        self.ranked_list_bottleneck_nodes_analysis(
                            mode, extended=extended)))

    def min_latency(self) -> float:
        return self._analysis.min_latency()

    def iterator_duration(self) -> float:
        return self._analysis.iterator_duration()

    def iterator_wallclock_duration(self) -> float:
        return self._analysis.iterator_wallclock_duration()

    def iterator_variance(self) -> float:
        return self._analysis.iterator_variance()

    def iterator_autotune_output_time(self) -> float:
        return self._analysis.iterator_autotune_output_time()

    def current_rate(self, mode: Optional[str]=None) -> float:
        """Returns the data pipeline's maximum rate"""
        return self.bottleneck_node_analysis(mode).expected_parallel_max_rate()

    def actual_rate(self) -> float:
        return self._analysis.observed_rate()

    def remaining_CPU_cores(self) -> float:
        """The number of cores that are not being used."""
        extra_cores = (self._analysis._performance_model
                           .fractional_cores_unutilized())
        return extra_cores

    def disk_upper_bounds(self, disk_bandwidth: float) -> float:
        """ Convenience function to compute upper bound given a bandwidth (in
        byte)"""
        if disk_bandwidth <= 0:
            raise ValueError("Disk bandwidth cannot be less than 0 "
                             "({})".format(disk_bandwidth))
        bytes_per_element = self.disk_bytes_per_root_element()
        if not bytes_per_element:
            return None
        return disk_bandwidth / self.disk_bytes_per_root_element()

    def disk_bytes_per_root_element(self) -> float:
        """The amount of disk bytes necessary to compute a single minibatch"""
        return self._analysis.disk_bytes_per_root_element()

    def projected_dataset_working_set_size(self) -> float:
        return self._analysis.projected_dataset_working_set_size()

    def upper_bounds(self,
                     keep_p_busy: bool=False,
                     global_rate_comparison=True,
                     mode: Optional[str]=None) -> float:
        """The maximum rate that can be achieved if the bottleneck node is
        parallelized.

        keep_p_busy: set to True for a pessiminstic estimate that conserves the
        busy factor"""
        node = self.bottleneck_node_analysis(mode)
        extra_cores = self.remaining_CPU_cores()
        # TODO(mkuchnik): reinstantiate new model with parallelism higher
        rate = node.expected_parallel_max_rate(extra_cores=extra_cores)
        if keep_p_busy:
            rate *= node.p_busy
        if global_rate_comparison:
            nodes = self.ranked_list_bottleneck_nodes_analysis(mode)
            assert nodes[0].name == node.name
            if len(nodes) >= 2:
                second_node_rate = nodes[1].expected_parallel_max_rate()
                if keep_p_busy:
                    second_node_rate *= nodes[1].p_busy
                if rate > second_node_rate:
                    # Node1 has p*c cores
                    # Node2 has (1-p)*c cores
                    # To get equality, we want Rate[p*c] = Rate[(1-p)*c]
                    # Rate[c] = per_core_rate * c
                    # p * per_core_rate_1 * c == (1-p) * per_core_rate_2 * c
                    # p / (1-p) = per_core_rate_2 / per_core_rate_1
                    # p^2 / (p - p^2) = ratio
                    # p^2 = ratio * (p - p^2)
                    # p^2 = ratio * p - ratio * p^2
                    # p^2 * (1+ratio) = ratio * p
                    # p^2 * (1+ratio) - p * ratio + 0 = 0
                    # self.expected_per_core_max_rate * parallelism
                    ratio = (nodes[1].expected_per_core_max_rate
                             / nodes[0].expected_per_core_max_rate)
                    # Quadratic formula
                    a = 1+ratio
                    b = -ratio
                    c = 0
                    p = (-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a)
                    assert p >= 0 and p <= 1
                    n2_extra_cores = (1 - p) * extra_cores
                    second_node_rate = nodes[1].expected_parallel_max_rate(
                        extra_cores=n2_extra_cores)
                    if keep_p_busy:
                        second_node_rate *= nodes[1].p_busy
                    rate = min(second_node_rate, rate)
        return rate

    def LP_upper_bounds(self, use_existing_usage: bool=False, debug: bool=False,
                        naive: bool=False,
                        return_current_throughput=False,
                        consider_outer_parallelism=True):
        if not check_cvxpy():
          message = (
              'Failed to import cvxpy. You must `pip install cvxpy` '
              'for `LP_upper_bounds` to work.')
          if 'IPython.core.magics.namespace' in sys.modules:
            # We don't raise an exception here in order to avoid crashing
            # notebook tests where graphviz is not available.
            logger.error(message)
            return
          else:
            raise ImportError(message)
        model = self._analysis._performance_model
        return PerformanceRecommendationv1.LP_upper_bounds_inner(
            model, use_existing_usage=use_existing_usage, debug=debug,
            naive=naive, return_current_throughput=return_current_throughput,
            consider_outer_parallelism=consider_outer_parallelism,
        )

    @staticmethod
    def LP_upper_bounds_inner(model, use_existing_usage=False,
                              debug=False, naive=False, process_correction=True,
                              return_current_throughput=False,
                              consider_outer_parallelism=True):
        """Calculates the upper bounds of plumber using a linear program
        The optimal throughput can be calculated according to the program:

        maximize T(theta) s.t., sum_i theta_i <= 1.0
        T(theta) = min[f_1(theta_1), ..., f_N(theta_N)]
        """
        # TODO(mkuchnik): deprecated debug flag
        recommendation = model.recommendation()
        ranked_nodes = \
                recommendation.ranked_list_bottleneck_nodes_analysis()
        num_cores = recommendation._analysis.global_state.machine_info.num_cores
        if naive:
            # For naive, we use wallclock time
            CPU_Util_clock = model.CPU_Util()
        else:
            CPU_Util_clock = model.CPU_Util(calculation_mode="CPU_clock")
        names = [n.name for n in ranked_nodes]
        if naive:
            rates = [n._expected_per_core_max_rate_naive for n in ranked_nodes]
        else:
            rates = [n.expected_per_core_max_rate for n in ranked_nodes]
        def parallel_check_fn(node):
            if consider_outer_parallelism:
                return node.is_parallel_node() or node.has_outer_parallelism
            else:
                return node.is_parallel_node()
        is_parallel = [parallel_check_fn(n) for n in ranked_nodes]
        parallelism = [n.parallelism for n in ranked_nodes]
        if naive:
            num_cores_used = [n._num_cores_used_naive for n in ranked_nodes]
        else:
            num_cores_used = [n.num_cores_used for n in ranked_nodes]
        num_cores_used = np.array(num_cores_used)
        logger.debug("rates:\n{}".format(rates))
        logger.debug("is_parallel:\n{}".format(is_parallel))
        logger.debug("parallelism:\n{}".format(parallelism))
        if return_current_throughput:
            current_throughput = min(num_cores_used * rates)
        if process_correction:
            # Correct using known process time
            process_time = model.total_CPU_process_time()
            correction = np.sum(num_cores_used) / process_time
            num_cores_used /= correction
        N = len(rates)
        theta_min = np.zeros(N)
        if use_existing_usage:
            for i in range(len(num_cores_used)):
                c = num_cores_used[i]
                if not is_parallel[i] and c > 1.:
                    logger.warning(
                        "WARNING: cores used greater than 1 for sequential"
                        " node: {}".format(names[i]))
                    c = min(c, 1)
                theta_min[i] = c
            modeling_cores_used = np.sum(theta_min)
            modeling_bias = CPU_Util_clock * num_cores - modeling_cores_used
            logger.debug("modeling_bias: {}".format(modeling_bias))
            if modeling_bias < 0.:
                logger.warning("WARNING: modeling bias < 0")
                modeling_bias = 0.
        else:
            modeling_bias = 0.
        theta_min = cvxpy.Parameter(pos=True, shape=(N,), value=theta_min,
                                 name='theta_min')
        theta_max = num_cores * np.ones(N)
        for i, is_par in enumerate(is_parallel):
            if not is_par:
                theta_max[i] = 1
        theta_max = cvxpy.Parameter(pos=True, shape=(N,), value=theta_max,
                                    name='theta_max')
        theta = cvxpy.Variable(N)
        expression = cvxpy.multiply(rates, theta)
        constraints = [
            cvxpy.sum(theta) <= num_cores - modeling_bias,
            theta <= theta_max,
            theta >= theta_min,
        ]
        objective_fn = cvxpy.min(expression)
        problem = cvxpy.Problem(cvxpy.Maximize(objective_fn), constraints)
        t1 = time.time()
        problem.solve()
        t2 = time.time()
        logger.debug("Solve took {} seconds".format(t2 - t1))
        logger.debug("theta=\n{}".format(theta.value))
        max_throughput = problem.value
        theta_hat = theta.value
        num_cores_allocated = theta_hat
        params_dict = {name: theta_i for name, theta_i in zip(names, theta_hat)}
        logger.debug("num_cores=\n{}".format(num_cores_allocated))
        logger.debug("num_cores=\n{}".format(np.round(num_cores_allocated)))
        logger.debug("throughput={}".format(max_throughput))
        logger.debug("params={}".format(params_dict))
        rets = [max_throughput, params_dict]
        if return_current_throughput:
            rets.append(current_throughput)
        return tuple(rets)

    def analysis_str(self) -> str:
        """Returns a debug string for what the user should inspect"""
        return ""

    # The following are implementation dependent
    def bottleneck_node_analysis(
            self, mode: Optional[str]=None) -> 'AnalysisTreeNode':
        """Returns a pointer to the bottleneck node"""
        return self._analysis.bottleneck_node(mode)

    def ranked_list_bottleneck_nodes_analysis(
        self, mode: Optional[str]=None, extended: bool=True) -> List['AnalysisTreeNode']:
        return self._analysis.ranked_list_bottleneck_nodes(mode,
                                                           extended=extended)

    def node_names(self) -> Iterator[NodeName]:
        return self._analysis.node_names()

    def nodes(self) -> Iterator['AnalysisTreeNode']:
        return self._analysis.nodes()

    def lookup_name(self, node_name: NodeName) -> 'AnalysisTreeNode':
        """Returns the node corresponding to the node_name"""
        return self._analysis.lookup_name(node_name)


class TreeNode(object):
  """Maps a running graphdef to a nodes name and op.

  Used for determining connectivity and, optionally, statistics of a node.

  Represents tensorflow graphdef nodes with their current runtime
  statistics. Names are runtime names (not type of node).

  Note: state (e.g., graphdef attributes) are held in `state` while connectivity
  information is held within the node. In general, the connectivity information
  is useful for traversal, but external queries should return `state`.
  """
  def __init__(self, name: str, op: str):
    self.input = []  # Input nodes into this node
    self.function = [] # Nested functions within the node
    self.name = NodeName(name)
    self.op = NodeOp(op)
    self.state = None
    self.metadata = {}

  def add_input(self, x: 'TreeNode') -> None:
    self.input.append(x)

  def add_functions(self, fs: List['TreeNode']) -> None:
    self.function.extend(fs)

  def add_state_annotations(self, state: NodeState) -> None:
    assert isinstance(state, NodeState)
    self.state = state

  def add_metadata(self, metadata: dict) -> None:
    self.metadata = metadata

  def content_dict(self) -> Dict:
    return dict(self.state._asdict()) if self.state else {}

  @property
  def parallelism(self) -> int:
      if self.state:
          return self.state.parallelism
      else:
          return 1

  def __str__(self) -> str:
    if not self.input:
      return str(self.name)
    else:
      input_strs = []
      for i in self.input:
        input_strs.append(str(i))
      input_str = ",".join(input_strs)
      return str(self.name) + "(" + input_str + ")"

  def __repr__(self) -> str:
    """Long string."""
    content_dict = self.content_dict()
    self_repr = str(self.name) + ":" + str(content_dict)
    if not self.input:
      return self_repr
    else:
      input_strs = []
      for i in self.input:
        input_strs.append(repr(i))
      input_str = ",".join(input_strs)
      return self_repr + "(" + input_str + ")"

  def has_state(self) -> bool:
      return bool(self.state)

  def is_src_node(self) -> bool:
      return self.op in SRC_DATASET_OP_NODES

  def is_disk_node(self) -> bool:
      return self.op in DISK_DATASET_OP_NODES

  def is_cache_node(self) -> bool:
      return self.op in CACHE_DATASET_OP_NODES

  def is_tensor_slice_node(self) -> bool:
      return self.op in TENSOR_SLICE_DATASET_OP_NODES

  def is_take_node(self) -> bool:
      return self.op in TAKE_DATASET_OP_NODES

  def is_range_node(self) -> bool:
      return self.op in RANGE_DATASET_OP_NODES

  def is_repeat_node(self) -> bool:
      return self.op in REPEAT_DATASET_OP_NODES

  def is_interleave_node(self) -> bool:
      return self.op in INTERLEAVE_DATASET_OP_NODES

  def is_snapshot_node(self) -> bool:
      return self.op in SNAPSHOT_DATASET_OP_NODES

  def is_zip_node(self) -> bool:
      return self.op in ZIP_DATASET_OP_NODES

  def is_group_by_window_node(self) -> bool:
      return self.op in GROUP_BY_WINDOW_DATASET_OP_NODES

  def is_parallel_node(self) -> bool:
      return self.op in PARALLEL_DATASET_OP_NODES

  def owned_nodes(self) -> Iterator['TreeNode']:
      return iter(self.function)

def import_graphdef(graph_def: input_pipeline_analysis_pb2.PipelineSnapshot
                    ) -> Dict[str, TreeNode]:
    """Converts a graphdef into a dictionary of TreeNodes 'dataset' is root"""
    def is_tfdata_node(node):
        # NOTE(mkuchnik): 'dataset' is a meta-node and not a 'Dataset'-type op
        return "Dataset" in node.op or "dataset" in node.name

    def find_function_attrs_in_node(node) -> List:
        return list(filter(lambda x: x in node.attr, FUNCTION_ATTR_NAMES))

    def find_functions_of_function(f):
      """Find nested function nodes e.g., f1 calls f2."""
      return [node for node in f.node_def if find_function_attrs_in_node(node)]

    def find_datasets_in_function(graph_def, f_name: str,
                                  datasets: Optional[List]=None):
      """Find datasets in a function."""
      datasets = [] if datasets is None else datasets
      for f in graph_def.library.function:
          if f.signature.name == f_name:
              for node in f.node_def:
                  if is_tfdata_node(node):
                      datasets.append(node)
              child_f_nodes = find_functions_of_function(f)
              for child_node in child_f_nodes:
                  f_attrs = find_function_attrs_in_node(child_node)
                  assert len(f_attrs) == 1, \
                      "Found other than 1 function_attr: {}".format(
                          child_node.name)
                  child_f_name = child_node.attr[f_attrs[0]].func.name
                  find_datasets_in_function(graph_def, child_f_name, datasets)
      return datasets

    def fetch_metadata(graph_def, node):
        """Get metadata to nodes from graphdef (e.g., filenames)"""
        if node.op == "ParallelInterleaveDatasetV4":
            maybe_tensor_slice_name = node.input[0]
            maybe_tensor_slice_node = find_node_in_graph_def(
                graph_def, maybe_tensor_slice_name)
            def get_num_files(maybe_tensor_slice_node):
                # out_types = node.attr["Toutput_types"].list.type
                # TODO(mkuchnik): Check that DT_STRING
                const_node_name = maybe_tensor_slice_node.input[0]
                logger.info("TensorSliceDataset num_files in "
                             "{}".format(const_node_name))
                const_node = find_node_in_graph_def(graph_def, const_node_name)
                if const_node:
                    value_attr = const_node.attr["value"].tensor
                    dim = value_attr.tensor_shape.dim
                    if len(dim) != 1:
                        return None
                    num_files = int(dim[0].size)
                    return {"num_files": num_files}
                else:
                    logger.warning("Failed to find {}".format(const_node_name))
            if maybe_tensor_slice_node.op == "TensorSliceDataset":
                return get_num_files(maybe_tensor_slice_node)
            elif maybe_tensor_slice_node.op == "ShuffleDatasetV3":
                # Sometimes, shuffle is used on files
                maybe_tensor_slice_name = maybe_tensor_slice_node.input[0]
                maybe_tensor_slice_node = find_node_in_graph_def(
                    graph_def, maybe_tensor_slice_name)
                return get_num_files(maybe_tensor_slice_node)
            logger.warning("Did not see TensorSliceDataset in "
                            "{}".format(node.name))
        elif node.op == "TakeDataset":
            const_node_name = node.input[1]
            const_node = find_node_in_graph_def(graph_def, const_node_name)
            if const_node:
                # TODO(mkuchnik): Outer parallelism
                value_attr = const_node.attr["value"].tensor
                assert len(value_attr.int64_val) == 1, \
                    "Expected 1 length constant"
                take_size = int(value_attr.int64_val[0])
                return {"take_size": take_size}
            else:
                logger.warning("Failed to find {}".format(const_node_name))
        elif node.op == "RepeatDataset":
            # TODO(mkuchnik): Add ShuffleRepeat?
            const_node_name = node.input[1]
            const_node = find_node_in_graph_def(graph_def, const_node_name)
            if const_node:
                value_attr = const_node.attr["value"].tensor
                assert len(value_attr.int64_val) == 1, \
                    "Expected 1 length constant"
                repeat_size = int(value_attr.int64_val[0])
                return {"repeat_size": repeat_size}
            else:
                logger.warning("Failed to find {}".format(const_node_name))
        return None

    def find_tree_and_nested_nodes(graph_def):
      tree_node_lookup = dict()  # Name to node
      nested_nodes = collections.defaultdict(list)  # Name to function names
      for node in graph_def.node:
          if is_tfdata_node(node):
              n = TreeNode(node.name, node.op)
              metadata = fetch_metadata(graph_def, node)
              if metadata:
                  n.add_metadata(metadata)
              tree_node_lookup[node.name] = n
              for input_name in node.input:
                  if input_name in tree_node_lookup:
                      input_node = tree_node_lookup[input_name]
                      n.add_input(input_node)
              f_attrs = find_function_attrs_in_node(node)
              for f in f_attrs:
                  # NOTE(mkuchnik): Some datasets have many f for one node
                  nested_nodes[node.name].append(node.attr[f].func.name)
      return tree_node_lookup, nested_nodes

    def find_node_in_graph_def(graph_def, node_name):
        for node in graph_def.node:
            if node.name == node_name:
                return node
        return None

    def create_TreeNode(name, op, metadata):
        n = TreeNode(name, op)
        if metadata:
            n.add_metadata(metadata)
        return n

    tree_node_lookup, nested_nodes = find_tree_and_nested_nodes(graph_def)
    logger.debug("Tree node lookup:\n{}".format(tree_node_lookup))
    logger.debug("Nested nodes:\n{}".format(nested_nodes))
    if not len(tree_node_lookup):
        logger.warning("Found 0 tf.data nodes for analysis")
    for dataset_name, f_names in nested_nodes.items():
        for f_name in f_names:
            datasets = find_datasets_in_function(graph_def, f_name)
            logger.debug("Found datasets in function: '{}'\n{}".format(
                f_name, pprint.pformat(datasets)))
            dataset_names_ops = [(x.name, x.op, fetch_metadata(graph_def, x))
                                 for x in datasets]
            new_nodes = list(
                map(lambda x: create_TreeNode(*x), dataset_names_ops))
            tree_node_lookup[dataset_name].add_functions(new_nodes)
            for (new_dataset_name, _, _), new_node in zip(dataset_names_ops,
                                                       new_nodes):
              tree_node_lookup[new_dataset_name] = new_node
    return tree_node_lookup

@tf_export("data.experimental.analysis.ResumeDataset", v1=[])
class ResumeDataset(dataset_ops.DatasetSource):
  """Creates a dataset on a given `device` given a graph def."""

  def __init__(self, graph_def, element_spec: tensor_spec.TensorSpec,
               device=None):
    with ops.name_scope("resume_dataset"):
      graph_def = ops.convert_to_tensor(
          graph_def, dtype=dtypes.string, name="graph_def")
    if ops.device:
        with ops.device(device):
            variant_tensor = ged_ops.dataset_from_graph(graph_def)
    else:
        variant_tensor = ged_ops.dataset_from_graph(graph_def)
    if element_spec is None:
        raise ValueError("element_spec cannot be {}".format(element_spec))
    self._elem_spec = element_spec
    super(ResumeDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._elem_spec


@tf_export("data.experimental.analysis.RestructuredDataset", v1=[])
class RestructuredDataset(dataset_ops.UnaryDataset):
  """An internal helper for changing the element spec of a dataset."""

  def __init__(self, dataset, structure):
    self._input_dataset = dataset
    self._structure = structure

    variant_tensor = self._input_dataset._variant_tensor  # pylint: disable=protected-access
    super(RestructuredDataset, self).__init__(dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._structure
