
?#
;
Placeholder/_0Placeholder*
shape:*
dtype0
>
Const/_3Const*
valueB B *
dtype02
	Const/_13
?
Const/_4Const*?
value?B? r?
,/job:localhost/replica:0/task:0/device:CPU:0	localhost4StatefulPartitionedCall/CacheDatasetV2/MemoryCache_0 ??????*'N10tensorflow4data18MemoryCacheManagerE*
dtype0
H
Const/_6Const*
valueB	 R
?????????*
dtype0	2
	Const/_16
3
Const/_8Const*
dtype0	*
value
B	 R?
2
Const/_9Const*
value	B
 Z *
dtype0

3
	Const/_11Const*
value	B	 R*
dtype0	
?
	Const/_14Const*?
value?B? r?
,/job:localhost/replica:0/task:0/device:CPU:0	localhost6StatefulPartitionedCall/CacheDatasetV2_1/MemoryCache_1 ??????*'N10tensorflow4data18MemoryCacheManagerE*
dtype0
4
	Const/_19Const*
value
B	 R?*
dtype0	
/
ConstConst*
value	B	 R*
dtype0	
j
TensorSliceDataset/_1TensorSliceDatasetPlaceholder/_0*
Toutput_types
2*
output_shapes
: 
?
MapDataset/_2
MapDatasetTensorSliceDataset/_1*
preserve_cardinality(*
output_shapes

:  *C
f>R<
!__inference_Dataset_map_lambda_15
_tf_data_function(*
use_inter_op_parallelism(*
output_types
2*

Targuments
 
|
CacheDatasetV2/_5CacheDatasetV2MapDataset/_2Const/_3Const/_4*
output_shapes

:  *
output_types
2
t
RepeatDataset/_7RepeatDatasetCacheDatasetV2/_5Const/_6*
output_shapes

:  *
output_types
2
?
BatchDatasetV2/_10BatchDatasetV2RepeatDataset/_7Const/_8Const/_9*
output_types
2*
parallel_copy( *"
output_shapes
:?  
x
TakeDataset/_12TakeDatasetBatchDatasetV2/_10	Const/_11*
output_types
2*"
output_shapes
:?  
?
CacheDatasetV2/_15CacheDatasetV2TakeDataset/_12Const/_3	Const/_14*"
output_shapes
:?  *
output_types
2
{
RepeatDataset/_17RepeatDatasetCacheDatasetV2/_15Const/_6*
output_types
2*"
output_shapes
:?  
?
MapDataset/_18
MapDatasetRepeatDataset/_17**
output_shapes
:?????????  *E
f@R>
#__inference_Dataset_map_heavy_fn_48
_tf_data_function(*
output_types
2*
use_inter_op_parallelism(*

Targuments
 *
preserve_cardinality(
?
PrefetchDataset/_20PrefetchDatasetMapDataset/_18	Const/_19*
slack_period *
legacy_autotune(*
output_types
2**
output_shapes
:?????????  *
buffer_size_min 
?
intra_op_parallelismMaxIntraOpParallelismDatasetPrefetchDataset/_20Const*
output_types
2**
output_shapes
:?????????  
?
ModelDataset/_21ModelDatasetintra_op_parallelism*
output_types
2*

ram_budget **
output_shapes
:?????????  *

cpu_budget *
	algorithm 2
ModelDataset/_22
:
dataset_RetvalModelDataset/_21*
index *
T0
+
SinkIdentityModelDataset/_21*
T0?
?
?
#__inference_Dataset_map_heavy_fn_48

args_0
identity?
PartitionedCallPartitionedCallargs_0* 
_read_only_resource_inputs
 * 
fR
__inference_heavy_fn_45*
Tout
2*
executor_type *-
config_proto

CPU

GPU 2J 8? *
config *
_collective_manager_ids
 *
Tin
27
IdentityIdentityPartitionedCall:output:0*
T01
	FakeSink0IdentityIdentity:output:0*
T0"
identityIdentity:output:0*
_tf_data_function(:S O
 
_user_specified_nameargs_0
+
_output_shapes
:?????????  
?
=
!__inference_Dataset_map_lambda_15

args_0
identity<
CastCastargs_0*

SrcT0*
Truncate( *

DstT0I
ones/shape_as_tensorConst*
dtype0*
valueB"        7

ones/ConstConst*
valueB
 *  ??*
dtype0[
onesFillones/shape_as_tensor:output:0ones/Const:output:0*
T0*

index_type0,
mulMulCast:y:0ones:output:0*
T0&
IdentityIdentitymul:z:0*
T01
	FakeSink0IdentityIdentity:output:0*
T0"
identityIdentity:output:0*
_tf_data_function(:> :

_output_shapes
: 
 
_user_specified_nameargs_0
?
.
__inference_heavy_fn_45
x
identity@
MatMulBatchMatMulV2xx*
adj_x( *
adj_y(*
T0.
IdentityIdentityMatMul:output:0*
T01
	FakeSink0IdentityIdentity:output:0*
T0"
identityIdentity:output:0*
_tf_data_function(:N J
+
_output_shapes
:?????????  

_user_specified_namex"?T?v??ₕҍ?Ώ )      ??2MapDataset/_188@?????H?????X?????????`?h???p???X?v??肕??? )      ??2intra_op_parallelism8@?????H?????X?????????`?h???p???M???? )      ??2CacheDatasetV2/_158@???H???X?????????`?h???p???U?vٛₕ???~ )      ??2RepeatDataset/_178@?????H?????X?????????`?h???p???W?v??肕???H )      ??2PrefetchDataset/_208@?????H?????X?????????`?h???p???????????>(1)\?????@:110"*	?"?L??n@??"K???@?jA??@2ɺ???ҋ??????Ӌ?