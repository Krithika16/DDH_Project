	? v??Uw@? v??Uw@!? v??Uw@	????"@????"@!????"@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6? v??Uw@?H?<????1???? ?t@AX?L??~??I{??B1@YUPQ?+?A@*	???M??@2?
NIterator::Model::MaxIntraOpParallelism::ShuffleAndRepeat::BatchV2::TensorSlice????i5@!Lh-?QM@)????i5@1Lh-?QM@:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::ShuffleAndRepeat::BatchV2m???<B@!?2?4(?X@)????u.@1;?<]?D@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism[?a/@B@!G???X@)?,??o???1l??t???:Preprocessing2F
Iterator::Modelyu??AB@!      Y@)q???"M??1`?'?u`??:Preprocessing2o
8Iterator::Model::MaxIntraOpParallelism::ShuffleAndRepeat?f??=B@!UX??4?X@)??&N???1<?,?pb??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9????"@I@??2????Q?RR?RV@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?H?<?????H?<????!?H?<????      ??!       "	???? ?t@???? ?t@!???? ?t@*      ??!       2	X?L??~??X?L??~??!X?L??~??:	{??B1@{??B1@!{??B1@B      ??!       J	UPQ?+?A@UPQ?+?A@!UPQ?+?A@R      ??!       Z	UPQ?+?A@UPQ?+?A@!UPQ?+?A@b      ??!       JGPUY????"@b q@??2????y?RR?RV@