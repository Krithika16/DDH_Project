	!?1??Sw@!?1??Sw@!!?1??Sw@	??g*?g"@??g*?g"@!??g*?g"@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6!?1??Sw@B\9{g???1Ͽ]???t@A??*l???I[rP¬	@Y???,A,A@*	???ҡ\?@2?
NIterator::Model::MaxIntraOpParallelism::ShuffleAndRepeat::BatchV2::TensorSliceg??e?4@!?=?r?xM@)g??e?4@1?=?r?xM@:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::ShuffleAndRepeat::BatchV2?u?+.?A@!?Q3???X@)E?J?-@1[e??2sD@:Preprocessing2F
Iterator::Model?ΤM?A@!      Y@)1(?hr1??1M??25??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?f?v??A@!?L?Y?X@)?S?{F"??1|????O??:Preprocessing2o
8Iterator::Model::MaxIntraOpParallelism::ShuffleAndRepeat???-?A@!?]??X@)??BW"P??1?9????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9??g*?g"@I?@??6??Q?"\??ZV@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	B\9{g???B\9{g???!B\9{g???      ??!       "	Ͽ]???t@Ͽ]???t@!Ͽ]???t@*      ??!       2	??*l?????*l???!??*l???:	[rP¬	@[rP¬	@![rP¬	@B      ??!       J	???,A,A@???,A,A@!???,A,A@R      ??!       Z	???,A,A@???,A,A@!???,A,A@b      ??!       JGPUY??g*?g"@b q?@??6??y?"\??ZV@