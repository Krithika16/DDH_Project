	?Or?$w@?Or?$w@!?Or?$w@	Oq3D!@Oq3D!@!Oq3D!@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?Or?$w@̴?++M??1Gsd??t@A??^?sa??I?@e??
@Y???????@*	?I/S?@2?
NIterator::Model::MaxIntraOpParallelism::ShuffleAndRepeat::BatchV2::TensorSlicef?O7P|3@!.%?9$M@)f?O7P|3@1.%?9$M@:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::ShuffleAndRepeat::BatchV2? ??q?@@!Ć??v?X@)??o'?+@1Z菇??D@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismϼvߵ@@!???-??X@)??=?>t??1ܝ??J??:Preprocessing2F
Iterator::Model8j??{?@@!      Y@)7?֊6ǉ?1}(? ?F??:Preprocessing2o
8Iterator::Model::MaxIntraOpParallelism::ShuffleAndRepeat?I@@!????S?X@)4,F]k???1?9N?3Н?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9Oq3D!@I@?Ċ?W??QEɦ?~V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	̴?++M??̴?++M??!̴?++M??      ??!       "	Gsd??t@Gsd??t@!Gsd??t@*      ??!       2	??^?sa????^?sa??!??^?sa??:	?@e??
@?@e??
@!?@e??
@B      ??!       J	???????@???????@!???????@R      ??!       Z	???????@???????@!???????@b      ??!       JGPUYOq3D!@b q@?Ċ?W??yEɦ?~V@