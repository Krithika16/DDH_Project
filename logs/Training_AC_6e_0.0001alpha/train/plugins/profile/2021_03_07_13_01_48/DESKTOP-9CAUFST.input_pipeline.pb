	?O??v??@?O??v??@!?O??v??@	??U????U??!??U??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?O??v??@?@A?:Mty?@Y??	hRE@*	3333??@2n
7Iterator::Model::ShuffleAndRepeat::BatchV2::TensorSlice???Q?:@!jhMfg0R@)???Q?:@1jhMfg0R@:Preprocessing2a
*Iterator::Model::ShuffleAndRepeat::BatchV2V}??b??@!???yJ?U@)*??D?@1?j?O,@:Preprocessing2F
Iterator::Model??JY?hB@!      Y@)vOj?@1pي??@:Preprocessing2X
!Iterator::Model::ShuffleAndRepeat?鷯+A@!?hR?QW@)????@14'֨v@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??U??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?@?@!?@      ??!       "      ??!       *      ??!       2	?:Mty?@?:Mty?@!?:Mty?@:      ??!       B      ??!       J	??	hRE@??	hRE@!??	hRE@R      ??!       Z	??	hRE@??	hRE@!??	hRE@JCPU_ONLYY??U??b 