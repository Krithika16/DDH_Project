	??oW?@??oW?@!??oW?@	??^??????^????!??^????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??oW?@M?O?5@A?U??]2?@Yp_?/q@*	?????RA2n
7Iterator::Model::ShuffleAndRepeat::BatchV2::TensorSlice@??H.?n@!?????V@)??H.?n@1?????V@:Preprocessing2a
*Iterator::Model::ShuffleAndRepeat::BatchV2J{?/L\p@!@uxX@)?{??P?1@1l2???m@:Preprocessing2X
!Iterator::Model::ShuffleAndRepeat?T????p@!?N??X@)????<?
@1~??????:Preprocessing2F
Iterator::Model?,C??p@!      Y@)????9?@1??r?X???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??^????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	M?O?5@M?O?5@!M?O?5@      ??!       "      ??!       *      ??!       2	?U??]2?@?U??]2?@!?U??]2?@:      ??!       B      ??!       J	p_?/q@p_?/q@!p_?/q@R      ??!       Z	p_?/q@p_?/q@!p_?/q@JCPU_ONLYY??^????b 