?	P??n??@P??n??@!P??n??@	?3??????3?????!?3?????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$P??n??@Ș???!@Af?c]|??@Yc?ZB>J@*2333?<?@)      ?=2n
7Iterator::Model::ShuffleAndRepeat::BatchV2::TensorSlice ???Q?A@!??8 ?R@)???Q?A@1??8 ?R@:Preprocessing2a
*Iterator::Model::ShuffleAndRepeat::BatchV2?1??%F@!???}!W@)M?O? @1A??]?e1@:Preprocessing2X
!Iterator::Model::ShuffleAndRepeatDio???F@!?????X@)y?&1???1p????,@:Preprocessing2F
Iterator::ModelA??ǘ?G@!      Y@)?c?ZB??1+?#??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?3?????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Ș???!@Ș???!@!Ș???!@      ??!       "      ??!       *      ??!       2	f?c]|??@f?c]|??@!f?c]|??@:      ??!       B      ??!       J	c?ZB>J@c?ZB>J@!c?ZB>J@R      ??!       Z	c?ZB>J@c?ZB>J@!c?ZB>J@JCPU_ONLYY?3?????b Y      Y@q?eHͯ??"?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 