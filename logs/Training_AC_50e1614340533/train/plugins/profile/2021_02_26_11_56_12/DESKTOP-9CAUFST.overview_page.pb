?	3333?>?@3333?>?@!3333?>?@	??_?k????_?k??!??_?k??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$3333?>?@?ڊ?e???A?W[??4?@Y:??H3@*	    `??@2n
7Iterator::Model::ShuffleAndRepeat::BatchV2::TensorSlice??H??(@!i??_sP@)??H??(@1i??_sP@:Preprocessing2a
*Iterator::Model::ShuffleAndRepeat::BatchV2??q??2@!?????X@)33333s@1???@@:Preprocessing2X
!Iterator::Model::ShuffleAndRepeat???K7?2@!e'???X@)?Y??ڊ??1?:3Pr??:Preprocessing2F
Iterator::ModelZd;?O?2@!      Y@)????Mb??1nl?cߑ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??_?k??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ڊ?e????ڊ?e???!?ڊ?e???      ??!       "      ??!       *      ??!       2	?W[??4?@?W[??4?@!?W[??4?@:      ??!       B      ??!       J	:??H3@:??H3@!:??H3@R      ??!       Z	:??H3@:??H3@!:??H3@JCPU_ONLYY??_?k??b Y      Y@q?W?P?k??"?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 