?	????7??@????7??@!????7??@	lT
q???lT
q???!lT
q???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$????7??@??~j??2@A?^)˔??@Yb??4?rh@*	?????hA2n
7Iterator::Model::ShuffleAndRepeat::BatchV2::TensorSlice =
ףp?c@!M??@rU@)=
ףp?c@1M??@rU@:Preprocessing2a
*Iterator::Model::ShuffleAndRepeat::BatchV2??B?i?e@!:\7,?W@)Ș????.@1C?D?ϥ @:Preprocessing2X
!Iterator::Model::ShuffleAndRepeat?&1?tf@!s_%-wX@)?c?]K@1??$??\@:Preprocessing2F
Iterator::Model?6?[?f@!      Y@)]?C??k@1?T[?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9lT
q???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??~j??2@??~j??2@!??~j??2@      ??!       "      ??!       *      ??!       2	?^)˔??@?^)˔??@!?^)˔??@:      ??!       B      ??!       J	b??4?rh@b??4?rh@!b??4?rh@R      ??!       Z	b??4?rh@b??4?rh@!b??4?rh@JCPU_ONLYYlT
q???b Y      Y@q?	Z????"?
device?Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 