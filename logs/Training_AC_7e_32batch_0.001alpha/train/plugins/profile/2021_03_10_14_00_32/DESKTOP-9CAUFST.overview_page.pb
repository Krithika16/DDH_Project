?	???h?)?@???h?)?@!???h?)?@	?H?bc????H?bc???!?H?bc???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???h?)?@??ܵ?|$@A ?~???@Y$(~???J@*	????9g?@2n
7Iterator::Model::ShuffleAndRepeat::BatchV2::TensorSlice ??ܵ?l>@!g???˓P@)??ܵ?l>@1g???˓P@:Preprocessing2a
*Iterator::Model::ShuffleAndRepeat::BatchV2_)?ǢC@!??y??eU@)?s??!@1*B䛭H3@:Preprocessing2X
!Iterator::Model::ShuffleAndRepeat??\mŦE@!|
o?E?W@)r????@1Q̩?u?!@:Preprocessing2F
Iterator::ModelpΈ???F@!      Y@)6<?R??@1KX)?{@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?H?bc???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??ܵ?|$@??ܵ?|$@!??ܵ?|$@      ??!       "      ??!       *      ??!       2	 ?~???@ ?~???@! ?~???@:      ??!       B      ??!       J	$(~???J@$(~???J@!$(~???J@R      ??!       Z	$(~???J@$(~???J@!$(~???J@JCPU_ONLYY?H?bc???b Y      Y@q8`??s??"?
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