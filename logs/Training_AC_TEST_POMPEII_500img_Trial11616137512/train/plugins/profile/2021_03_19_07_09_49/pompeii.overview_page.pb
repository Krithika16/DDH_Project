?	!?1??Sw@!?1??Sw@!!?1??Sw@	??g*?g"@??g*?g"@!??g*?g"@"w
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
	B\9{g???B\9{g???!B\9{g???      ??!       "	Ͽ]???t@Ͽ]???t@!Ͽ]???t@*      ??!       2	??*l?????*l???!??*l???:	[rP¬	@[rP¬	@![rP¬	@B      ??!       J	???,A,A@???,A,A@!???,A,A@R      ??!       Z	???,A,A@???,A,A@!???,A,A@b      ??!       JGPUY??g*?g"@b q?@??6??y?"\??ZV@?"d
8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??????!??????0"b
7gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput??O8??!?V?W????0"5
model/conv2d_1/Relu_FusedConv2D%?7(Q{??!????D???"C
%gradient_tape/model/conv2d_1/ReluGradReluGrad??Zý??!=?/<????"V
5gradient_tape/model/max_pooling2d/MaxPool/MaxPoolGradMaxPoolGrad???IƑ?!spy????"-
IteratorGetNext/_1_SendZg?E$??!??:?
???"c
8gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropInputConv2DBackpropInput y?Ye??!{	֯[??0"A
#gradient_tape/model/conv2d/ReluGradReluGrad\?Q??!ax?YyZ??"d
8gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter~2?????!C?ġ????0"d
8gradient_tape/model/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?t????!???1???0Q      Y@Y?@&?d@a???C??W@q????<???y&8???F?"?	
both?Your program is MODERATELY input-bound because 9.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 