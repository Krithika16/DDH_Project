?	?Or?$w@?Or?$w@!?Or?$w@	Oq3D!@Oq3D!@!Oq3D!@"w
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
@B      ??!       J	???????@???????@!???????@R      ??!       Z	???????@???????@!???????@b      ??!       JGPUYOq3D!@b q@?Ċ?W??yEɦ?~V@?"d
8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??y޲?!??y޲?0"b
7gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput???kQ??!?}?/????0"5
model/conv2d_1/Relu_FusedConv2D????Eџ?!M0\Ku???"C
%gradient_tape/model/conv2d_1/ReluGradReluGrad?Vw?????!(k?h???"V
5gradient_tape/model/max_pooling2d/MaxPool/MaxPoolGradMaxPoolGrad?e:;:???!?g?'????"-
IteratorGetNext/_1_Send?IR??׍?!}??j???"c
8gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropInputConv2DBackpropInput???4?_??!?K
c??0"A
#gradient_tape/model/conv2d/ReluGradReluGrad????;D??!#fæX??"d
8gradient_tape/model/conv2d_4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter ?b?_??!?O??<???0"d
8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???&???!???????0Q      Y@Y?@&?d@a???C??W@q?cɽ??y?/oqR?"?	
both?Your program is MODERATELY input-bound because 8.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
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