?	? v??Uw@? v??Uw@!? v??Uw@	????"@????"@!????"@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6? v??Uw@?H?<????1???? ?t@AX?L??~??I{??B1@YUPQ?+?A@*	???M??@2?
NIterator::Model::MaxIntraOpParallelism::ShuffleAndRepeat::BatchV2::TensorSlice????i5@!Lh-?QM@)????i5@1Lh-?QM@:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::ShuffleAndRepeat::BatchV2m???<B@!?2?4(?X@)????u.@1;?<]?D@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism[?a/@B@!G???X@)?,??o???1l??t???:Preprocessing2F
Iterator::Modelyu??AB@!      Y@)q???"M??1`?'?u`??:Preprocessing2o
8Iterator::Model::MaxIntraOpParallelism::ShuffleAndRepeat?f??=B@!UX??4?X@)??&N???1<?,?pb??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9????"@I@??2????Q?RR?RV@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?H?<?????H?<????!?H?<????      ??!       "	???? ?t@???? ?t@!???? ?t@*      ??!       2	X?L??~??X?L??~??!X?L??~??:	{??B1@{??B1@!{??B1@B      ??!       J	UPQ?+?A@UPQ?+?A@!UPQ?+?A@R      ??!       Z	UPQ?+?A@UPQ?+?A@!UPQ?+?A@b      ??!       JGPUY????"@b q@??2????y?RR?RV@?"d
8gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?cz???!?cz???0"b
7gradient_tape/model/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput?~D??;??!?Dl???0"5
model/conv2d_1/Relu_FusedConv2D!_)X?~??!b?????"C
%gradient_tape/model/conv2d_1/ReluGradReluGrad? ??&???!xr??9???"V
5gradient_tape/model/max_pooling2d/MaxPool/MaxPoolGradMaxPoolGrad*?vP???!?J??=???"-
IteratorGetNext/_1_Send?4]???!&?????"A
#gradient_tape/model/conv2d/ReluGradReluGrad?ވ??c??!?WQ??"c
8gradient_tape/model/conv2d_12/Conv2D/Conv2DBackpropInputConv2DBackpropInput?#p?\??!K???Y??0"d
8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter5l?????!%;K~???0"d
8gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?LY????!۹ 5????0Q      Y@Y?@&?d@a???C??W@q$??????y\"34ؖG?"?	
both?Your program is MODERATELY input-bound because 9.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
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