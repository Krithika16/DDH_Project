	?????t?@?????t?@!?????t?@	???千?????千??!???千??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?????t?@?St$????A????m?@Y?	??*@*	????Y<?@2n
7Iterator::Model::ShuffleAndRepeat::BatchV2::TensorSlicevq?m @!??s"??N@)vq?m @1??s"??N@:Preprocessing2a
*Iterator::Model::ShuffleAndRepeat::BatchV2Ϊ??V?*@!??&???X@)?rh???@1U??iMLC@:Preprocessing2X
!Iterator::Model::ShuffleAndRepeat??ͪ??*@!??<??X@)?l??????1??j????:Preprocessing2F
Iterator::ModelZ??ڊ?*@!      Y@)?W[?????1?ˤ?Ǽ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???千??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?St$?????St$????!?St$????      ??!       "      ??!       *      ??!       2	????m?@????m?@!????m?@:      ??!       B      ??!       J	?	??*@?	??*@!?	??*@R      ??!       Z	?	??*@?	??*@!?	??*@JCPU_ONLYY???千??b 