	?/L????@?/L????@!?/L????@	??*???????*?????!??*?????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?/L????@??z6;?@A?ŏ?{?@Y??St$?n@*	    ??A2n
7Iterator::Model::ShuffleAndRepeat::BatchV2::TensorSlice ;?O??rj@!u-?pJV@);?O??rj@1u-?pJV@:Preprocessing2a
*Iterator::Model::ShuffleAndRepeat::BatchV2?al@!???5?W@)d?]K??.@1m?jD@:Preprocessing2F
Iterator::Modelsh??|?m@!      Y@)33333?@1??y?Or@:Preprocessing2X
!Iterator::Model::ShuffleAndRepeat???S?m@!?2̂mtX@)?ZB>?Y@1̔?~'@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??*?????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??z6;?@??z6;?@!??z6;?@      ??!       "      ??!       *      ??!       2	?ŏ?{?@?ŏ?{?@!?ŏ?{?@:      ??!       B      ??!       J	??St$?n@??St$?n@!??St$?n@R      ??!       Z	??St$?n@??St$?n@!??St$?n@JCPU_ONLYY??*?????b 