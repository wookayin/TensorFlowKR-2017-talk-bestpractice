다시 코드로 돌아와서,

```python
class VGG19():
    def __init__(self):
        self.conv1 = layers.convolution2d(image, 64, [3, 3], scope='conv1_1')
        ...
        self.conv5_4 = layers.convolution2d(self.conv5_3, 512, [3, 3], scope='conv5_4')
        ...
        self.fc6 = layers.fully_connected(self.pool5, 4096, scope='fc6')
        self.fc7 = layers.fully_connected(self.fc6, 4096, scope='fc7')
        self.output = layers.fully_connected(self.fc7, 1000, scope='output')
```

```python
def train():
    image = read_image_dataset()   # [batch_size, 224, 224, 3] Tensor

*   net = VGG19(image)
*   predict_logit = net.output   # [1000] Tensor
    convmap = net.conv5_3
```

--

- ~~`conv5_4` feature map 등 다른 endpoint를 가져오고 싶다면?~~ .oc-green-5[해결!]
- Endpoint tensor (ReLU output) 말고, 더 안쪽에 있는 preactivation, 혹은 batch_norm output을 보려면??
- 각 layer에서 쓰인 Variable 도 갖고오고 싶은데...
- 각 layer에서 loss가 각각 발생한다면? (e.g. l2 regularization)




---


## Scope is Your Friend

.center[.img-100[ ![Scope](images/diagram-im2txt.png) ]]

```python
with tf.name_scope("Img2Txt"):
    input = preprocess_input()  # [224, 224, 3]
    with tf.name_scope("Encoder"):
        conv1 = layers.convolution2d(input, 64, [11, 11], 4, scope="conv1")
        pool1 = layers.max_pool2d(conv1, [3, 3], 2, scope="pool1")
        # ...
        fc7 = layers.fully_connected(fc6, 4096, scope="fc7")
    image_repr = fc7
    with tf.name_scope("Decoder"):
        fc = layers.fully_connected(image_repr, 500, scope="fc")
        txt = tf.nn.rnn(rnn_cell, inputs=...)
```



---

## Collection: Cross-cutting Concerns

.center[.img-80[ ![Cross-concern](images/diagram-crosscutting-before.png) ]]





---

## Collection: Cross-cutting Concerns

.center[.img-80[ ![Cross-concern](images/diagram-crosscutting-after.png) ]]




--

- 우리의 .blue[핵심 관심(**Core Concerns**)]은 모델 아키텍쳐이다 (e.g. layer, scope)
- 각 layer(scope) 마다 발생하는 summary, loss, variable 등은 <br/>
  .blue[횡단 관심(**Cross-Cutting Concerns**)] : collection 으로 처리
- 이 때 collection은 `tf.Graph` 에 전역(global)으로 존재하는 싱글턴임


---

[`tf.get_variable()`](https://www.tensorflow.org/api_docs/python/state_ops/sharing_variables#get_variable):

```python
tf.get_variable(name, shape=None, dtype=None, initializer=None,
  regularizer=None, trainable=True, collections=None, caching_device=None,
  partitioner=None, validate_shape=True, custom_getter=None)
```

.small[

- `regularizer`: A (Tensor -> Tensor or None) function; the result of applying it on a newly created variable will be added to the collection `GraphKeys.REGULARIZATION_LOSSES` and can be used for regularization.

- `collections`: List of graph collections keys to add the Variable to. Defaults to `[GraphKeys.GLOBAL_VARIABLES]` (see `tf.Variable`).

]

--

가져오기:

```python
tf.get_collection(key, scope=None)
```

```python
# GraphKeys.TRAINABLE_VARIABLES
trainable_variables = tf.trainable_variables()

# GraphKeys.MOVING_AVERAGE_VARIABLES (for batch normalization)
ma_variables = tf.moving_average_variables()

# Get all the weight variables in current CNN scope
conv_filters = tf.get_collection(tf.GraphKeys.WEIGHTS, scope='CNN/')
```


---

### Summary

```python
*tf.summary.image('InputImage', image)
with tf.name_scope("Encoder") as sc:
    _, endpoints = vgg.vgg_19(image, is_training=False)
    conv5_4 = endpoints[sc + '/conv5/conv5_4']

*tf.summary.histogram('Encoder/conv5_4', conv5_4)

with tf.name_scope("Decoder") as sc:
    output_mask = deconvnet(image)
*   tf.summary.image('Output', output_mask)
```

```python
merged_summary = tf.summary.merge()
```

---

### Losses (Bleeding-Edge Feature)

```python
class MyModel(object):
    ...
    def build_graph(self):
        ...
        self.encoder_loss = tf.nn.softmax_cross_entropy_with_logits(logits, gt_label)
        ...
        self.decoder_loss = seq2seq.sequence_loss_by_example(...)
        ...
```

```python
total_loss = self.encoder_loss + self.decoder_loss + self.regularization_loss
train_op = create_train_op(total_loss, optimizer)
```

--

Summary 모으듯이... 이렇게도 할 수 있어요

```python
# The loss tensor are collected into ops.GraphKeys.LOSSES
tf.losses.sparse_softmax_cross_entropy(...)

# The loss tensor are collected into ops.GraphKeys.LOSSES
sequence_loss = seq2seq.sequence_loss_by_example(...)
tf.losses.add_loss(sequence_loss)
```

```python
total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
```



---

### `Collection`

Standard collections:

```python
GLOBAL_VARIABLES
LOCAL_VARIABLES
MODEL_VARIABLES
TRAINABLE_VARIABLES
SUMMARIES
QUEUE_RUNNERS
MOVING_AVERAGE_VARIABLES
REGULARIZATION_LOSSES
WEIGHTS
BIASES
ACTIVATIONS
```

`batch_norm` 이라는 새로운 Collection을 만들어 쓸 수도 있고

```python
with slim.arg_scope([slim.batch_norm], variables_collections=["batch_norm"]) as scope:
    net = layers.convolution2d(net, 64, [3, 3], stride=2,
                               normalizer_fn=layers.batch_norm,
                               normalizer_params=...)
    # ...
```


---

layout: false

### Scope is Your Friend

.best-practice[모델을 디자인할 때, `name_scope`으로 잘 정돈된 variable, ops 를 만들자]

```python
model_vars = tf.contrib.framework.get_model_variables()
slim.model_analyzer.analyze_vars(model_vars, print_info=True)
```

```YAML
---------
Variables: name (type shape) [size]
---------
InceptionResnetV2/Conv2d_1a_3x3/weights:0 (float32_ref 3x3x3x32) [864, bytes: 3456]
InceptionResnetV2/Conv2d_1a_3x3/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV2/Conv2d_1a_3x3/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV2/Conv2d_1a_3x3/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV2/Conv2d_2a_3x3/weights:0 (float32_ref 3x3x32x32) [9216, bytes: 36864]
InceptionResnetV2/Conv2d_2a_3x3/BatchNorm/beta:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV2/Conv2d_2a_3x3/BatchNorm/moving_mean:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV2/Conv2d_2a_3x3/BatchNorm/moving_variance:0 (float32_ref 32) [32, bytes: 128]
InceptionResnetV2/Conv2d_2b_3x3/weights:0 (float32_ref 3x3x32x64) [18432, bytes: 73728]
InceptionResnetV2/Conv2d_2b_3x3/BatchNorm/beta:0 (float32_ref 64) [64, bytes: 256]
InceptionResnetV2/Conv2d_2b_3x3/BatchNorm/moving_mean:0 (float32_ref 64) [64, bytes: 256]
InceptionResnetV2/Conv2d_2b_3x3/BatchNorm/moving_variance:0 (float32_ref 64) [64, bytes: 256]
InceptionResnetV2/Conv2d_3b_1x1/weights:0 (float32_ref 1x1x64x80) [5120, bytes: 20480]
InceptionResnetV2/Conv2d_3b_1x1/BatchNorm/beta:0 (float32_ref 80) [80, bytes: 320]
InceptionResnetV2/Conv2d_3b_1x1/BatchNorm/moving_mean:0 (float32_ref 80) [80, bytes: 320]
```



---

### Scope: 장점

- 계층적으로 scoping (e.g. `inception/conv1/weights:0`) 되어있고 종류별(weight, bias 등)로 suffix를 잘 붙여둔 변수를 가지고 있으면...
- 준비된 [각종 헬퍼 API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/framework/python/ops/variables.py#L57) (e.g. `tf.contrib.framework.get_XXX`)로
원하는 variable나 op들을 .green[어디서든] 쉽게 불러올 수 있습니다

```python
# scope, suffix 그리고 collection의 조합으로 변수 불러오기
tf.contrib.framework.get_variables(
    scope=None, suffix=None,
    collection=ops.GraphKeys.GLOBAL_VARIABLES
)

# 그외에... (전부 tf.get_variables를 호출합니다)
tf.contrib.framework.get_variables_by_name(given_name, scope)
tf.contrib.framework.get_variables_by_suffix(suffix, scope)
tf.contrib.framework.get_model_variables(scope, suffix)

```

가져와서 어디에 쓸까요?



---

**Pre-train된 모델의 파라미터를 load할 때** 쓰고

```python
variables_to_restore = slim.get_variables_to_restore(exclude=["fc8"])
init_assign_op, init_feed_dict = assign_from_checkpoint(
    checkpoint_path, variables_to_restore)
```

--

**디버깅** 할 때도 씁니다

```python
var_to_debug = tf.contrib.framework.get_variables(suffix="Logits")[0]
var_to_debug = tf.Print(var_to_debug, [var_to_debug], "Logits: ", first_n=20)
```

--

e.g. 특정 scope에 있는 variable 모두 가져오기

```python
with tf.variable_scope("encoder") as sc:
    with arg_scope([conv2d], weights_initializer=tf.contrib.layers.xavier_initializer()):
        conv1 = conv2d(input, 96, 3, 3, scope="conv1", name=name)
        conv2 = conv2d(conv1, 64, 3, 3, scope="conv2", name=name)

    self.encoder_variables = tf.contrib.framework.get_variables(sc)
```

---

### Practice: TensorFlow `Variable` (변수)를 만드는 3가지 방법

- `tf.Variable`,
- `tf.get_variable()`,
- 그 외 contrib 또는 [slim 에 있는 wrapper][slim-variables-wrapper] (TBD)


[slim-variables-wrapper]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim#variables

차이점은 무엇일까? 어느 것을 써야 할까요?



--

**정답** : .best-practice[`tf.Variable()` 대신 .green[`tf.get_variable()`] 등의 wrapper를 쓰자]

- `tf.Variable` 는 무조건 생성만 하는거니까 Parameter Sharing이 불가
- `tf.Variable` 로는 `tf.contrib.layers.initializers`를 쓸 수 없음

```python
# Xavier Initialization is GOOD!
weights_var = tf.get_variable(
    "weights",
    shape=shape,
*   initializer=tf.contrib.layers.xavier_initializer()
)
```


