name: highlevel
template: recipe

# Using .oc-cyan-3[high-level] API extensively

### Recommend using slim, contrib

---

layout: false

## High-level API를 적극적으로 사용하자

- .red[DO NOT REINVENT THE WHEEL]
- 초반 TensorFlow core는 low-level 위주였지만,
  요즘에는 쉽게 사용할 수 있도록 다양한 층위의 모듈화와 추상화가 되고 있다
    - 데이터, 변수, 레이어, 학습, 평가, 네트워크까지! (e.g. [TF-slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim))
- Tensorflow 도 Model Zoo가 있습니다!


.small[
.oc-red-6[Legacy Codes] (Not recommended): <br/>
[`tensorflow-vgg`](https://github.com/machrisaa/tensorflow-vgg)
[`tensorflow-vgg16`](https://github.com/ry/tensorflow-vgg16)
[`tensorflow-resnet`](https://github.com/ry/tensorflow-resnet)
[`resnet-tf`](https://github.com/xuyuwei/resnet-tf)
]



---

## High-level APIs for TensorFlow

.blue[**Highly Recommended (Official)**]

- [`tf.contrib.layers`][contrib-layers]
  .oc-red-4[&nbsp;&nbsp;&#x2014;&#x2014; 레이어 수준의 추상화 (Utility)]
    - 현재는 `contrib`에 있지만 대부분 official API에 포함됨
- [**TensorFlow slim**][slim]       .oc-red-4[&nbsp;&nbsp;&#x2014;&#x2014; Layer + 모델 + Trainer 등등...]
- 그 외 수많은 `contrib` utilities (e.g. [`tf.contrib.rnn`][contrib-rnn])

.orange[**각종 Wrapper: 취향의 영역 (Optional)**]

- [Keras][keras]
- TensorFlow learn (a.k.a. skflow)
- [TF Learn][tflearn]  .small.oc-grape-4[공식 tf.contrib.learn과 다른 것임. 헷갈리지 말것]
- [prettytensor]
- [sugartensor]

[contrib]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/
[contrib-layers]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/layers
[contrib-rnn]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn.py
[slim]: https://github.com/tensorflow/models/tree/master/slim
[keras]: https://keras.io/
[tflearn]: http://tflearn.org/
[prettytensor]: https://github.com/google/prettytensor
[sugartensor]: https://github.com/buriburisuri/sugartensor



---

template: centertext



<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">I hoped TensorFlow would standardize our code but it&#39;s low level so we&#39;ve diverged on layers over it: Slim, PrettyTensor, Keras, TFLearn ...</p>&mdash; Andrej Karpathy (@karpathy) <a href="https://twitter.com/karpathy/status/765734518594547712">August 17, 2016</a></blockquote>


Andrej Karpathy도 예전에 이를 지적한 적이 있습니다.

---

## 선택 시 고려할 것들

TensorFlow core (contrib 포함)에 있는 것을 가급적 먼저 선택하되
.dogdrip[믿고쓰는 구글],



--

- 내가 원하는 기능이 충분히 들어있는가?
- 충분히 확장성이 있는가?
  - (복잡한) 새로운 기능을 추가하기 위하여 library 내부 혹은 밑단을 고치거나 library 전체를 들어내야할 필요는 없는가?
  -  그 framework의 style .oc-red-7[만]을 고집하지는 않는가?
  - 다른 것과 섞어 쓰기 충분한가? 특히, TensorFlow core API 와 호환성이 좋은가?
  - e.g. `XXX_initializer()`

---

#### e.g. [TF Learn][TFlearn]


Toy Example 을 벗어나 더 복잡한 네트워크를 학습시킬 수 있을까?

```python
# TF learn
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam',
                    loss='categorical_crossentropy',
                    learning_rate=0.001)
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96, run_id='cifar10_cnn')
```

- 만약, 다른 optimizer, activation을 쓰고 싶다면? (.dogdrip[string...??])
- 만약, fit에 내가 원하는 옵션이 없다면?

.gray[➡ 결국엔 다 뜯어고쳐야 할 수도 있으니 처음에 잘 선택합시다]



---

## Examples of [`tf.contrib`][contrib]: Common Layers

`tf.nn.convolution()` 보단 우리는 3x3 필터인게 더 관심이 있습니다
(`contrib`에는 레이어 수준의 추상화, `slim`에는 모델수준의 추상화까지 담겨있다)

```python
with tf.name_scope("conv1"):
    input_ch = inputs.get_shape().as_list()[-1]
    weights = tf.get_variable("weights", shape=(ksize, ksize, input_ch, num_outputs))
    biases = tf.get_variable("biases", shape=(num_outputs,))
    convolve = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding=padding)
    conv1 = tf.nn.bias_add(convolve, bias)
```

.dogdrip[바퀴 다시 만들지 말고] 있는건 잘 가져다 쓰면 편합니다!

```python
with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
*   net = tf.contrib.layers.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
    net = tf.contrib.layers.max_pool2d(net, [3, 3], 2, scope='pool1')
    net = tf.contrib.layers.conv2d(net, 192, [5, 5], scope='conv2')
    # ...
```

기타: `layers.batch_norm`, `tf.one_hot_encoding`






---

## Examples of [`tf.contrib`][contrib]: `train_op`

미분 (`tf.gradient`)하고, gradient clipping 해서, `train_op` 구하기

Before:

```python
self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
train_vars = tf.trainable_variables()
gradients = tf.gradient(model.loss, train_vars)

# gradient clipping by max norm
gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)
self.train_op = self.optimizer.apply_gradients(
    zip(grads, train_vars),
    global_step=self.global_step
)
```



--

After :
<span data-balloon-pos="down" data-balloon-break data-balloon="tf.contrib.layers.optimize_loss(&#10; loss,&#10; global_step,&#10; learning_rate,&#10; optimizer,&#10; gradient_noise_scale=None,&#10; gradient_multipliers=None,&#10; clip_gradients=None,&#10; learning_rate_decay_fn=None, &#10; update_ops=None,&#10; variables=None,&#10; name=None,&#10; summaries=None,&#10; colocate_gradients_with_ops=False)">
[`tf.contrib.layers.optimize_loss()`][tf_optimize_loss]
</span>
를 써서 한방에:

[tf_optimize_loss]: https://www.tensorflow.org/api_docs/python/contrib.layers/optimization#optimize_loss

```python
self.train_op = tf.contrib.layers.optimize_loss(
    loss=model.total_loss,
    global_step=self.global_step,
    learning_rate=self.learning_rate,
    optimizer='Adam',
    clip_gradients=max_grad_norm
)
```



---

## Examples of [slim][slim]: Load Pretrained Models

- Caffe 에서는 `.prototxt` (모델) 및 `.caffemodel` (weight) 을 사용
- 하지만, Tensorflow는 checkpoint 로 pre-train된 모델을 다룹니다
- [Model Zoo](https://github.com/tensorflow/models/tree/master/slim#pre-trained-models) 에는 VGG\*, ResNet\*, Inception\* InceptionResnet\* 등 널리 사용되는 CNN을 파라미터와 함께 제공하고 있습니다

```python
import tensorflow as tf
import nets.inception_resnet_v2 as inception_resnet_v2

# Load the images and labels.
images, labels = ...

# Create the Inception-Resnet model.
*with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
*    logits, end_points = inception_resnet_v2.inception_resnet_v2(images)

# Define the loss functions and get the total loss.
loss = slim.losses.softmax_cross_entropy_with_logits(logits, labels)

# ...train helper
```



---

.img-100[![](images/tensorflow-model-zoo.png)]

Check out the [Tutorial - Pretrained Models](https://github.com/tensorflow/models/tree/master/slim#pre-trained-models)



---

#### 이미 잘 학습된 파라미터를 가져와 파인튜닝하기


1. 체크포인트로부터 모델 시작하기
2. 체크포인트로부터 모델 시작하되 변수명이 매치되지 않는 경우
3. 체크포인트로부터 모델의 일부만 가져오기(e.g. 마지막 레이어)
4. 메모리의 값을 모델에 넣기

--

아래처럼 [slim](https://github.com/tensorflow/tensorflow/blob/0.12.1/tensorflow/contrib/slim/python/slim/learning.py#L120) 이 다 해결해줍니다.

```python
# ...build model
# Create the initial assignment op
checkpoint_path = 'inception_resnet_v2_2016_08_30.ckpt'
variables_to_restore = slim.get_model_variables()
*init_assign_op, init_feed_dict = assign_from_checkpoint(
*    checkpoint_path, variables_to_restore)

# Create an initial assignment function.
def InitAssignFn(sess):
    sess.run(init_assign_op, init_feed_dict)

# Run training.
slim.learning.train(train_op, my_log_dir, init_fn=InitAssignFn)
```

