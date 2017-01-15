name: abstraction
title: Writing Model Codes
template: recipe

# Writing .oc-cyan-3[Model] Codes

### Abstraction Patterns in TensorFlow


---

layout: false

## A Question?

모델(혹은 일부)을 구현할 때...
함수(function)처럼 짜야할까요?

```python
def vgg19(image):
    conv1 = layers.convolution2d(image, 64, [3, 3], scope='conv1')
    ...
    output = layers.fully_connected(fc7, 1000, scope='output')
    return output

def train():
    image = read_image_dataset()  # [batch_size, 224, 224, 3] Tensor or placeholder
*   predict_logit = vgg19(image)  # [1000] Tensor
```

.small[
e.g. [inception](https://github.com/tensorflow/models/tree/master/inception/inception), [slim nets](https://github.com/tensorflow/models/tree/master/slim/nets)
]

---

## A Question?

아니면 클래스(class) 스타일로 짜야할까요?

```python
class VGG19():
    def __init__(self):
        self.conv1 = layers.convolution2d(image, 64, [3, 3], scope='conv1')
        ...
        self.output = layers.fully_connected(fc7, 1000, scope='output')

def train():
    image = read_image_dataset()   # [batch_size, 224, 224, 3] Tensor
*   net = VGG19(image)
*   predict_logit = vgg19.output   # [1000] Tensor
```

혹은 

```python
def train():
    net = VGG19()
*   predict_logit = net.build_graph(image)  # [1000] Tensor
```

.small[
e.g. [im2txt](https://github.com/tensorflow/models/tree/master/im2txt)
]

---

## A Question: Model (Component) 구현 스타일

정답......
--
은 없습니다

--
...만 .dogdrip[그때그때 달라요] (둘 다 좋은 practice입니다)

.blue[**요구사항 및 고려사항**]을 정리해보자면:

--

- (a) `input` to `output` 매핑이 가능해야 함 (일종의 Black box 처럼 보기)
- (b) 중간 tensor (layer)의 activation 에 접근이 가능해야 하고
- (c) Model (component)에 존재하는 variable 을 가져올 수 있어야 하고
- (d) 작성한 컴포넌트를 library 처럼 다른 곳에 붙이거나 <br>
      여러 모델을 합성(e.g. Ensemble)하기 용이해야 하고
- 등등등...

---





Requirements

```python
def vgg19(image):
    conv1_1 = layers.convolution2d(image, 64, [3, 3], scope='conv1_1')
    ...
    conv5_4 = layers.convolution2d(conv5_3, 512, [3, 3], scope='conv5_4')
    fc6 = layers.fully_connected(conv5_4, 4096, scope='fc6')
    fc7 = layers.fully_connected(fc6, 4096, scope='fc7')
    output = layers.fully_connected(fc7, 1000, scope='output')
    return output
```

```python
def train():
    image = read_image_dataset()  # [batch_size, 224, 224, 3]
    predict_logit = vgg19(image)  # [1000] Tensor
```

- (b) `conv5_4` feature map 등 다른 endpoint를 가져오고 싶다면?
- (c) 각 layer별 variable 들을 가져오고 싶다면?

---

.oc-red-6[Oh....]

```python
def vgg19(image):
    conv1_1 = layers.convolution2d(image, 64, [3, 3], scope='conv1_1')
    ...
    conv5_4 = layers.convolution2d(conv5_3, 512, [3, 3], scope='conv5_4')
    fc6 = layers.fully_connected(conv5_4, 4096, scope='fc6')
    fc7 = layers.fully_connected(fc6, 4096, scope='fc7')
    output = layers.fully_connected(fc7, 1000, scope='output')
*   return output, conv1_1, conv1_2, ..., conv5_4, fc6, fc7
```

```python
def train():
    image = read_image_dataset()  # [batch_size, 224, 224, 3]

    output, _, ..., conv5_4, _, _ = vgg19(image)
```

- (b) .orange[`conv5_4` feature map 등 다른 endpoint를 가져오고 싶다면?]


---

.oc-orange-6[Hmm...?]


```python
def vgg19(image):
    net = {}
    net['conv1_1'] = layers.convolution2d(image, 64, [3, 3], scope='conv1_1')
    ...
    net['conv5_4'] = layers.convolution2d(net['conv5_3'], 512, [3, 3], scope='conv5_4')
    net['fc6'] = layers.fully_connected(net['conv5_4'], 4096, scope='fc6')
    net['fc7'] = layers.fully_connected(net['fc6'], 4096, scope='fc7')
    net['output'] = layers.fully_connected(net['fc7'], 1000, scope='output')
*   return output, net
```

```python
def train():
    image = read_image_dataset()  # [batch_size, 224, 224, 3]

*   output, end_points = vgg19(image)
    convmap = end_points['conv5_4']
```

- (b) .orange[`conv5_4` feature map 등 다른 endpoint를 가져오고 싶다면?]



---

.oc-orange-6[Hmm...?]


```python
class VGG19():
    def __init__(self):
        self.conv1 = layers.convolution2d(image, 64, [3, 3], scope='conv1_1')
        ...
        self.conv5_4 = layers.convolution2d(self.conv5_3, 512, [3, 3], scope='conv5_4')
        self.fc6 = layers.fully_connected(self.conv5_4, 4096, scope='fc6')
        self.fc7 = layers.fully_connected(self.fc6, 4096, scope='fc7')
        self.output = layers.fully_connected(fc7, 1000, scope='output')
```

```python
def train():
    image = read_image_dataset()   # [batch_size, 224, 224, 3] Tensor

*   net = VGG19(image)
*   predict_logit = net.output   # [1000] Tensor
    convmap = net.conv5_3
```

- (b) .orange[`conv5_4` feature map 등 다른 endpoint를 가져오고 싶다면?]

---

.best-practice[Class style로 짜더라도, input tensor는 외부에서 주입(inject)받도록 하자]
<br/>: (a) input to output 매핑이 가능해야 함 (일종의 Black box 처럼 보기)

.oc-red-7[*A Bad Style*]
```python
class VGG19():

    def __init__(self):
*       self.image = tf.placeholder(image, [224, 224, 3], name='input')
        self.conv1 = layers.convolution2d(self.image, 64, [3, 3], scope='conv1_1')
        ...
        self.conv5_4 = layers.convolution2d(self.conv5_3, 512, [3, 3], scope='conv5_4')
        self.fc6 = layers.fully_connected(self.conv5_4, 4096, scope='fc6')
        self.fc7 = layers.fully_connected(self.fc6, 4096, scope='fc7')
        self.output = layers.fully_connected(fc7, 1000, scope='output')
```

```python
def train():
*   net = VGG19()
    predict_logit = net.output
    train_op = optimizer.minimize(net.loss)

    with tf.Session(...) as session:
        for s in range(MAX_STEPS):
*           session.run(train_op, feed_dict={self.image: ...})
```


---

.best-practice[Class style로 짜더라도, input tensor는 외부에서 주입(inject)받도록 하자]
<br/>: (a) input to output 매핑이 가능해야 함 (일종의 Black box 처럼 보기)

.oc-green-7[*A Better Style*]
```python
class VGG19():

    def __init__(self, input):
*       self.input = input  # assumes shape of [batch_size, H, W, 3]
        self.conv1 = layers.convolution2d(self.input, 64, [3, 3], scope='conv1_1')
        ...
        self.conv5_4 = layers.convolution2d(self.conv5_3, 512, [3, 3], scope='conv5_4')
        self.fc6 = layers.fully_connected(self.conv5_4, 4096, scope='fc6')
        self.fc7 = layers.fully_connected(self.fc6, 4096, scope='fc7')
        self.output = layers.fully_connected(fc7, 1000, scope='output')
```

```python
def train():
*   input_image = tf.placeholder([batch_size, 224, 224, 3], tf.float32)
    # input_image는 placeholder가 아닌 임의의 다른 텐서도 될 수 있음 (e.g. queued)

*   net = VGG19(input_image)
    predict_logit = net.output
    train_op = optimizer.minimize(net.loss)

    with tf.Session(...) as session:
        for s in range(MAX_STEPS):
*           session.run(train_op, feed_dict={input_image: ...})
```

---

## Model 코드 작성 시 생각

개인적인 생각: end code를 짤 때는 class style이 여러가지 장점이 있음.

- 단, class가 해야할 것까지만 하도록 하고, 불필요한 기능을 많이 넣어 지나치게 방대해지지 않도록 유의하자!
- 앞의 요구사항 (a)-(d)을 잘 고려하면, 깔끔한 설계가 가능하고
  여러 모델을 재조합하기 등의 복잡한 요구사항에도 잘 대응할 수 있다.


