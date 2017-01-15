name: projectstructure
template: recipe

# Basic Project Structure

---

layout: false

## A Typical ML Structure

- (1) training/evaluation에 필요한 **Dataset**을 준비한다.
- (2) Model training loop 및 pipeline을 구현한다.
- (3) Model 코드를 구현한다.
- .dogdrip[(4) 실험! 실험!]

---

구체적으로:

- Dataset wrapper 또는 data loader 코드를 짜야하고
  - .gray[`data_loader.py`, `input_ops.py`, ...]
  - .gray[`preprocess_data.py`, ...]
- .green[Model] 코드를 짜야하고 (TensorFlow computation graph를 빌드)
  - .gray[`model.py`, `models/xxxnet.py`, ...]
- Model을 training 하는 코드를 짜야하고 (.green[Trainer])
  - .gray[`train.py`]
- Evaluation 하는 코드를 짜야한다 (.green[Evaluator])
  - .gray[`eval.py`]


이 때,
--

.best-practice[Model 과 train/eval 을 분리(decouple)시키는 것이 필요하다.]
<br/>
다시말해, `model.py` 와 `train.py` 를 나누자 !!
<sup>[1]</sup> <span class="footnote">[1] 경우에 따라 train/test 를 한 코드에서 할거면 대신 `main.py --train` 등의 flag를 쓰는 것도 방법!</span>

---

## Model 과 Training 코드의 분리

- .best-practice[Model 과 train/eval 을 분리(decouple)시키는 것이 필요하다.]

- **단일 책임의 원칙** [(Single Responsibility Principle)][wiki-SRP]: 관심사의 분리
  - "각 모듈은 한 가지의 .border-red.border-dotted[역할] 또는 .border-red.border-dotted[책임]을 가져야 한다"

--

<p>

- Model 은 어떤 역할을 해야 하는가?
  * ✅ `build_graph()`: Computation Graph 를 만드는 것
  * ❓ .dogdrip[Checkpoint Load & Save] → Trainer 의 역할
  * ❓ .dogdrip[Training: e.g. `sess.run(train_op)`] → Trainer 의 역할

[wiki-SRP]: https://en.wikipedia.org/wiki/Single_responsibility_principle





---

.oc-red-7[*A Bad Style*] : `model.py` — 간단하게 짜면 아래처럼 되겠지만...

```python
class MyModel(object):
    def __init__(self, session, config):
*       self.session = session
        self.config = config

        self.input_image = tf.placeholder(...)
        self.output = layers.fully_connected(input_image, 10)  # dummy model

*   def train(self):
        self.train_op = self.optimizer.minimize(self.loss)
        run_training_loop(self.session, self.train_op)

def main():
    ...
    with tf.Session(...) as session:
        model = MyModel(session)
        model.train()
```

---

.oc-green-7[*A Good Style*]
<sup>[\*]</sup> .footnote[[\*] 굉장히 많은 코드를 생략하거나 간략화했음]
: `model.py` 에서 training 관련 코드 분리

```python
# model.py
class MyModel(object):
    def __init__(self, config):
        self.config = config
        self.input_image = tf.placeholder(...)
        self.gt_label = tf.placeholder(...)
        self.output = layers.fully_connected(input_image, 10)  # dummy model
        self.loss = tf.nn.softmax_cross_entropy_with_logits(output, self.gt_label)
```

```python
# train.py
def train(model):
    train_op = optimizer.minimize(model.loss)
    with tf.Session(...) as session:
        run_training_loop(session, self.train_op)

def main():
    ...
    model = MyModel(config=...)
    train(model)
```

model 의 constructor에 `session`을 넣어주지 않음에 유의

---

## Model 과 Training 코드의 분리

왜 Model을 train/eval과 분리해야 하는가? .dogdrip[Why SRP?]

- .oc-blue-6[모델을 여러개 짜는 경우]: training 코드 1개, model 코드 N개
  - 모델과 trainer가 강결합 되어있다면 자연스럽게 중복이 발생
- .oc-blue-6[모델 A와 B를 합쳐서 (연결하여)] 더 큰 네트워크 AB를 만들 때?
  - AB의 training 코드는 어떻게..?

마찬가지 이유로,
.best-practice[model은 `session` 과 결합하지 않고 분리하는 것이 좋다!]
  - model: graph 및 TensorFlow op을 생성(build)하는 역할
  - `session`: 생성된 graph (TensorFlow ops)를 .oc-blue-6[실행]하기 위한 context


---

하지만 model과 session의 완전한 decoupling이 다소 어려운 경우도 있음!

- model의 single step이 복잡하거나 모델 내부 tensor에 대해 잦은 접근을 해야 하는 경우 등 (동작의 detail을 모델에 위임)
  [(example: seq2seq)](https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/seq2seq_model.py#L195)

```python
class Seq2seqModel(object):
    def __init__(self, ..., forward_only=False):
        ...
*   def step(self, session, encoder_inputs, decoder_inputs,
             target_weights, bucket_id, forward_only):
        ...
    # def generate(self, input)
```

```python
def train():
  # ...
  with tf.Session() as sess:
    # ...
    encoder_inputs, decoder_inputs, target_weights = \
        model.get_batch(train_set, bucket_id)
*   _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
*                                target_weights, bucket_id, forward_only=False)
```

- model 코드에서는 `self.session`이 아닌, `session`을 argument로 외부에서 주입받아<sup>injection</sup> 사용하고 있음

