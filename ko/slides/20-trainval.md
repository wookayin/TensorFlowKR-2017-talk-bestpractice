name: trainval
template: recipe

# Training Loop

### Training, Validation, Evaluation, ...

---

## 요구사항

- Load Dataset
  - .oc-blue-5[Training Set] and .oc-blue-5[Validation Set]
- Build Model
- Training 준비
  - model analyzer 등 디버깅 정보 출력...
  - variabile initialization (`tf.global_variables_initializer()`)
  - summary 준비 (`tf.summary.merge_all()`)
  - `train_op` 만들기
- while not converged ...
  - Prepare a single mini-batch from dataset
  - Run `train_op` (training single step)
  - 주기적으로 validation step 실행
  - 가끔씩(??) evaluation 1epoch 실행

---

## Traditional Training Loop

Training과 Validation을 한 프로세스에서 같이 하려면

```python
# flag for train/validation
is_training = tf.placeholder_with_default(
    tf.constant(False, dtype=tf.bool),
    shape=(), name='is_training'
)
# ...
# Using contrib's smart_cond for branching two queues
x_images, y_labels = tf.contrib.layers.utils.smart_cond(
    is_train,
    lambda: (x_images_train, y_labels_train),
    lambda: (x_images_valid, y_labels_valid)
)
# ...
while True:
    session.run(train_op, feed_dict={is_training: True})

    # Periodically do validation
    if step % FLAGS.steps_per_validation == 0:
        session.run(train_op, feed_dict={is_training: False})
```

- `is_traing` 을 `tf.Tensor`로 만들고 `feed_dict`으로 True/False를 넘깁니다

---

혹은 training 용 graph 를 만들고, validation 용 (inference mode) graph 를 만들고,
  variable를 share한다.

- [Tutorial 참고](https://www.tensorflow.org/how_tos/variable_scope/index)
- RNN 같은 경우는 training 할 때 와 generation 할 때 graph가 달라져야 할 필요가 있음
- 앞에서 이야기한대로 모델을 구현할 때 scope를 잘 정의해야 쉽게 할 수 있음


---

## Training Loop와 Validation/Evaluation Loop를 분리

더 큰 데이터셋으로 복잡한 모델을 학습시키려면, <br/>
.best-practice[train.py와 eval.py를 다른 프로세스에서 돌리는 것을 고려해보자]

.img-100[![](images/train_valid_command-with-info.png)]

---

## Training Loop와 Validation/Evaluation Loop를 분리

별도 프로세스에서, 임의의 checkpoint 와 모델을 로딩해서 온갖 evaluation metric을 계산함

e.g. .small[https://github.com/tensorflow/models/tree/master/im2txt/im2txt]

.small[
- show_and_tell_model.py
- evaluate.py
- train.py
]


---

## Training Loop와 Validation/Evaluation Loop를 분리

장점

- evaluation 1epoch 도는 것이 굉장히 오래 걸리는 경우가 많은데, <br/>
  train/validation/eval 을 분리해두면 .blue[필요할 때만] (비동기적으로) evaluation을 돌릴 수 있습니다
- 하나의 GPU에서는 training 만 돌기 떄문에 .green[전체 학습 시간을 많이 아낄 수 있음]
- validation을 CPU only로도 돌릴 수 있다! .gray[4 GPU로 트레이닝]
- 특정 checkpoint 에 대해서 evaluation을 유연하게 돌려볼 수 있음
- evaluation 은 느리기 때문에 작은 데이터셋에 돌릴 수 있게끔 flexible 해야한다

단점

- 만약 GPU가 1대라면? eval.py는 CPU로 돌려야만 합니다
- Validation 로스를 보면서 Early Stopping하고 싶을 수 있습니다
- 코딩이 약간 번거로움..


---


train/validation 모두 동시에 tensorboard로 같이 보면 좋습니다

.img-100[![](images/train_valid-epoch6.png)]



---

### slim 등에서 evaluation loop utility도 제공함


train.py은 `train-dir/`에 event와 checkpoint를 저장하고

```python
logdir = config['train_dir']
slim.learning.train(
    train_op,
    logdir=logdir,
    number_of_steps=1000,
    save_summaries_secs=300,
    save_interval_secs=600
)
```

eval.py은 `train-dir/` 에 checkpoint가 생길 때마다 `eval_op`을 실행



```python
# ... Create evaluation metrics
slim.evaluation_loop(
    master='',
    checkpoint_dir,
    logdir=logdir,
    num_evals=num_evals,
    eval_op=names_to_updates.values(),
    summary_op=tf.summary.merge_all(),
    eval_interval_secs=600
)
```

