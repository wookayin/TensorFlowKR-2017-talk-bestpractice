name: data
template: recipe

# Dataset

### Use Queue!



---

layout: false

## Dataset

머신러닝의 핵심!!

- 데이터셋은 수많은 종류가 있많지만 공통 인터페이스로 표준화는 것은 .red[불가능]
- 제각각 다른 인터페이스와 사용법을 갖고 있음
- 그러나 결국 TF 코드에서 필요한 것은,
  <br/> input tensor에 (training을 위해) <u>적절한 값</u>을 넣어주는 Pipeline





---


## We need a Swiss Army Knife (Dataset Utility)!

- 궁극적으로는 TF model에 feed할 수 있는 배치(batched) input을 제공해야 함
  - .gray[e.g. image: (32,224,224,3) float32, label: (32,) int32]
- (TF와 상관없이) Dataset 접근에 대한 Fluent API가 필요하다!
  - .gray[e.g. 특정 id 에 대한 image, label, GT annotation 등 raw data 접근]
  - .gray[e.g. number of dataset, 모든 example (1 epoch) enumerate 하기]
  - .gray[Over-engineering 하지 않고 필요한 만큼만 기능을 개발할 것!]


- batch를 구성할 때 feed 할 tensor 이외에 각종 부가정보들도 도움이 된다.

```python
In [1]: batch_chunk.keys()
Out[1]: ['image_id', 'image', 'caption_length', 'caption_raw', 'caption_words']

In [2]: batch_chunk['image_id'][0]
Out[2]: 174888

In [3]: batch_chunk['caption_raw'][0]
Out[3]: 'A man eating a piece of pizza on a paper plate.'
```




---

## TensorFlow Data Reading 101 <sup>[\*]</sup>

.footnote[[\*] Data I/O 하는 방법에 대한 자세한 설명과 코드 예제는 공식 문서 [HOWTO Reading Data][howto-reading-data] 를 참고하세요]

[howto-reading-data]: https://www.tensorflow.org/how_tos/reading_data/

--

- **Feeding** : `feed_dict` 를 통해 직접 Data를 placeholder에 넣어줌
  - 💚 .green[쉽다!] 데이터 처리를 python 코드로 해도 됨 (.green[높은 자유도])
  - 😡 .red[느림]... 데이터 처리 도중에 GPU가 쉼 (low GPU utilization)

--

- **Pre-loading** : 메모리(RAM or Graph)에 한번에 다 올려놓고 시작한다.
  - 💚 빠름 / 😡 메모리 부족... 올리는 것도 느림...
  - .dogdrip[Toy dataset을 제외하고는 꿈도 희망도 없음]

--

- **Reading from files** : .blue[Threading & Queue를 사용한 I/O]
  - 💚 .green[Efficient and Fast], even for large dataset.
  - 😡 개념이 다소 복잡하고... 구현이 번거롭고 까다로움

---

### 개인적인 경험담 ...

- Feeding 을 쓰면 (CPU code로 직접 batch를 만들어서 넣어주면)
  - 편하긴 한데 너무 느려서... (training과 parallel하게 돌 수 없음)
  - `dataset.next_batch()` 함수를 직접 구현하는 것이 생각보다 번거로움

--

<p>

- **사실상의 대부분 Use Cases** 에서 .blue[Queue]를 사용한 I/O 처리는 필수적!
  - .dogdrip[Training이 아주 느려도 상관없다면 Pass]
  - `tf.train.batch()`, `tf.train.shuffle_batch()` 을 쓰면
      (i) batch tensor로 묶어주는 것 (ii) data shuffling 을 편하게 할 수 있음

- 시작할 때는 `feed_dict()`로 짜고 나중에 queue를 사용한 방식으로 옮기는 것도 괜찮지만,
  처음부터 queue를 쓰는 것이 구현 시간을 많이 아낄 수 있다.
  - [Tutorial on Threading and Queues](https://www.tensorflow.org/how_tos/threading_and_queues/)


