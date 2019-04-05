Toward Best Practices of TensorFlow Code Patterns
==================================================

**Jongwook Choi** ([@wookayin][wookayin]) and
**Beomjun Shin** ([@shastakr][shastakr])

A talk in [the 2nd meetup][tfkr-meetup] of [TensorFlow Korea Group][tfkr-facebook] <br/>
January 14th, 2017


<hr>
<p align="center">
🚧 Warning!! 🚧
</p>
<p align="center">
Many contents in this talk are <b>outdated</b> (and thus less recommended) as they were based on older versions of TensorFlow.
<br/>
TensorFlow now has more handy APIs and is moving toward <a href="https://www.tensorflow.org/alpha/guide/effective_tf2">TF 2.0</a>
that encourages many better practices.
<br/>
This repository is kept only for archival purposes.
</p>

<hr>

[https://wookayin.github.io/TensorFlowKR-2017-talk-bestpractice/][github-pages]


Slides
------

- :kr: [**발표자료 (In Korean)**](https://wookayin.github.io/TensorFlowKR-2017-talk-bestpractice/ko/)
- :us: Lecture Slide (In English): ~_Coming Soon!_~


Abstract (In Korean)
-----------------

이번 발표에서는 (주로 연구 프로젝트를 위한) TensorFlow 코드를 작성할 때의
Best Practice 그리고 Code Patterns 에 대해 이야기합니다.
잠정적으로 다음과 같은 내용들을 다룰 예정입니다:

- 구현 및 설계 시의 각종 요구사항/고려사항
- TensorFlow의 모델 모듈화 및 추상화 패턴 (+횡단관심의 처리 방법)
- High-level API 적극적으로 사용하기 (tf.contrib.layers, tf.slim, tf.learn 등)
- (i) 모델 코드 작성 (ii) 데이터셋 및 트레이닝 상용구 (iii) 실험 및 설정 관리 (iv) 기타 자잘한 팁 등에 관한 패턴과 코드 예제


Abstract
--------

In this talk, we aim to deliver several best practices, guidelines, and code patterns for writing TensorFlow codes,
mainly when conducting research projects.
A tentative and incomplete list of topics:

- Common requirements and frequent design concerns in implementation
- Patterns of model modularization and abstraction in TensorFlow (+ cross-cutting concerns)
- Using modern and high-level APIs (tf.contrib.layers, tf.slim, tf.learn, etc.)
- Patterns and code examples on: (i) styles of model writing, (ii) dataset loader and training boilerplates, (iii) management of experiments and configuration,
  and (iv) other general and miscellaneous tips.


Note
----

The talk will given in Korean, but I will self-translate the material into English shortly after the talk.

The topics are mainly based on the authors' personal experience, so may contain some opinionated suggestions.
Of course, there cannot be the only answer:
Please contact me if you have a suggestion, a question, or an idea that can improve the contents of this talk!

[tfkr-meetup]: http://onoffmix.com/event/86620
[tfkr-facebook]: https://www.facebook.com/groups/TensorFlowKR/
[github-pages]: https://wookayin.github.io/TensorFlowKR-2017-talk-bestpractice/

[wookayin]: https://github.com/wookayin
[shastakr]: https://github.com/shastakr
