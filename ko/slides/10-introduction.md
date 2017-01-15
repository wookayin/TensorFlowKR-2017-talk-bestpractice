name: centertext
class: center, middle, centertext
layout: true

---

name: introduction
nav-marker: &#x249C;
template: centertext

.img-33[![TensorFlow](images/tensorflow-logo.png)]

**Code Patterns**

**Best Practices**


<script type="text/javascript" data-showslide>
    var $p = $slide.find("p");
    $p.css('opacity', 0).each(function(i) {
        $(this).stop(true).delay(i * 1000).fadeTo(300, 1);
    });
</script>

---

'Best Practice' 라고 써놨지만 <br/>
어떤 '정답'을 이야기하고자 하는 건 아닙니다. <br/>

---

미천한 경험을 돌이켜보면.. <br/>
.dogdrip[삽질 기록]

---

문서화되지 않은 TF 기능 <br/>
하루가 다르게 새로 추가되는 새로운 기능 <br/>
.dogdrip[원래 삽질하며 배우는 거라지만..]

---

보고 따라할만한 모범적인 좋은 example 이 없었음 <sup>[\*]</sup> <br/>
.dogdrip[초기에 공개된 코드들의 퀄리티는 ㅠㅠ] <br/>

.footnote[[\*] 그러나 최근에는 [Google's TensorFlow Model 예제](http://github.com/tensorflow/models)들이 많이 올라왔습니다.]

---

일단 데드라인이 얼마 안 남았으니까 .dogdrip[모든 것의 발단] <br/>
어떻게할지 잘 모르겠으니 일단 막 짜고 나중에 고쳐야지 <br/> .dogdrip[하지만 그런 일은 일어나지 않았다]

---

**이대로는 안되겠다**


---

.img-80[![](images/teaser-01-note.png)]

---

.img-80[![](images/teaser-03-note.png)]

---

.oc-blue-5['이럴 때는 이렇게 코드를 짜는 것이 좋겠다']

--
<br/>
와 같은

(i) 설계 시 고려사항 <br/>
(ii) 각종 자잘한 Tip <br/>
(iii) 기타 고민거리 <br/>

--

...등을 다른 분들과 나누고 싶었습니다


---

어떤 이야기들을 할 것인가?


---

layout:false

## TensorFlow 로 딥러닝 코드 짜기

- (1) Training/evaluation에 필요한 **Dataset** 을 준비한다.
- (2) Model training loop 및 pipeline을 구현한다.
- (3) Model 코드를 구현한다.
- (4) 실험! 실험!

쉬워 보이지만...

---

## TensorFlow 로 딥러닝 코드 짜기

(1) Training/evaluation에 필요한 **Dataset** 을 .green[**"잘"**] 준비한다.

--

- 적절한 **전처리**(pre-processing)도 필요하고
- **효율적인 I/O** 가 가능해야 하고: e.g. Queue? Threading?
- 다른 종류 혹은 **여러가지 데이터셋**을 쓸 필요도 있고
- 디버깅에 필요한 **부가적인 정보**들도 담고 있어야 하고

좀 더 욕심을 내면..

- **하나의 효율적인 I/O** 인터페이스가 **여러 데이터셋**에 적용하게끔 설계하고싶고
- **여러 전처리 방식(ImageNet, Inception 등)**도 필요에 따라 쉽게 조합가능하고

... **잘** 만들고 싶다!


---

## TensorFlow 로 딥러닝 코드 짜기

(2) Model training loop 및 pipeline을 구현한다.

--

- Optimizer: loss gradient 계산하고, variable을 update하는 op을 만듬
- 주기적으로 (몇 step마다) validation도 해야겠지?
- 가끔씩 validation set에 대해 full evaluation도 해봐야지!
- pre-trained 로딩도 해야하고.. checkpoint 저장도 해야하고..
- 디버깅을 위해서 Logging 과 summary 도 필요하네!
- .dogdrip[아 귀찮아 매번 똑같이 짜야함....]

---

## TensorFlow 로 딥러닝 코드 짜기

(3) (여러가지) Model 코드를 구현한다.

--

- 예쁘게 잘... .dogdrip[암호문 싫어요]
- 다른 네트워크 component를 .dogdrip[복&붙 없이] 잘 가져다 붙이려면..?
  - .gray[e.g. End-to-end learning, Using ResNet-151 encoder]
- Multi-GPU 로 손쉽게 확장하려면?
- 디버깅과 Trouble-shooting이 쉽게 하려면 어떻게 해야할까?
  - .gray[Model variable/parameter 찍어보기 등등...]


---

## TensorFlow 로 딥러닝 코드 짜기

(4) 실험... 변경... 실험...

--

- Hyperparameter나 Model configuration은 어떻게 관리하지?
- 모델이 많아지거나 조금씩만 달라지는 부분이 많아진다면
  <br/> 코드 복붙(...) 없이 어떻게 variant 들을 잘 관리할 수 있을까?


---

## 개인적인 경험

- .red[(1) training/evaluation에 필요한 **Dataset** 을 **잘** 준비한다.]
- .red[(2) Model training loop 및 pipeline을 구현한다.]
- .green[(3) Model 코드를 구현한다.]
- .red[(4) 실험! 실험!]


--

.green["모델의 핵심 부분"] 보다,
이런 .red[부가적인 glue code] 등을 구현하는 것이
<br/>
더 힘들고 시간이 많이 걸리는 일이었습니다.
.dogdrip[복잡한 기능과 요구사항이 많음]


---

class: middle

![Schulley et al., 2015](images/Sculley2015-debt-mlcode.png)

*Only a tiny fraction of the code in many ML systems is actually devoted to learning or prediction*.

.small[Schulley et al, [Hidden Technical Debt in Machine Learning Systems][schulley-2015], In NIPS 2015.]

[schulley-2015]: https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf


---

## 두 가지 핵심 목표

- .green[**신속하게 (Be Agile)**] : .red[모델의 핵심 부분에 집중]하여 .blue[빠르게] 구현하고 실험할 수 있도록 하자!
- .green[**예쁘게 (Be Pretty)**] : .dogdrip[보기 좋은 떡이 먹기도 좋다!] 재사용이 가능하고, 유지보수가 쉽고, 확장이 쉽도록 (복잡한 요구사항에 대응), 잘 정돈되고 기능별로 적절하게 모듈화된/decoupled된 코드를 작성하자.


다양한 Example 위주로 살펴봅시다.


---

## Disclaimer

- 물론 유일한 '정답'은 없습니다. .dogdrip[제가 지금 틀렸을 수도 있죠!]
- 상황에 따라서는 유연하거나 대충 .dogdrip[빨리]하는 것이 필요할 수도 있습니다.
  - e.g. .gray[Toy Problem, Prototyping]
- 다만, .oc-green-7[어느 일정 규모] 이상의 Deep Learning 코드를 작성하고 각종 실험을 할 때 (Research Problem을 푸는 관점에서)
  이러이러한 상황에서는 이렇게 하면 좋더라.. 하는
  .oc-blue-6[개인적인 의견]이나 .oc-blue-6[경험담]을 주로 포함하고자 했습니다.<sup>[1]</sup>
  .footnote[[1] 아쉽게도, 이 짧은 talk에서 모든 내용, 자잘한 Tip, Case Example을 모두 다룰 수는 없었습니다ㅠㅠ]
- 다소 당연한 얘기가 많더라도 양해해주세요
- 잠시 후 Poster session 에서도 많은 논의와 토론 환영합니다!

---

name: recipe
class: center, middle, inverse, recipe
layout: true

---

template: recipe

# Common Patterns and Practices


---

layout: false

## 목차

- [(1) Basic Project Structure](#projectstructure)
- [(2) Model Style](#abstraction)
- [(3) High Level API](#highlevel)
- [(4) Validation while Training](#trainval)
- [(5) Dataset](#data)
- [(6) Other Tips](#tips)

