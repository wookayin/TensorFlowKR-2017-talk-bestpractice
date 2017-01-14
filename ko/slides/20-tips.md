name: tips
template: recipe

# TensorFlow: tips

---

layout: false

## 설정 관리

실험 또 실험 .dogdrip[삽질 또 삽질]하는 우리의 일상

- 논문에서 주어진 옵션과 파라미터대로 일단 돌려본다
- 옵션을 바꿔본다
- 파라미터를 바꿔본다
- 파라미터를 추가하게 된다
- ... Configuration HELL 발생!

잘 관리하지 않으면 벌어지는 일들

- 분명 이 실험이 잘됬는데 이 실험의 셋팅을 모르겠다
- 두 실험을 비교할 때 조작변인을 잘못 체크한다

---

## 설정 관리 tip

- 일단 옵션과 파라미터를 분리해서 생각합니다
    - 옵션: max_epoch, train_dir, queue_size ...
    - 파라미터: batch_size, learning_rate, layer_size ...
- 파라미터는 모델 파라미터와 학습 파라미터로 나눌 수 있습니다
    - 학습 파라미터: batch_size, learning_rate, ...
    - 모델 파라미터: layer_size, image_size, ...
- FLAGS 로 다 해결이 되나요? ... 쉽지 않습니다.
    - 구글도 FLAGS를 버리고 있어요. 재사용이 안되고 ipdb가 안됩니다.
- 위를 염두해두고 커맨드라인을 제공합니다
- 현재 돌고 있는 실험의 설정을 재사용가능하게끔 덤프해둡니다

