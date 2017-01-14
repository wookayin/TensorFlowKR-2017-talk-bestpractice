name: monitoring
template: recipe

# Best Practice 7

---

layout: false

## 옵저버 패턴으로 모니터링 객체를 만들자

.TODO[공부를 하고 다시 돌아오자]

```python
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
                         test_set.data,
                         test_set.target,
                         every_n_steps=50)

classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000,
               monitors=[validation_monitor])
```
