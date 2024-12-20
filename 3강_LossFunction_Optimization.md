Linear Classifier 모델을 사용하게 되면 최종적으로 해당 클래스의 점수를 얻을 수 있습니다. 하지만 이때 각 점수들은 아무런 기준이 없이 나오게됩니다. 그래서 해당 값이 높은건지, 아니면 낮은건지 판단할수가 없습니다. 그래서 모든 점수들에 대해서 만일 점수가 ∞에 가깝다면 1에 가까운 수를 보이고, 점수가 -∞에 가깝다면 0에 가까운수를 보이고, score가 0에 가깝다면 0.5에 가까운 수를 내보내서 확률적으로 바라보고싶었습니다. 그래서 모든 값을 다음과 같이 매핑해주는 함수인 sigmoid 함수를 사용하게 됩니다.

$$
\sigma(x) = {1 \over 1 + e^{-s}}
$$

만일 binary classification인 경우를 예로 들어보겠습니다. 
우리가 확률이라고 부를수 있는 2가지 조건은 아래와 같습니다.

1. 모든 값이 0과1 사이에 존재해야한다.
2. 모든 값의 합이 1이어야한다.

그래서 각 클래스에 대한 값을 $s_1, s_2$로 정의해보겠습니다. 그리고 $s = s_1 - s_2$로 정의해서 만일 s의 값이 크다면 class 1에 속할 확률이 높고, s = 0 이면 구분할수 없고, s < 0 이면 class2에 속할 확률이 높다는 것으로 해석할수 있습니다.

그래서 우리는 클래스에 대한 확률 분포를 아래와 같이 표현할수 있습니다.

$$
p(c_1) = {1 \over 1 + e^{-(s_1 - s_2)}}
$$

$$
p(c_2) = {1 \over 1 + e^{-(s_2 - s_1)}}
$$

위의 식을 조금더 쉽게 표현하기 위해서 $p(c_1)$  분포에는 $e^{s_1}$ 을  $p(c_2)$  에는 $e^{s_2}$ 을 분모, 분자에 곱해서 표현해보면 아래와 같이 수식이 바뀌는것을 확인할 수 있습니다.

$$
p(c_1) = {e^{s_1}\over e^{s_1} + e^{s_2}}
$$

$$
p(c_2) = {e^{s_2}\over e^{s_1} + e^{s_2}}
$$

해당 식을 통해서 우리는 모든 값이 0과1 사이에 존재하며, 모든 값의 합이 1임을 확인하였습니다. 그래서 우리는 이를 확률의 관점에서 생각해 볼 수 있습니다. 그렇다고 해서 전체 분포에서 class 1이 나올 확률을 의미하는지는 알순 없습니다. 단순히 값들을 확률로 생각해서 해당 값이 나올 확률 정도로 해석해야됩니다.

그리고 만일 binary classification이 아니라 multi classification인 경우 위의 식을 확장해서 아래와 같이 표현할수 있고, 이를 softmax라고 합니다.

$$
p(c_n) = {e^{s_t} \over e^{s_1} + e^{s_2} + e^{s_3} + ... + e^{s_n}}
$$

그렇다면 우리가 이제 가중치를 우리가 원하는 결과로 만들기 위해서 학습을 진행해야 됩니다.
우리가 학습을 하기 위해서는 2가지 요소가 필요합니다.

1. Loss function : $\hat{y}$ 와 $y$ 의 차이를 수치화 해주는 함수.
2. Optimization : Loss function을 이용해서 실제로 가중치를 업데이트 해주는 과정

Loss function에 대해서 Binary classification 관점에서 2가지 loss에 대해 알아보도록 하겠습니다.

## Discrimination setting인 경우

discrimination setting의 경우 binary classification이고, Label = { -1, + 1 } 인 가정입니다. 그래서 홰당 경우의 경우 간단하게 margin based loss를 사용하게 됩니다. 

margin based loss는 $y\hat{y}$ 을 사용해서 loss function을 정의하게 됩니다.

정답인 경우 : $y\hat{y} > 0$ 

오답인 경우 : $y\hat{y} < 0$

$y\hat{y}$ 의 값이 크고 정답인 경우 → 높은 확률로 정답이라고 에측함  → 학습이 굉장히 잘됨을 의미.

$y\hat{y}$의 값이 크고 오답인 경우 → 높은 확률로 정답이라고 예측했지만 오답 → 학습이 많이 잘못됨을 의미.

이러한 경우 아래와 같은 함수들을 사용할 수 있습니다.


<br>
![image](https://github.com/user-attachments/assets/92b1539f-862a-42db-9d78-3d057f4ff7b8)

<br>


다음과 같이 나타낼수 있ㅇ습니다.

- 0/1 loss : 정답/오답을 단순히 비교해서 맞는 경우 loss = 0을 만들어 줍니다.
- Log loss : $log(1 + e^{-y\hat{y}})$ → Logistic regression에서 주로 사용합니다. 값이 0과 1사이에 매핑되어 유용하게 사용됩니다.
- Exponential loss : $e^{-y\hat{y}}$ → Label 자체가 틀린 경우 잘못 학습되기에 주의해서 사용해야됨.
- Hinge Loss : $max(0, 1 - y\hat{y})$ → gradient를 구하기 유용함 ( SVM 에서 사용 )

하지만 Discrimination setting의 경우 클래수의 수를 늘리는데 한계가 존재합니다.

## Probabilistic Setting

해당 경우 cross entropy loss를 주로 사용하게 됩니다.

$$
L = -{1\over N}\sum^N_i \sum^K_j y_{ik}log(\hat{y_{ik}})
$$

위와 같이 표현할 수 있습니다.

여기서  $y = [0, 0, 1, 0 ,0]$ 다음과 같이 표현되고, $\hat{y} = [0.1, 0.2, 0.5, 0.2, 0.1]$ 처럼 표현되어 정답인 라벨에 대해서만 loss 값을 구하게 됩니다. 이렇게 정답인 라벨에 대해서만 Loss 값을 구하는 이유는 굳이 다르게 예측한 클래스의 loss 값을 사용해서 update를 진행하지 않기에 오로지 정답인 Loss만을 사용하도록 합니다.

그리고 -  값을 사용해서 모든 가중합의 음수를 구하게 됩니다. 이는 사실 log(x) 함수가 아니라 -log(x) 함수에서 나오게 된 값입니다.

![image](https://github.com/user-attachments/assets/b250ae6b-7d0e-4278-80d5-56c826176c43)

<br>

다음과 같이 우리가 구한 값들은 결국 확률 값으로 나오게 되고, 만일 1과 가까운 값이 나오게 되면 loss 값을 0에 가깝게 설정해야됩니다. 그리고 확률 값이 0에 가깝다면 loss값을 ∞에 가까운 값을 주기 위해서 $-log(x)$함수를 사용해서 표현하게 됩니다.

## Optimization

최적화란, 우리가 loss function이 주어진 경우 해당 값을 최소로 만드는 가중치를 찾아가는 과정입니다.

<br>

![image](https://github.com/user-attachments/assets/ea0652b5-8655-49e6-9a91-7b5ad4dbc655)


<br>

위와 같이 loss function이 2차 함수인 경우 해당 값을 최소화 하기 위해서는 gradient의 방향과 반대 방향으로 이동해야 합니다. 그래서 optimization을 수식으로 정해보면 간단하게 아래와 같이 표현해볼 수 있습니다.

$$
\theta_{new} = \theta_{old} - \eta{\delta L \over \delta \theta_{old}}
$$

즉, 기울기의 반대 방향으로 $\eta$ 만큼 이동하도록 설계하게 됩니다.

이렇게 모든 데이터에 대해서 모든 gradient를 계산하게 되면 많은 양의 메모리를 필요로 합니다. 그래서 Batch 단위로 데이터를 나눠서 최적화 하게 됩니다. 이를 SGD라고 합니다. Batch의 크기가 너무 작으면 너무 적은 데이터만 보고 가중치를 업데이트 하기에 불안정하게 학습이 진행되고, Batch의 크기가 너무 크게 되면 연산량 대비 얻을 수 있는 정확도 향상이 줄어드는 diminishing return이 발생합니다. 그래서 배치 사이즈를 잘 조절하는것도 중요한 요소입니다.

그리고 학습은 gradient들이 0에 가까워질때 까지 반복하게 됩니다. 그리고 overfitting이나 하이퍼 파라미터를 조정하기 위해서 cross validation 방법을 사용하게 됩니다.

반드시 test 데이터는 마지막 추론에만 사용을 하게 되며, 우리는 test데이터 대신 ovefitting이나 하이퍼 파라미터를 조정하기 위해서 사용하는 validation을 추가하고, validation과 test 데이터는 IID 하다는 가정하에 사용되게 됩니다.

