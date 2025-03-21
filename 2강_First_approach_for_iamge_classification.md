machine learning가 나오게 된 배경은 고전적인 방법을 사용하는 경우 사람의 개입이 많기에 이를 최소화 하는 Data driven 방식을 사용하도록 설계되었습니다.

## Nearest Neighbor

우리가 기본적으로 이미지 데이터가 비슷한지, 비슷하지 않은지 확인하는 가장 근본적인 방법은 두 이미지가 얼마나 다른지 픽셀단위로 비교하는 방식입입니다.

Nearest Neighbor 방식은 단순히 test이미지와 가장 비슷한 Train 이미지를 찾고, 해당 라벨로 test 이미지를 에측하는 방법입니다. 그래서 train시 특별히 모델을 학습을 진행하지 않아 $O(1)$ 의 복자도를 갖습니다. 그리고 inference시 모든 학습 데이터들과 test 데이터를 픽셀단위에서 비교해서 유사도를 구해야하기에 $O(N)$ 의 복잡도를 갖게 됩니다. 

이렇게 설계된 모델이 과연 좋을까요????

→ 만일 100장의 이미지에 대해서 예측을 하고 싶은경우 모든 이미지와 비교를 100번 해야되기 때문에 연산량이 너무나도 많게 됩니다. 그래서 해당 방법은 너무 느리다는 단점이 존재합니다.

그래서 픽셀단위에서 이미지를 비교하기에, 만일 모든 이미지를 1픽셀씩 오른쪽으로 이동시키게 된다면 두 이미지는 유사도가 떨어지게 됩니다. 하지만 인간의 의미적 관점에서 보면 두 이미지는 결국 똑같은 이미지입니다. 그래서 Nearest Neighbor 는 아래와 같은 2가지 치명적인 문제점이 존재합니다.

1. 모든 Test 데이터에 대해 모든 픽셀을 비교하기에 추론이 너무나 느리다.
2. 인간이 이해하는 의미와는 전혀 다른 방식으로 이미지를 비교한다.

## Linear Classifier

그래서 추론시 높은 복잡도를 갖기 보다, 학습시 더 많은 복잡도를 갖게만들어서 한번의 학습으로 인해 추론을 빠르게 하는 모델을 만들어야 합니다. 그리고 이러한 접근 방식이 **Parametric Approach** 입니다.

이미지의 모든 픽셀들을 비교하기 보다 특정 라벨의 이미지의 픽셀들의 패턴을 학습하고자 하는 아이디어 입니다.

그래서 만일 이미지의 크기가 N x M 크기라면 이와 동일한 크기의 가중치 행렬이 필요합니다.
그래서 N X M 크기의 행렬을 point wise하게 곱한후 해당 값들을 모두 더해줍니다. 즉 ,이미지와 가중치의 가중합을 구해줍니다. 이를 수식으로 표현하면 아래와 같이 표현할 수 있습니다.

$$
y_1= \sum_{i=1}^{n*m}image_i W_i
$$

위의 식에서는 간단하게 표현하기 위해서 2차원 이미지를 flatten하여 표현했습니다. 아무튼 이렇게 해서 $y_1$ , 즉 1번 클래스의 패턴이 들어간 $W_1$을 학습하게 됩니다.
만일 클래스가 C개라면 우리는 총 C개의 W를 갖게 됩니다. 그리고 최종적으로 W 가중치의 크기는 결국 (C, N*M) 이될것입니다.

그리고 항상 bais가 붙어야됩니다. bais는 각 데이터와 관련없는 기본적인 값을 의미합니다. 예를들어 설명하면 시험 점수가 주로 3번이 많다고 가정한다면, 모르는 문제가 나오는 경우 3번으로 찍게됩니다. basis가 바로 이러한 역할을 해줍니다. 간단하게 데이터의 전체적인 분포를 보조해주는 역할일 진행합니다.
그래서 위에서 배운 Linear Classifier를 표현해보면 아래와 같습니다.

$F(W,x) = Wx + b$ 그리고 이를 조금더 편하기 표현하기 위해서 $F(W,x) = [W, b] \begin{bmatrix} x \\ 1 \end{bmatrix}$ 다음과 같이 표현 방식을 바꿀수 있습니다. 즉 여기서 $[W, b]$ 를 W로 표현을 하고 $\begin{bmatrix} x \\ 1 \end{bmatrix}$  을 X로 표현하게 되면 최종적으로 $F(W,x) = Wx$ 로 표현이 가능하게 됩니다. 그래서 주로 Linear Classifier 를 $F(W,x) = Wx$로 표현하지만 언제나 bais term이 존재함을 기억해야됩니다.

그리고 이렇게 만들어진 W 가중치들에 대해서 우리가 0 ~ 255 값을 갖도록 픽셀들을 정규화해주고 시각화를 해보면 특정 패턴을 갖는 모양을 볼수 있습니다. 하지만 만일 다양한 패턴을 갖는 이미지가 들어오는 경우 우리가 이해할수 없이 복잡한 패턴을 보입니다. 즉 하나의 가중치로 이미지의 패턴을 표현하지 못하는 경우입니다.
<br>

![image](https://github.com/user-attachments/assets/183063b0-bc54-4441-8570-8984ea9bc7bf)
