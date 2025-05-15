# 3강

Linear Regression은 데이터들이 선형적인 관계를 가지고 있음을 가정하고 모델링한 모델이다.
특히 Linear Regression 모델의 목표가 왜 RSS, 즉 에러 제곱을 최소화 하는 식으로 유도되는지를, IID 가정과 MLE를 통해서 증명한다. 
최종적으로 Linear Regression 모델은 데이터의 노이즈가 IID + Gaussian 분포를 따른다는 가정하에 MLE를 풀게 되면 오차제곱의 평균을 최소화해야한다는 식을 얻게 된다. ( 데이터가 행렬인 경우 OLS 추정 방법과 동일하다 )

<br>

# 4강 

Linear Regression의 입력으로 카테고리컬 데이터가 들어오는 경우 이를 N - 1개의 베타로 변경해서 사용하거나 one-hot-encoding 처럼 클래스 개수 만큼 [0,1]로 만들어서 처리하는 방법이 있다. 
그리고 각 변수들끼리의 시너지 효과를 고려하여 더욱 복잡한 모델을 만들수 있고, 불필요한 모델 파라미터의 경우 다양한 종류의 Feature Selection 방법을 통해서 최적화 할수 있다.

<br>

# 5강 

y가 카테고리컬 형태인 classification 문제를 풀기 위해서는 최종 출력이 카테고리를 나타내는 인덱스로 매핑되어야한다. 이를 Linear Regression 모델에서 출력값을 확률로 매핑시켜주는 식으로 모델링한게 Logistic Regression이다. 그러기에 Logistic Regressio은 log odds가 선형적이라는 가정을 기반으로 모델링 되어있다. 그리고 MLE가 closed-form이 아니기에 gradient Descent 방법을 통해서 $\beta$ 를 최적화 해야한다.

<br>
# 6강 

Discriminative 모델과 Generative 모델을 비교한다.
Discriminative 모델의 경우 p(y|x)를 특수한 가정을 통해서 직접적으로 모델링을 한다. logistic Regression에서 log odds가 Linear 하다는 가정을 한것 처럼. <br> 하지만 Generative model의 경우 p(x|y)p(y) 라는 베이즈 정리를 통해서 간접적으로 p(y|x)를 모델링 하는 방법입니다. 그리고 그 때 모델링하는 p() 함수를 가우시안 함수로 가정해서 모델링하는 것을 LDA와 QDA 로 구분할수 있습니다.