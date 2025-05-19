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


<br>

# 7강
Train data를 계속학습시 Valid Loss가 올라가는 Overfitting 현상은 Train data만의 디테일한 패턴까지 학습하기에 발생합니다. 이를 (1) train만의 디테일을 학습하기 이전에 valid loss를 보고 학습을 중단하는 "Early Stopping" 방법 (2) 데이터에 맞는 모델의 복잡도를 모델이 스스로 결정할수 있도록 강제로 유도할 수 있는 "Regularizatoin" 으로 완화할수 있습니다. 특히 L1의 경우 Sparse한 모델을, L2의 경우 w를 0이 아닌 값들로 조정하는 과정을 기하학적으로 이해할 수 있습니다.


<br>

# 8강
데이터가 한정되어 있는 경우 해당 데이터에서 IID 조건하에서 복원 추출하여 여러 모델을 학습 시킨 후 결과를 평균 내는 Bagging 방법으로 일반화 성능을 향상 시킬 수 있습니다. 그리고 Bagging의 경우 언제나 Variance가 낮아지기 때문에 많은 모델을 쓰면 쓸수록 성능이 나빠지진 않습니다. 그리고 모델이 잘 예측하지 못하는 데이터에 가중치를 주는 Boosting 방법 또한 좋은 방법입니다. 

추가로 Model을 Tree로 사용하고 특징 $\sqrt{p}$ 개를 랜덤으로 뽑아 여러 모델을 합치는 방법을 Random Forest라고 합니다. 


<br>

# 9강 
SVM은 이전의 모델들과는 다르게 확률값을 사용하지 않고 직접 decision boundary와의 거기를 최대화 하는 방법론입니다. 두 데이터를 분류하는 hyper plane에 수직인 법선 벡터로 부터 가장 가까운 거리가 1이라는 가정을 두고 라그장주 승수법을 통해서 최적화된 값을 찾을 수 있습니다. 그리고 노이즈를 대비하여 마진 내 데이터 존재가 가능한 soft SVM 또한 가능합니다.

그리고 모든 모델들의 LOSS는 결국 모두 동일한 작동을 하며, 결국 틀린 경우 Loss를 많이 주고 맞은 경우 Loss를 줄이는 모두 동일한 흐름으로 모델을 최적화 시키고 있음을 알수 있습니다.