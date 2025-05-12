# 3강

Linear Regression은 데이터들이 선형적인 관계를 가지고 있음을 가정하고 모델링한 모델이다.

특히 Linear Regression 모델의 목표가 왜 RSS, 즉 에러 제곱을 최소화 하는 식으로 유도되는지를, IID 가정과 MLE를 통해서 증명한다. 

최종적으로 Linear Regression 모델은 데이터의 노이즈가 IID + Gaussian 분포를 따른다는 가정하에 MLE를 풀게 되면 오차제곱의 평균을 최소화해야한다는 식을 얻게 된다. ( 데이터가 행렬인 경우 OLS 추정 방법과 동일하다 )