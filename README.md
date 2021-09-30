# DGU_Independent_Capstone_Design
2020_03~2020_04

[ 연구 주제 ]
 - 차세대 통신 시스템을 위한 심층 강화 학습
 - 'Tensor Flow'를 통한 머신러닝 기초 학습

 [ 연구 내용 ]
 나는 머신러닝에 대해 기본적인 이해를 하고, 머신러닝의 핵심 알고리즘인 'Linear Regression'과 'Logistic Regression'에 대해서 연구했다. 연구한 내용을 정리하고, 주된 Keyword들을 분석해 보겠다.
 
- Machine Learning 핵심 알고리즘
 * Linear Regression
 * Logistic Regression

- Deep Learning 핵심 알고리즘
 * Neural Networks
 * Convolutional Neural Networks
 * Recurrent Neural Networks


 - 머신러닝이란 일종의 프로그램이다. 스스로 데이터를 보고 '학습'해서 배우는 능력을 갖는 프로그램이다. 여기서 '학습'의 종류는 2가지가 있다. 'Supervised Learning'과 'Unsupervised Learning'이 있다. 

 - Supervised Learning
 라벨이 정해진 데이터를 갖고 학습하는 것. 
 ex) 이미지를 보고 개,고양이 구분 / 알파고 ...

 - UnSupervised Learning
 라벨이 정해지지 않은 데이터를 갖고 학습하는 것. 프로그램이 스스로 라벨링을 한다. 
 ex) 구글 뉴스 그룹핑 / 비슷한 단어들 그룹핑 ...

 - Training Data Set: supervied learning에서 '라벨이 정해진 데이터'

 - Supervied Machine Learning의 순서
  1. Training-Testing-Validation Data Set을 나눈다.
  2. Training을 시킨다. (이 때 Model이 생긴다.)
  3. Model을 통해 Test를 한다.


 - Supervised Learning의 3종류
  * Regression	ex)시험점수
  * Binary Classification	  ex)Pass/Fail 분류
  * Multi-Label Classification  ex)학점(A,B,C,D,F) 분류

[ Modeling의 핵심 ]
 - Hypothesis
 H(x) = wx + b

 - Loss Function
 어떤 hypothesis가 가장 적절한지 결정한다.(+수식참고)

 - Minimize Loss
 Loss값을 최소화 해주는 역활을 한다. 여기선 'Gradient Descent Algorithm'을 활용한다. Gradient descent algorithm을 하는 방법은 우선, 아무지점에서 시작한다. 그 후 loss fucntion의 w,b값을 조금씩 조정 해준다. 조정된 값에서 나온 loss와 그 전 loss의 경사도를 분석한다(미분). 이 과정을 반복해 준다면 최저점에 도달 할 수 있다.
 TensorFlow에서는 Gradient Descent Optimizer를 통해 minimize loss가 가능하다.

  * Convex Function
 이는 Minimize Loss시에 최저점이 유일 한 것을 말한다. 만약, 최저점이 여러가지라면 한 머신러닝 모델에서 여러가지 값이 나오게 된다. 따라서, 최저점을 하나 갖는 것은 매우 중요하다. Loss Function은 꼭 Convex Function이어야 한다. 

[ Model 만드는 순서 (TF기준) ]
 1. training data set을 불러온다.
 2. model의 틀을 만들어 준다.
 3. Hypothesis, Loss Function, Minimize를 해준다.
 4. Model을 training해준다.
 5. Model이 완성된다.


 [ 머신러닝 할 때 Tip! ]
 - Learning Rate 조절하는 방법
 Learning Rate이란 (Minimize Loss)시에 step size를 결정하는 값이다. 이 값이 너무 크면 loss값이 범위를 넘어갈 수 있다. 반대로 너무 작으면 최저점까지 내려가지지를 않는다. 따라서 적당한 값을 잘 설정해야 한다. (처음에는 0.01로 설정해 보고 값의 추이를 분석해서 값을 조정하라!) (+그림참고)

 - Data Preprocessing(데이터전처리)
 Traing data set의 값들간에 차이가 너무 크면, 분석이 잘 안될 수 있다. 이때 'Normalization'을 해주면 해결이 가능하다. Standardization은 대표적인 nomalization 방법이다. 평균,분산,확률을 통해 수식으로 나타낼 수 있으며, 파이썬 에서는 한 줄의 코드로 구현가능하다. (+수식참고)

 - Overfitting
 Train data에는 딱 맞지만, 실제 data, test data에는 맞지 않는 경우를 말한다. 이를 해결하는 방안은 3가지가 있다. 첫째, Training data를 많이 갖는다. 둘째, feature의 갯수를 줄인다. 중복된 feature를 제거해 준다. 셋째, 'Regularization'을 해준다. Regularization이란 'Minimize Loss'시에 수식에 틈(regularization strength)을 추가시켜 주는 것이다. 이 또한 파이썬에서는 한줄의 코드로 구현가능하다. (+수식참고)

 - Original Data Set을 나누는 방법
 {Training(&Validation) | Testing} 으로 나누어 준다. 이 때, validation data set이란 regulation을 해주기 위한 data set이다. 이 3가지를 적절히 나누어 주는 것이 중요하다. 



 [ 연구 실습 ]

[Tool]
 - TensorFlow 2.0
 - Python 3.7
 - Jupyter Notebook

[Story]
 - 주제: 미세먼지로 인한 소비/경제/행동변화에 따른 사회적 영향 분석.
 - 예측: 1. 오전8시의 미세먼지양이 많을 시 그날 하루 (헬스&뷰티 분야의 	  매출)은 평균보다 높을 것이다.
 	2. 오전8시의 미세먼지양이 많을 시 그날 하루 총 매출은 평균매출보	  다 많은 매출을 올릴까? 
 - 전략: 서울시 노원구 상계10동을 거점으로, 이 지역의 1년동안 미세먼지 	측정량과, 이 지역의 매출 추이를 비교해 본다.

[사용한 데이터]
  빅콘테스트에서 제공받은 파일 중 총2개의 엑셀파일 이용.
	1. 미세먼지 -> 환경기상데이터 -> 노원구 -> (csv파일)V10O1610293
	2. 미세먼지 -> 유통데이터 -> (xlsx파일)GS리테일_동별매출지수용기준값확인AMT_NEW -> (엑셀파일 내 6번째 sheet)종합테이블

[모델 설명]
  1. 218개의 훈련 샘플, 55개의 검증 샘플, 69개의 테스트 샘플을 제작한다.
  2. 판다스 데이터 프레임으로 부터 tf.data 데이터 셋을 만들기 위한 함수를 제작한다.
  3. 작은batch크기를 설정해 미리 시연해 본다.
  4. 사용할 열을 선택한다.
  5. 특성 층을 만든다.
  6. 모델을 생성, 컴파일, 훈련한다.

  모델의 예상 성능 -> 70%
  모델의 실제 성능 -> 53%
  후기: 정제된 데이터의 잘못인지, 훈련 시 문제였는지, 생각보다 모델의 성능이 낮았다.
