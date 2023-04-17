import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

#다중회귀
# 데이터 불러오기
train_data = pd.read_csv("train.csv")

# 상관 관계 분석을 위한 heatmap 출력
# 상관 관계 분석 정보 -> 운동시간, 온도, BPM, 칼로리 소모량 / 키, 몸무게
sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm', linewidths=0.2)
plt.show()
# 변수 선택
input = train_data[["Exercise_Duration", "Body_Temperature(F)", "BPM"]].to_numpy() # 독립 변수
target = train_data["Calories_Burned"].to_numpy() # 종속 변수

# input, target 상관관계 그래프
plt.plot(input[:, 0]* 10000)
plt.plot(input[:, 1]* 10000)
plt.plot(input[:, 2]* 10000)
plt.plot(target* 10000)
plt.xlim([0, 100])
plt.legend(("Exercise_Duration",  "Body_Temperature(F)","BPM","Calories_Burned"),loc='upper right')
plt.show()

poly = PolynomialFeatures(degree=7, include_bias=False)
poly.fit(input)
train_poly = poly.transform(input)
print(np.shape(train_poly))

# 데이터 분할
input_train, input_test, target_train, target_test = train_test_split(
    train_poly, target, test_size=0.2)
# , random_state=42
# 스케일링
ss = StandardScaler()
ss.fit(input_train)

# 모델 훈련
lr = LinearRegression()
lr.fit(input_train, target_train)
# 정확도
print("다중회귀")
print(lr.score(input_train, target_train))
print(lr.score(input_test, target_test))

# Ridge
print("-------Ridge ----")
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(input_train, target_train)
#y_pred = ridge.predict(input_test)
print(ridge.score(input_train, target_train))
print(ridge.score(input_test, target_test))

import matplotlib.pyplot as plt

train_score = []
test_score =[]
alpha_list = [ 0.001, 0.01, 0.1, 1, 10, 100]

for alpha in alpha_list:
    ridge = Ridge(alpha=alpha)
    ridge.fit(input_train, target_train)
    train_score.append(ridge.score(input_train, target_train))
    test_score.append(ridge.score(input_test, target_test))

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

ridge = Ridge(alpha= 0.1)
ridge.fit(input_train, target_train)

print(ridge.score(input_train, target_train))
print(ridge.score(input_test, target_test))


# ## 적절한 규제 강도 찾기
# alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
# for alpha in alpha_list:
#     ridge = Ridge(alpha=alpha)
#     ridge.fit(X_train_scaled, target_train)
#     train_score.append(ridge.score(X_train_scaled, target_train))
#     test_score.append(ridge.score(X_test_scaled, target_test))
# # plt.plot(np.log10(alpha_list), train_score)
# # plt.plot(np.log10(alpha_list), test_score)
# # plt.xlabel('alpha')
# # plt.ylabel('R^2')
# # plt.show()
#
# ridge = Ridge(alpha=0.1)
# ridge.fit(X_train_scaled, target_train)
# print("Ridge")
# print(ridge.score(X_train_scaled, target_train))
# print(ridge.score(X_test_scaled, target_test))
#
# # ## Lasso
#
# # X_test = pipeline.transform(ss)
# # Y_pred = lr.predict(X_test)
