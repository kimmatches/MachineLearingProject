import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

#다중회귀
# 데이터 불러오기
train_data = pd.read_csv("train.csv")
# test_data = pd.read_csv("test.csv")

# 상관 관계 분석을 위한 heatmap 출력
# sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm', linewidths=0.2)
# plt.show()
# 상관 관계 분석 정보 -> 운동시간, 온도, BPM, 칼로리 소모량 / 키, 몸무게

# 변수 선택
# X = train_data[["Exercise_Duration", "Body_Temperature(F)", "BPM"]].to_numpy() # 독립 변수
# y = train_data["Calories_Burned"].to_numpy() # 종속 변수

input = train_data[["Exercise_Duration", "Body_Temperature(F)", "BPM"]] # 독립 변수
target = train_data["Calories_Burned"]

# 데이터 분할
input_train, input_test, target_train, target_test = train_test_split(input, target, test_size=0.2, random_state=42)

ss = StandardScaler()
ss.fit(input_train)
X_train_scaled = ss.transform(input_train)
X_test_scaled = ss.transform(input_test)

## K-NN
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(X_train_scaled, target_train)

# print(kn.score(X_train_scaled, target_train)) # 0.3498333333333
# print(kn.score(X_test_scaled, target_test)) # 0.0526666666667

# print(kn.classes_)

# print(kn.predict(X_test_scaled[:5]))

proba = kn.predict_proba(X_test_scaled[:5])
# print(np.round(proba, decimals=4))

## Logistic
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(X_train_scaled, target_train)

# print(lr.score(X_train_scaled, target_train))
# print(lr.score(X_test_scaled, target_test))

proba = lr.predict_proba(X_test_scaled[:5])
# print(np.round(proba, decimals=3))

# print(lr.coef_.shape, lr.intercept_.shape)

## SGD Classifier
from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss='log', max_iter=10, random_state=42)
sc.fit(X_train_scaled, target_train)

# print(sc.score(X_train_scaled, target_train))
# print(sc.score(X_test_scaled, target_test))

## LinearRegression
poly = PolynomialFeatures(include_bias=False)

poly.fit(input_train)
train_poly = poly.transform(input_train)

print(train_poly.shape)

test_poly = poly.transform(input_test)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_poly, target_train)

print(lr.score(train_poly, target_train))
print(lr.score(test_poly, target_test))

## Ridge
ss.fit(train_poly)

X_train_scaled = ss.transform(train_poly)
X_test_scaled = ss.transform(test_poly)

from sklearn.linear_model import Ridge

ridge = Ridge()
# ridge.fit(X_train_scaled, target_train)

# print(ridge.score(X_train_scaled, target_train))
# print(ridge.score(X_test_scaled, target_test))

train_score = []
test_score = []

## 적절한 규제 강도 찾기
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, target_train)
    train_score.append(ridge.score(X_train_scaled, target_train))
    test_score.append(ridge.score(X_test_scaled, target_test))
# plt.plot(np.log10(alpha_list), train_score)
# plt.plot(np.log10(alpha_list), test_score)
# plt.xlabel('alpha')
# plt.ylabel('R^2')
# plt.show()

ridge = Ridge(alpha=0.1)
ridge.fit(X_train_scaled, target_train)

print(ridge.score(X_train_scaled, target_train))
print(ridge.score(X_test_scaled, target_test))

## Lasso
