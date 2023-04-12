import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#다중회귀
# 데이터 불러오기
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# 상관 관계 분석을 위한 heatmap 출력
sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm', linewidths=0.2)
plt.show()

# 상관 관계 분석 정보 -> 운동시간, 온도, BPM, 칼로리 소모량 / 키, 몸무게

# 변수 선택
input = train_data[["Exercise_Duration", "Body_Temperature(F)", "BPM"]].to_numpy() # 독립 변수
target = train_data["Calories_Burned"].to_numpy() # 종속 변수
X_test = test_data[["Exercise_Duration", "Body_Temperature(F)", "BPM"]].to_numpy()
# plt.plot(input[:, 0]* 10000)
# plt.plot(input[:, 1])
# plt.plot(input[:, 2]* 10000)
# plt.plot(target)
# plt.show()

# # 데이터 전처리
# X = X.fillna(X.mean()) # 결측치 처리
# scaler = StandardScaler()
# X = scaler.fit_transform(X) # 스케일링

poly = PolynomialFeatures(degree=7, include_bias=False)
poly.fit(input)
train_poly = poly.transform(input)
print(np.shape(train_poly))


tr_in, ts_in, tr_out, ts_out = train_test_split(
    train_poly, target, test_size=0.2, random_state=42)
# # 데이터 분할
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 정확도
lr=LinearRegression()
lr.fit(tr_in, tr_out)
print(lr.score(tr_in, tr_out))
print(lr.score(ts_in, ts_out))


# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # 모델 평가
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# rmse = mse ** 0.5
# r2 = r2_score(y_test, y_pred)
#
# print("MSE: ", mse)
# print("RMSE: ", rmse)
# print("R-squared: ", r2)

