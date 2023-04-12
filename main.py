import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 불러오기
data = pd.read_csv("train.csv")

# 상관 관계 분석을 위한 heatmap 출력
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.2)
plt.show()
# 상관 관계 분석 정보 -> 운동시간, 온도, BPM, 칼로리 소모량 / 키, 몸무게

# 변수 선택
X = data[["Exercise_Duration", "BPM", "Weight(lb)"]] # 독립 변수
y = data["Calories_Burned"] # 종속 변수

# 데이터 전처리
X = X.fillna(X.mean()) # 결측치 처리
scaler = StandardScaler()
X = scaler.fit_transform(X) # 스케일링

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("MSE: ", mse)
print("RMSE: ", rmse)
print("R-squared: ", r2)

