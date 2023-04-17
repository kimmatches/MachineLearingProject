import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDClassifier
import numpy as np
#K-NN, 로지스틱, SGD

# 데이터 불러오기
train_data = pd.read_csv("train.csv")

input = train_data[["Exercise_Duration", "Body_Temperature(F)", "BPM"]].to_numpy() # 독립 변수
target = train_data["Calories_Burned"].to_numpy()

# 데이터 분할
input_train, input_test, target_train, target_test = train_test_split(
    input, target, test_size=0.2, random_state=42)

ss = StandardScaler()
ss.fit(input_train)
X_train_scaled = ss.transform(input_train)
X_test_scaled = ss.transform(input_test)

#K-NN
kn = KNeighborsClassifier(n_neighbors=10)
kn.fit(X_train_scaled, target_train)

print("K-NN")
print(kn.score(X_train_scaled, target_train)) # 0.3498333333333
print(kn.score(X_test_scaled, target_test)) # 0.0526666666667
proba = kn.predict_proba(X_test_scaled[:5])


## Logistic
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(X_train_scaled, target_train)
print("Logistic")
print(lr.score(X_train_scaled, target_train))
print(lr.score(X_test_scaled, target_test))

proba = lr.predict_proba(X_test_scaled[:5])

## SGD Classifier

sc = SGDClassifier(loss='log', max_iter=10, random_state=42)
sc.fit(X_train_scaled, target_train)
print("SGD Classifier")
print(sc.score(X_train_scaled, target_train))
print(sc.score(X_test_scaled, target_test))
