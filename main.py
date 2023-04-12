import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_csv("train.csv")

# 상관 관계 분석을 위한 heatmap 출력
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.2)
plt.show()
# 어어어어