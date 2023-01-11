import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

dir = 'C:/Users/15/Desktop/DataSet/'

sns.set("talk","darkgrid",font_scale=1,font="sans-serif",color_codes=True)
df = pd.read_csv(dir+'[Dataset]_Module11_(Viral).csv')
print(df.head())
print(df.columns);print(df.shape)
print(df.isnull().sum());print(df.describe())

# URL 열은 문자열만 있고 클러스터링에 도움이 되지 않아 삭제 합니다.
X = df.drop('url',axis=1)
print(X.columns)
features = []
for col in df.columns:
    features.append(col)

df[features].std().plot(kind='bar', figsize=(10,6), title="Features Standard Deviation")
plt.show()

#TASK 1: 데이터 세트 X에서 timedelta 열을 삭제하고 결과를 X에 저장하기
X=X.drop(' timedelta', axis= 1)
print(X.columns)

feat = []
for value in X.iloc[:,0:20].columns:
    feat.append(value)

print(feat)
cm = np.corrcoef(X[feat].values.T)
print(cm)
sns.set(font_scale = 1.0)
fig = plt.figure(figsize=(10, 8))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=feat, xticklabels=feat)
plt.title('Features Correlation Heatmap')
plt.show()


# K-Means 알고리즘은 KMeans 함수로 호출할 수 있습니다. 클러스터 수를 인수로 전달합니다.
kmeans = KMeans(n_clusters=4)
# kmeans.fit 함수로 kmeans 모델을 피팅합니다.
kmeans_output = kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()

#Task 2: X의 처음 10개 열을 선택하고 결과를 변수 X2에 저장하기
X2 = X.iloc[:,:10]
print(X2.columns)
print(X2.shape)
# 이제 클러스터가 6개인 X2에 K-Means 알고리즘을 적용해 보겠습니다.
kmeans = KMeans(n_clusters=6)
kmeans_output = kmeans.fit(X2)
y_kmeans = kmeans.predict(X2)
plt.scatter(X2.iloc[:, 0], X2.iloc[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()

# X의 하위 집합 데이터 세트 X3를 사용하겠습니다.
X3 = X.iloc[:, 5:13]
# 이제 클러스터가 6개인 X2에 K-Means 알고리즘을 적용해 보겠습니다.
kmeans = KMeans(n_clusters=6)
kmeans_output = kmeans.fit(X2)
y_kmeans = kmeans.predict(X2)
plt.scatter(X2.iloc[:, 0], X2.iloc[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()