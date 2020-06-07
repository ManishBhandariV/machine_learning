import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from matplotlib.colors import ListedColormap
import seaborn as sns



#dataset1
mean_set1 = (0,0)
cov = [[1,0], [0,1]]
mean_set2 = [2.5,0]

x1, x2 =np.random.default_rng().multivariate_normal(mean_set1, cov, 100).T

y1= []
for i in range(100):
    y1.append('class1')

x3, x4 =np.random.default_rng().multivariate_normal(mean_set2, cov, 100).T
y2= []
for j in range(100):
    y2.append('class2')

zipped_dataset_1 = list(zip(x1,x2,y1))
dataset_1 = pd.DataFrame(zipped_dataset_1, columns=['x1','x2','y'])
#print(dataset_1)

zipped_dataset_2 = list(zip(x3,x4,y2))
dataset_2 = pd.DataFrame(zipped_dataset_2, columns=['x1','x2','y'])
#print(dataset_2)

# #plotting
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
#
# ax1.scatter(x1, x2, s=10, c='b', marker="s", label='first')
# ax1.scatter(x3,x4, s=10, c='r', marker="o", label='second')
# plt.show()
#
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(dataset_1.iloc[:,0], dataset_1.iloc[:,1], s=10, c='b', marker="s", label='first')
ax1.scatter(dataset_2.iloc[:,0],dataset_2.iloc[:,1], s=10, c='r', marker="o", label='second')
plt.show()

data = pd.concat([dataset_1,dataset_2],ignore_index= True, keys=['Feature1','Feature2','Class'])
#print(data)
shuffled_data = data.reindex(np.random.permutation(data.index))
#print(shuffled_data)

#plotting different classes
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
#
# color = {'class1': 'red', 'class2': 'blue'}
# ax1.scatter(shuffled_data.iloc[:,0], shuffled_data.iloc[:,1], s=10, c=shuffled_data.iloc[:,2].apply(lambda x:color[x]), marker="s", label='first')
# plt.show()

x = shuffled_data.iloc[:,0:2]
y = shuffled_data.iloc[:,2]


train_data , test_data, train_labels, test_labels = train_test_split(x,y,test_size=1/3,random_state=42, stratify=y)
# print(np.shape(train_data))
#print(np.shape(train_labels))
#print(train_labels)

model = LogisticRegression().fit(train_data,train_labels)
prediction = model.predict(test_data)
accuracy = accuracy_score(test_labels,prediction)
print(accuracy)
print(classification_report(test_labels,prediction,['class1', 'class2']))


#plotting a decision boundary
xx,yy = np.mgrid[-5:5:0.01,-5:5:0.01]
grid = np.c_[xx.ravel(),yy.ravel()]
probs = model.predict_proba(grid)[:,1].reshape(xx.shape)

f,ax = plt.subplots(figsize = (8,6))
ax.contour(xx,yy,probs,levels= [0.5], cmap= 'Greys', vmin= 0, vmax= 0.6)
color = {'class1': 'red', 'class2': 'blue'}
ax.scatter(train_data.iloc[:,0],train_data.iloc[:,1],c=train_labels.iloc[:].apply(lambda x:color[x]),
           s=50, cmap='RdBu', vmin=0.2,vmax=1.2,edgecolor='white',linewidth=1)
ax.set(aspect = 'equal', xlim=(-5,8),xlabel= "$F_1$", ylabel = "$F_2$")
plt.show()









#dataset2

# mean_set1 = (0,0)
# cov = [[1,0], [0,1]]
# mean_set2 = [2.25,0]
#
# x1, x2 =np.random.default_rng().multivariate_normal(mean_set1, cov, 100).T
# x3, x4 =np.random.default_rng().multivariate_normal(mean_set2, cov, 100).T
# y1= []
# for i in range(100):
#     y1.append('class1')
# y2= []
# for i in range(100):
#     y2.append('class2')
# y1 = pd.DataFrame(y1)
# y2 = pd.DataFrame(y2)
#
# #plotting
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
#
# ax1.scatter(x1, x2, s=10, c='b', marker="s", label='first')
# ax1.scatter(x3,x4, s=10, c='r', marker="o", label='second')
# plt.show()



