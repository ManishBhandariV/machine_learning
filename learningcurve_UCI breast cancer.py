import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

data = pd.read_csv("wdbc.data", header= None)
#print(data.head())
y = data.iloc[:,1]
#print(y.head())

data = data.drop(columns= 1, )
data = data.drop(columns=0)
data = minmax_scale(data)
#print(data)

#print(data.head())

x_train, x_test, y_train, y_test = train_test_split(data, y, test_size= 1/3 , stratify= y)

#using for loop NB
acc_NB = []
for i in range(5):
 x_train, x_test, y_train, y_test = train_test_split(data, y, test_size= 1/3 , stratify= y)

 #error = []
 accuracy = []
 test_set_size = [0.8,0.6,0.4,0.2]  #to get diffierent proportion of training set in the loop
 for i in test_set_size:
     x_train_r, x_valid, y_train_r, y_valid = train_test_split(x_train, y_train, test_size= i, random_state=42, stratify= y_train)
     #print(np.shape(x_train_r))
     model = MultinomialNB(fit_prior= True)
     model.fit(x_train_r,y_train_r)
     lbl = model.predict(x_test)
     #error.append(np.mean(lbl != y_test))
     accuracy.append(accuracy_score(y_test, lbl))
     x_train , y_train = x_train, y_train
 model = MultinomialNB(fit_prior= True)
 model.fit(x_train, y_train)
 lbl = model.predict(x_test)
 #error.append(np.mean(lbl!= y_test))
 accuracy.append(accuracy_score(y_test, lbl))
 acc_NB.append(accuracy)
 #print(np.shape(x_train))
 train_size = [0.2,0.4,0.6,0.8,1]
 #print('train size = {}'.format(train_size))
 #print(error)
 #print(accuracy)

train_size = [0.2,0.4,0.6,0.8,1]
#print('train size = {}'.format(train_size))
#print(np.mean(acc_NB, axis= 0))
meanacc_NB = np.mean(acc_NB, axis= 0)

#using Lgistic Regression

acc_LG = []
for i in range(5):
 x_train, x_test, y_train, y_test = train_test_split(data, y, test_size= 1/3 , stratify= y)

 #error = []
 accuracy = []
 test_set_size = [0.8,0.6,0.4,0.2]  #to get diffierent proportion of training set in the loop
 for i in test_set_size:
     x_train_r, x_valid, y_train_r, y_valid = train_test_split(x_train, y_train, test_size= i, random_state=42, stratify= y_train)
     #print(np.shape(x_train_r))
     model = MultinomialNB(fit_prior= True)
     model.fit(x_train_r,y_train_r)
     lbl = model.predict(x_test)
     #error.append(np.mean(lbl != y_test))
     accuracy.append(accuracy_score(y_test, lbl))
     x_train , y_train = x_train, y_train
 model = MultinomialNB(fit_prior= True)
 model.fit(x_train, y_train)
 lbl = model.predict(x_test)
 #error.append(np.mean(lbl!= y_test))
 accuracy.append(accuracy_score(y_test, lbl))
 acc_LG.append(accuracy)
 #print(np.shape(x_train))
 train_size = [0.2,0.4,0.6,0.8,1]
 #print('train size = {}'.format(train_size))
 #print(error)
 #print(accuracy)

train_size = [0.2,0.4,0.6,0.8,1]
# print('train size = {}'.format(train_size))
# print(np.mean(acc_LG, axis= 0))
meanacc_LG = np.mean(acc_LG, axis= 0)


#plots
train_size = [0.2,0.4,0.6,0.8,1]
fig = plt.figure()
ax1 = fig.add_subplot(111)
#fig(figsize=(12, 6))
ax1.plot(train_size, meanacc_NB, color='red', linestyle='dashed', marker='o',
          markerfacecolor='blue', markersize=10)
ax1.plot(train_size, meanacc_LG, color='blue', linestyle='dashed', marker='s',
          markerfacecolor='blue', markersize=10)
plt.title('Test Accuracy v/s Training Data Size')
plt.xlabel('Training Data Size')
plt.ylabel('Test Accuracy')
plt.show()
# plt.figure(figsize=(12, 6))
#
# plt.title('Test Accuracy v/s Training Data Size')
# plt.xlabel('Training Data Size')
# plt.ylabel('Test Accuracy')
# plt.show()
#
# # #plotting
# # fig = plt.figure()
# # ax1 = fig.add_subplot(111)
# #
# # ax1.scatter(x1, x2, s=10, c='b', marker="s", label='first')
# # ax1.scatter(x3,x4, s=10, c='r', marker="o", label='second')
# # plt.show()




# using learning_curve
train_sizes, train_scores, valid_scores = learning_curve(MultinomialNB(),data,
                                                         y,train_sizes=[0.2,0.4,0.6,0.8,1],cv=5)
print(train_sizes)
print(train_scores)
print(valid_scores)






