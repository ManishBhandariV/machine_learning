import pandas as pd
import numpy as np
from sklearn.model_selection import  train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


# importing data
data = pd.read_csv("iris.data")

#splitting labels from data
x = data.iloc[:,0:4]
y = data.iloc[:,4]
#class_names = [' Iris-setosa','Iris-versicolor','Iris-virginica']

#splitting into train,test and valid
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.253, random_state=0)
x_train, x_valid, y_train, y_valid = train_test_split(x_train,y_train, test_size=0.330, random_state=0)


# all_acc = cross_val_score(KNeighborsClassifier(n_neighbors=3),x_train, y_train)
# print(all_acc)

#k_nearesr_neighbour classifier
model = make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=5))
model.fit(x_train,y_train)
#lbl_valid = model.predict(x_valid)
lbl_test = model.predict(x_test)
#print(confusion_matrix(y_valid,lbl_valid))
accuracy =1- (np.mean(lbl_test!= y_test))
print(accuracy)
cmt = confusion_matrix(y_test,lbl_test, labels=[' Iris-setosa','Iris-versicolor','Iris-virginica'])
print(cmt)

#5foldcv
score = []
for i in range(1,7,2):
 classifier = KNeighborsClassifier(n_neighbors=i)
 scores = cross_val_score(classifier,x_train,y_train, cv=5,scoring='accuracy')
 print('scores for n={}'.format(i),scores)
 print('mean score for n={}'.format(i),scores.mean())
 mean_score = scores.mean()
 score.append(mean_score)


classifier = KNeighborsClassifier(n_neighbors= 5)
classifier.fit(x_train, y_train)
lbl = classifier.predict(x_test)
print('test accuracy of selected model=',accuracy_score(y_test,lbl))



#plotting error for different k values
k = [1,3,5,7,9]
error = []
for i in k:
    model = make_pipeline(StandardScaler(),KNeighborsClassifier(i))
    model.fit(x_train,y_train)
    lbl = model.predict(x_valid)
    error.append(np.mean(lbl!= y_valid))
print(error)
#
plt.figure(figsize=(12, 6))
plt.plot(range(1,10,2), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

#using grid search to estimate best parameter

grid_params = {'n_neighbors': [1,3,5], 'weights':['uniform', 'distance'],'metric': ['euclidean','manhattan']}

model = GridSearchCV(KNeighborsClassifier(),grid_params,verbose=1,cv= 5, n_jobs=-1)
results = model.fit(x_train,y_train)
print(results.best_score_)
print(results.best_estimator_)
print(results.best_params_)