from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pandas as pd
import numpy as np
from mlxtend.evaluate import paired_ttest_5x2cv

categories = ['alt.atheism','comp.sys.mac.hardware', 'rec.sport.baseball','talk.politics.guns']
data = fetch_20newsgroups(categories=categories).data
labels = fetch_20newsgroups(categories=categories).target
vectorizor = CountVectorizer()
data = vectorizor.fit_transform(data)

#print(fetch_20newsgroups().data[1])

train_data, test_data, train_labels, test_labels = train_test_split(fetch_20newsgroups().data, fetch_20newsgroups().target,
                                                  test_size=1/3, random_state=42, stratify=fetch_20newsgroups().target)

model_1 = make_pipeline(CountVectorizer(lowercase= True, stop_words= 'english'),SelectKBest(score_func= mutual_info_classif, k=10000 ), LogisticRegression())
model_2 = make_pipeline(CountVectorizer(lowercase= True, stop_words= 'english'),SelectKBest(score_func= mutual_info_classif, k=10000), DecisionTreeClassifier())
model_1.fit(train_data, train_labels)
score_1 = model_1.score(test_data, test_labels)
model_2.fit(train_data,train_labels)
score_2 = model_2.score(test_data, test_labels)
#
print('Logistic Regression Accuracy:{}'.format(score_1*100))
print('Decision Tree Accuracy:{}'.format(score_2*100))

t, p = paired_ttest_5x2cv(estimator1=LogisticRegression(),
                          estimator2=DecisionTreeClassifier(),
                          X=data, y=labels,
                          random_seed=1)

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)
