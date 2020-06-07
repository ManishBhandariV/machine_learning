from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import  SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator


data = fetch_20newsgroups()
#print(data.target_names)

categories = ['alt.atheism','comp.sys.mac.hardware', 'rec.sport.baseball','talk.politics.guns']

train = fetch_20newsgroups(subset='train',categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

#vectorizer = TfidfVectorizer()
#vectors = vectorizer.fit_transform(train.data)
# # print(vectorizer.get_feature_names())
# # print(newsgroup_train.filenames.shape)
#print(vectors.shape)

stop_words = STOPWORDS
model = make_pipeline(CountVectorizer(stop_words=stop_words),SelectKBest(score_func=mutual_info_classif,k=12500), SGDClassifier(verbose=1))
model.fit(train.data, train.target)



labels = model.predict(test.data)
accuracy = accuracy_score(test.target, labels)
print(accuracy)


train_scores, valid_scores = learning_curve(model,train.data,train.target)
print(train_scores)
print(valid_scores)



mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

def predict_category(s, train= train, model = model):
    pred = model.predict([s])
    return train.target_names[pred[0]]
print(predict_category('Dhyanchand'))



