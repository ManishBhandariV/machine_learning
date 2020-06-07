import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups


category = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
            'comp.sys.mac.hardware','comp.windows.x','misc.forsale', 'rec.autos', 'rec.motorcycles',]
            # 'rec.sport.baseball', 'rec.sport.hockey','sci.crypt','sci.electronics', 'sci.med', 'sci.space',
            # 'soc.religion.christian','talk.politics.guns','talk.politics.mideast',
            # 'talk.politics.misc', 'talk.religion.misc']                 #to get text related to any particular category
                                                                        # here i have selected all categories

newsgroups_train = fetch_20newsgroups(subset= 'train', categories=category)


from pprint import pprint

pprint(list(newsgroups_train.filenames))       #the data has two coloumns: filenames and target
pprint(list(newsgroups_train.target_names))      #prints the unique class labels

print(newsgroups_train.filenames.shape)         #number of rows in filenames
print(newsgroups_train.target.shape)             #number of rows in target
#
print(newsgroups_train.target_names[5])        #target_names at index 5
print(newsgroups_train.data[394])              # prints data in the file at given index i.e the file from
                                               # filenames at a given index
#text = newsgroups_train.data[5]                #to create wordcloud of data at index 5 i.e file at index 5
text = "".join(data for data in newsgroups_train.data) #joining all files to create one single file
#print(text)
wordcloud = WordCloud().generate(text)          #generating word cloud
plt.imshow(wordcloud , interpolation= 'bilinear')
plt.axis("off")
plt.show()








