import pandas as pd 
import numpy as np
import nltk
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from nltk.util import pr
from nltk.corpus import stopwords

data= pd.read_csv('labeled_data.csv')

stemer=nltk.SnowballStemmer('english')

data['lables'] =data['class'].map({0:"Hate Speech", 1:" Offesinve Language", 2:"Normal"})

print(data.head())

# selecting imprtant clumns
data=data[['tweet','lables']]
print(data.head())

# cleaning the data set 
def clean_text(text):
    text=text.lower()
    text=re.sub('\[.*?\]', '', text)
    text=re.sub('https?://\S+|www\.\S+', '', text)
    text=re.sub('<.*?>+', '', text)
    text=re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text=re.sub('\n', '', text)
    text=re.sub('\w*\d\w*', '', text)
    return text