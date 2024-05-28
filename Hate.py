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
print(data.head())

data['lables'] =data['class'].map({0:"Hate Speech", 1:" Offesinve Language", 2:"Normal"})

