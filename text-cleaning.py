import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Librerie per la pulizia del testo
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
#Machine learning 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
#La funzione cambia la prima virgola con 'sub'
#Attivare nel caso in cui si debba convertire un nuovo file
def substitute_first_comma(sep_to_sub=',', sub='<'):
    my_file = open("data/spam.csv", "r")
    content_list = my_file.readlines()
    converted_content = []
    for line in content_list:
        index_of_comma = line.find(sep_to_sub)
        new_line = line[0:index_of_comma] + sub +  line[index_of_comma+1:]
        converted_content.append(new_line)
    
    new_file = open("data/converted/spam_converted.csv", "w")
    for line in converted_content:
        new_file.write(line)
#Trasforma il testo in piccolo
def low_text(input):
    #print("input value is: ", input)
    return "".join([i.lower() for i in input])

substitute_first_comma(",", "<>")

#MAIN()--------
#Carichiamo il dataset
df = pd.read_csv('data/converted/spam_converted.csv', sep='<>', names=['label', 'text'], engine='python')
print(df.head(5))
#Text Cleaning
df['text'] = df['text'].map(lambda x:low_text(x))
corpus = []
for i in range(0, df.shape[0]):
    text = re.sub('[^a-zA-z]', ' ' , df['text'][i])
    text = text.lower()
    text = text.split()
    porter = PorterStemmer()
    text = [porter.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus.append(text)
#NaN Handling
print("NaN BEFORE handling")
print("NaN in Label column: ", df['label'].isnull().sum())
print("NaN in Text column: ", df['text'].isnull().sum())
df.dropna(inplace=True)
print("NaN POST handling")
print("NaN in Label column: ", df['label'].isnull().sum())
print("NaN in Text column: ", df['text'].isnull().sum())
#Trasformiamo il corpus in vector ( X )
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()
#Mappiamo le label ( y )
cl = {'ham':1, 'spam':0}
df['label'] = df['label'].map(cl)
y = df['label'].values
