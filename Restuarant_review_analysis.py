#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the data into a dataframe which is in a .tsv format
df = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t')

#data preprocessing begins from here
#preprocessing of data is necessary, as it cleans the data and helps to build a better model that can be used.
#data cleaning includes removal of stop words, conversion from upper to lower case, removal of alpha-numeric characters or removal of puntuation marks

#naive data cleaning performed on just one review
review = df['Review'][0]
print(review)
#import re class that is used to remove numeric or alpha-numeric characters
import re
#the sub function of the re class substitutes spaces where the given pattern is satisfied.
#[^a-zA-Z] = replace everything in the review except alphabets (lower and capital)
review  = re.sub('[^a-zA-Z]', ' ', review)
print(review)

#we need to tokenize the review so that we get words instead of a string
review = review.split()
print(review)

#now the cleaned text is in a string format and we need to remove unnecessary noise present in the data
#the unnecessary noise is the words that have no unique existence eg, articles, prepositions,conjunctions, etc
#we need to import stopwords.words present in the nltk library
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
review = [word for word in review if not word in stopwords]
print(review)

#now as the stopwords are removed, we need to do stemming so that the same word wont be considered differently.
from nltk.stem import PorterStemmer
ps = PorterStemmer()
review = [ps.stem(word) for word in review]
print(review)

#o/p: Wow... Loved this place.
#Wow    Loved this place 
#['Wow', 'Loved', 'this', 'place']
#['Wow', 'Loved', 'place']
#['wow', 'love', 'place']

#so this is how we can do the preprocessing manually 

#onvert dataframe into a list of lists
corpus =[]
ps = PorterStemmer()
for i in range(0,1000):
    #keeping only words in the review text
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    #convert everything to lower case
    review = review.lower()
    #split it into words
    review = review.split()
    #remove stopwords and do stemming
    review = [ps.stem(word) for word in review if not word in stopwords]
    #the words are joined together to form a sentence
    review = ' '.join(review)
    #the cleaned text is appended to corpus
    corpus.append(review)




#to reduce the manual work, sklearn has tools for perforimg cleansing of text
#CountVectorizer is used to convert a collection of text documents(reviews) into a matrix of tokens
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X = cv.fit_transform(corpus).toarray()

y = df.iloc[:,-1].values

#now perform classification

#split data into training and test data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size = 0.2, random_state = 0)

#classification being performed using DecisionTrees
from sklearn.tree import DecisionTreeClassifier
classifier_1 = DecisionTreeClassifier()

classifier_1.fit(xtrain,ytrain)

y_pred = classifier_1.predict(xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest,y_pred)
cm
o/p: array([[70, 27],
       [41, 62]], dtype=int64)

from sklearn.naive_bayes import GaussianNB
classifier_2 = GaussianNB()

classifier_2.fit(xtrain,ytrain)

y_pred_1 = classifier_2.predict(xtest)

cm = confusion_matrix(ytest,y_pred_1)
cm
o/p: array([[55, 42],
       [12, 91]], dtype=int64)

from sklearn.ensemble import RandomForestClassifier
classifier_3 = RandomForestClassifier(n_estimators = 300, random_state = 0)

classifier_3.fit(xtrain,ytrain)

y_pred_3 = classifier_3.predict(xtest)

cm = confusion_matrix(ytest,y_pred_3)
cm
o/p: array([[87, 10],
       [49, 54]], dtype=int64)
