#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import gzip
import pandas as pd
from urllib.request import urlopen
import csv
import shutil
import os
import time
import re
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk.download('stopwords')
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import numpy as np


# In[ ]:


import pickle
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import MiniBatchKMeans


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# In[ ]:


from collections import Counter
from random import seed
from random import randrange
from tabulate import tabulate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report


# In[ ]:


#Dataset Link: https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt
import requests

base_url = "https://s3.amazonaws.com/amazon-reviews-pds/tsv/"
file_name = "amazon_reviews_us_Wireless_v1_00"
r = requests.get(base_url+file_name+".tsv.gz", allow_redirects=True)
#Downlaod data from the server to local machine
open(file_name+".tsv.gz", 'wb').write(r.content)
#Unzip downlaoded data
with gzip.open(file_name+".tsv.gz", 'rb') as f_in:
    with open(file_name+".tsv", 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
#Remove old zipped data
os.remove(file_name+".tsv"+".gz")
#Store data into a pandas dataframe
chunk = pd.read_csv(file_name+".tsv", sep='\t',error_bad_lines=False,chunksize=1000000,low_memory=False)
df = pd.concat(chunk)
#Remove unzipped file from local machine
os.remove(file_name+".tsv")
print(len(df))
print(df.head())
print("Before cleanning reviews")
print(df["review_body"].head(5))
def clean_dataset(X):
  review = X
  stemmer = WordNetLemmatizer()

  # Remove all the special characters
  review = re.sub(r'\W', ' ', review)
  
  # remove all single characters
  review = re.sub(r'\s+[a-zA-Z]\s+', ' ', review)
  
  # Remove single characters from the start
  review = re.sub(r'\^[a-zA-Z]\s+', ' ', review) 
  
  # Substituting multiple spaces with single space
  review = re.sub(r'\s+', ' ', review, flags=re.I)
  
  # Removing prefixed 'b'
  review = re.sub(r'^b\s+', '', review)
  
  # Converting to Lowercase
  review = review.lower()
  
  # Lemmatization
  review = review.split()
  review = [stemmer.lemmatize(word) for word in review]
  review = ' '.join(review)

  #Removing all  stopwords.
  review = remove_stopwords(review)
  return review

start_time = time.time()

X = df["review_body"].apply(lambda x: clean_dataset(str(x)))
print(f"Total time to clean reviews: {time.time()-start_time}")
print("After cleanning reviews")
print(X.head(5))
y = df["star_rating"]
from google.colab import drive
drive.mount('/content/drive')
datasetPath = "/content/drive/My Drive/Natural Language Processing/Project/Cleaned Dataset/"
#Saving cleaned data to CSV File
X.to_csv(datasetPath+'X_cleaned.csv', index=False) 
y.to_csv(datasetPath+'y_cleaned.csv', index=False) 


# In[ ]:


from google.colab import drive
#Reading saved clean file
drive.mount('/content/drive')
datasetPath = "/content/drive/My Drive/Natural Language Processing/Project/Cleaned Dataset/"
start_time = time.time()
X = pd.read_csv(datasetPath+'X_cleaned.csv')
y = pd.read_csv(datasetPath+'y_cleaned.csv')
print(f"Total time to load reviews: {time.time()-start_time}")

#Removing null values
result = pd.concat([X,y], axis=1).dropna()
X = result["review_body"]
y = result["star_rating"]
#Cnverting to 5 ratings 0 - 4 from 1 - 5
def reset_ratings(r):
  if r == 1:
    return 0;
  if r == 2:
    return 1;
  if r == 3:
    return 2;
  if r == 4:
    return 3;
  if r == 5:
    return 4;
y = y.apply(lambda x: reset_ratings(int(x)))
#Calculating Targets or Classes
target_names = []
labels = np.unique(y)
for label in labels:
  target_names.append('Star Rating '+str(label))
print(f"Target Names: {[target_names]}")
#Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train = np.array(X_train).reshape(X_train.shape[0],)
X_test = np.array(X_test).reshape(X_test.shape[0],)
#Convert all rating to type int
y_train = np.array(y_train).astype(int)
y_test = np.array(y_test).astype(int)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


def transform(array):    
    vectorizer = CountVectorizer(max_features=100, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    vectorized = vectorizer.fit_transform(array)
    transformer = TfidfTransformer()
    transformed = transformer.fit_transform(vectorized)
    return np.array(transformed.toarray())


# In[ ]:


start_time = time.time()
print("Creating Count Vectors for training data")
X_train = transform(X_train)
print(np.array(X_train).shape)
print("Creating Count Vectors for testing data")
X_test = transform(X_test)
print(np.array(X_test).shape)
print(f"Total time to vectorize: {time.time() - start_time}")


# In[ ]:


# convert to sparse matrix to dense matrix
start_time = time.time()
print("Converting sparse matrix to dense matrix")
X_train = csr_matrix(X_train)
X_test = csr_matrix(X_test)
print(f"Total time to vectorize: {time.time() - start_time}")


# In[ ]:


# #scaling vector between 0 - 1 
# scaler = MinMaxScaler()
# scaler.fit(X_train)
# X_train =scaler.transform(X_train)

# scaler = MinMaxScaler()
# scaler.fit(X_test)
# X_test = scaler.transform(X_test)

X_train = X_train.toarray()
X_test= X_test.toarray()


# 

# In[ ]:


from sklearn.linear_model import SGDClassifier
n_batches = 10
current_batch = 0
increment = (int(len(X_train)/n_batches))
clf = SGDClassifier(loss="hinge", penalty="l2")

print(f"Classifying using SVM")
start_time0 = time.time()
for i in range(n_batches):
  start_time = time.time()
  next_batch = current_batch + increment
  clf.partial_fit(X_train[current_batch:next_batch], y_train[current_batch:next_batch], classes=labels)
  current_batch = next_batch
  print(f"Batch {i} : {current_batch} of {len(X_train)} : {time.time() - start_time} seconds ")
print(f"Total training time: {time.time() - start_time0} ")
start_time = time.time()
svm_accuracy = accuracy_score(y_test, clf.predict(X_test))
print(f"Test dataset accuracy for classifier is {svm_accuracy} ")
print(classification_report(y_test, clf.predict(X_test), target_names=target_names))
print(f"Total time to test: {time.time()-start_time}")


# In[ ]:


start_time = time.time()
print(f"Classifying using multinomialNB")
clf = MultinomialNB()
clf.fit(X_train, y_train)
print(f"Total time to train: {time.time()-start_time}")
start_time = time.time()
nb_accuracy = accuracy_score(y_test, clf.predict(X_test))
print(f"Test dataset accuracy is {nb_accuracy} ")
print(classification_report(y_test, clf.predict(X_test), target_names=target_names))
print(f"Total time to test: {time.time()-start_time}")


# In[ ]:


n_batches = 1
current_batch = 0
increment = (int(len(X_train)/n_batches))
clf = MiniBatchKMeans(n_clusters=len(labels),random_state=0,batch_size=n_batches)

print(f"Classifying using KNN")
start_time0 = time.time()
for i in range(n_batches):
  start_time = time.time()
  next_batch = current_batch + increment
  clf.partial_fit(X_train[current_batch:next_batch])
  current_batch = next_batch
  print(f"Batch {i} : {current_batch} of {len(X_train)} : {time.time() - start_time} seconds ")
print(f"Total training time: {time.time() - start_time0} ")
start_time = time.time()
knn_accuracy = accuracy_score(y_test, clf.predict(X_test))
print(f"Test dataset accuracy for classifier is {knn_accuracy} ")
print(classification_report(y_test, clf.predict(X_test), target_names=target_names))
print(f"Total time to test: {time.time()-start_time}")

