#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import numpy as np


# In[2]:


import pickle
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import MiniBatchKMeans


# In[3]:


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


# In[4]:


#https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt
import requests
base_url = "https://s3.amazonaws.com/amazon-reviews-pds/tsv/"
file_name = "amazon_reviews_us_Wireless_v1_00"
r = requests.get(base_url+file_name+".tsv.gz", allow_redirects=True)
open(file_name+".tsv.gz", 'wb').write(r.content)

with gzip.open(file_name+".tsv.gz", 'rb') as f_in:
    with open(file_name+".tsv", 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

os.remove(file_name+".tsv"+".gz")
chunk = pd.read_csv(file_name+".tsv", sep='\t',error_bad_lines=False,chunksize=1000000,low_memory=False)
df = pd.concat(chunk)
#Removing null values
df = df.dropna()
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
def combined_features(row):
    return row['product_title'] + ' '+row['review_headline']+' '+ row['review_body']

X = df.apply(combined_features, axis=1)

X = df["review_body"].apply(lambda x: clean_dataset(str(x)))
print(f"Total time to clean reviews: {time.time()-start_time}")
print("After cleanning reviews")
print(X.head(5))
y = df["star_rating"]
from google.colab import drive
drive.mount('/content/drive')
datasetPath = "/content/drive/My Drive/Natural Language Processing/Project/Cleaned Dataset/"
#Saving to CSV File
X.to_csv(datasetPath+'X_cleaned.csv', index=False) 
y.to_csv(datasetPath+'y_cleaned.csv', index=False) 


# In[5]:


from google.colab import drive
drive.mount('/content/drive')
datasetPath = "/content/drive/My Drive/Natural Language Processing/Project/Cleaned Dataset/"
start_time = time.time()
X = pd.read_csv(datasetPath+'X_cleaned.csv')
y = pd.read_csv(datasetPath+'y_cleaned.csv')
print(f"Total time to load reviews: {time.time()-start_time}")

#Removing null values
result = pd.concat([X,y], axis=1).dropna()
result.head(10)
X = result["review_body"]
y = result["star_rating"]

#Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train = np.array(X_train).reshape(X_train.shape[0],)
X_test = np.array(X_test).reshape(X_test.shape[0],)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[6]:


# Tfidf vectorizer:
vectorizer = TfidfVectorizer(max_df=0.7, max_features=1000,
                             min_df=5, stop_words='english',
                             use_idf=True)

# Build the tfidf vectorizer from the training data ("fit"), and apply it 
# ("transform").
X_train_tfidf = vectorizer.fit_transform(X_train)

print(f"  Actual number of tfidf features: {X_train_tfidf.get_shape()}")

print("\nPerforming dimensionality reduction using LSA")
t0 = time.time()

# Project the tfidf vectors onto the first N principal components.
# Though this is significantly fewer features than the original tfidf vector,
# they are stronger features, and the accuracy is higher.
svd = TruncatedSVD(100)
lsa = make_pipeline(svd, Normalizer(copy=False))

# Run SVD on the training data, then project the training data.
X_train_lsa = lsa.fit_transform(X_train_tfidf)

print("  done in %.3fsec" % (time.time() - t0))

# explained_variance = svd.explained_variance_ratio_.sum()
# print("  Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

# Now apply the transformations to the test data as well.
X_test_tfidf = vectorizer.transform(X_test)
X_test_lsa = lsa.transform(X_test_tfidf)


# In[7]:


#scaling vector between 0 - 1 
scaler = MinMaxScaler()
scaler.fit(X_train_lsa)
X_train_lsa_scaled =scaler.transform(X_train_lsa)

scaler = MinMaxScaler()
scaler.fit(X_test_lsa)
X_test_lsa_scaled =scaler.transform(X_test_lsa)

#Convert all rating to type int
y_train = np.array(y_train).astype(int)
y_test = np.array(y_test).astype(int)


# In[8]:


y_test_old = y_test 
y_train_old = y_train


# In[9]:


y_test = y_test_old 
y_train = y_train_old
#Using only 3 ratings:
def rating(r):
  if r == 1:
    return 0;
  if r == 2:
    return 0;
  if r == 3:
    return 1;
  if r == 4:
    return 2;
  if r == 5:
    return 2;
ratings = lambda t: rating(t)
vfunc = np.vectorize(ratings)
y_train = vfunc(y_train)
y_test = vfunc(y_test)

print(np.unique(y_train))


# In[10]:


target_names = []
labels = np.unique(y_test)
for label in labels:
  target_names.append('Star Rating '+str(label))


# In[12]:


from sklearn.linear_model import SGDClassifier
n_batches = 10
current_batch = 0
increment = (int(len(X_train_lsa_scaled)/n_batches))
clf = SGDClassifier(loss="hinge", penalty="l2")

print(f"Classifying using SVM")
start_time0 = time.time()
for i in range(n_batches):
  start_time = time.time()
  next_batch = current_batch + increment
  clf.partial_fit(X_train_lsa_scaled[current_batch:next_batch], y_train[current_batch:next_batch], classes=labels)
  current_batch = next_batch
  print(f"Batch {i} : {current_batch} of {len(X_train_lsa_scaled)} : {time.time() - start_time} seconds ")
print(f"Total training time: {time.time() - start_time0} ")
start_time = time.time()
svm_accuracy = accuracy_score(y_test, clf.predict(X_test_lsa_scaled))
print(f"Test dataset accuracy for classifier is {svm_accuracy} ")
print(classification_report(y_test, clf.predict(X_test_lsa_scaled), target_names=target_names))
print(f"Total time to test: {time.time()-start_time}")


# In[15]:


start_time = time.time()
print(f"Classifying using multinomialNB")
clf = MultinomialNB()
clf.fit(X_train_lsa_scaled, y_train)
print(f"Total time to train: {time.time()-start_time}")
start_time = time.time()
nb_accuracy = accuracy_score(y_test, clf.predict(X_test_lsa_scaled))
print(f"Test dataset accuracy is {nb_accuracy} ")
print(classification_report(y_test, clf.predict(X_test_lsa_scaled), target_names=target_names))
print(f"Total time to test: {time.time()-start_time}")


# In[16]:


n_batches = 1
current_batch = 0
increment = (int(len(X_train_lsa_scaled)/n_batches))
clf = MiniBatchKMeans(n_clusters=len(labels),random_state=0,batch_size=n_batches)

print(f"Classifying using KNN")
start_time0 = time.time()
for i in range(n_batches):
  start_time = time.time()
  next_batch = current_batch + increment
  clf.partial_fit(X_train_lsa_scaled[current_batch:next_batch])
  current_batch = next_batch
  print(f"Batch {i} : {current_batch} of {len(X_train_lsa_scaled)} : {time.time() - start_time} seconds ")
print(f"Total training time: {time.time() - start_time0} ")
start_time = time.time()
knn_accuracy = accuracy_score(y_test, clf.predict(X_test_lsa_scaled))
print(f"Test dataset accuracy for classifier is {svm_accuracy} ")
print(classification_report(y_test, clf.predict(X_test_lsa_scaled), target_names=target_names))
print(f"Total time to test: {time.time()-start_time}")

