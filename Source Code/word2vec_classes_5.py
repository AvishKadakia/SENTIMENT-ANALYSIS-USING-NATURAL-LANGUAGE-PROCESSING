#!/usr/bin/env python
# coding: utf-8

# In[8]:


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


# In[9]:


import pickle
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import MiniBatchKMeans


# In[10]:


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


# In[11]:


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


# In[12]:


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


# In[13]:


# Creating the model and setting values for the various parameters
num_features = 100  # Word vector dimensionality
min_word_count = 40 # Minimum word count
num_workers = 4     # Number of parallel threads
context = 10        # Context window size
downsampling = 1e-3 # (0.001) Downsample setting for frequent words

# Initializing the train model
from gensim.models import word2vec
print("Training model....")
start_time = time.time()
model = word2vec.Word2Vec(X_train,                          workers=num_workers,                          size=num_features,                          min_count=min_word_count,                          window=context,
                          sample=downsampling)

# To make the model memory efficient
model.init_sims(replace=True)

# Saving the model for later use. Can be loaded using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)
# Function to average all word vectors in a paragraph
def featureVecMethod(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    
    #Converting Index2Word which is a list to a set for better speed in the execution.
    index2word_set = set(model.wv.index2word)
    
    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    
    # Dividing the result by number of words to get average
    featureVec = np.divide(featureVec, nwords)
    return featureVec
# Function for calculating the average feature vector
def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        # Printing a status message every 100000th review
        if counter%100000 == 0:
            print("Review %d of %d"%(counter,len(reviews)))
            
        reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)
        counter = counter+1
        
    return reviewFeatureVecs


X_train = getAvgFeatureVecs(X_train, model, num_features)
X_test = getAvgFeatureVecs(X_test, model, num_features)
print(f"Total time for word2 vec: {time.time()-start_time}")


# In[14]:


#scaling vector between 0 - 1 
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train =scaler.transform(X_train)

scaler = MinMaxScaler()
scaler.fit(X_test)
X_test = scaler.transform(X_test)



# 

# In[22]:


from google.colab import drive
drive.mount('/content/drive')
datasetPath = "/content/drive/My Drive/Natural Language Processing/Project/Cleaned Dataset/"
#Saving to CSV File
np.savetxt(datasetPath+'X_train_w2v.csv', X_train, delimiter=",") 
np.savetxt(datasetPath+'X_test_w2v.csv', X_test, delimiter=",") 


# In[23]:



from google.colab import drive
drive.mount('/content/drive')
datasetPath = "/content/drive/My Drive/Natural Language Processing/Project/Cleaned Dataset/"
start_time = time.time()
X_train_temp = pd.read_csv(datasetPath+'X_train_w2v.csv')
X_test_temp = pd.read_csv(datasetPath+'X_test_w2v.csv')
print(f"Total time to load reviews: {time.time()-start_time}")


# In[27]:


X_train_cleaned = X_train[~np.isnan(X_train)]
X_test_cleaned = X_test[~np.isnan(X_test)]


# In[29]:


X_train_cleaned= X_train_cleaned.reshape(-1,100)
X_test_cleaned = X_test_cleaned.reshape(-1,100)


# In[38]:


y_train_cleaned = y_train[0:-1]
y_test_cleaned = y_test[0:-2]


# In[40]:


from sklearn.linear_model import SGDClassifier
n_batches = 10
current_batch = 0
increment = (int(len(X_train_cleaned)/n_batches))
clf = SGDClassifier(loss="hinge", penalty="l2")

print(f"Classifying using SVM")
start_time0 = time.time()
for i in range(n_batches):
  start_time = time.time()
  next_batch = current_batch + increment
  clf.partial_fit(X_train_cleaned[current_batch:next_batch], y_train_cleaned[current_batch:next_batch], classes=labels)
  current_batch = next_batch
  print(f"Batch {i} : {current_batch} of {len(X_train_cleaned)} : {time.time() - start_time} seconds ")
print(f"Total training time: {time.time() - start_time0} ")
start_time = time.time()
svm_accuracy = accuracy_score(y_test_cleaned, clf.predict(X_test_cleaned))
print(f"Test dataset accuracy for classifier is {svm_accuracy} ")
print(classification_report(y_test_cleaned, clf.predict(X_test_cleaned), target_names=target_names))
print(f"Total time to test: {time.time()-start_time}")


# In[42]:


start_time = time.time()
print(f"Classifying using multinomialNB")
clf = MultinomialNB()
clf.fit(X_train_cleaned, y_train_cleaned)
print(f"Total time to train: {time.time()-start_time}")
start_time = time.time()
nb_accuracy = accuracy_score(y_test_cleaned, clf.predict(X_test_cleaned))
print(f"Test dataset accuracy is {nb_accuracy} ")
print(classification_report(y_test_cleaned, clf.predict(X_test_cleaned), target_names=target_names))
print(f"Total time to test: {time.time()-start_time}")


# In[43]:


n_batches = 1
current_batch = 0
increment = (int(len(X_train_cleaned)/n_batches))
clf = MiniBatchKMeans(n_clusters=len(labels),random_state=0,batch_size=n_batches)

print(f"Classifying using KNN")
start_time0 = time.time()
for i in range(n_batches):
  start_time = time.time()
  next_batch = current_batch + increment
  clf.partial_fit(X_train_cleaned[current_batch:next_batch])
  current_batch = next_batch
  print(f"Batch {i} : {current_batch} of {len(X_train_cleaned)} : {time.time() - start_time} seconds ")
print(f"Total training time: {time.time() - start_time0} ")
start_time = time.time()
knn_accuracy = accuracy_score(y_test_cleaned, clf.predict(X_test_cleaned))
print(f"Test dataset accuracy for classifier is {svm_accuracy} ")
print(classification_report(y_test_cleaned, clf.predict(X_test_cleaned), target_names=target_names))
print(f"Total time to test: {time.time()-start_time}")

