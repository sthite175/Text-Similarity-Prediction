###----------------------IMPORT ALL LIBRARIES-----------------------------------------------------
import numpy as np 
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from contractions import fix
from unidecode import unidecode

from nltk.stem import WordNetLemmatizer, LancasterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.util import ngrams

import warnings
warnings.filterwarnings('ignore')


###-----------------------TEXT DATA PREPROCESSING---------------------------------------------------
# Remove blank and tab spaces
def remove_blank(data):
    clean_text = data.replace("\\n"," ").replace("\t"," ")
    return clean_text

# Expand the text like Ex. don't >> do not
def expand_text(data):
    clean_text = fix(data)
    return clean_text

# Remove Stopwords
stopword_list = stopwords.words("english")
stopword_list.remove("no")
stopword_list.remove("nor")
stopword_list.remove("not")

# Clean text
# Lowercase the text, Remove Punctuation, Remove numbers
def clean_text(data):
    tokens = word_tokenize(data)
    clean_data = [word.lower() for word in tokens if (word.lower() not in punctuation) and (word.lower() not in stopword_list) and ( len(word)>2) and (word.isalpha()) ]
    return clean_data

# Lemmatization to convert word into their root form
def lemmatization(data):
    lemmatizer = WordNetLemmatizer()
    final_text = []
    for word in data:
        lemmatized_word = lemmatizer.lemmatize(word)
        final_text.append(lemmatized_word)
    return final_text 


###----------------------VECTORS OF ALL WORDS--------------------------------------------
def vectorizer(list_of_docs,model):
    
    feature =[] # save rew vector 
    for rew in list_of_docs: # iterating over reviews
        zero_vector = np.zeros(model.vector_size) # keyerror
        vectors =[] # to append vector of each word
        for word in rew : # iterating over all words in a review
            if word in model.wv: # checking if word is there in our vocab
                try :
                    vectors.append(model.wv[word]) # appending vector of each word
                except KeyError:
                    continue
        if vectors: # if Vectors is a empty list or not 
            vectors = np.asarray(vectors) # converting multiple arrays into a single array
            avg_vec = vectors.mean(axis=0) # avg of all vectors
            feature.append(avg_vec) # appending the avg vector
        else :
            feature.append(zero_vector) # handling key error
    return feature

###--------------------------END-----------------------------------------------------------------
