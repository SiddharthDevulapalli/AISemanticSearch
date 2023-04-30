from sklearn.feature_extraction.text import TfidfVectorizer

# from sklearn.ensemble import RandomForestClassifier
import pickle

# from util import plot_roc
# spacy_tok

import re
import nltk
import string
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
import numpy as np
import random
# import tensorflow as tf
import os
from numpy.linalg import norm
from functools import reduce
import pandas as pd
import json

class SearchModel(object):
    def __init__(self, data):
        """Simple Search
        Attributes:
            clf: sklearn classifier model
            vectorizor: TFIDF vectorizer or similar
        """
        # self.clf = MultinomialNB()
        # self.vectorizer = TfidfVectorizer(tokenizer=spacy_tok)
        # self.vectorizer = TfidfVectorizer()
        self.data = data

    def tokenize(self, lyriclist):
        lyriclist = pd.Series(lyriclist)
        return lyriclist.apply(lambda x: nltk.word_tokenize(x))

    def lemmatizedocs(self, lyriclist):
        lemmatizer = WordNetLemmatizer()
        lyriclist = pd.Series(lyriclist)
        lyriclist = lyriclist.apply(lambda x: ' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(x)]))
        return lyriclist

    def backgroundvocals(self, lyriclist):
        lyriclist = pd.Series(lyriclist)
        return lyriclist.apply(lambda x: re.sub("[\(\[].*?[\)\]]", "", x))

    def removestopwords(self, lyriclist):
        lyriclist = pd.Series(lyriclist)
        stop_words = set(stopwords.words('english'))
        return lyriclist.apply(lambda x: ' '.join([w for w in nltk.word_tokenize(x) if not w.lower() in stop_words]))

    def removepunctuations(self, lyriclist):
        lyriclist = pd.Series(lyriclist)
        return lyriclist.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

    def removeothers(self, lyriclist):
        lyriclist = self.removestopwords(lyriclist)
        lyriclist = self.removepunctuations(lyriclist)
        return lyriclist

    def queryparser(self, query, bgvocals=False, lemma=False, rmother=False):
        lyriclist = query.split()
        lyriclist = pd.Series(lyriclist)
        lyriclist = lyriclist.apply(lambda x: x.lower())

        if bgvocals:
            lyriclist = self.backgroundvocals(lyriclist)

        if lemma:
            lyriclist = self.lemmatizedocs(lyriclist)

        if rmother:
            lyriclist= self.removeothers(lyriclist)
        # print(lyriclist, "------------------------------------------------------------")
        lyriclist = self.tokenize(lyriclist)
        lyriclist = [i[0] for i in lyriclist]

        return lyriclist

    def parser(self, df, bgvocals=False, lemma=False, rmother = False):
        df['lyrics'] = df['lyrics'].apply(lambda x: x.lower())

        if bgvocals:
            df['lyrics'] = self.backgroundvocals(df['lyrics'])

        if lemma:
            df['lyrics'] = self.lemmatizedocs(df['lyrics'])

        if rmother:
            df['lyrics']= self.removeothers(df['lyrics'])

        df['tokens'] = self.tokenize(df['lyrics'])

        return df

    def getdocvecs(self, allsongs, word2vec):
        doc_vecs = []
        print("computing doc vecs")
        for song in allsongs:
            
            listoflists = [word2vec[token] for token in song]
            n = len(listoflists)
            sum_list = list(map(sum, zip(*listoflists)))
            avg_list = list(map(lambda x: x/n, sum_list))
            doc_vecs.append(avg_list)

        return doc_vecs

    def desm(self, document_vectors, query, word2vec):
        similarities = []
        print("desm starts")
        for dvec in document_vectors:
            querydocscore = 0
            querylen = 0
            for term in query.split(" "):
                termvec = word2vec[term]
                simscore = np.dot(termvec, dvec)/(norm(termvec)*norm(dvec))
                querylen += 1
                querydocscore += simscore
            
            querydocscore = querydocscore/querylen
            similarities.append(querydocscore)
        data_tuples = list(zip(similarities, self.data['song'], self.data['lyrics']))
        printdf = pd.DataFrame(data_tuples, columns=['Scores','DocumentID','Document'])
        printdf['Rank'] = printdf['Scores'].rank(method='first', ascending = False).astype(int)
        printdf = printdf.sort_values(by=['Rank'])
        printdf = printdf[["Rank", "Scores", "DocumentID", "Document"]]
        return printdf

    def matching(self, query):
        song_sentences = list(self.parser(self.data)['tokens'])
        model = Word2Vec(sentences = song_sentences, min_count=1, seed = 0, workers = 1)
        # model.train(song_sentences, total_examples=model.corpus_count, epochs=20)
        # model.save('word2vec.model')
        word_vectors = model.wv
        document_vectors = self.getdocvecs(song_sentences, word_vectors)
        displaydf = self.desm(document_vectors, query, word_vectors)
        return displaydf.head(10)