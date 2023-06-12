from IPython.display import clear_output

import pandas as pd
import numpy as np
import mysql.connector
import sqlalchemy as sa
import getpass

from scipy import sparse

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import Ridge

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')

ENGLISH_WORDS = set(words.words())
clear_output()
class TfidfModel:
    def __init__(self, data, id, docs):
        self.data = data
        self.id = id
        self.id_data = data[id]
        self.docs_data = data[docs]
        # self.feature_vectors = None
        self.indices = pd.Series(data.index, index=data[id])
        # self.sim_matrix = None

    def preprocess_text(self, docs):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        tokens_list = []
        for doc in docs:
            tokens = nltk.word_tokenize(doc)
            tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.isalpha() and token.lower() in ENGLISH_WORDS]        
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
            tokens_list.append(tokens)
        return tokens_list

    def cal_tfidf(self, docs):
        tfidf = TfidfVectorizer()
        self.feature_vectors = tfidf.fit_transform(docs)  
        self.sim_matrix = cosine_similarity(self.feature_vectors)  
        
    def get_feature_vectors(self, ids=[]):
        if ids:
            result_vectors = []
            for id in ids:
                result_vectors.append(self.feature_vectors[self.indices[id]])
            return (ids, sparse.vstack(result_vectors))
        else:
            return (self.indices.index, self.feature_vectors)

    def update_features(self, new_matrix):
        self.feature_vectors = new_matrix
        self.sim_matrix = cosine_similarity(self.feature_vectors)

    def update_user(self, items, ratings, alpha=1):
        self.user_items = items
        _, self.items_vectors = self.get_feature_vectors(self.user_items)
        # ratings = np.array(ratings)
        self.user_ratings = ratings
        self.ridge = Ridge(alpha=alpha).fit(self.items_vectors, self.user_ratings)

    def get_personal_recommendations(self, top=5):
        a = []
        for id in self.id_data.tolist():
            if id not in self.user_items:
                a.append(id)
        ids, remained = self.get_feature_vectors(a)
        predicts = self.ridge.predict(remained).flatten()
        pre_dict = dict(zip(ids, predicts))
        pre_dict = dict(sorted(pre_dict.items(), key=lambda item: item[1],reverse=True)[:top])
        return list(pre_dict.keys())

    def score(self, measurement, test_data, test_ratings):
        _, test_vectors = self.get_feature_vectors(test_data)
        predicts = self.ridge.predict(test_vectors).flatten()
        score = measurement(predicts, test_ratings)
        return score

    def evaluatePR(self, test_data, top=5):
        tp = 0
        recommended_items = self.get_personal_recommendations(top)
        for item in recommended_items:
            if item in test_data:
                tp += 1
        return tp

    def get_recommendations(self, id, top=5):
        idx = self.indices[id]
        sim_scores = list(enumerate(self.sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top + 1]
        item_indices = [i[0] for i in sim_scores]
        return self.data[self.id].iloc[item_indices].tolist()

    def reduce_features(self):
        parameters = {
            'n_clusters': None,
            'metric': 'euclidean',
            'linkage': 'ward',
            'distance_threshold': 1.38,
            'compute_distances': True
        }
        print('reducing features!')
        model = AgglomerativeClustering(**parameters).fit(self.feature_vectors.toarray())
        labels = model.labels_
        x_new = SelectKBest(chi2, k=4000).fit_transform(self.feature_vectors, labels)
        self.update_features(x_new)
        clear_output()

    def fit(self, is_feature_reduced=False):
        print('initializing features')
        tokens_list = [[' '.join(doc)][0] for doc in self.preprocess_text(self.docs_data)]
        self.cal_tfidf(tokens_list)
        clear_output()
        if is_feature_reduced:
        # reducing features
            self.reduce_features()