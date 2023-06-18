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
    def __init__(self):
        pass

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
        self.user_profiles = None
        self.feature_vectors = new_matrix
        self.sim_matrix = cosine_similarity(self.feature_vectors)

    def get_user_profile(self, items, ratings, alpha=1):
        self.user_items = items
        _, self.items_vectors = self.get_feature_vectors(self.user_items)
        self.user_ratings = ratings
        ridge = Ridge(alpha=alpha).fit(self.items_vectors, self.user_ratings)
        return (ridge, sparse.csr_matrix(ridge.coef_))

    def fit_users(self, user_data, id_col, item_col, rating_col, top=100):
        # self.reduce_features()
        self.user_data = user_data
        user_list = user_data[id_col].unique().tolist()
        user_profiles = []
        self.user_indices = pd.Series(range(0,len(user_list)),index=user_list)
        total = len(user_list)
        for index, user in enumerate(user_list):
            print(f'fitting {index+1}/{top}.')
            this_user = user_data.loc[user_data[id_col] == user]
            _, profile = self.get_user_profile(this_user[item_col].astype(str).tolist(),
                                            this_user[rating_col].astype(str).tolist(), 100)
            user_profiles.append(profile)
            clear_output(wait=True)
            if index + 1 == top:
                break
        self.user_profiles = sparse.vstack(user_profiles)
        self.user_matrices = cosine_similarity(self.user_profiles)

    def get_personal_recommendations(self, id, n_users=5, n_items=10):
        this_user = self.user_data.loc[self.user_data[0] == id]
        clf, _ = self.get_user_profile(this_user[1].astype(str).tolist(),
                                       this_user[2].astype(str).tolist(), 100)
        user_items = self.user_data[1].loc[self.user_data[0] == id].tolist()
        a = []
        for iid in self.indices.index.tolist():
            if iid not in self.user_items:
                a.append(iid)
        ids, remained = self.get_feature_vectors(a)        
        predicts = clf.predict(remained).flatten()
        pre_dict = dict(zip(ids, predicts))
        pre_dict = dict(sorted(pre_dict.items(), key=lambda item: item[1],reverse=True)[:n_items])

        idx = self.user_indices[id]
        sim_scores = list(enumerate(self.user_matrices[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_users + 1]
        sim_users_indices = [i[0] for i in sim_scores]
        sim_users = self.user_indices[sim_users_indices].tolist()
        pi = []
        for user in sim_users:
            items = self.user_data.loc[self.user_data[0] == user]
            like_items = items[1].loc[items[2] >= 4].tolist()
            pi = pi + like_items[:n_items]
        pi = list(set(pi))
        recs = [str(item) for item in pi if item not in user_items]
        return (list(pre_dict.keys()), recs)


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
        model = AgglomerativeClustering(
            **parameters).fit(self.feature_vectors.toarray())
        labels = model.labels_
        x_new = SelectKBest(chi2, k=4000).fit_transform(
            self.feature_vectors, labels)
        self.update_features(x_new)
        clear_output()

    def fit(self, data, id, docs, is_feature_reduced=False):
        self.indices = pd.Series(data.index, index=data[id])
        print('initializing features')
        tokens_list = [[' '.join(doc)][0]
                       for doc in self.preprocess_text(data[docs])]
        self.cal_tfidf(tokens_list)
        clear_output()
        if is_feature_reduced:
            self.reduce_features()