import math
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
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

class Doc2VecModel:
    def __init__(self, data, data_id, data_document):
        self.data = data
        self.data_id = data_id
        self.data_document = data_document
        self.documents = data[data_document]
        self.ids = data[data_id]
        self.indices = pd.Series(data.index, index=self.ids)
        self.model = None
        self.docvecs = []
        self.sim_matrix = None

    def preprocess_doc(self):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        docs_list = []
        for doc in self.documents:
            tokens = word_tokenize(doc)
            tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.isalpha() and token.lower() in ENGLISH_WORDS]        
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
            docs_list.append(' '.join(tokens))
        return docs_list

    def get_recommendations(self, book_id, num_recommends=5):
        idx = self.indices[book_id]
        sim_scores = list(enumerate(self.sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommends]
        item_indices = [i[0] for i in sim_scores]
        return self.data[self.data_id].iloc[item_indices].tolist()

    def get_docvecs(self):
        return self.indices, self.docvecs

    def fit(self):
        processed_docs = self.preprocess_doc()
        documents = [TaggedDocument(doc, [index]) for index, doc in enumerate(processed_docs)]
        self.model = Doc2Vec(documents, vector_size=50, window=5, workers=4, epochs=5)
        data_len = len(self.ids)
        for i in range(0, data_len):
            self.docvecs.append(self.model.docvecs[i])
        self.sim_matrix = cosine_similarity(self.docvecs)