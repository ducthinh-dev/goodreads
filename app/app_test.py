# first_time = True
# if first_time:
#     import sys
#     from os import path
#     dir = path.abspath('__file__')
#     sys.path.append(path.dirname(dir))
#     import streamlit as st
#     import pandas as pd
#     import numpy as np
#     import sqlalchemy as sa
#     import mysql.connector

#     from setup import setup
#     # from models.doc2vec import Doc2VecModel
#     from models.tfidf import TfidfModel
#     connector = setup()
#     books_query = sa.text(
#         'select * from books;'
#     )
#     books = pd.read_sql_query(books_query, con=connector.connect())

#     descr_query = sa.text(
#         'select * from processed_description;'
#     )
#     descr = pd.read_sql_query(descr_query, con=connector.connect())
#     ratings = pd.read_csv('./data/main/ratings_train.dat', sep=':', header=None)
#     ratings_test = pd.read_csv('./data/main/ratings_test.dat', sep=':', header=None)
#     st.title('DEMO APP')
#     model = TfidfModel(descr, 'goodreads_book_id', 'processed_descr')
#     model.fit()
#     indices = pd.Series(books.index,index=books['goodreads_book_id'].astype(int))
#     first_time = False

# with st.form('id'):
#     input_id = st.text_input('Enter book id')
#     no_recommends = st.number_input('Enter number of recommends', min_value=2, max_value=100, step=1)
#     submit_button = st.form_submit_button('Submit')

# if submit_button:
#     recommends = model.get_recommendations(int(input_id), no_recommends)
#     chosen_book = books.loc[books.goodreads_book_id == input_id]
#     st.write(f'recommends for {chosen_book.title.values[0]}')
#     recommended_book = books[['title', 'average_rating', 'description']].iloc[indices[recommends]]
#     recommended_book.reset_index(inplace=True)
#     recommended_book.drop(columns=['index'], inplace=True)
#     st.write(recommended_book)


import mysql.connector
import sqlalchemy as sa
import numpy as np
import pandas as pd
import streamlit as st
import sys
from os import path
dir = path.abspath('__file__')
sys.path.append(path.dirname(dir))

from setup import setup
from models.tfidf import TfidfModel
import time
from PIL import Image
import requests

import joblib

st.set_page_config(
    page_title='Books recommender',
    page_icon = '📚'
)
def get_rec(model):
    with st.form('id'):
        input_id = st.text_input('Enter book id')
        no_recommends = st.number_input('Enter number of recommends', min_value=2, max_value=100, step=1)
        submit_button = st.form_submit_button('Submit')

    if submit_button:
        recommends = model.get_recommendations(int(input_id), no_recommends)
        chosen_book = books.loc[books.goodreads_book_id == input_id]
        st.write(f'recommends for {chosen_book.title.values[0]}')
        recommended_book = books[['title', 'average_rating', 'description']].iloc[indices[recommends]]
        recommended_book.reset_index(inplace=True)
        recommended_book.drop(columns=['index'], inplace=True)
        st.write(recommended_book)

def book_page():
    st.title('Books')
    data_btn = st.button('Get data')
    model_btn = st.button('Get model')
    if data_btn:
        get_data(book=True)
        success = st.success('Books data ready!', icon='📚')
        time.sleep(3)
        success.empty()
    if model_btn:
        model = get_model()
        success = st.success('Model ready!', icon='🤖')
        time.sleep(3)
        success.empty()
        get_rec(model)
    # if st.button('Get image'):
        # url = r'https://picsum.photos/200/400'
        # img = st.image(Image.open(requests.get(url, stream=True).raw))
        
def user_page():
    st.title('Users')
    if st.button('Get train data'):
        get_data(user=True)
        success = st.success('Train data ready!', icon='🧑‍🏫')
        time.sleep(3)
        success.empty()
        
    if st.button('Get test data'):
        success = st.success('Test data ready!', icon='🧑‍🏫')
        time.sleep(3)
        success.empty()

pages = {
    'By book': book_page,
    'By user': user_page
}

def get_data(book=False, user=False, test=False):
    global books, indices, users, users_test
    if book:
        connector = setup()
        query = sa.text(
            'select * from books;'
        )
        books = pd.read_sql_query(query, con=connector.connect())
        indices = pd.Series(books.index,index=books['goodreads_book_id'].astype(int))
    if user:
        users = pd.read_csv(r'D:/project/goodreads/data/main/ratings_train.dat', sep=':', header=None)
    if test:
        users_test = pd.read_csv(r'D:/project/goodreads/data/main/ratings_test.dat', sep=':', header=None)

def get_model(model_type='tfidf'):
    if model_type == 'tfidf':
        model = joblib.load(r'D:\project\goodreads\dumbed_models\tfidf.joblib')
    return model

def main():
    st.sidebar.title('')
    page = st.sidebar.radio('Get recommends by:', list(pages.keys()))
    pages[page]()


if __name__ == '__main__':
    main()
