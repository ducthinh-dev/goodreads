import joblib
import requests
from PIL import Image
import time
from models.tfidf import TfidfModel
from setup import setup
import mysql.connector
import sqlalchemy as sa
import numpy as np
import pandas as pd
import streamlit as st
import sys
from os import path
dir = path.abspath('__file__')
sys.path.append(path.dirname(dir))


st.set_page_config(
    page_title='Books recommender',
    page_icon='📚'
)


def get_recommendations(model, top):
    return model


def book_page():
    st.title('Books')
    get_data(book=True)
    change_model = st.button('Change model')
    if change_model:
        st.session_state["model"] = get_model(model_type='doc2vec')
        st.session_state["model_type"] = 'doc2vec'
    st.write('MODEL: ', st.session_state.model_type.upper())

    with st.form('id'):
        input_id = st.text_input('Enter book id')
        no_recommends = st.number_input(
            'Enter number of recommends', min_value=2, max_value=100, step=1)
        submit_button = st.form_submit_button('Submit')
    if submit_button:
        books = st.session_state.books
        indices = st.session_state.indices
        recommends = st.session_state.model.get_recommendations(
            int(input_id), no_recommends)
        chosen_book = books.loc[books.goodreads_book_id == input_id]
        st.write(f'recommends for {chosen_book.title.values[0]}')
        recommended_book = books[[
            'title', 'average_rating', 'description']].iloc[indices[recommends]]
        recommended_book.reset_index(inplace=True)
        recommended_book.drop(columns=['index'], inplace=True)
        st.write(recommended_book)


def user_page():
    st.title('Users')
    get_data(user=True, test=True)
    books = st.session_state.books
    indices = st.session_state.indices
    users = st.session_state.users
    users_test = st.session_state.test
    change_model = st.button('Change model')
    if change_model:
        st.session_state.model = get_model(model_type='doc2vec')
        st.session_state.model_type = 'doc2vec'
    st.write('MODEL: ', st.session_state.model_type.upper())

    with st.form('id'):
        input_id = st.text_input('Enter user id')
        no_recommends = st.number_input(
            'Enter number of recommends', min_value=2, max_value=100, step=1)
        submit_button = st.form_submit_button('Submit')
    if submit_button:
        user_profile = users.loc[users[0] == int(input_id)]
        uitems = user_profile[1].astype(int).tolist()
        uratings = user_profile[2].tolist()
        st.session_state.model.update_user(uitems, uratings, 100)
        recommends = st.session_state.model.get_personal_recommendations(no_recommends)
        recommended_book = books[[
            'title', 'average_rating', 'description']].iloc[indices[recommends]]
        recommended_book.reset_index(inplace=True)
        recommended_book.drop(columns=['index'], inplace=True)
        st.write(recommended_book)


pages = {
    'By book': book_page,
    'By user': user_page
}


def get_data(book=False, user=False, test=False):
    if book:
        connector = setup()
        query = sa.text(
            'select * from books;'
        )
        books = pd.read_sql_query(query, con=connector.connect())
        indices = pd.Series(
            books.index, index=books['goodreads_book_id'].astype(int))
        st.session_state.books = books
        st.session_state.indices = indices
    if user:
        users = pd.read_csv(
            r'D:/project/goodreads/data/main/ratings_train.dat', sep=':', header=None)
        st.session_state.users = users
    if test:
        users_test = pd.read_csv(
            r'D:/project/goodreads/data/main/ratings_test.dat', sep=':', header=None)
        st.session_state.test = users_test


def get_model(model_type='tfidf'):
    if model_type == 'tfidf':
        model = joblib.load(r'D:\project\goodreads\dumbed_models\tfidf.joblib')
    if model_type == 'doc2vec':
        model = 'doc2vec'
    return model


def main():
    st.sidebar.title('')
    page = st.sidebar.radio('Get recommends by:', list(pages.keys()))
    pages[page]()


if __name__ == '__main__':
    st.session_state.model = get_model()
    st.session_state.model_type = 'tfidf'
    main()
