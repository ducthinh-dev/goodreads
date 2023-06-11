import sys
import os
from os import path

dir = path.abspath('__file__')
sys.path.append(path.dirname(dir))

import streamlit as st
import pandas as pd
import numpy as np

import sqlalchemy as sa
import mysql.connector

from setup import setup
# from models.doc2vec import Doc2VecModel
from models.tfidf import TfidfModel

connector = setup()
#====QUERIES DATA==============================
first_time = True
if first_time:
    books_query = sa.text(
        "select * from books;"
    )
    books = pd.read_sql_query(books_query, con=connector.connect())

    descr_query = sa.text(
        "select * from processed_description;"
    )
    descr = pd.read_sql_query(descr_query, con=connector.connect())
    st.title('DEMO APP')
    model = TfidfModel(descr, 'goodreads_book_id', 'processed_descr')
    model.fit()
    indices, _ = model.get_tfidf_scores()
    first_time = False

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