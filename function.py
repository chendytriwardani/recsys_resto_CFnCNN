import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
from keras.layers import Add, Activation, Lambda
from keras.models import Model
from keras.layers import Input, Reshape, Dot
from keras.layers import Embedding


def upload_and_display_file(file_uploader, file_type):
    uploaded_file = st.file_uploader(f"Upload file CSV {file_type}", accept_multiple_files=True, key=f"file_uploader_{file_type}")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Nama File {file_type} Anda = {uploaded_file.name}")
        st.dataframe(df)
    else:
        st.write(f"Belum ada file {file_type} yang diunggah.")

def split_data(data):
    X = data[['user_id', 'resto_id']].values
    y = data['Rating Restoran'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

class EmbeddingLayer:
    def __init__(self, n_items, n_factors, input_length):
        self.n_items = n_items
        self.n_factors = n_factors
        self.input_length = input_length

    def __call__(self, x):
        x = Embedding(input_dim=self.n_items, output_dim=self.n_factors, input_length=self.input_length)(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(self.n_factors, activation='relu')(x)
        return x


def Recommender(n_users, n_rests, n_factors, min_rating, max_rating, max_text_length):
    user = Input(shape=(1,))
    u = EmbeddingLayer(n_users, n_factors, input_length=1)(user)
    ub = EmbeddingLayer(n_users, 1, input_length=1)(user)

    restaurant = Input(shape=(1,))
    m = EmbeddingLayer(n_rests, n_factors, input_length=max_text_length)(restaurant)
    mb = EmbeddingLayer(n_rests, 1, input_length=max_text_length)(restaurant)

    x = Dot(axes=1)([u, m])
    x = Add()([x, ub, mb])
    x = Activation('sigmoid')(x)
    x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)

    model = Model(inputs=[user, restaurant], outputs=x)
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    return model

def trainingModel(epochs, batch_size):
    epochs = epochs
    batch_size = batch_size
