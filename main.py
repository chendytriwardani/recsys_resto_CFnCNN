import streamlit as st
import functions as f

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


st.title("UTS PPW KELAS A (MODELLING)")
st.write("-------------------------------------------------------------------------------------------------------------------------")
st.write("*Nama  : Chendy Tri Wardani")
st.write("*NIM   : 200411100041*")
st.write("-------------------------------------------------------------------------------------------------------------------------")
upload_data, modeling, recommendation = st.tabs(["Upload Data", "Modeling", "Recommendation"])


with upload_data:
    st.write("# Upload File")
    st.write("Recsys Restoran")

    # Menampilkan file yang diunggah
    f.upload_and_display_file("resto", "resto")
    f.upload_and_display_file("user", "user")
    f.upload_and_display_file("rating", "rating")



with modeling:
    st.title("# Modeling")


with recommendation:
    st.title("# recommendation")
