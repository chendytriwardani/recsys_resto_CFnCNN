import streamlit as st
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
    st.write("""# Upload File""")
    st.write("Recsys Restoran")
    file_resto = st.file_uploader("Upload file CSV resto", accept_multiple_files=True, key="file_uploader_resto")
    file_user = st.file_uploader("Upload file CSV user", accept_multiple_files=True, key="file_uploader_user")
    file_rating = st.file_uploader("Upload file CSV user", accept_multiple_files=True, key="file_uploader_rating")
    for uploaded_file_resto in file_resto:
        df1 = pd.read_csv(uploaded_file_resto)
        st.write("Nama File Anda = ", uploaded_file_resto.name)
        st.dataframe(df1)
    for uploaded_file_user in file_user:
        df2 = pd.read_csv(uploaded_file_user)
        st.write("Nama File Anda = ", uploaded_file_user.name)
        st.dataframe(df2)
    for uploaded_file_rating in file_rating:
        df3 = pd.read_csv(uploaded_file_rating)
        st.write("Nama File Anda = ", uploaded_file_rating.name)
        st.dataframe(df3)


with modeling:
    st.title("# Modeling")


with recommendation:
    st.title("# recommendation")
