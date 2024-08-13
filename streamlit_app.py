import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
import pickle
# import spacy

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer

# from gensim import models
# from gensim.models import Phrases
# from gensim.models.phrases import Phraser

st.title("Phân Tích và Dự Báo Điểm GPA")

with st.expander('Data'):
    st.write('**Raw data**')
    df = pd.read_csv('https://raw.githubusercontent.com/lam-01/Data/main/Student_performance_data_2.csv')
    st.write(df)

# Hiển thị dữ liệu ban đầu
st.subheader("Dữ liệu ban đầu")
st.write(df.head())

# Thực hiện One-Hot Encoding cho các biến phân loại
st.subheader("Áp dụng One-Hot Encoding cho các biến phân loại")
cat_cols = ['Sports', 'Volunteering', 'ParentalSupport', 'Music', 'Extracurricular', 'ParentalEducation', 'Gender', 'Tutoring', 'Ethnicity']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

st.write("Dữ liệu sau khi áp dụng One-Hot Encoding:")
st.write(df_encoded.head())

# Phân tách dữ liệu
st.subheader("Phân tách dữ liệu")
X = df_encoded.drop(['GradeClass', 'StudentID'], axis=1)
y = df_encoded['GradeClass']

st.write("Biến đầu vào (X):")
st.write(X.head())

st.write("Biến đầu ra (y):")
st.write(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write(f"Train set size: {X_train.shape[0]} samples")
st.write(f"Test set size: {X_test.shape[0]} samples")

# Áp dụng SMOTE để cân bằng dữ liệu
smote = SMOTE(sampling_strategy='auto')
X_res, y_res = smote.fit_resample(X_train, y_train)

st.write(f"Train set size after SMOTE: {X_res.shape[0]} samples")

# Mô hình 
clf = RandomForestRegressor(max_depth=2, random_state=42)
clf.fit(X_train, y_train)

## Hàm dự đoán 
prediction = clf.predict(X)
prediction_proba = clf.predict_proba(X)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['0', '1', '2','3','4']
df_prediction_proba.rename(columns={0:'A', 1:'B', 2: 'C', 3:'D', 4:'F'})
