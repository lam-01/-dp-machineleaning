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
# Encode X
X = df_encoded[1:]
input_row = df_encoded[:1]

# Encode y
target_mapper = {'0': A ,
                 '1': B ,
                 '2': C ,
                 '3': D ,
                 '4': F ,
                }
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
  st.write('**Encoded X (input penguin)**')
  input_row
  st.write('**Encoded y**')
  y


######################################### LOAD MÔ HÌNH ###########################################

# # Tải các mô hình hồi quy đã huấn luyện
# with open('linear_regression_model.pkl', 'rb') as f:
#     linear_regression_model = pickle.load(f)

# with open('decision_tree_model.pkl', 'rb') as f:
#     decision_tree_model = pickle.load(f)

# with open('random_forest_model.pkl', 'rb') as f:
#     random_forest_model = pickle.load(f)

# with open('svr_model.pkl', 'rb') as f:
#     svr_model = pickle.load(f)

# with open('knn_model.pkl', 'rb') as f:
#     knn_model = pickle.load(f)

# models = {
#     'Linear Regression': linear_regression_model,
#     'Decision Tree': decision_tree_model,
#     'Random Forest': random_forest_model,
#     'Support Vector Machine': svr_model,
#     'K-Nearest Neighbors': knn_model
# }

# # Hàm dự đoán
# def predict_gpa(model, input_data):
#     prediction = model.predict([input_data])
#     return prediction[0]

# # Triển khai giao diện Streamlit
# def main():
#     st.title('GPA Prediction')
    
#     # Input dữ liệu từ người dùng
#     study_time = st.slider('Study Time Weekly', 0, 40, 20)
#     absences = st.slider('Absences', 0, 30, 0)
#     gender = st.selectbox('Gender', ['Male', 'Female'])
#     ethnicity = st.selectbox('Ethnicity', ['Caucasian', 'African American', 'Asian', 'Other'])
#     parental_education = st.selectbox('Parental Education', ['None', 'High School', 'Some College', 'Bachelor\'s', 'Higher'])

#     # Chuyển đổi các giá trị phân loại sang định dạng số (OneHotEncoding)
#     gender_encoded = 1 if gender == 'Male' else 0
#     ethnicity_encoded = [1, 0, 0, 0]  # Thay bằng mã hóa tương ứng cho các lựa chọn khác nhau
#     parental_education_encoded = [0, 0, 0, 0, 1]  # Tương tự cho giáo dục của cha mẹ

#     # Kết hợp tất cả đầu vào thành một vector
#     input_data = [study_time, absences] + ethnicity_encoded + parental_education_encoded + [gender_encoded]

#     # Chọn mô hình dự đoán
#     model_choice = st.selectbox('Choose the regression model', list(models.keys()))

#     # Dự đoán GPA khi nhấn nút Predict
#     if st.button('Predict!'):
#         model = models[model_choice]
#         gpa = predict_gpa(model, input_data)
#         st.success(f'Predicted GPA: {gpa:.2f}')

# if __name__ == '__main__':
#     main()
