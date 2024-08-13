import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
# import spacy

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer

# from gensim import models
# from gensim.models import Phrases
# from gensim.models.phrases import Phraser

# Tiêu đề ứng dụng
st.title("Phân Tích và Dự Báo Điểm GPA")

# Tải dữ liệu

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/lam-01/Data/main/Student_performance_data_2.csv')
  df

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

# # Huấn luyện mô hình Linear Regression
# lr = LinearRegression()
# lr.fit(X_res, y_res)
# y_pred = lr.predict(X_test)


# # Mô hình tiền xử lý ngôn ngữ của Spacy (dùng cho kỹ thuật Lemmatization)
# with open('spacy_nlp.pkl', 'rb') as f:
#     spacy_nlp = pickle.load(f)

# # Mô hình tạo các từ ghép 2 chữ (Bigrams)
# with open('bigrams_phraser.pkl', 'rb') as f:
#     bigrams_phraser = pickle.load(f)
    
# # Mô hình tạo các từ ghép 3 chữ (Trigrams)
# with open('trigrams_phraser.pkl', 'rb') as f:
#     trigrams_phraser = pickle.load(f)

# # Mô hình vectơ hóa bằng kỹ thuật Bag of Word
# with open('bow_vectorizer.pkl', 'rb') as f:
#     bow_vectorizer = pickle.load(f)
    
# # Mô hình vectơ hóa bằng kỹ thuật TF-IDF
# with open('tfidf_vectorizer.pkl', 'rb') as f:
#     tfidf_vectorizer = pickle.load(f)

# ######################################### LOAD MÔ HÌNH ###########################################

# # Load các mô hình hồi quy đã huấn luyện
# with open('linear_regression_model.pkl', 'rb') as f:
#     linear_regression_model = pickle.load(f)

# with open('random_forest_model.pkl', 'rb') as f:
#     random_forest_model = pickle.load(f)

# with open('svr_model.pkl', 'rb') as f:
#     svr_model = pickle.load(f)

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
#     gender = 1 if gender == 'Male' else 0
#     ethnicity_encoded = [1, 0, 0, 0]  # Thay bằng mã hóa tương ứng cho các lựa chọn khác nhau
#     parental_education_encoded = [0, 0, 0, 0, 1]  # Tương tự cho giáo dục của cha mẹ

#     # Kết hợp tất cả đầu vào thành một vector
#     input_data = [study_time, absences] + ethnicity_encoded + parental_education_encoded + [gender]

#     # Chọn mô hình dự đoán
#     model_choice = st.selectbox('Choose the regression model', ('Linear Regression', 'Random Forest', 'SVR'))

#     # Dự đoán GPA khi nhấn nút Predict
#     if st.button('Predict!'):
#         if model_choice == 'Linear Regression':
#             gpa = predict_gpa(linear_regression_model, input_data)
#         elif model_choice == 'Random Forest':
#             gpa = predict_gpa(random_forest_model, input_data)
#         else:
#             gpa = predict_gpa(svr_model, input_data)
        
#         st.success(f'Predicted GPA: {gpa:.2f}')

# if __name__ == '__main__':
#     main()

# Hiển thị kết quả
st.subheader("Kết quả Linear Regression")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"Mean Squared Error: {mse}")
st.write(f"R2 Score: {r2}")

# Huấn luyện mô hình Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_res, y_res)
y_pred_rf = rf.predict(X_test)

# Hiển thị kết quả
st.subheader("Kết quả Random Forest")
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
st.write(f"Mean Squared Error: {mse_rf}")
st.write(f"R2 Score: {r2_rf}")

# Trực quan hóa dữ liệu
st.subheader("Trực quan hóa dữ liệu")

fig, ax = plt.subplots()
sns.scatterplot(x=X['Absences'], y=y, ax=ax)
plt.title('Absences vs GPA')
st.pyplot(fig)


