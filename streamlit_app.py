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



st.title("Phân Tích và Dự Báo Điểm GPA")

#with st.expander('Data'):
#     st.write('**Raw data**')
df = pd.read_csv('https://raw.githubusercontent.com/lam-01/Data/main/Student_performance_data_2.csv')
   #st.write(df)

# Hiển thị dữ liệu ban đầu
# st.subheader("Dữ liệu ban đầu")
# st.write(df.head())
# with st.expander('Data visualization'):
#   st.scatter_chart(data=df, x='Abesence', y='GPA', color='GradeClass')
#   st.scatter_chart(data=df, x='StudyTimeWeekly', y='GPA', color='GradeClass')

# Thực hiện One-Hot Encoding cho các biến phân loại
# st.subheader("Áp dụng One-Hot Encoding cho các biến phân loại")
cat_cols = ['Sports', 'Volunteering', 'ParentalSupport', 'Music', 'Extracurricular', 'ParentalEducation', 'Gender', 'Tutoring', 'Ethnicity']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# st.write("Dữ liệu sau khi áp dụng One-Hot Encoding:")
# st.write(df_encoded.head())

# Phân tách dữ liệu
# st.subheader("Phân tách dữ liệu")
X = df_encoded.drop(['GradeClass', 'StudentID'], axis=1)
y = df_encoded['GradeClass']

# st.write("Biến đầu vào (X):")
# st.write(X.head())

# st.write("Biến đầu ra (y):")
# st.write(y.head())

# Áp dụng SMOTE để cân bằng dữ liệu
smote = SMOTE(sampling_strategy='auto')
X_res, y_res = smote.fit_resample(X, y)


# st.write(f"Train set size after SMOTE: {X_res.shape[0]} samples")

# chia tập dữ liệu 
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# st.write(f"Train set size: {X_train.shape[0]} samples")
# st.write(f"Test set size: {X_test.shape[0]} samples")

 
# Xây dựng
with st.container:
    st.header('Input features')

    gender_map = {"Male": 0, "Female": 1}
    gender_selected = st.selectbox('Gender', ('Male', 'Female'))
    gender_encoded = gender_map[gender_selected]

    ethnicity_map = {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3}
    ethnicity_selected = st.selectbox('Ethnicity', ('Caucasian', 'African American', 'Asian', 'Other'))
    ethnicity_encoded = ethnicity_map[ethnicity_selected]

    parental_education_map = {"None": 0, "High School": 1, "Some College": 2, "Bachelor": 3, "Higher": 4}
    parental_education_selected = st.selectbox('ParentalEducation', ('None', 'High School', 'Some College', 'Bachelor', 'Higher'))
    parental_education_encoded = parental_education_map[parental_education_selected]

    tutoring_map = {"Yes": 1, "No": 0}
    tutoring_selected = st.selectbox('Tutoring', ('Yes', 'No'))
    tutoring_encoded = tutoring_map[tutoring_selected]

    parental_support_map = {"None": 0, "Low": 1, "Moderate": 2, "High": 3, "Very High": 4}
    parental_support_selected = st.selectbox('ParentalSupport', ('None', 'Low', 'Moderate', 'High', 'Very High'))
    parental_support_encoded = parental_support_map[parental_support_selected]

    extracurricular_map = {"Yes": 1, "No": 0}
    extracurricular_selected = st.selectbox('Extracurricular', ('Yes', 'No'))
    extracurricular_encoded = extracurricular_map[extracurricular_selected]

    volunteering_map = {"Yes": 1, "No": 0}
    volunteering_selected = st.selectbox('Volunteering', ('Yes', 'No'))
    volunteering_encoded = volunteering_map[volunteering_selected]

    study_time_weekly = st.number_input('Study Time Weekly (hours)', min_value=0, max_value=20)
    absences = st.number_input('Absences', min_value=0, max_value=30)

    # Create a DataFrame for the input features
    data = {
        'Gender': gender_encoded,
        'Ethnicity': ethnicity_encoded,
        'ParentalEducation': parental_education_encoded,
        'Tutoring': tutoring_encoded,
        'ParentalSupport': parental_support_encoded,
        'Extracurricular': extracurricular_encoded,
        'Volunteering': volunteering_encoded,
        'StudyTimeWeekly': study_time_weekly,
        'Absences': absences,
       
    }


input_df = pd.DataFrame(data, index=[0])
input_penguins = pd.concat([input_df, X], axis=0)

# with st.expander('Input features'):
#   st.write('**Input data**')
#   st.dataframe(input_df)
#   st.write('**Combined data**')
#   st.dataframe(input_penguins)

# Mô hình 
clf = RandomForestRegressor(max_depth=2, random_state=42)
clf.fit(X_train, y_train)

# ## Hàm dự đoán 
# # Dự đoán giá trị đầu ra
# predictions = clf.predict(X_test)  
# # Hiển thị kết quả
# st.write(f"Dự đoán: {predictions}")
# Hàm dự đoán
def predict_gpa(mode, X_test):
    prediction = mode.predict(X_test)
    return prediction[0]
# Hàm chuyển đổi GPA sang GradeClass
def gpa_to_grade_class(gpa):
    if gpa >= 3.5:
        return 'A'
    elif gpa >= 3.0:
        return 'B'
    elif gpa >= 2.5:
        return 'C'
    elif gpa >= 2.0:
        return 'D'
    else:
        return 'F'

# Dự đoán GPA khi nhấn nút Predict
if st.button('Predict GPA'):
    gpa_prediction = predict_gpa(clf, X_test)
    grade_class = gpa_to_grade_class(gpa_prediction)
    st.success(f'Predicted GPA: {gpa_prediction:.2f}')
    st.success(f'Grade Class: {grade_class}')

