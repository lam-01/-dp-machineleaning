import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🤖 Machine Learning App')

st.info('This is app builds a machine learning model!')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/lam-01/Data/main/Student_performance_data_2.csv')
  df


st.write('**X**')
X=df.drop('StudentID',axis=1)
X

st.write('**y**')
y=df.StudentID
y

with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='Absences', y='GPA', color='GradeClass')
  st.scatter_chart(data=df, x='StudyTimeWeekly', y='GPA', color='GradeClass')

with st.sidebar:
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

with st.expander('Input features'):
    st.write('**Input data**')
    st.dataframe(input_df)
    st.write('**Combined data**')
    st.dataframe(input_penguins)
# Data preparation
# Encode categorical variables
encode = ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']

# Check if all columns in 'encode' are present in input_data_combined
missing_cols = [col for col in encode if col not in input_data_combined.columns]
if missing_cols:
    st.error(f"Columns missing in input_data_combined: {missing_cols}")
else:
    df_encoded = pd.get_dummies(input_data_combined, columns=encode)

    # Proceed with the rest of the code
    X = df_encoded[1:]  # Using all rows except the first one (input row)
    input_row = df_encoded[:1]  # The first row is the input row

    with st.expander('Data preparation'):
        st.write('**Encoded X (input student data)**')
        input_row


