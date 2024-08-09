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
  st.scatter_chart(data=df, x='Absences', y='GPA',color='GradeClass')
  st.scatter_chart(data=df, x='StudyTimeWeekly', y='GPA',color='GradeClass')


with st.sidebar:
  st.header('Input features')
   Gender = st.selectbox('Gender', ( 'Male', 'Famale'))
   Ethnicity = st.selectbox('Ethnicity',( 'Caucasian', 'African American', 'Asian', 'Other'))
   ParentalEducation = st.selectbox('ParentalEducation', ('None', 'High School', 'Some College', 'Bachelor, 'Higher')
   Tutoring = st.selectbox('Tutoring',( 'Yes','No'))
   ParentalSupport = st.selectbox('ParentalSupport',( 'None', 'Low', 'Moderate', 'High', 'Very High'))
   Extracurricular = st.selectbox('Extracurricular',( 'Yes','No'))
   Volunteering = st.selectbox('Volunteering',( 'Yes','No'))
  
  
  # Create a DataFrame for the input features
  data = {'sex': Gender,
          'Ethnicity': Ethinicity,
          'ParentalEducation': ParentalEducation,
          'Tutoring': Tutoring,
          'ParentalSupport': ParentalSupport,
          'Extracurricular': Extracurricular,
          'Volunteering': Volunteering,
          
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, X], axis=0)

with st.expander('Input features'):
  st.write('**Input penguin**')
  input_df
  st.write('**Combined penguins data**')
  input_penguins
  


