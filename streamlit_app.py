import streamlit as st
import pandas as pd

st.title('Dự đoán kết quả học tập của học sinh ')

st.info('Đây là app dự đoán kết quả')
with st.expander('Data'):
     st.write('**Raw data**')
df=pd.read_csv('https://raw.githubusercontent.com/lam-01/Data/main/Student_performance_data_2.csv' )
df

