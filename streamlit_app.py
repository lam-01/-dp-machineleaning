import streamlit as st
import pandas as pd

st.title('Dự đoán kết quả học tập của học sinh ')

st.write('Đây là app dự đoán kết quả')
df=pd.read_csv('https://raw.githubusercontent.com/lam-01/Data/main/Student_performance_data_2.csv' )
df
