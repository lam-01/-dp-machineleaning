# import streamlit as st
# import pandas as pd

# st.title('Dự đoán kết quả học tập của học sinh ')

# st.write('Đây là app dự đoán kết quả')
# df=pd.read_csv('https://raw.githubusercontent.com/lam-01/Data/main/Student_performance_data_2.csv' )
# df
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title('Data Exploration and Visualization')

# Load data
@st.cache_data
def load_data():
    # Giả sử có một file CSV chứa dữ liệu bạn đã xử lý
    df = pd.read_csv('https://raw.githubusercontent.com/lam-01/Data/main/Student_performance_data_2.csv')
    return df

df = load_data()

# Show raw data
if st.checkbox('Show raw data'):
    st.subheader('Raw Data')
    st.write(df)

# Sidebar options
st.sidebar.header('Visualization Settings')
plot_type = st.sidebar.selectbox('Choose plot type', ['Histogram', 'Boxplot', 'Scatter Plot', 'Bar Plot'])

# Select column for visualization
col_to_plot = st.sidebar.selectbox('Choose column to plot', df.columns)

# Visualization based on user input
if plot_type == 'Histogram':
    st.subheader(f'Histogram of {col_to_plot}')
    fig, ax = plt.subplots()
    sns.histplot(df[col_to_plot], kde=True, ax=ax)
    st.pyplot(fig)

elif plot_type == 'Boxplot':
    st.subheader(f'Boxplot of {col_to_plot}')
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col_to_plot], ax=ax)
    st.pyplot(fig)

elif plot_type == 'Scatter Plot':
    st.subheader(f'Scatter Plot of {col_to_plot} vs GPA')
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[col_to_plot], y=df['GPA'], ax=ax)
    st.pyplot(fig)

elif plot_type == 'Bar Plot':
    st.subheader(f'Bar Plot of {col_to_plot}')
    fig, ax = plt.subplots()
    sns.countplot(x=df[col_to_plot], ax=ax)
    st.pyplot(fig)

# Correlation heatmap
if st.sidebar.checkbox('Show Correlation Heatmap'):
    st.subheader('Correlation Heatmap')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
