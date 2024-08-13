import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
#from imblearn.over_sampling import SMOTE

# Tiêu đề ứng dụng
st.title("Phân Tích và Dự Báo Điểm GPA")

# Tải dữ liệu
with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/lam-01/Data/main/Student_performance_data_2.csv')
  df

  st.write('**X**')
  X_raw = df.drop('species', axis=1)
  X_raw
# df = load_data()

# Hiển thị dữ liệu ban đầu
st.subheader("Dữ liệu ban đầu")
st.write(df.head())

# Phân tách dữ liệu
st.subheader("Phân tách dữ liệu")
X = df[['StudyTimeWeekly', 'Absences']]
y = df['GPA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write(f"Train set size: {X_train.shape[0]} samples")
st.write(f"Test set size: {X_test.shape[0]} samples")

# Áp dụng SMOTE để cân bằng dữ liệu
smote = SMOTE(sampling_strategy='auto')
X_res, y_res = smote.fit_resample(X_train, y_train)

st.write(f"Train set size after SMOTE: {X_res.shape[0]} samples")

# Huấn luyện mô hình Linear Regression
lr = LinearRegression()
lr.fit(X_res, y_res)
y_pred = lr.predict(X_test)

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
