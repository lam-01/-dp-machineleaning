#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Brand Mobile Classification Model

Using HOG (Histogram of Oriented Gradients) + SVM

"""


# ============ Cell 3 ============

from google.colab import drive
drive.mount('/content/drive')



# ============ Cell 4 ============

import pandas as pd
data =pd.read_csv('/content/drive/MyDrive/data2.csv')
print(data)



# ============ Cell 5 ============

import pandas as pd

# Kiểm tra các cột trong file
print("Các cột trong file CSV:", data.columns)

# Giả sử cột nhãn có tên là 'Label', liệt kê các nhãn
if 'brand' in data.columns:
    labels = data['brand'].unique()
    print("Danh sách các nhãn:", labels)
else:
    print("Không tìm thấy cột 'Label' trong file CSV. Hãy kiểm tra tên cột.")



# ============ Cell 6 ============

import os
import requests

# Tạo thư mục để lưu hình ảnh
os.makedirs('images', exist_ok=True)

# Sắp xếp dữ liệu theo index để đảm bảo ảnh được tải theo thứ tự
data_sorted = data.sort_index()

# Tải hình ảnh từ các liên kết trong cột "images_links"
for index, row in data_sorted.iterrows():
    image_url = row['image']
    response = requests.get(image_url)
    if response.status_code == 200:
        # Đảm bảo tên file có định dạng sắp xếp tự nhiên (ví dụ: 001.jpg, 002.jpg)
        image_filename = os.path.join('images', f'image_{str(index).zfill(4)}.jpg')
        with open(image_filename, 'wb') as image_file:
            image_file.write(response.content)
        print(f"Downloaded: {image_filename}")
    else:
        print(f"Failed to download: {image_url}")

# Kiểm tra file trong thư mục và in ra theo thứ tự
files_in_directory = sorted(os.listdir('images'))
print("Files in directory (sorted):", files_in_directory)



# ============ Cell 7 ============

import os
import numpy as np
from skimage import io, transform
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian
import cv2
import random

def create_augmented_data(image_dir, output_dir):
    """Tạo dữ liệu tăng cường từ ảnh gốc"""
    os.makedirs(output_dir, exist_ok=True)

    # Đọc tất cả các file ảnh
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

    for img_file in image_files:
        # Đọc ảnh
        img_path = os.path.join(image_dir, img_file)
        img = io.imread(img_path)

        # Tên file gốc không có phần mở rộng
        base_name = os.path.splitext(img_file)[0]

        # 1. Xoay ảnh
        angles = [90, 180, 270]
        for angle in angles:
            rotated = rotate(img, angle, resize=True)
            io.imsave(os.path.join(output_dir, f"{base_name}_rotate_{angle}.jpg"),
                     (rotated * 255).astype(np.uint8))

        # 2. Lật ảnh
        flipped_h = np.fliplr(img)
        flipped_v = np.flipud(img)
        io.imsave(os.path.join(output_dir, f"{base_name}_flip_h.jpg"), flipped_h)
        io.imsave(os.path.join(output_dir, f"{base_name}_flip_v.jpg"), flipped_v)

        # 3. Thay đổi độ sáng
        brightness_factors = [0.7, 1.3]  # Giảm và tăng độ sáng
        for factor in brightness_factors:
            brightened = np.clip(img * factor, 0, 255).astype(np.uint8)
            io.imsave(os.path.join(output_dir, f"{base_name}_bright_{factor}.jpg"),
                     brightened)

        # 4. Thêm nhiễu Gaussian
        noisy = random_noise(img, mode='gaussian', var=0.01)
        io.imsave(os.path.join(output_dir, f"{base_name}_noise.jpg"),
                 (noisy * 255).astype(np.uint8))

        # 5. Làm mờ ảnh
        blurred = gaussian(img, sigma=1)
        io.imsave(os.path.join(output_dir, f"{base_name}_blur.jpg"),
                 (blurred * 255).astype(np.uint8))

        # 6. Cắt ngẫu nhiên (random crop)
        h, w = img.shape[:2]
        crop_size = (int(h * 0.8), int(w * 0.8))
        start_h = np.random.randint(0, h - crop_size[0])
        start_w = np.random.randint(0, w - crop_size[1])
        cropped = img[start_h:start_h + crop_size[0],
                     start_w:start_w + crop_size[1]]
        io.imsave(os.path.join(output_dir, f"{base_name}_crop.jpg"), cropped)

        # 7. Thay đổi tỷ lệ (scale)
        scales = [0.8, 1.2]
        for scale in scales:
            scaled = transform.resize(img,
                                   (int(h * scale), int(w * scale)),
                                   mode='reflect')
            io.imsave(os.path.join(output_dir, f"{base_name}_scale_{scale}.jpg"),
                     (scaled * 255).astype(np.uint8))

        # 8. Biến dạng phối cảnh (perspective transform)
        def random_perspective_transform(image):
            h, w = image.shape[:2]
            src_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
            dst_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])

            # Thêm nhiễu vào các điểm đích
            dst_points += np.random.uniform(-50, 50, size=dst_points.shape)

            # Tính ma trận biến đổi
            transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            warped = cv2.warpPerspective(image, transform_matrix, (w, h))
            return warped

        perspective = random_perspective_transform(img)
        io.imsave(os.path.join(output_dir, f"{base_name}_perspective.jpg"),
                 perspective)

def create_augmented_dataframe(original_df, augmented_image_dir, max_augmented_per_image=5):
    """Tạo DataFrame mới bao gồm cả dữ liệu gốc và tập con dữ liệu tăng cường."""
    augmented_files = os.listdir(augmented_image_dir)
    new_rows = []

    for idx, row in original_df.iterrows():
        base_name = f"image_{str(idx).zfill(4)}"
        # Thêm dữ liệu gốc
        new_rows.append({
            'brand': row['brand'],
            'image_path': os.path.join('images', f'{base_name}.jpg')
        })

        # Thêm dữ liệu tăng cường
        aug_files = [f for f in augmented_files if f.startswith(base_name)]

        # Lựa chọn ngẫu nhiên một số lượng nhất định ảnh tăng cường
        selected_aug_files = random.sample(aug_files, min(len(aug_files), max_augmented_per_image))

        for aug_file in selected_aug_files:
            new_rows.append({
                'brand': row['brand'],
                'image_path': os.path.join(augmented_image_dir, aug_file)
            })

    return pd.DataFrame(new_rows)

# Sử dụng các hàm
if __name__ == "__main__":
    # Tạo thư mục cho dữ liệu tăng cường
    augmented_dir = 'augmented_images'

    # Tạo dữ liệu tăng cường
    create_augmented_data('images', augmented_dir)

    # Tạo DataFrame mới với dữ liệu tăng cường
    augmented_df = create_augmented_dataframe(data, augmented_dir)

    # In thông tin về kích thước dữ liệu mới
    print(f"Kích thước dữ liệu gốc: {len(data)}")
    print(f"Kích thước dữ liệu sau khi tăng cường: {len(augmented_df)}")



# ============ Cell 8 ============

# Lưu DataFrame mới vào tệp CSV
    augmented_df.to_csv('augmented_data.csv', index=False)



# ============ Cell 9 ============

import pandas as pd
import numpy as np
from skimage.feature import hog
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import transform
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Bước 1: Đọc tệp CSV và chia tập dữ liệu
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['brand'])
    return train_df, test_df


# Bước 2: Hàm xử lý ảnh và trích xuất đặc trưng HOG
def process_image(image_path):
    img = imread(image_path)
    if len(img.shape) == 3:  # Nếu ảnh có 3 kênh (RGB hoặc RGBA), chuyển về grayscale
        img = rgb2gray(img)
    img_resized = transform.resize(img, (128, 128))  # Resize ảnh
    features, _ = hog(img_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                      visualize=True)
    return features

# Trích xuất đặc trưng HOG từ DataFrame
def extract_features(df):
    features = []
    labels = []
    for _, row in df.iterrows():
        img_path = row['image_path']
        label = row['brand']
        hog_features = process_image(img_path)
        features.append(hog_features)
        labels.append(label)
    return np.array(features), np.array(labels)

# Bước 3: Huấn luyện mô hình SVM
def train_svm(X_train, y_train):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    # svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    # svm_model= SVC(kernel='rbf', C=100, gamma=0.01 )
    svm_model = SVC(kernel='poly', C=100, gamma=1)
    svm_model.fit(X_train, y_train_encoded)
    return svm_model, label_encoder

# Bước 4: Đánh giá mô hình
def evaluate_model(svm_model, label_encoder, X_test, y_test):
    y_test_encoded = label_encoder.transform(y_test)
    y_pred = svm_model.predict(X_test)
    y_test_labels = label_encoder.inverse_transform(y_test_encoded)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    print(classification_report(y_test_labels, y_pred_labels))
    print(f"Độ chính xác: {accuracy_score(y_test_labels, y_pred_labels):.4f}")

# Bước 5: Lưu mô hình
def save_model(svm_model, label_encoder, model_path="svm_model.pkl", encoder_path="label_encoder.pkl"):
    joblib.dump(svm_model, model_path)
    joblib.dump(label_encoder, encoder_path)
    print(f"Mô hình đã được lưu vào {model_path} và {encoder_path}")

#Bước 6: Triển khai mô hình để dự đoán ảnh mới
def predict_new_image(image_path, model_path="svm_model.pkl", encoder_path="label_encoder.pkl"):
    svm_model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    features = process_image(image_path)
    prediction = svm_model.predict([features])
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

# Chạy toàn bộ pipeline
if __name__ == "__main__":
    # Tải dữ liệu và chia tập
    train_df, test_df = load_data("augmented_data.csv")

    # Trích xuất đặc trưng
    X_train, y_train = extract_features(train_df)
    X_test, y_test = extract_features(test_df)

    # Huấn luyện mô hình
    svm_model, label_encoder = train_svm(X_train, y_train)
    print("Mô hình SVM đã được huấn luyện.")

    # Đánh giá mô hình
    print("\nĐánh giá trên tập kiểm tra:")
    evaluate_model(svm_model, label_encoder, X_test, y_test)

    # Lưu mô hình
    save_model(svm_model, label_encoder)

    # Dự đoán cho ảnh mới
    new_image_path = "/content/augmented_images/image_0000_blur.jpg"
    predicted_brand = predict_new_image(new_image_path)
    print(f"Hãng điện thoại dự đoán: {predicted_brand}")



# ============ Cell 10 ============

# # Khởi tạo mô hình với tham số tùy chỉnh
# svm_model = SVC(kernel='poly', C=10, gamma=0.01)
# svm_model.fit(X_train, y_train)

# # Đánh giá hiệu suất
# y_pred = svm_model.predict(X_test)
# print(classification_report(y_test, y_pred))
# print(f"Độ chính xác: {accuracy_score(y_test, y_pred):.4f}")



# ============ Cell 11 ============

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC # Import the SVC class

# Tạo lưới tham số
param_grid = {
    'C': [0.1, 1],
    'kernel': ['linear', 'rbf'],
    'gamma': [1, 0.1]
}

# Khởi tạo mô hình SVM và GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5)
grid.fit(X_train, y_train)

# Tham số tốt nhất
print("Best Parameters:", grid.best_params_)
print("Best Cross-validation Score:", grid.best_score_)

# Dự đoán và đánh giá
y_pred = grid.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))



# ============ Cell 12 ============

import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.filters import gaussian

# Đường dẫn đến ảnh mẫu
sample_image_path = "/content/images/image_0000.jpg"

# 1. Đọc ảnh
original_image = imread(sample_image_path)

# 2. Chuyển đổi sang ảnh xám
gray_image = rgb2gray(original_image)

# 3. Chuẩn hóa kích thước ảnh
resized_image = resize(gray_image, (128, 128))

# 4. Loại bỏ nhiễu bằng Gaussian filter
denoised_image = gaussian(resized_image, sigma=1)

# Hiển thị kết quả
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
axes[0].imshow(original_image)
axes[0].set_title("Ảnh gốc")
axes[0].axis("off")

axes[1].imshow(gray_image, cmap="gray")
axes[1].set_title("Ảnh xám")
axes[1].axis("off")

axes[2].imshow(resized_image, cmap="gray")
axes[2].set_title("Ảnh chuẩn hóa kích thước")
axes[2].axis("off")

axes[3].imshow(denoised_image, cmap="gray")
axes[3].set_title("Ảnh sau loại bỏ nhiễu")
axes[3].axis("off")

plt.tight_layout()
plt.show()



# ============ Cell 13 ============

import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import transform

# Bước 2: Hàm xử lý ảnh và trích xuất đặc trưng HOG
def process_image(image_path):
    img = imread(image_path)
    if len(img.shape) == 3:
        img = rgb2gray(img)
    img_resized = transform.resize(img, (128, 128))
    features, hog_image = hog(img_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                              visualize=True)
    return img_resized, hog_image, features

# Hiển thị ảnh gốc, ảnh HOG và các đặc trưng
def display_images_and_features(image_path):
    # Trích xuất đặc trưng HOG và ảnh gốc
    img_resized, hog_image, features = process_image(image_path)

    # Tạo hình ảnh để hiển thị
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax = axes.ravel()

    # Hiển thị ảnh gốc
    ax[0].imshow(img_resized, cmap=plt.cm.gray)
    ax[0].set_title("Ảnh Gốc")
    ax[0].axis('off')

    # Hiển thị ảnh HOG
    ax[1].imshow(hog_image, cmap=plt.cm.gray)
    ax[1].set_title("Ảnh với Đặc Trưng HOG")
    ax[1].axis('off')

    plt.show()

    # In ra các đặc trưng HOG
    print("Đặc Trưng HOG:")
    print(features)

# Đường dẫn đến ảnh mà bạn muốn xử lý
image_path = '/content/images/image_0000.jpg'  # Thay bằng đường dẫn ảnh của bạn

# Hiển thị ảnh gốc, ảnh HOG và in ra các đặc trưng HOG
display_images_and_features(image_path)



# ============ Cell 14 ============

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import os

# Bước 1: Đọc tệp CSV và chia tập dữ liệu
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['brand'])
    return train_df, test_df

# Bước 2: Xử lý ảnh và chuyển đổi thành mảng cho CNN
def process_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)  # Chuyển đổi ảnh thành mảng numpy
    img_array = img_array / 255.0  # Chuẩn hóa giá trị pixel từ [0, 255] thành [0, 1]
    return img_array

# Trích xuất ảnh từ DataFrame
def extract_images(df):
    images = []
    labels = []
    for _, row in df.iterrows():
        img_path = row['image_path']
        label = row['brand']
        img_array = process_image(img_path)
        images.append(img_array)
        labels.append(label)
    return np.array(images), np.array(labels)

# Bước 3: Xây dựng mô hình CNN
def build_cnn_model(input_shape=(128, 128, 3), num_classes=0):
    model = Sequential()

    # Lớp Convolutional 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Lớp Convolutional 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Lớp Convolutional 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # Flatten để đưa vào lớp fully connected
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout để giảm overfitting
    model.add(Dense(num_classes, activation='softmax'))  # Lớp đầu ra với softmax

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Bước 4: Huấn luyện mô hình CNN
def train_cnn(X_train, y_train, X_val, y_val, input_shape=(128, 128, 3), num_classes=10, epochs=10, batch_size=32):
    cnn_model = build_cnn_model(input_shape=input_shape, num_classes=num_classes)
    cnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    return cnn_model

# Bước 5: Đánh giá mô hình
def evaluate_model(cnn_model, X_test, y_test, label_encoder):
    y_pred = cnn_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Chuyển đổi xác suất thành nhãn dự đoán
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred_classes)
    print(classification_report(y_test_labels, y_pred_labels))
    print(f"Độ chính xác: {accuracy_score(y_test_labels, y_pred_labels):.4f}")

# Chạy toàn bộ pipeline
if __name__ == "__main__":
    # Tải dữ liệu và chia tập
    train_df, test_df = load_data("augmented_data.csv")

    # Trích xuất ảnh và nhãn
    X_train, y_train = extract_images(train_df)
    X_test, y_test = extract_images(test_df)

    # Chuẩn hóa nhãn
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Huấn luyện mô hình CNN
    print("Đang huấn luyện mô hình CNN...")
    cnn_model = train_cnn(X_train, y_train_encoded, X_test, y_test_encoded, epochs=10)

    # Đánh giá mô hình
    print("\nĐánh giá trên tập kiểm tra:")
    evaluate_model(cnn_model, X_test, y_test_encoded, label_encoder)



# ============ Cell 15 ============

import pandas as pd
import numpy as np
from skimage.feature import hog
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import transform
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Bước 1: Đọc tệp CSV và chia tập dữ liệu
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['brand'])
    return train_df, test_df


# Bước 2: Hàm xử lý ảnh và trích xuất đặc trưng HOG
def process_image(image_path):
    img = imread(image_path)
    if len(img.shape) == 3:  # Nếu ảnh có 3 kênh (RGB hoặc RGBA), chuyển về grayscale
        img = rgb2gray(img)
    img_resized = transform.resize(img, (128, 128))  # Resize ảnh
    features, _ = hog(img_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                      visualize=True)
    return features

# Trích xuất đặc trưng HOG từ DataFrame
def extract_features(df):
    features = []
    labels = []
    for _, row in df.iterrows():
        img_path = row['image_path']
        label = row['brand']
        hog_features = process_image(img_path)
        features.append(hog_features)
        labels.append(label)
    return np.array(features), np.array(labels)

# Bước 3: Tuning các tham số SVM bằng GridSearchCV
def tune_svm(X_train, y_train):
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 1.0]
    }
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    print(f"Tối ưu tham số: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Bước 4: Huấn luyện mô hình SVM với tham số tối ưu
def train_svm(X_train, y_train):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Tuning và huấn luyện mô hình SVM với GridSearchCV
    svm_model = tune_svm(X_train, y_train_encoded)

    return svm_model, label_encoder

# Bước 5: Đánh giá mô hình
def evaluate_model(svm_model, label_encoder, X_test, y_test):
    y_test_encoded = label_encoder.transform(y_test)
    y_pred = svm_model.predict(X_test)
    y_test_labels = label_encoder.inverse_transform(y_test_encoded)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    print(classification_report(y_test_labels, y_pred_labels))
    print(f"Độ chính xác: {accuracy_score(y_test_labels, y_pred_labels):.4f}")

# Bước 6: Đánh giá mô hình với cross-validation
def cross_validate_model(svm_model, X_train, y_train):
    scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Độ chính xác trung bình từ Cross-Validation: {scores.mean():.4f}")

# Chạy toàn bộ pipeline
if __name__ == "__main__":
    # Tải dữ liệu và chia tập
    train_df, test_df = load_data("augmented_data.csv")

    # Trích xuất đặc trưng
    X_train, y_train = extract_features(train_df)
    X_test, y_test = extract_features(test_df)

    # Huấn luyện mô hình với tối ưu hóa tham số
    svm_model, label_encoder = train_svm(X_train, y_train)
    print("Mô hình SVM đã được huấn luyện với tham số tối ưu.")

    # Đánh giá mô hình trên tập kiểm tra
    print("\nĐánh giá trên tập kiểm tra:")
    evaluate_model(svm_model, label_encoder, X_test, y_test)

    # Đánh giá mô hình với cross-validation
    print("\nĐánh giá mô hình với Cross-Validation:")
    cross_validate_model(svm_model, X_train, y_train)

    # # Lưu mô hình
    # save_model(svm_model, label_encoder)

    # # Dự đoán cho ảnh mới
    # new_image_path = "/content/augmented_images/image_0000_blur.jpg"
    # predicted_brand = predict_new_image(new_image_path)
    # print(f"Hãng điện thoại dự đoán: {predicted_brand}")



# ============ Cell 16 ============

from sklearn.model_selection import RandomizedSearchCV

# Tạo dải tham số
param_distributions = {
    'C': np.logspace(-3, 3, 10),
    'kernel': ['linear', 'rbf'],
    'gamma': np.logspace(-3, 3, 10)
}

# Khởi tạo RandomizedSearchCV
random_search = RandomizedSearchCV(SVC(), param_distributions, n_iter=20, random_state=42, cv=5)
random_search.fit(X_train, y_train)

# Tham số tốt nhất
print("Best Parameters:", random_search.best_params_)
print("Best Cross-validation Score:", random_search.best_score_)

# Dự đoán và đánh giá
y_pred = random_search.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))



# ============ Cell 17 ============

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

label_counts = Counter(y_train)  # Đếm số lượng từng nhãn
labels, counts = zip(*label_counts.items())  # Tách nhãn và số lượng

# Chuyển đổi sang DataFrame để dễ sử dụng
df_labels = pd.DataFrame({'Label': labels, 'Count': counts})

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
sns.barplot(data=df_labels, x='Label', y='Count', palette='viridis')
plt.title('Số lượng nhãn trước khi tăng cường dữ liệu', fontsize=16)
plt.xlabel('Nhãn', fontsize=14)
plt.ylabel('Số lượng', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()



# ============ Cell 18 ============

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

# Đếm số lượng nhãn
original_counts = Counter(y_train)  # Trước khi tăng cường


augmented_y = augmented_df['brand'].values
augmented_counts = Counter(augmented_y)  # Sau khi tăng cường
# Chuyển đổi sang DataFrame
labels_original, counts_original = zip(*original_counts.items())
labels_augmented, counts_augmented = zip(*augmented_counts.items())

df_original = pd.DataFrame({'Label': labels_original, 'Count': counts_original, 'Type': 'Trước khi tăng cường'})
df_augmented = pd.DataFrame({'Label': labels_augmented, 'Count': counts_augmented, 'Type': 'Sau khi tăng cường'})

# Kết hợp hai DataFrame
df_combined = pd.concat([df_original, df_augmented])

# Vẽ biểu đồ ghép
plt.figure(figsize=(12, 8))
sns.barplot(data=df_combined, x='Label', y='Count', hue='Type', palette='viridis')
plt.title('So sánh số lượng nhãn trước và sau khi tăng cường', fontsize=16)
plt.xlabel('Nhãn', fontsize=14)
plt.ylabel('Số lượng', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Loại dữ liệu', fontsize=12)
plt.show()



# ============ Cell 19 ============

import matplotlib.pyplot as plt
from PIL import Image
import os

# Lấy danh sách file trong thư mục và sắp xếp theo thứ tự bảng chữ cái
sample_images = sorted(os.listdir('images'))[:5]  # Lấy 5 ảnh đầu tiên theo thứ tự

# Hiển thị các ảnh
fig, axes = plt.subplots(1, len(sample_images), figsize=(15, 5))
for ax, image_file in zip(axes, sample_images):
    img_path = os.path.join('images', image_file)
    img = Image.open(img_path)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(image_file)
plt.show()



# ============ Cell 20 ============

import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import io, color, transform

# Hàm trích xuất đặc trưng HOG và trả về cả visualization
def extract_hog_features_with_visualization(image_path):
    image = io.imread(image_path)
    if image.ndim == 3:  # Chuyển sang ảnh xám nếu là ảnh màu
        image = color.rgb2gray(image)
    # Resize image to a fixed size (e.g., 128x128)
    image = transform.resize(image, (128, 128))
    features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                              cells_per_block=(1, 1), visualize=True)
    return image, features, hog_image

# Chọn một số ảnh để hiển thị
sample_images = sorted(os.listdir('images'))[:3]  # Lấy 3 ảnh đầu tiên

# Lấy nhãn từ dữ liệu gốc
image_labels = data_sorted['brand'].tolist()

# Hiển thị ảnh gốc và ảnh HOG
fig, axes = plt.subplots(len(sample_images), 2, figsize=(10, 10))
for i, image_file in enumerate(sample_images):
    img_path = os.path.join('images', image_file)
    original_image, _, hog_image = extract_hog_features_with_visualization(img_path)

    # Hiển thị ảnh gốc
    axes[i, 0].imshow(original_image, cmap='gray')
    axes[i, 0].axis('off')
    # Thêm nhãn vào tiêu đề ảnh gốc
    label = image_labels[i]  # Nhãn tương ứng với thứ tự ảnh
    axes[i, 0].set_title(f"Original: {image_file}\nLabel: {label}")

    # Hiển thị ảnh HOG
    axes[i, 1].imshow(hog_image, cmap='gray')
    axes[i, 1].axis('off')
    axes[i, 1].set_title("HOG Visualization")

plt.tight_layout()
plt.show()



# ============ Cell 21 ============

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

    # Chuyển đổi nhãn thành dạng one-hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Đảm bảo rằng đầu vào là dạng (số lượng mẫu, chiều rộng, chiều cao, số kênh)
    X_train = X_train.reshape(-1, 128, 128, 1)  # Giả sử ảnh có kích thước 128x128 và là ảnh xám
    X_test = X_test.reshape(-1, 128, 128, 1)

    return X_train, X_test, y_train, y_test

# Bước 2: Xây dựng mô hình CNN
def build_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout để tránh overfitting
    model.add(Dense(5, activation='softmax'))  # Giả sử có 5 lớp (các hãng điện thoại)

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Bước 3: Huấn luyện mô hình
def train_cnn_model(X_train, y_train, X_test, y_test):
    model = build_cnn_model()
    model.summary()  # Hiển thị kiến trúc mô hình
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    return model, history

# Bước 4: Đánh giá mô hình
def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc

# Main
csv_file = 'augmented_data.csv'  # Thay thế với đường dẫn tới tệp CSV của bạn
X_train, X_test, y_train, y_test = load_and_preprocess_data(csv_file)

# Huấn luyện mô hình CNN
model, history = train_cnn_model(X_train, y_train, X_test, y_test)

# Đánh giá mô hình trên tập kiểm tra
evaluate_model(model, X_test, y_test)



# ============ Cell 22 ============

# Dự đoán trên một ảnh mới
new_image_path = '/content/images/image_0000.jpg'
_, new_features, _ = extract_hog_features_with_visualization(new_image_path)
prediction = model.predict([new_features])
print("Predicted label:", prediction[0])
