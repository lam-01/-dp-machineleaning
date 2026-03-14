#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brand Mobile Classification Model
Using HOG (Histogram of Oriented Gradients) + SVM
Converted from Jupyter Notebook
"""

import os
import numpy as np
import pandas as pd
import requests
from skimage import io, transform, color
from skimage.feature import hog
from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import gaussian
import cv2
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============ PHẦN 1: TẢI VÀ XỬ LÝ DỮ LIỆU ============

def load_data_from_csv(csv_file):
    """Đọc dữ liệu từ file CSV"""
    data = pd.read_csv(csv_file)
    print("Các cột trong file CSV:", data.columns)
    
    if 'brand' in data.columns:
        labels = data['brand'].unique()
        print("Danh sách các nhãn:", labels)
    else:
        print("Không tìm thấy cột 'brand' trong file CSV")
    
    return data


def download_images(data, output_dir='images'):
    """Tải hình ảnh từ URL"""
    os.makedirs(output_dir, exist_ok=True)
    
    data_sorted = data.sort_index()
    
    for index, row in data_sorted.iterrows():
        image_url = row['image']
        response = requests.get(image_url)
        if response.status_code == 200:
            image_filename = os.path.join(output_dir, f'image_{str(index).zfill(4)}.jpg')
            with open(image_filename, 'wb') as image_file:
                image_file.write(response.content)
            print(f"Downloaded: {image_filename}")
        else:
            print(f"Failed to download: {image_url}")
    
    files_in_directory = sorted(os.listdir(output_dir))
    print(f"Tổng file trong thư mục: {len(files_in_directory)}")
    return files_in_directory


# ============ PHẦN 2: TĂNG CƯỜNG DỮ LIỆU ============

def create_augmented_data(image_dir, output_dir):
    """Tạo dữ liệu tăng cường từ ảnh gốc"""
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = io.imread(img_path)
        
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
        brightness_factors = [0.7, 1.3]
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
        
        # 6. Cắt ngẫu nhiên
        h, w = img.shape[:2]
        crop_size = (int(h * 0.8), int(w * 0.8))
        start_h = np.random.randint(0, h - crop_size[0])
        start_w = np.random.randint(0, w - crop_size[1])
        cropped = img[start_h:start_h + crop_size[0],
                     start_w:start_w + crop_size[1]]
        io.imsave(os.path.join(output_dir, f"{base_name}_crop.jpg"), cropped)
        
        # 7. Thay đổi tỷ lệ
        scales = [0.8, 1.2]
        for scale in scales:
            scaled = transform.resize(img,
                                   (int(h * scale), int(w * scale)),
                                   mode='reflect')
            io.imsave(os.path.join(output_dir, f"{base_name}_scale_{scale}.jpg"),
                     (scaled * 255).astype(np.uint8))
        
        # 8. Biến dạng phối cảnh
        def random_perspective_transform(image):
            h, w = image.shape[:2]
            src_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
            dst_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
            dst_points += np.random.uniform(-50, 50, size=dst_points.shape)
            transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            warped = cv2.warpPerspective(image, transform_matrix, (w, h))
            return warped
        
        perspective = random_perspective_transform(img)
        io.imsave(os.path.join(output_dir, f"{base_name}_perspective.jpg"),
                 perspective)


def create_augmented_dataframe(original_df, augmented_image_dir, max_augmented_per_image=5):
    """Tạo DataFrame mới bao gồm cả dữ liệu gốc và dữ liệu tăng cường"""
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
        selected_aug_files = random.sample(aug_files, min(len(aug_files), max_augmented_per_image))
        
        for aug_file in selected_aug_files:
            new_rows.append({
                'brand': row['brand'],
                'image_path': os.path.join(augmented_image_dir, aug_file)
            })
    
    return pd.DataFrame(new_rows)


# ============ PHẦN 3: TRÍCH XUẤT ĐẶC TRƯNG HOG ============

def process_image(image_path):
    """Xử lý ảnh và trích xuất đặc trưng HOG"""
    img = io.imread(image_path)
    if len(img.shape) == 3:
        img = color.rgb2gray(img)
    img_resized = transform.resize(img, (128, 128))
    features, _ = hog(img_resized, pixels_per_cell=(8, 8), 
                      cells_per_block=(2, 2), visualize=True)
    return features


def extract_features(df):
    """Trích xuất đặc trưng HOG từ DataFrame"""
    features = []
    labels = []
    for _, row in df.iterrows():
        img_path = row['image_path']
        label = row['brand']
        try:
            hog_features = process_image(img_path)
            features.append(hog_features)
            labels.append(label)
        except Exception as e:
            print(f"Lỗi xử lý ảnh {img_path}: {e}")
    return np.array(features), np.array(labels)


# ============ PHẦN 4: HUẤN LUYỆN MÔ HÌNH SVM ============

def train_svm(X_train, y_train):
    """Huấn luyện mô hình SVM với tối ưu hóa tham số"""
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    # Tối ưu hóa tham số
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train_encoded)
    
    print(f"Tham số tối ưu: {grid_search.best_params_}")
    print(f"Điểm cross-validation tốt nhất: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, label_encoder


def evaluate_model(svm_model, label_encoder, X_test, y_test):
    """Đánh giá mô hình"""
    y_test_encoded = label_encoder.transform(y_test)
    y_pred_encoded = svm_model.predict(X_test)
    
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    print(f"Độ chính xác: {accuracy:.4f}")
    print("\nBáo cáo phân loại:")
    print(classification_report(y_test_encoded, y_pred_encoded, 
                               target_names=label_encoder.classes_))
    
    return accuracy


def cross_validate_model(svm_model, X_train, y_train):
    """Đánh giá mô hình với cross-validation"""
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    scores = cross_val_score(svm_model, X_train, y_train_encoded, cv=5)
    print(f"Điểm Cross-Validation: {scores}")
    print(f"Trung bình: {scores.mean():.4f} (+/- {scores.std():.4f})")


def save_model(svm_model, label_encoder):
    """Lưu mô hình"""
    joblib.dump(svm_model, 'svm_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Mô hình đã được lưu!")


def predict_new_image(image_path):
    """Dự đoán nhãn cho ảnh mới"""
    svm_model = joblib.load('svm_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    
    features = process_image(image_path)
    features = features.reshape(1, -1)
    
    pred_encoded = svm_model.predict(features)
    pred_label = label_encoder.inverse_transform(pred_encoded)
    
    return pred_label[0]


# ============ PHẦN 5: VISUALIZE DỮ LIỆU ============

def plot_label_distribution(y_train, title="Phân phối nhãn"):
    """Vẽ biểu đồ phân phối nhãn"""
    label_counts = Counter(y_train)
    labels, counts = zip(*label_counts.items())
    
    df_labels = pd.DataFrame({'Label': labels, 'Count': counts})
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_labels, x='Label', y='Count', palette='viridis')
    plt.title(title, fontsize=16)
    plt.xlabel('Nhãn', fontsize=14)
    plt.ylabel('Số lượng', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_augmentation_comparison(y_train, augmented_df):
    """So sánh phân phối dữ liệu trước và sau augmentation"""
    original_counts = Counter(y_train)
    augmented_y = augmented_df['brand'].values
    augmented_counts = Counter(augmented_y)
    
    labels_original, counts_original = zip(*original_counts.items())
    labels_augmented, counts_augmented = zip(*augmented_counts.items())
    
    df_original = pd.DataFrame({
        'Label': labels_original, 
        'Count': counts_original, 
        'Type': 'Trước khi tăng cường'
    })
    df_augmented = pd.DataFrame({
        'Label': labels_augmented, 
        'Count': counts_augmented, 
        'Type': 'Sau khi tăng cường'
    })
    
    df_combined = pd.concat([df_original, df_augmented])
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_combined, x='Label', y='Count', hue='Type', palette='viridis')
    plt.title('So sánh số lượng nhãn trước và sau khi tăng cường', fontsize=16)
    plt.xlabel('Nhãn', fontsize=14)
    plt.ylabel('Số lượng', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Loại dữ liệu', fontsize=12)
    plt.tight_layout()
    plt.show()


def visualize_sample_images(image_dir, num_samples=5):
    """Hiển thị ảnh mẫu"""
    from PIL import Image
    
    sample_images = sorted(os.listdir(image_dir))[:num_samples]
    
    fig, axes = plt.subplots(1, len(sample_images), figsize=(15, 5))
    for ax, image_file in zip(axes, sample_images):
        img_path = os.path.join(image_dir, image_file)
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(image_file)
    plt.tight_layout()
    plt.show()


def visualize_hog_features(image_dir, num_samples=3):
    """Hiển thị ảnh gốc và HOG features"""
    from skimage.feature import hog
    
    sample_images = sorted(os.listdir(image_dir))[:num_samples]
    
    fig, axes = plt.subplots(len(sample_images), 2, figsize=(10, 10))
    for i, image_file in enumerate(sample_images):
        img_path = os.path.join(image_dir, image_file)
        image = io.imread(img_path)
        
        if image.ndim == 3:
            image = color.rgb2gray(image)
        image = transform.resize(image, (128, 128))
        
        features, hog_image = hog(image, orientations=8, 
                                 pixels_per_cell=(16, 16),
                                 cells_per_block=(1, 1), visualize=True)
        
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Original: {image_file}")
        
        axes[i, 1].imshow(hog_image, cmap='gray')
        axes[i, 1].axis('off')
        axes[i, 1].set_title("HOG Visualization")
    
    plt.tight_layout()
    plt.show()


# ============ PHẦN 6: HÀM MAIN ============

def main():
    """Hàm main để chạy toàn bộ pipeline"""
    
    print("=" * 60)
    print("BRAND MOBILE CLASSIFICATION")
    print("Using HOG + SVM")
    print("=" * 60)
    
    # 1. Tải dữ liệu
    print("\n1. LOADING DATA...")
    csv_file = "data2.csv"  # Thay đổi đường dẫn phù hợp
    
    try:
        data = load_data_from_csv(csv_file)
        print(f"Tổng ảnh: {len(data)}")
    except Exception as e:
        print(f"Lỗi tải dữ liệu: {e}")
        return
    
    # 2. Tăng cường dữ liệu
    print("\n2. DATA AUGMENTATION...")
    augmented_dir = 'augmented_images'
    if not os.path.exists(augmented_dir):
        create_augmented_data('images', augmented_dir)
        augmented_df = create_augmented_dataframe(data, augmented_dir)
        augmented_df.to_csv('augmented_data.csv', index=False)
        print(f"Dữ liệu gốc: {len(data)}")
        print(f"Dữ liệu sau augmentation: {len(augmented_df)}")
    else:
        augmented_df = pd.read_csv('augmented_data.csv')
        print(f"Đã tải dữ liệu augmentation từ file")
    
    # 3. Chia tập dữ liệu
    print("\n3. SPLITTING DATA...")
    train_df, test_df = train_test_split(augmented_df, test_size=0.2, 
                                         random_state=42, 
                                         stratify=augmented_df['brand'])
    print(f"Tập huấn luyện: {len(train_df)}")
    print(f"Tập kiểm tra: {len(test_df)}")
    
    # 4. Trích xuất đặc trưng
    print("\n4. EXTRACTING FEATURES...")
    X_train, y_train = extract_features(train_df)
    X_test, y_test = extract_features(test_df)
    print(f"Kích thước X_train: {X_train.shape}")
    print(f"Kích thước X_test: {X_test.shape}")
    
    # 5. Huấn luyện mô hình
    print("\n5. TRAINING SVM MODEL...")
    svm_model, label_encoder = train_svm(X_train, y_train)
    
    # 6. Đánh giá mô hình
    print("\n6. EVALUATING MODEL...")
    print("\nKết quả trên tập kiểm tra:")
    evaluate_model(svm_model, label_encoder, X_test, y_test)
    
    print("\nKết quả Cross-Validation:")
    cross_validate_model(svm_model, X_train, y_train)
    
    # 7. Lưu mô hình
    print("\n7. SAVING MODEL...")
    save_model(svm_model, label_encoder)
    
    print("\n" + "=" * 60)
    print("HOÀN THÀNH!")
    print("=" * 60)


if __name__ == "__main__":
    main()
