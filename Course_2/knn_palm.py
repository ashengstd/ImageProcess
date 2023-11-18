import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from lightgbm import LGBMClassifier
import os

# 1. 读取图像并准备数据
def load_data(directory):
    images = []
    labels = []

    # 遍历每个人的文件夹
    for person_folder in os.listdir(directory):
        person_path = os.path.join(directory, person_folder)
        if os.path.isdir(person_path):
            # 遍历每个人的图像
            for file_name in os.listdir(person_path):
                file_path = os.path.join(person_path, file_name)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                # 将图像调整为相同的大小，例如 (32, 32)
                image = cv2.resize(image, (128, 128))
                images.append(image.flatten())  # 将图像转换为一维数组
                labels.append(int(person_folder))

    return np.array(images), np.array(labels)

# 2. 加载数据并划分为训练集和测试集
dataset_directory = 'output'  # 替换为你的实际目录
X, y = load_data(dataset_directory)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 3. 创建并训练SVM分类器
svm_classifier = SVC(kernel='linear')  # 可根据需要调整SVM的参数
svm_classifier.fit(X_train, y_train)

# 4. 预测并评估SVM模型
y_pred_svm = svm_classifier.predict(X_test)

accuracy_svm = metrics.accuracy_score(y_test, y_pred_svm)
print(f'SVM Accuracy: {accuracy_svm:.2f}')

# 5. 创建并训练KNN分类器
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # 可根据需要调整邻居数量
knn_classifier.fit(X_train, y_train)

# 6. 预测并评估KNN模型
y_pred_knn = knn_classifier.predict(X_test)

accuracy_knn = metrics.accuracy_score(y_test, y_pred_knn)
print(f'KNN Accuracy: {accuracy_knn:.2f}')

# 6. 创建并训练LightGBM分类器
lgbm_classifier = LGBMClassifier()
lgbm_classifier.fit(X_train, y_train)

# 7. 预测并评估LightGBM模型
y_pred_lgbm = lgbm_classifier.predict(X_test)

accuracy_lgbm = metrics.accuracy_score(y_test, y_pred_lgbm)
print(f'LightGBM Accuracy: {accuracy_lgbm:.2f}')


# 打印性能指标
# print("Classification Report:")
# print(metrics.classification_report(y_test, y_pred))

# print("\nConfusion Matrix:")
# print(metrics.confusion_matrix(y_test, y_pred))
