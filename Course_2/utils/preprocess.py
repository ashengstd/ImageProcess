import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor


#gabor滤波
def gabor(gray_img, ksize=9, sigma=1.0, gamma=0.5, lamda=5, psi=-np.pi/2):
    filters = []
    for alpha in np.arange(0, np.pi, np.pi / 6):
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, alpha, lamda, gamma, psi, ktype=cv2.CV_64F)
        filters.append(kernel)

    gabor_img = np.zeros(gray_img.shape, dtype=np.uint8)

    for kern in filters:
        fimg = cv2.filter2D(gray_img, ddepth=cv2.CV_8U, kernel=kern)
        gabor_img = cv2.max(gabor_img, fimg)

    return gabor_img

def remove_lower_connect_components(image):
    # 寻找连通分量
    connectivity = 8
    output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

    # 获取连通分量信息
    num_labels = output[0]
    labels = output[1]
    stats = output[2]

    # 计算总面积
    total_area = np.sum(stats[:, cv2.CC_STAT_AREA])

    # 标准化连通分量的面积
    normalized_areas = stats[:, cv2.CC_STAT_AREA] / total_area

    # 定义标准化面积阈值
    threshold_normalized_area = 0.01

    # 去除标准化面积小于阈值的连通分量
    filtered_image = np.zeros_like(labels)
    for i in range(1, num_labels):
        if normalized_areas[i] >= threshold_normalized_area:
            filtered_image[labels == i] = 255
    return filtered_image

def process_image(person_folder_path, output_person_folder, image_file):
    image_path = os.path.join(person_folder_path, image_file)

    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 检查图像是否成功加载
    if image is None:
        print(f"Error loading the image {image_file}. Please check the image path.")
        return

    # 应用Gabor滤波器到图像
    gabor_result = gabor(image, ksize=9, sigma=1.0, gamma=0.5, lamda=5, psi=-np.pi/2)

    # 调整对比度
    alpha = 2.5  # 调整因子，可以根据需要调整
    adjusted_image = cv2.convertScaleAbs(gabor_result, alpha=alpha, beta=0)

    # 高斯滤波
    blurred_img = cv2.GaussianBlur(adjusted_image, (0, 0), 1.0)

    # 通过减去高斯滤波结果来增强纹理
    enhanced = cv2.addWeighted(adjusted_image, 2, blurred_img, -1, 0)

    # 二值化图像
    _, binary_img = cv2.threshold(enhanced, 54, 255, cv2.THRESH_BINARY)

    # 连通分量
    final_img = remove_lower_connect_components(binary_img)

    # 保存处理后的图像到输出文件夹
    output_image_path = os.path.join(output_person_folder, image_file)
    cv2.imwrite(output_image_path, final_img)

def process_images_in_folder(dataset_root, output_root):
    # 检查数据集文件夹是否存在
    if not os.path.exists(dataset_root):
        print(f"Dataset folder {dataset_root} does not exist.")
        return None
    
    # 创建输出文件夹
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    for person_folder in os.listdir(dataset_root):
        person_folder_path = os.path.join(dataset_root, person_folder)

        # 确保当前路径是一个文件夹
        if os.path.isdir(person_folder_path):

            # 创建当前人的输出文件夹
            output_person_folder = os.path.join(output_root, person_folder)
            if not os.path.exists(output_person_folder):
                os.makedirs(output_person_folder)

            with ThreadPoolExecutor() as executor:
                # 遍历文件夹中的图像文件，并行处理
                futures = [executor.submit(process_image, person_folder_path, output_person_folder, image_file)
                        for image_file in os.listdir(person_folder_path)]

                # 等待所有任务完成
                for future in futures:
                    future.result()
    print("Image processing and saving completed.")

if __name__ == "__main__":
    # 指定数据集的根目录
    dataset_root = 'Dataset'
    output_root = 'output'
    process_images_in_folder(dataset_root=dataset_root, output_root=output_root)
