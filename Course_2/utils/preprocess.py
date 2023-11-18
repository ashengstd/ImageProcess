import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def gabor_kernel(ksize, sigma, gamma, lamda, alpha, psi):
    sigma_x = sigma
    sigma_y = sigma / gamma

    ymax = xmax = ksize // 2  # 9//2
    xmin, ymin = -xmax, -ymax

    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))  # 生成网格点坐标矩阵
    x_alpha = x * np.cos(alpha) + y * np.sin(alpha)
    y_alpha = -x * np.sin(alpha) + y * np.cos(alpha)
    exponent = np.exp(-.5 * (x_alpha ** 2 / sigma_x ** 2 + y_alpha ** 2 / sigma_y ** 2))
    kernel = exponent * np.cos(2 * np.pi / lamda * x_alpha + psi)
    return kernel


def gabor(gray_img, ksize=9, sigma=1.0, gamma=0.5, lamda=5, psi=-np.pi/2):#gabor滤波
    filters = []
    for alpha in np.arange(0, np.pi, np.pi / 6):
        kern = gabor_kernel(ksize=ksize, sigma=sigma, gamma=gamma,lamda=lamda, alpha=alpha, psi=psi)
        filters.append(kern)

    gabor_img = np.zeros(gray_img.shape, dtype=np.uint8)

    i = 0
    for kern in filters:
        fimg = cv2.filter2D(gray_img, ddepth=cv2.CV_8U, kernel=kern)
        gabor_img = cv2.max(gabor_img, fimg)
        i += 1

    return gabor_img

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

    # 对Gabor滤波结果应用拉普拉斯算子
    laplacian_result = cv2.Laplacian(gabor_result, cv2.CV_32F)

    # 调整对比度
    alpha = 2.5  # 调整因子，可以根据需要调整
    adjusted_image = cv2.convertScaleAbs(laplacian_result, alpha=alpha, beta=0)

    ksize = 5
    blurred_img = cv2.GaussianBlur(adjusted_image, (ksize, ksize), sigmaX=5)

    # 二值化图像
    _, binary_img = cv2.threshold(blurred_img, 54, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)

    # 保存处理后的图像到输出文件夹
    output_image_path = os.path.join(output_person_folder, image_file)
    cv2.imwrite(output_image_path, opened_img)

def process_images_in_folder(dataset_root, output_root):
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
            print(f"Processing images in folder: {person_folder}")

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
