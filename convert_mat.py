import os
import scipy.io
import cv2
import numpy as np

# 設定檔案路徑
images_folder = "/Users/mavischen/Desktop/ImageProcessing/Image_d1218/datasets/crosswalk_2/img_Categories"  # 圖像資料夾
annotations_folder = "/Users/mavischen/Desktop/ImageProcessing/Image_d1218/datasets/crosswalk_2/label_txt"  # .txt 標註文件的資料夾
output_folder = "/Users/mavischen/Desktop/ImageProcessing/Image_d1218/datasets/crosswalk_2/label_mat_2"  # 儲存 .mat 文件的資料夾

os.makedirs(output_folder, exist_ok=True)  # 確保輸出目錄存在

# 遍歷每個 .txt 標註文件
for txt_file in os.listdir(annotations_folder):
    if txt_file.endswith(".txt"):
        image_name = txt_file.replace(".txt", ".jpg")
        box_coords = []  # 儲存所有標註框的座標
        obj_contours = []  # 儲存所有物體的輪廓

        # 圖像路徑
        img_path = os.path.join(images_folder, image_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Cannot find or open image: {image_name}")
            continue
        img_h, img_w = img.shape[:2]

        # 讀取標註文件內容
        with open(os.path.join(annotations_folder, txt_file), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    class_id = int(parts[0])  # 類別 ID（此處未使用）
                    x_center, y_center, width, height = map(float, parts[1:])
                except ValueError:
                    continue

                # 計算邊界框座標
                x_min = int((x_center - width / 2) * img_w)
                y_min = int((y_center - height / 2) * img_h)
                x_max = int((x_center + width / 2) * img_w)
                y_max = int((y_center + height / 2) * img_h)
                box_coords.append([x_min, x_max, y_min, y_max])

                # 假設輪廓為邊界框的四個角點
                obj_contours.append([
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max]
                ])

        # 組織數據，將輪廓展平為數組
        obj_contours_np = np.array(obj_contours, dtype=np.float32).reshape(-1, 2).T
        box_coords_np = np.array(box_coords, dtype=np.int32)

        # 保存為 .mat 文件
        mat_file_name = txt_file.replace(".txt", ".mat")
        scipy.io.savemat(
            os.path.join(output_folder, mat_file_name),
            {
                "__header__": b"MATLAB 5.0 MAT-file, Platform: PCWIN",
                "__version__": "1.0",
                "__globals__": [],
                "box_coord": box_coords_np,
                "obj_contour": obj_contours_np
            }
        )
        print(f"[INFO] Saved {mat_file_name}")
