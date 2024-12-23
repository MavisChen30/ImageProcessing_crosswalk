from scipy.io import loadmat

# 輸入你的.mat文件路徑
data = loadmat('/Users/mavischen/Desktop/ImageProcessing/Image_d1218/datasets/crosswalk_2/label_mat_2/10000003.mat')
# data = loadmat('/Users/mavischen/Desktop/ImageProcessing/d1210/datasets/Annotations/car_side/annotation_0001.mat')

# 打印文件內的所有鍵
print(data.keys())

# # 檢查 annotations 鍵的結構
# annotations = data['annotations']
#
# # 打印 annotations 的結構
# print(annotations)

print(data['__header__'])
print(data['__version__'])
print(data['__globals__'])
print(data['box_coord'])
print(data['obj_contour'])
