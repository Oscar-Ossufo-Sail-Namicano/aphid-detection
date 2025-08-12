import cv2
import os

input_dir = r"D:\Unirovuma\Monography_namicano\datasets\Eval\forTrain"
output_dir = "imagens_redimensionadas"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    print(file)
    img = cv2.imread(os.path.join(input_dir, file))
    resized = cv2.resize(img, (608, 608))
    cv2.imwrite(os.path.join(output_dir, file), resized)
