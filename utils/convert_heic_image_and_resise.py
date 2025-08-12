import os
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

input_folder = r"D:\Unirovuma\Monography_namicano\datasets\Eval\forTrain\toYolo_need_resize"
output_folder = "imagens_redimensionadas"
os.makedirs(output_folder, exist_ok=True)

TARGET_SIZE = (608, 608)

try:
    resample_method = Image.Resampling.LANCZOS
except AttributeError:
    resample_method = Image.ANTIALIAS

counter = 142

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".heic") or filename.lower().endswith(".jpg"):
        heic_path = os.path.join(input_folder, filename)
        png_filename = f"{counter}.png"
        png_path = os.path.join(output_folder, png_filename)
        try:
            image = Image.open(heic_path)
            image = image.convert("RGB")
            # só redimensiona se o tamanho for diferente do alvo
            if image.size != TARGET_SIZE:
                image = image.resize(TARGET_SIZE, resample_method)
            image.save(png_path, "PNG")
            print(f"Convertido e renomeado: {filename} -> {png_filename}")
            counter += 1
        except Exception as e:
            print(f"Erro ao converter {filename}: {e}")


