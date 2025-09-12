import os
import cv2
from openpyxl import Workbook

# ===== CONFIGURAÇÕES =====
diretorio_imagens = r"D:\Unirovuma\Monography_namicano\app\utils\imagens"
cfg_path = "yolov4-custom.cfg"
weights_path = "yolov4-custom_best.weights"
classes_path = "classe.names"  # se usar YOLO custom, troque para o seu arquivo

# Carregar nomes das classes
with open(classes_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Criar modelo YOLOv4 (seguindo sua estrutura)
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

# Mapeamento sigla -> periodo / condicao
mapa_periodo_condicao = {
    "MA": ("Manha", "Antes da Rega"),
    "MD": ("Manha", "Após a Rega"),
    "TA": ("Tarde", "Antes da Rega"),
    "TD": ("Tarde", "Após a Rega")
}

# ===== CRIAR PLANILHA =====
wb = Workbook()
ws = wb.active
ws.title = "Dados"
ws.append(["Imagem", "periodo", "condicao", "Prototipo"])

# ===== PROCESSAR IMAGENS =====
for arquivo in os.listdir(diretorio_imagens):
    if arquivo.lower().endswith((".jpg", ".jpeg", ".png")):
        caminho_img = os.path.join(diretorio_imagens, arquivo)
        nome_base = os.path.splitext(arquivo)[0]
        sigla = nome_base[:2]

        periodo, condicao = mapa_periodo_condicao.get(sigla, ("", ""))

        # Carregar imagem
        img = cv2.imread(caminho_img)

        # Rodar detecção
        class_ids, scores, boxes = model.detect(img, confThreshold=0.5, nmsThreshold=0.4)

        # Contagem de detecções
        contagem = len(class_ids) if class_ids is not None else 0

        # Adicionar à planilha
        ws.append([nome_base, periodo, condicao, contagem])
        print(f"{arquivo}: {contagem} detecções")

# Salvar planilha
wb.save("afideos_salvos.xlsx")
print("Arquivo 'resultados_yolo.xlsx' criado com sucesso!")
