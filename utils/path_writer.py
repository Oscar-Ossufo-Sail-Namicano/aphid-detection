import os

# Caminho da pasta onde estão as imagens
pasta_imagens = r"D:\Unirovuma\Monography_namicano\app\utils\obj"

# Nome do arquivo de saída
arquivo_saida = "train.txt"

# Extensões de imagem que deseja considerar
extensoes = (".png", ".jpg", ".jpeg")

# Lista para guardar os caminhos
caminhos = []

# Percorre todos os arquivos da pasta
for arquivo in os.listdir(pasta_imagens):
    if arquivo.lower().endswith(extensoes):
        caminho_completo = 'data/obj/'+arquivo
        caminhos.append(caminho_completo)

# Salva no arquivo .txt
with open(arquivo_saida, "w", encoding="utf-8") as f:
    for caminho in caminhos:
        f.write(caminho + "\n")

print(f"{len(caminhos)} caminhos salvos em {arquivo_saida}")
