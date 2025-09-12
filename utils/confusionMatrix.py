import pandas as pd

# Carregar dados
df = pd.read_csv(r"D:\Unirovuma\Monography_namicano\app\resultados_analise/dados_com_metricas.csv")

# Remover linhas com valores ausentes em VP, VN, FN, FP
df_clean = df.dropna(subset=["vp", "vn", "fn", "fp"])

# Somar valores
vp_total = df_clean["vp"].sum()
vn_total = df_clean["vn"].sum()
fn_total = df_clean["fn"].sum()
fp_total = df_clean["fp"].sum()

# Criar matriz de confusão
matriz_confusao = pd.DataFrame(
    [[vp_total, fp_total],
     [fn_total, vn_total]],
    index=["Positivo Real", "Negativo Real"],
    columns=["Positivo Previsto", "Negativo Previsto"]
)

print(matriz_confusao)
