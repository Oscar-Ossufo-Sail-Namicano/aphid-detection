import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression

# 1. Ler dados
df = pd.read_csv("dados_afideos.csv", encoding="latin-1")

# 2. Calcular métricas
df['Acuracia'] = (df['TP'] + df['TN']) / (df['TP'] + df['TN'] + df['FP'] + df['FN'])
df['Precisao'] = df['TP'] / (df['TP'] + df['FP'])
df['Recall'] = df['TP'] / (df['TP'] + df['FN'])
df['F1'] = 2 * (df['Precisao'] * df['Recall']) / (df['Precisao'] + df['Recall'])

# 3. Médias por grupo
metricas_media = df.groupby(['Condicao', 'Periodo'])[['Acuracia', 'Precisao', 'Recall', 'F1']].mean()
print("\nMédias por Condição e Período:\n", metricas_media)

# 4. Função para ANOVA de duas vias
def two_way_anova(df, metrica):
    modelo = ols(f'{metrica} ~ C(Condicao) * C(Periodo)', data=df).fit()
    anova_tabela = sm.stats.anova_lm(modelo, typ=2)
    return anova_tabela

# 5. ANOVA para cada métrica
anova_resultados = {}
for metrica in ['Acuracia', 'Precisao', 'Recall', 'F1']:
    anova_resultados[metrica] = two_way_anova(df, metrica)
    print(f"\nANOVA para {metrica}:\n", anova_resultados[metrica])

# 6. Gráficos
plt.figure(figsize=(10,6))
sns.barplot(x='Periodo', y='Acuracia', hue='Condicao', data=df, ci='sd')
plt.title("Acurácia por Condição e Período")
plt.ylim(0, 1)
plt.legend(title="Condição")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# 7. Salvar relatório
with open("relatorio_resultados.txt", "w") as f:
    f.write("Relatório de Análise de Desempenho do Protótipo\n")
    f.write("="*60 + "\n\n")
    f.write("Médias por Condição e Período:\n")
    f.write(metricas_media.to_string() + "\n\n")
    for metrica, tabela in anova_resultados.items():
        f.write(f"ANOVA para {metrica}:\n")
        f.write(tabela.to_string() + "\n\n")



if 'Afideos Reais' in df.columns and 'Afideos Detectados' in df.columns:
    correlacao_pearson = stats.pearsonr(df['Afideos Detectados'], df['Afideos Reais'])
    print(f'\nCorrelação entre Infestação Automática e Manual: r={correlacao_pearson[0]:.2f}, p={correlacao_pearson[1]:.4f}')

    # Regressão linear
    X = df[['Afideos Detectados']].values.reshape(-1, 1)
    y = df['Afideos Reais']
    modelo = LinearRegression()
    modelo.fit(X, y)
    print(f'Coef. Angular (Inclinação): {modelo.coef_[0]:.2f}')
    print(f'Intercepto: {modelo.intercept_:.2f}')
else:
    print('\nColunas de infestação não encontradas para correlação.')

print("\n✅ Relatório salvo como 'relatorio_resultados.txt'")