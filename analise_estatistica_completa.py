"""
                OSCAR OSSUFO SAIL NAMICANO
análise avançada 2x2 para protótipo de detecção de afídeos
Uso:
    pip install pandas matplotlib seaborn scipy statsmodels numpy tabulate
    python analise_prototipo_avancada.py
Entrada esperada (CSV): dados_afideos.csv com colunas:

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")

# ---------- Config ----------
INPUT_CSV = "afideos_salvos.xlsx"
OUT_DIR = "resultados_analise_final"
os.makedirs(OUT_DIR, exist_ok=True)
np.set_printoptions(precision=4, suppress=True)

# ---------- Funções utilitárias ----------
def safe_div(n, d):
    """Divide com proteção contra divisão por zero -> retorna np.nan se indefinido."""
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.array(n, dtype=float) / np.array(d, dtype=float)
    r[~np.isfinite(r)] = 0.5
    return r

def compute_metrics(df):
    """Adiciona colunas Acurácia, Precisão, Recall, F1 tratando zeros e NaNs."""
    total = df[['vp','fp','fn','vn']].sum(axis=1)
    df['Acurácia'] = safe_div(df['vp'] + df['vn'], total)
    df['Precisão'] = safe_div(df['vp'], df['vp'] + df['fp'])
    df['Recall'] = safe_div(df['vp'], df['vp'] + df['fn'])
    df['F1'] = safe_div(2 * df['Precisão'] * df['Recall'], df['Precisão'] + df['Recall'])
    return df

def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return path

def interpret_p(p):
    if np.isnan(p):
        return "p não calculado"
    if p < 0.001:
        return "p < 0.001 (muito significativo)"
    if p < 0.01:
        return "p < 0.01 (muito significativo)"
    if p < 0.05:
        return "p < 0.05 (significativo)"
    return "p >= 0.05 (não significativo)"

# ---------- Scheirer-Ray-Hare (SRH) implementation ----------
def scheirer_ray_hare(data, factorA, factorB, dv):
    """
    Implementa Scheirer-Ray-Hare test (two-way non-parametric).
    data: DataFrame
    factorA: nome coluna fator A (e.g., 'Condição')
    factorB: nome coluna fator B (e.g., 'Período')
    dv: nome da variável dependente (string)

    Retorna dict com estatísticas para A, B, A:B e residuals.
    """
    df = data[[factorA, factorB, dv]].dropna().copy()
    # Ranks across all observations
    df['rank'] = stats.rankdata(df[dv])
    N = df.shape[0]

    # Sums and counts
    a_levels = df[factorA].unique()
    b_levels = df[factorB].unique()
    a = len(a_levels)
    b = len(b_levels)

    grand_mean_rank = df['rank'].mean()

    # Sum of squares total on ranks
    SST = ((df['rank'] - grand_mean_rank) ** 2).sum()

    # Compute sums per factor/interaction
    sum_ranks_A = df.groupby(factorA)['rank'].sum()
    n_A = df.groupby(factorA).size()
    SS_A = sum((sum_ranks_A - n_A * grand_mean_rank) ** 2 / n_A)

    sum_ranks_B = df.groupby(factorB)['rank'].sum()
    n_B = df.groupby(factorB).size()
    SS_B = sum((sum_ranks_B - n_B * grand_mean_rank) ** 2 / n_B)

    sum_ranks_AB = df.groupby([factorA, factorB])['rank'].sum()
    n_AB = df.groupby([factorA, factorB]).size()
    SS_AB = sum((sum_ranks_AB - n_AB * grand_mean_rank) ** 2 / n_AB) - SS_A - SS_B

    # Error (residual) sum of squares
    SS_error = SST - SS_A - SS_B - SS_AB

    # Mean squares (using df)
    df_A = a - 1
    df_B = b - 1
    df_AB = (a - 1) * (b - 1)
    df_error = N - a * b

    MS_A = SS_A / df_A if df_A > 0 else np.nan
    MS_B = SS_B / df_B if df_B > 0 else np.nan
    MS_AB = SS_AB / df_AB if df_AB > 0 else np.nan
    MS_error = SS_error / df_error if df_error > 0 else np.nan

    # H statistics approximated as F-like: H = SS_factor / MS_error
    H_A = SS_A / MS_error if (not np.isnan(MS_error) and MS_error != 0) else np.nan
    H_B = SS_B / MS_error if (not np.isnan(MS_error) and MS_error != 0) else np.nan
    H_AB = SS_AB / MS_error if (not np.isnan(MS_error) and MS_error != 0) else np.nan

    # p-values using chi-square approx with corresponding df
    p_A = 1 - stats.chi2.cdf(H_A, df_A) if not np.isnan(H_A) else np.nan
    p_B = 1 - stats.chi2.cdf(H_B, df_B) if not np.isnan(H_B) else np.nan
    p_AB = 1 - stats.chi2.cdf(H_AB, df_AB) if not np.isnan(H_AB) else np.nan

    results = {
        'A': {'SS': SS_A, 'df': df_A, 'MS': MS_A, 'H': H_A, 'p': p_A},
        'B': {'SS': SS_B, 'df': df_B, 'MS': MS_B, 'H': H_B, 'p': p_B},
        'A:B': {'SS': SS_AB, 'df': df_AB, 'MS': MS_AB, 'H': H_AB, 'p': p_AB},
        'Error': {'SS': SS_error, 'df': df_error, 'MS': MS_error}
    }
    return results

# ---------- Carregamento ----------
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Arquivo '{INPUT_CSV}' não encontrado no diretório. Coloque o CSV e rode novamente.")

df = pd.read_excel(INPUT_CSV)
expected_cols = {'auto', 'manual', 'vp', 'vn', 'fp', 'fn',	'Condição', 'Período'}
if not expected_cols.issubset(set(df.columns)):
    """
    # Renomear colunas para o formato esperado
    df.rename(columns={
        'Afídeos Detectados': 'Afideos_Detectados',
        'Afídeos Reais': 'Afideos_Reais'
    }, inplace=True)
    """

    raise ValueError(f"O CSV deve conter colunas: {expected_cols}. Colunas encontradas: {set(df.columns)}")

# ---------- Calcula métricas ----------
df = compute_metrics(df)
# Salva métricas por amostra
df.to_csv(os.path.join(OUT_DIR, "dados_com_metricas.csv"), index=False)

# ---------- Resumo descritivo ----------
metrics = ['Acurácia','Precisão','Recall','F1']
group_stats = df.groupby(['Condição','Período'])[metrics].agg(['mean','std','count'])
# Flatten columns
group_stats.columns = ['_'.join(col).strip() for col in group_stats.columns.values]

# Escreve resumo inicial
report_lines = []
report_lines.append("Relatório de Análise - Protótipo Detecção de Afídeos (2x2)\n")
report_lines.append("Resumo descritivo por Condição × Período:\n")
report_lines.append(tabulate(group_stats.reset_index(), headers='keys', tablefmt='github', showindex=False))
report_lines.append("\n\n")

# ---------- ANOVA de duas vias + verificação pressupostos ----------
anova_results = {}
assumption_checks = {}

for met in metrics:
    # Modelo
    formula = f"{met} ~ C(Condição) * C(Período)"
    # dropna for metric
    df_met = df[['Condição','Período',met]].dropna()
    # Fit model
    try:
        model = ols(formula, data=df_met).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
    except Exception as e:
        anova_table = None

    anova_results[met] = anova_table

    # Checar pressupostos: normalidade dos resíduos e homocedasticidade
    check = {}
    if anova_table is None:
        check['normality_p'] = np.nan
        check['levene_p'] = np.nan
    else:
        resid = model.resid
        # Shapiro normality on residuals
        try:
            sh = stats.shapiro(resid)
            check['normality_stat'] = float(sh.statistic)
            check['normality_p'] = float(sh.pvalue)
        except Exception:
            check['normality_stat'] = np.nan
            check['normality_p'] = np.nan
        # Levene test across groups (Condição x Período groups)
        groups = []
        for (a,b), sub in df_met.groupby(['Condição','Período']):
            groups.append(sub[met].values)
        try:
            lev = stats.levene(*groups, center='median')  # robust
            check['levene_stat'] = float(lev.statistic)
            check['levene_p'] = float(lev.pvalue)
        except Exception:
            check['levene_stat'] = np.nan
            check['levene_p'] = np.nan

    assumption_checks[met] = check

# ---------- Executa Scheirer-Ray-Hare quando necessário ----------
srh_results = {}
for met in metrics:
    need_nonparam = False
    ck = assumption_checks[met]
    # If normality p < 0.05 OR levene p < 0.05 => assumptions violated
    if not np.isnan(ck.get('normality_p', np.nan)) and ck['normality_p'] < 0.05:
        need_nonparam = True
    if not np.isnan(ck.get('levene_p', np.nan)) and ck['levene_p'] < 0.05:
        need_nonparam = True
    if need_nonparam:
        srh = scheirer_ray_hare(df, 'Condição', 'Período', met)
        srh_results[met] = srh
    else:
        srh_results[met] = None

# ---------- Gráficos para todas as métricas ----------
for met in metrics:
    # Barplot: média ± dp
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x='Período', y=met, hue='Condição', data=df, ci='sd', ax=ax)
    ax.set_title(f"{met} por Condição e Período (média ± DP)")
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    path = save_fig(fig, f"bar_{met}.png")
    report_lines.append(f"Gráfico salvo: {path}\n")

    # Boxplot (dispersion)
    fig2, ax2 = plt.subplots(figsize=(8,5))
    sns.boxplot(x='Período', y=met, hue='Condição', data=df, ax=ax2)
    ax2.set_title(f"{met} por Condição e Período (boxplot)")
    path2 = save_fig(fig2, f"box_{met}.png")
    report_lines.append(f"Gráfico salvo: {path2}\n")

# ---------- Compee secção de resultados estatisticos ----------
report_lines.append("\nResultados dos testes estatísticos (ANOVA 2 vias):\n\n")
for met in metrics:
    report_lines.append(f"--- Métrica: {met} ---\n")
    anova_table = anova_results[met]
    report_lines.append("ANOVA (type II):\n")
    if anova_table is None:
        report_lines.append("ANOVA não pôde ser ajustada para esta métrica (dados insuficientes ou erro).\n")
    else:
        report_lines.append(tabulate(anova_table.reset_index(), headers='keys', tablefmt='github', showindex=False))
        report_lines.append("\n")
    # Assumption checks
    ck = assumption_checks[met]
    report_lines.append("Checagem de pressupostos:\n")
    report_lines.append(f"  - Normalidade dos resíduos (Shapiro): stat={ck.get('normality_stat')}, p={ck.get('normality_p')}\n")
    report_lines.append(f"  - Homogeneidade de variâncias (Levene, center=median): stat={ck.get('levene_stat')}, p={ck.get('levene_p')}\n")
    # Interpretacao automatica
    if not np.isnan(ck.get('normality_p', np.nan)) and ck['normality_p'] < 0.05:
        report_lines.append("  -> Resíduos NÃO seguem normalidade (p < 0.05).\n")
    else:
        report_lines.append("  -> Resíduos parecem seguir normalidade (p >= 0.05) ou teste não aplicável.\n")
    if not np.isnan(ck.get('levene_p', np.nan)) and ck['levene_p'] < 0.05:
        report_lines.append("  -> Variâncias NÃO são homogêneas (p < 0.05).\n")
    else:
        report_lines.append("  -> Variâncias parecem homogêneas (p >= 0.05) ou teste não aplicável.\n")

    # If SRH was run
    if srh_results[met] is not None:
        report_lines.append("\nAlternativa não-paramétrica aplicada: Scheirer–Ray–Hare (SRH)\n")
        srh = srh_results[met]
        report_lines.append(f"  A: H={srh['A']['H']:.4f}, df={srh['A']['df']}, p={srh['A']['p']:.4f} -> {interpret_p(srh['A']['p'])}\n")
        report_lines.append(f"  B: H={srh['B']['H']:.4f}, df={srh['B']['df']}, p={srh['B']['p']:.4f} -> {interpret_p(srh['B']['p'])}\n")
        report_lines.append(f"  A:B: H={srh['A:B']['H']:.4f}, df={srh['A:B']['df']}, p={srh['A:B']['p']:.4f} -> {interpret_p(srh['A:B']['p'])}\n")
    else:
        # Interpretacao automatica dos p-valores da ANOVA (se existente)
        if anova_table is not None:
            for factor in ['C(Condição)', 'C(Período)', 'C(Condição):C(Período)']:
                if factor in anova_table.index:
                    pval = float(anova_table.loc[factor, 'PR(>F)'])
                    report_lines.append(f"  {factor}: F={anova_table.loc[factor,'F']:.4f}, p={pval:.4f} -> {interpret_p(pval)}\n")
    report_lines.append("\n\n")

if 'manual' in df.columns and 'auto' in df.columns:
    correlacao_pearson = stats.pearsonr(df['auto'], df['manual'])
    print(f'\nCorrelação entre Infestação Automática e Manual: r={correlacao_pearson[0]:.2f}, p={correlacao_pearson[1]:.4f}')

    #y = 1.62*X + 0


    # Regressao linear
    X = df[['auto']].values.reshape(-1, 1)
    y = df['manual']
    from sklearn.linear_model import LinearRegression
    modelo = LinearRegression()
    modelo.fit(X, y)
    print(f'Coef. Angular (Inclinação): {modelo.coef_[0]:.2f}')
    print(f'Intercepto: {modelo.intercept_:.2f}')
    print(modelo.score(X, y))
else:
    print('\nColunas de infestação não encontradas para correlação.')

print("\nRelatório salvo como 'relatorio_resultados.txt'")

# ---------- Salva relatorio ----------
report_path = os.path.join(OUT_DIR, "relatorio_resultados_avancado.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("Análise concluída.")
print(f"Arquivos gerados na pasta: {OUT_DIR}")
print(f"Relatório: {report_path}")