import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('train.csv')
print(df.head())

AZUL = "#2E86AB"
CORAL = "#E84855"
VERDE = "#3BB273"
LARANJA = "#F4A261"
CINZA = "#6c757d"

sns.set_theme(style="whitegrid")
plt.rcParams.update({'axes.titlesize': 13, 'axes.labelsize': 11, 'figure.dpi': 150})

# FIGURA 1 — Distribuição do exam_score
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(df['exam_score'], bins=40, color=AZUL, alpha=0.7, edgecolor='white', density=True)
df['exam_score'].plot.kde(ax=ax, color=CORAL, linewidth=2.5)
ax.axvline(df['exam_score'].mean(), color=CORAL, linestyle='--', linewidth=1.5, label=f"Média: {df['exam_score'].mean():.1f}")
ax.axvline(df['exam_score'].median(), color=VERDE, linestyle=':', linewidth=1.5, label=f"Mediana: {df['exam_score'].median():.1f}")
ax.set_title('Distribuição das Notas de Exame', fontweight='bold')
ax.set_xlabel('Nota de Exame')
ax.set_ylabel('Densidade')
ax.legend()
plt.tight_layout()
plt.savefig('fig1_distribuicao_exam_score.png', bbox_inches='tight')
plt.show()

# FIGURA 2 — Matriz de correlação
df_enc = df.copy()
df_enc['sleep_quality_enc'] = df['sleep_quality'].map({'poor': 0, 'average': 1, 'good': 2})
df_enc['facility_rating_enc'] = df['facility_rating'].map({'low': 0, 'medium': 1, 'high': 2})
df_enc['exam_difficulty_enc'] = df['exam_difficulty'].map({'easy': 0, 'moderate': 1, 'hard': 2})
df_enc['internet_access_enc'] = df['internet_access'].map({'no': 0, 'yes': 1})

num_cols = ['study_hours','class_attendance','sleep_hours',
            'sleep_quality_enc','facility_rating_enc','exam_difficulty_enc','internet_access_enc','exam_score']
labels = ['Horas Estudo','Assiduidade','Horas Sono',
          'Qualidade Sono','Instalações','Dif. Exame','Internet','Nota Exame']

corr = df_enc[num_cols].corr()
corr.index = labels
corr.columns = labels

fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            vmin=-1, vmax=1, linewidths=0.5, annot_kws={'size': 8}, square=True, ax=ax)
ax.set_title('Matriz de Correlação entre Variáveis Numéricas', fontweight='bold')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('fig2_matriz_correlacao.png', bbox_inches='tight')
plt.show()

# FIGURA 3 — Boxplot por género
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.boxplot(data=df, x='gender', y='exam_score', hue='gender',
            order=['female', 'male',' other'],
            palette={'female': AZUL, 'male': CORAL, 'other': VERDE},
            width=0.5, legend=False, ax=ax)
ax.set_title('Distribuição da Nota de Exame por Género', fontweight='bold')
ax.set_xlabel('Género')
ax.set_ylabel('Nota de Exame')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Feminino', 'Masculino', 'Outro'])
plt.tight_layout()
plt.savefig('fig3_boxplot_genero.png', bbox_inches='tight')
plt.show()

# FIGURA 4 — Nota média por dificuldade do exame
fig, ax = plt.subplots(figsize=(6.5, 4.5))
order_diff = ['easy', 'moderate', 'hard']
labels_diff = ['Fácil', 'Moderado', 'Difícil']
means = [df[df['exam_difficulty'] == d]['exam_score'].mean() for d in order_diff]
stds  = [df[df['exam_difficulty'] == d]['exam_score'].std()  for d in order_diff]

bars = ax.bar(labels_diff, means, color=[VERDE, LARANJA, CORAL],
              edgecolor='white', width=0.55, yerr=stds, capsize=5,
              error_kw={'elinewidth': 1.2, 'ecolor': CINZA})
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, mean + 1.5,
            f'{mean:.1f}', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Nota Média por Dificuldade do Exame\n(com desvio padrão)', fontweight='bold')
ax.set_xlabel('Dificuldade do Exame')
ax.set_ylabel('Nota Média de Exame')
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig('fig4_nota_dificuldade.png', bbox_inches='tight')
plt.show()
