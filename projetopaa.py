import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

df = pd.read_csv(r'C:\Users\mafim\Downloads\train.csv')

# Paleta de cores
VERMELHO = "#C0392B"
VERDE = "#bae47d"
LARANJA = "#faac86"
AZUL = "#a8b8d7"
ROSA = "#eca8d1"
AMARELO = "#fde562"
BEGE = "#e9d4ae"

sns.set_theme(style="white")
plt.rcParams.update({'axes.titlesize': 13, 'axes.labelsize': 11, 
                     'figure.dpi': 150, 'axes.titleweight': 'bold'})

# FIGURA 1 — Distribuição das Notas de Exame
fig, ax = plt.subplots(figsize=(8, 4.5))

sns.histplot(df['exam_score'], bins=40, stat='density', kde=False,
             color=AZUL, alpha=0.6, edgecolor='white', ax=ax)

df['exam_score'].plot.kde(ax=ax, color=VERMELHO, linewidth=2.5, label='KDE')

ax.axvline(df['exam_score'].mean(), color=VERMELHO, linestyle='--', linewidth=1.5,
           label=f"Média: {df['exam_score'].mean():.1f}")
ax.axvline(df['exam_score'].median(), color=VERDE, linestyle=':', linewidth=1.5,
           label=f"Mediana: {df['exam_score'].median():.1f}")

ax.set(xlim=(0, 110),
       title='Distribuição das Notas de Exame',
       xlabel='Nota de Exame',
       ylabel='Densidade')

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
            'sleep_quality_enc','facility_rating_enc','exam_difficulty_enc',
            'internet_access_enc','exam_score']
labels = ['Horas Estudo','Assiduidade','Horas Sono',
          'Qualidade Sono','Instalações','Dif. Exame','Internet','Nota Exame']

corr = df_enc[num_cols].corr()
corr.index = labels
corr.columns = labels

cmap_rosa_verde = LinearSegmentedColormap.from_list(
    'rosa_verde', [VERMELHO, '#FFFFFF', '#17975a']
)

fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap_rosa_verde, center=0,
            vmin=-1, vmax=1, linewidths=0.5, annot_kws={'size': 8}, square=True, ax=ax)
ax.set_title('Matriz de Correlação entre Variáveis Numéricas')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('fig2_matriz_correlacao.png', bbox_inches='tight')
plt.show()

# FIGURA 3 — Boxplot por género
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.boxplot(data=df, x='gender', y='exam_score', hue='gender',
            order=['female', 'male', 'other'],
            palette={'female': ROSA, 'male': AZUL, 'other': VERDE},
            width=0.5, legend=False, ax=ax)

order = ['female', 'male', 'other']
for i, gender in enumerate(order):
    media = df[df['gender'] == gender]['exam_score'].mean()
    ax.text(i, media + 2, f'{media:.1f}', ha='center', fontsize=9, fontweight='bold')

ax.set_title('Distribuição da Nota de Exame por Género')
ax.set_xlabel('Género')
ax.set_ylabel('Nota de Exame')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Feminino', 'Masculino', 'Outro'])
plt.tight_layout()
plt.savefig('fig3_boxplot_genero.png', bbox_inches='tight')
plt.show()

# FIGURA 4 — Violin plot por dificuldade do exame
fig, ax = plt.subplots(figsize=(6.5, 4.5))
sns.violinplot(data=df, x='exam_difficulty', y='exam_score',
               order=['easy', 'moderate', 'hard'],
               palette={'easy': VERDE, 'moderate': AZUL, 'hard': ROSA},
               inner='box', linewidth=1.2, ax=ax)

order_diff = ['easy', 'moderate', 'hard']
for i, diff in enumerate(order_diff):
    media = df[df['exam_difficulty'] == diff]['exam_score'].mean()
    ax.text(i, 105, f'{media:.1f}', ha='center', fontsize=9, fontweight='bold')

ax.set_title('Distribuição da Nota de Exame por Dificuldade')
ax.set_xlabel('Dificuldade do Exame')
ax.set_ylabel('Nota de Exame')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Fácil', 'Moderado', 'Difícil'])
ax.set_ylim(15, 112)
plt.tight_layout()
plt.savefig('fig4_violin_dificuldade.png', bbox_inches='tight')
plt.show()
