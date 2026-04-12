import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('train.csv')

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





# ============================
# SELEÇÃO DE DADOS
# ============================

#criar coluna classe para a classificacao dos alunos
def atribuir_classe(score):
    if score >= 90: return 'A'
    elif score >= 75: return 'B'
    elif score >= 60: return 'C'
    elif score >= 50: return 'D'
    elif score >= 35: return 'E'
    else: return 'F'


df['classe']=df['exam_score'].apply(atribuir_classe)

#dataset para regressão e clustering
df_regressao = df.drop(columns=['classe'])

#dataset para classificação
df_classificacao = df.drop(columns=['exam_score'])

print("DATASET PARA REGRESSÃO E CLUSTERING:", df_regressao.shape)
print("DATASET PARA CLASSIFICAÇÃO:", df_classificacao.shape)


# ============================
# LIMPEZA DE DADOS
# ============================

# VALORES EM FALTA

print("=== Valores em falta ===")
falta = df.isnull().sum()
falta_percent = ((falta / len(df)) * 100).round(2)
falta_df = pd.DataFrame({'Valores em falta': falta, 'Percentual (%)': falta_percent})
print(falta_df)

# Decisão com base nos resultados obtidos
if falta.sum() == 0:
    print("\nNão há valores em falta. Não é necessária imputação.")

else:
    print("\nValores em falta encontrados. Será aplicada imputação.")
    # Para varivaeis numéricas, usar a mediana para imputação
    num_cols = df.select_dtypes(include=['number']).columns
    for col in cols:
        if df[col].isnull().sum() > 0:
            mediana = df[col].median()
            df[col].fillna(mediana, inplace=True)
            print(f"Imputação da mediana ({mediana}) para a coluna '{col}'.")

    #Para variaveis categóricas, usar a moda para imputação
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            moda = df[col].mode()[0]
            df[col].fillna(moda, inplace=True)
            print(f"Imputação da moda ('{moda}') para a coluna '{col}'.")


# VALORES DUPLICADOS
print("\n=== Valores duplicados ===")
n_dupli=df.duplicated().sum()
print(f"Registos duplicados encontrados: {n_dupli}")

if n_dupli == 0:
    print("Não há registos duplicados. Não é necessária a remoção.")

else:
    df.drop_duplicates(inplace=True)
    print(f"Registos duplicados removidos. Novo número de registos: {len(df)}")


# TIPOS DE DADOS
print("\n=== Tipos de dados ===")
print("Tipos de dados antes da correção:")
print(df.dtypes)

# Variaveis categoricas para string
cat_cols = ['gender', 'course', 'internet_access', 'sleep_quality', 'facility_rating', 'exam_difficulty', 'study_method']

for col in cat_cols:
    df[col] = df[col].astype(str)
    df_regressao[col] = df_regressao[col].astype(str)
    df_classificacao[col] = df_classificacao[col].astype(str)


# VAavariaveis numéricas 
df['age']=df['age'].astype(int)
df['study_hours']=df['study_hours'].astype(float)
df['class_attendance']=df['class_attendance'].astype(float)
df['sleep_hours']=df['sleep_hours'].astype(float)
df['exam_score']=df['exam_score'].astype(float)

print("\nTipos de dados após correção:")
print(df.dtypes)



# VALIDAÇÃO DOS INTERVALOS DE VALORES
print("\n=== Validação dos intervalos de valores ===")

#Definir intervalos válidos para cada variável numérica
intervalos = {
    'age': (17, 24),
    'study_hours': (0, 8),
    'class_attendance': (0, 100),
    'sleep_hours': (0, 12),
    'exam_score': (0, 100)
}


for col, (minimo, maximo) in intervalos.items():
    invalidos = df[(df[col] < minimo) | (df[col] > maximo)]
    if len(invalidos) == 0:
       print(f"Todos os valores da coluna '{col}' estão dentro do intervalo válido ({minimo} a {maximo}).")
    else:
        print(f"Valores inválidos encontrados na coluna '{col}':")
        print(invalidos[[col]])
        #Decisão: remover os registos com valores inválidos
        df.drop(invalidos.index, inplace=True)
        print(f"Registos com valores inválidos removidos. Novo número de registos: {len(df)}")
         

#verificar categorias inesperadas nas variáveis categóricas
print("\n=== Verificação de categorias inesperadas ===")

categorias_esperadas = {
    'gender': ['male','female','other'],
    'internet_access': ['yes','no'],
    'sleep_quality': ['poor','average','good'],
    'facility_rating': ['low','medium','high'],
    'exam_difficulty': ['easy','moderate','hard'],
    'study_method': ['self-study','group study','tutoring','online videos','coaching', 'mixed']

}


for col, esperadas in categorias_esperadas.items():
    encontradas = df[col].unique().tolist()
    inesperadas = [v for v in encontradas if v not in esperadas]
    if not inesperadas:
        print(f"Todas as categorias da coluna '{col}' são esperadas.")
    else:
        print(f"Categorias inesperadas encontradas na coluna '{col}': {inesperadas}")
        #Decisão: remover os registos com categorias inesperadas
        df = df[~df[col].isin(inesperadas)]
        print(f"Registos com categorias inesperadas removidos. Novo número de registos: {len(df)}")



# OUTLIERS
print("\n=== Análise de Outliers (IQR) ===")

num_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours', 'exam_score']

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower= Q1 -1.5*IQR
    upper= Q3 + 1.5*IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}:{len(outliers)} outliers (limite inferior: {lower:.2f}, limite superior: {upper:.2f})")

n_outliers = sum(
    len(df[(df[col] < df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))) |
           (df[col] > df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))])
    for col in num_cols
)


if n_outliers == 0:
    print("Não foram encontrados outliers pelo método IQR. O dataset não requer tratamento de outliers.")
else:
    print("Decisão: Manter os outliers identificados, pois representam alunos com comportamentos extremos reais. São exatamente esstes casos que pretendemos detectar.")

df_regressao.to_csv('df_regressao_limpo.csv', index=False)
df_classificacao.to_csv('df_classificacao_limpo.csv', index=False)
print("\nFicheiros guardados com sucesso!")

# ============================
# ENCODING
# ============================

df_reg = pd.read_csv('df_regressao_limpo.csv')
df_clas = pd.read_csv('df_classificacao_limpo.csv')

# ── LABEL ENCODING (variáveis ordinais) ───────────────────────

# Mapeamentos com ordem lógica
sleep_map      = {'poor': 0, 'average': 1, 'good': 2}
facility_map   = {'low': 0, 'medium': 1, 'high': 2}
difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}
internet_map   = {'no': 0, 'yes': 1}

for df in [df_reg, df_clas]:
    df['sleep_quality']   = df['sleep_quality'].map(sleep_map)
    df['facility_rating'] = df['facility_rating'].map(facility_map)
    df['exam_difficulty'] = df['exam_difficulty'].map(difficulty_map)
    df['internet_access'] = df['internet_access'].map(internet_map)

print("Label Encoding aplicado!")

# ── ONE-HOT ENCODING (variáveis nominais) ─────────────────────

nominal_cols = ['gender', 'course', 'study_method']

df_reg = pd.get_dummies(df_reg, columns=nominal_cols, drop_first=True, dtype=int)
df_clas = pd.get_dummies(df_clas, columns=nominal_cols, drop_first=True, dtype=int)

print("One-Hot Encoding aplicado!")
print(f"Colunas após encoding (regressão): {df_reg.shape[1]}")
print(f"Colunas após encoding (classificação): {df_clas.shape[1]}")

# (opcional) Guardar datasets codificados em ficheiros CSV
df_reg.to_csv('df_regressao_encoded.csv', index=False)
df_clas.to_csv('df_classificacao_encoded.csv', index=False)
print("Datasets codificados foram guardados com sucesso.")