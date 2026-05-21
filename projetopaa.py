import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, recall_score, precision_score,
                              f1_score, classification_report, confusion_matrix)


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

# LABEL ENCODING (variáveis ordinais)

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

# ONE-HOT ENCODING (variáveis nominais) 

nominal_cols = ['gender', 'course', 'study_method']

df_reg = pd.get_dummies(df_reg, columns=nominal_cols, drop_first=True, dtype=int)
df_clas = pd.get_dummies(df_clas, columns=nominal_cols, drop_first=True, dtype=int)

print("One-Hot Encoding aplicado!")
print(f"Colunas após encoding (regressão): {df_reg.shape[1]}")
print(f"Colunas após encoding (classificação): {df_clas.shape[1]}")

df_reg.to_csv('df_regressao_encoded.csv', index=False)
df_clas.to_csv('df_classificacao_encoded.csv', index=False)
print("Datasets codificados foram guardados com sucesso.")

# ============================
# NORMALIZAÇÃO
# ============================

num_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours']

minmax = MinMaxScaler()
df_reg_minmax = df_reg.copy()
df_clas_minmax = df_clas.copy()

minmax.fit(df_reg[num_cols])

df_reg_minmax[num_cols] = minmax.transform(df_reg[num_cols])
df_clas_minmax[num_cols] = minmax.transform(df_clas[num_cols])

# StandardScaler (para regressão linear)

scaler = StandardScaler()
df_reg_standard = df_reg.copy()
scaler.fit(df_reg[num_cols])
df_reg_standard[num_cols] = scaler.transform(df_reg[num_cols])

print("Normalização aplicada!")

with open('minmax_scaler.pkl', 'wb') as f:
    pickle.dump(minmax, f)
with open('standard_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

df_reg_minmax.to_csv('df_regressao_minmax.csv', index=False)
df_clas_minmax.to_csv('df_classificacao_minmax.csv', index=False)
df_reg_standard.to_csv('df_regressao_standard.csv', index=False)
print("\nFicheiros guardados com sucesso!")

# ============================
# FEATURE ENGINEERING
# ============================

SEED = 42

df_regressao = pd.read_csv("df_regressao_minmax.csv")
df_classificacao = pd.read_csv("df_classificacao_minmax.csv") 

for df in [df_regressao, df_classificacao]:
    df["estudo_assiduidade"] = df["study_hours"] * df["class_attendance"]
    
    df["sono_estudo"] = df["sleep_hours"] * df["study_hours"]
    
    limiar_estudo = df["study_hours"].quantile(0.25)
    limiar_assiduidade = df["class_attendance"].quantile(0.25)
    
    df["perfil_risco"] = (
        (df["study_hours"] < limiar_estudo) &
        (df["class_attendance"] < limiar_assiduidade)
    ).astype(int)

print("Feature Engineering aplicado!")

# ============================
# DIVISÃO TREINO/TESTE
# ============================
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from imblearn.over_sampling import SMOTE

# Regressão
X_reg = df_regressao.drop(columns=["exam_score", "id"])
y_reg = df_regressao["exam_score"]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=SEED
)

# Classificação
X_clf = df_classificacao.drop(columns=["classe", "id"])
y_clf = df_classificacao["classe"]

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=SEED, stratify=y_clf
)

print(f"\n=== DIVISÃO DOS DADOS ===")
print(f"Regressão    — Treino: {X_reg_train.shape} | Teste: {X_reg_test.shape}")
print(f"Classificação — Treino: {X_clf_train.shape} | Teste: {X_clf_test.shape}")
print(f"\nDistribuição de classes no treino:\n{y_clf_train.value_counts().sort_index()}")
print(f"\nDistribuição de classes no teste:\n{y_clf_test.value_counts().sort_index()}")

# ============================
# VALIDAÇÃO CRUZADA — K-FOLD (k=5)
# ============================

# Classificação — KFold 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Regressão — KFold 
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

print(f"\n=== VALIDAÇÃO CRUZADA ===")
print(f"Stratified K-Fold (k=5) definido para classificação")
print(f"K-Fold (k=5) definido para regressão")

# ============================
# SMOTE
# ============================

smote = SMOTE(random_state=SEED)
X_clf_train_bal, y_clf_train_bal = smote.fit_resample(X_clf_train, y_clf_train)

print(f"\n=== SMOTE APLICADO ===")
print(f"Antes  — {y_clf_train.value_counts().sort_index().to_dict()}")
print(f"Depois — {pd.Series(y_clf_train_bal).value_counts().sort_index().to_dict()}")


# ============================
# bibliotecas para classificação
# ============================

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix)

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

SEED = 42

#contruir modelos classificacao

le=LabelEncoder()
y_clf_train_enc = le.fit_transform(y_clf_train_bal)
y_clf_test_enc = le.transform(y_clf_test)

#arvores de decisao
dt= DecisionTreeClassifier(
    max_depth=10,
    min_samples_leaf=50,
    class_weight= 'balanced',
    random_state=SEED
)
dt.fit(X_clf_train_bal,y_clf_train_bal)
print('Arvore de decisao treinada.')

#random forest
rf_clf= RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight='balanced',
    random_state= SEED,
    n_jobs=1
)
rf_clf.fit(X_clf_train_bal,y_clf_train_bal)
print('random forest treinado.')

#XGBoost
xgb_clf= XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate= 0.1,
    eval_metric='mlogloss',
    random_state=SEED,
    n_jobs=1
)
xgb_clf.fit(X_clf_train_bal,y_clf_train_enc)
print('XGBoost treinado.')

#Ajuste dos hiperparametros
print('grid search-Random Forest')
param_grid={
    'n_estimators':[100,200],
    'max_depth':[10,15,20],
    'min_samples_leaf':[20,50]
}


grid_rf= GridSearchCV(
    RandomForestClassifier(class_weight='balanced',random_state=SEED,n_jobs=1),param_grid,cv=3,scoring='f1_macro',n_jobs=-1,verbose=1
)
grid_rf.fit(X_clf_train_bal,y_clf_train_bal)
print(f"Melhores parametros: {grid_rf.best_params_}")
print(f"Melhores F1-macro: {grid_rf.best_score_:.4f}")

rf_clf_otimizado=grid_rf.best_estimator_

# =============================
# AVALIACAO
# =============================

def avaliar_classificacao(modelo, X_test,y_test, nome, encoded= False):
    y_pred=modelo.predict(X_test)
    if encoded:
        y_pred=le.inverse_transform(y_pred)

    print(f"\n{'='*50}")
    print(f" MODELO:{nome}")
    print(classification_report(y_test,y_pred, target_names=['A','B','C','D','E','F']))

    accuracy= accuracy_score(y_test,y_pred)
    rec_macro=recall_score(y_test,y_pred,average='macro')
    rec_F= recall_score(y_test,y_pred,labels=['F'],average='macro')
    prec=precision_score(y_test,y_pred,average='macro')
    f1= f1_score(y_test, y_pred, average='macro')


    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall (Macro): {rec_macro:.4f}")
    print(f"Recall (F): {rec_F:.4f}") #mais importante para o nosso objetivo
    print(f"Precision (Macro): {prec:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")

    return {'Modelo':nome, 'Accuracy': round(accuracy,4), 'Recall_Macro': round(rec_macro,4), 'Recall_F': round(rec_F,4), 'Precision_Macro': round(prec,4), 'F1_Macro': round(f1,4)}

resultados_clf=[]
for modelo, nome, enc in[
    (dt, 'Árvore de Decisão', False),
    (rf_clf, 'Random Forest', False),
    (xgb_clf, 'XGBoost', True)
]:
    resultados_clf.append(avaliar_classificacao(modelo, X_clf_test, y_clf_test, nome, encoded=enc))

#Tabela comparativa dos resultados
df_resultados_clf = pd.DataFrame(resultados_clf).sort_values(by='Recall_F', ascending=False)
print("\n=== Tabela Comparativa dos Modelos de Classificação ===")
print(df_resultados_clf.to_string(index=False))
print(f"\n Melhor modelo: {df_resultados_clf.iloc[0]['Modelo']}")

#matriz de confusão do melhor modelo
nomes_modelos = ['Árvore de Decisão', 'Random Forest', 'XGBoost']
objetos_modelos= [dt,rf_clf_otimizado,xgb_clf]
melhor_nome = df_resultados_clf.iloc[0]['Modelo']
melhor_clf= objetos_modelos[nomes_modelos.index(melhor_nome)]

y_pred_melhor = melhor_clf.predict(X_clf_test)
if melhor_nome == 'XGBoost':
    y_pred_melhor = le.inverse_transform(y_pred_melhor)

fig, ax = plt.subplots(figsize=(8,6))
cm=confusion_matrix(y_clf_test, y_pred_melhor, 
                    labels=['A','B','C','D','E','F'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['A','B','C','D','E','F'], yticklabels=['A','B','C','D','E','F'], ax=ax)
ax.set_title(f'Matriz de Confusão - {melhor_nome}', fontsize=14, fontweight='bold')
ax.set_xlabel('Classe Prevista')
ax.set_ylabel('Classe Real')
plt.tight_layout()
plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=150)
plt.show()



# ============================
# CLUSTERING
# ============================
df_regressao     = pd.read_csv('df_regressao_minmax.csv')
X_clust = df_regressao.drop(columns=['exam_score', 'id'])
sample= X_clust.sample(10000, random_state=SEED)

inertias=[]
silhouettes=[]
k_range= range(2,9)

for k in k_range:
    km=KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels=km.fit_predict(sample)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(sample,labels))
    print(f"k={k} -Inércia: {km.inertia_:,.2f} | Silhouette: {silhouettes[-1]:.4f}")

#Grafico do joelho + silhouette
fig,axes = plt.subplots(1,2, figsize=(14,5))

axes[0].plot(k_range, inertias, 'o-', color='#2E75B6', linewidth=2, markersize=8)
axes[0].set_title('Método do Joelho — Inércia', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Número de Clusters (k)')
axes[0].set_ylabel('Inércia (WCSS)')
axes[0].set_facecolor('#F8F8F8')

axes[1].plot(k_range, silhouettes, 'o-', color='#70AD47', linewidth=2, markersize=8)
axes[1].set_title('Silhouette Score por k', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Número de Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_facecolor('#F8F8F8')

plt.tight_layout()
plt.savefig('joelho_silhouette.png', dpi=150, bbox_inches='tight')
plt.show()

# decisao do k otimo
k_otimo_sil=list(k_range)[silhouettes.index(max(silhouettes))]
diffs=np.diff(inertias)
k_otimo_joelho=list(k_range)[np.argmin(np.diff(diffs))+1]

print(f"\nMétodo do Joelho sugere: k = {k_otimo_joelho}")
print(f"Silhouette Score sugere: k = {k_otimo_sil}")
k_otimo = 4  # justificado por critérios de negócio
print("k=4 escolhido — 4 perfis de aluno acionáveis para o sistema SmartAlert")

# ==============================
# Construir modelos clustering
# ==============================

kmeans= KMeans(n_clusters= k_otimo, random_state= SEED, n_init=10)
clusters_kmeans=kmeans.fit_predict(X_clust)
sil_kmeans=silhouette_score(sample, kmeans.predict(sample))
print(f"\nK-Means — Silhouette Score: {sil_kmeans:.4f}")

# ============================
# PCA + K-MEANS
# ============================

from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=SEED)
X_clust_pca = pca.fit_transform(X_clust)

print(f"\nVariância explicada pelos 2 componentes: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# K-Means sem PCA
kmeans = KMeans(n_clusters=k_otimo, random_state=SEED, n_init=10)
clusters_kmeans = kmeans.fit_predict(X_clust)
sil_kmeans = silhouette_score(sample, kmeans.predict(sample))
print(f"K-Means sem PCA — Silhouette Score: {sil_kmeans:.4f}")

# K-Means com PCA
kmeans_pca = KMeans(n_clusters=k_otimo, random_state=SEED, n_init=10)
clusters_pca = kmeans_pca.fit_predict(X_clust_pca)
sil_pca = silhouette_score(X_clust_pca, clusters_pca)
print(f"K-Means com PCA — Silhouette Score: {sil_pca:.4f}")

# Visualização 2D dos clusters com PCA
fig, ax = plt.subplots(figsize=(9, 6))
cores = ['#2E75B6', '#70AD47', '#ED7D31', '#C00000']
sample_vis = np.random.choice(len(clusters_pca), 10000, replace=False)
for i in range(k_otimo):
    mask = clusters_pca[sample_vis] == i
    ax.scatter(X_clust_pca[sample_vis][mask, 0],
               X_clust_pca[sample_vis][mask, 1],
               s=5, alpha=0.4, color=cores[i], label=f'Cluster {i}')
ax.set_title('Clusters com PCA (2 Componentes Principais)', fontweight='bold', fontsize=14)
ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.legend()
plt.tight_layout()
plt.savefig('clusters_pca.png', dpi=150, bbox_inches='tight')
plt.show()

#Hierarchical
sample_idx=X_clust.sample(5000, random_state=SEED).index
hc=AgglomerativeClustering(n_clusters=k_otimo)
clusters_hc=hc.fit_predict(X_clust.loc[sample_idx])
sil_hc=silhouette_score(X_clust.loc[sample_idx],clusters_hc)
print(f"Hierarchical  — Silhouette Score: {sil_hc:.4f}")

print(f"\nMelhor método de clustering: "f"{'K-Means' if sil_kmeans >= sil_hc else 'Hierarchical'}")

#perfil medio por cluster
df_clust=X_clust.copy()
df_clust['cluster']=clusters_kmeans
df_clust['exam_score']=df_regressao['exam_score']

print("\n=== PERFIL MÉDIO POR CLUSTER ===")
perfil = df_clust.groupby('cluster')[['study_hours','class_attendance','sleep_hours','exam_score']].mean().round(3)
print(perfil)

with open('modelo_dt.pkl', 'wb') as f:
    pickle.dump(dt, f)
with open('modelo_rf_clf.pkl', 'wb') as f:
    pickle.dump(rf_clf_otimizado, f)
with open('modelo_xgb_clf.pkl', 'wb') as f:
    pickle.dump(xgb_clf, f)
with open('modelo_kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

#============================  
# Regressão Linear 
#============================

lr = LinearRegression()
lr.fit(X_reg_train, y_reg_train)

#============================
# Random Forest Regressor
#============================

rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=SEED,
    n_jobs=-1
)
rf_reg.fit(X_reg_train, y_reg_train)

#============================
# XGBoost Regressor
#============================

xgb_reg = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=SEED,
    n_jobs=-1
)
xgb_reg.fit(X_reg_train, y_reg_train)

#============================
# MÉTRICAS DE REGRESSÃO 
#============================

def avaliar_regressao(modelo, X_test, y_test, nome):
    y_pred = modelo.predict(X_test)
    y_mean = np.mean(y_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    rae  = np.sum(np.abs(y_test-y_pred)) / np.sum(np.abs(y_test-y_mean)) * 100
    nmae = mae / (y_test.max()-y_test.min()) * 100
    rse  = np.sum((y_test-y_pred)**2) / np.sum((y_test-y_mean)**2) * 100
    rrse = np.sqrt(rse)
    print(f"\n{'='*45}\n  {nome}\n{'='*45}")
    print(f"MAE:   {mae:.4f}  |  RAE:  {rae:.2f}%  |  NMAE: {nmae:.2f}%")
    print(f"MSE:   {mse:.4f}  |  RMSE: {rmse:.4f}")
    print(f"RSE:   {rse:.2f}%  |  RRSE: {rrse:.2f}%")
    print(f"R²:    {r2:.4f}  <- quanto maior melhor (máx. 1)")
    return {'Modelo':nome,'MAE':round(mae,4),'RMSE':round(rmse,4),
            'RAE%':round(rae,2),'R2':round(r2,4)}

resultados_reg = []
for modelo, nome in [(lr,'Regressão Linear'),
                     (rf_reg,'Random Forest'),
                     (xgb_reg,'XGBoost')]:
    resultados_reg.append(avaliar_regressao(modelo, X_reg_test, y_reg_test, nome))

# Tabela comparativa
df_reg = pd.DataFrame(resultados_reg).sort_values('RMSE')
print("\n=== MELHOR MODELO DE REGRESSÃO ===")
print(df_reg.to_string(index=False))
print(f"\n Melhor: {df_reg.iloc[0]['Modelo']}")

#============================
# MÉTRICAS DE CLASSIFICAÇÃO
#============================

def avaliar_classificacao(modelo, X_test, y_test, nome, encoded=False):
    y_pred = modelo.predict(X_test)
    if encoded:
        y_pred = le.inverse_transform(y_pred)
    print(f"\n{'='*45}\n  {nome}\n{'='*45}")
    print(classification_report(y_test, y_pred,
          target_names=['A','B','C','D','E','F']))
    acc    = accuracy_score(y_test, y_pred)
    recall_F = recall_score(y_test, y_pred, labels=['F'], average='macro')
    f1     = f1_score(y_test, y_pred, average='macro')
    return {'Modelo':nome,'Accuracy':round(acc,4),
            'Recall_F':round(recall_F,4),'F1_macro':round(f1,4)}

resultados_clf = []
for modelo, nome, enc in [
    (dt,              'Árvore de Decisão', False),
    (rf_clf_otimizado,'Random Forest',     False),
    (xgb_clf,         'XGBoost',           True)]:
    resultados_clf.append(avaliar_classificacao(modelo, X_clf_test,
                           y_clf_test, nome, enc))

df_clf = pd.DataFrame(resultados_clf).sort_values('Recall_F', ascending=False)
print("\n=== MELHOR MODELO DE CLASSIFICAÇÃO ===")
print(df_clf.to_string(index=False))
print(f"\n Melhor: {df_clf.iloc[0]['Modelo']}")

#============================
# GRÁFICO RSC (Regression Scatter Plot) 
#============================
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def plot_rsc(modelo, X_test, y_test, nome):
    y_pred = modelo.predict(X_test)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.2, s=5, color='#2E75B6')
    lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lim, lim, 'r--', linewidth=2, label='Previsão Perfeita')
    ax.set_xlabel('Valores Reais'); ax.set_ylabel('Valores Previstos')
    ax.set_title(f'RSC — {nome}', fontweight='bold')
    ax.legend(); plt.tight_layout()
    plt.savefig(f'rsc_{nome}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Guardado: rsc_{nome}.png")

#============================
# CURVA REC (Regression Error Characteristic) 
#============================

def plot_rec(modelos_dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#2E75B6','#70AD47','#C00000']
    for (nome, modelo), cor in zip(modelos_dict.items(), colors):
        y_pred = modelo.predict(X_test)
        erros  = np.abs(y_test - y_pred)
        tols   = np.linspace(0, erros.max(), 300)
        taxas  = [np.mean(erros <= t) for t in tols]
        auc    = np.trapezoid(taxas, tols) / tols[-1]
        ax.plot(tols, taxas, color=cor, linewidth=2,
                label=f'{nome} (AUC={auc:.3f})')
    ax.set_xlabel('Tolerância ao Erro'); ax.set_ylabel('Taxa de Acerto')
    ax.set_title('Curva REC — Comparação de Modelos', fontweight='bold')
    ax.legend(); ax.set_facecolor('#F8F8F8'); plt.tight_layout()
    plt.savefig('rec_comparacao.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Guardado: rec_comparacao.png")

# RSC para cada modelo individualmente
for modelo, nome in [(lr, 'Regressão Linear'),
                     (rf_reg, 'Random Forest'),
                     (xgb_reg, 'XGBoost')]:
    plot_rsc(modelo, X_reg_test, y_reg_test, nome)

# REC com todos os modelos juntos
modelos_dict = {
    'Regressão Linear': lr,
    'Random Forest':    rf_reg,
    'XGBoost':          xgb_reg
}
plot_rec(modelos_dict, X_reg_test, y_reg_test)

#============================
# REVISÃO DOS PARÂMETROS USADOS 
#============================

print("\n=== REVISÃO DOS PARÂMETROS USADOS ===")
modelos_params = {
    'Regressão Linear':   lr.get_params(),
    'Random Forest Reg':  rf_reg.get_params(),
    'XGBoost Reg':        xgb_reg.get_params(),
    'Árvore de Decisão':  dt.get_params(),
    'Random Forest Clf':  rf_clf_otimizado.get_params(),
    'XGBoost Clf':        xgb_clf.get_params(),
}
for nome, params in modelos_params.items():
    print(f"\n{nome}: {params}")