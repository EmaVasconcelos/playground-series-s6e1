# EduData — AcademicAlert

> **Paradigmas de Aprendizagem Automática** · Universidade do Minho · 2025/2026

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red?style=for-the-badge)
![R](https://img.shields.io/badge/R-4.x-276DC3?style=for-the-badge&logo=r&logoColor=white)
![CRISP-DM](https://img.shields.io/badge/Metodologia-CRISP--DM-purple?style=for-the-badge)

</div>

---

## Sobre o Projeto

O **EduData AcademicAlert** é um sistema de previsão de desempenho académico desenvolvido no âmbito da cadeira de Paradigmas de Aprendizagem Automática.

A ideia é simples: e se conseguíssemos saber **antes do exame** quais os alunos que precisam de ajuda?

Usando dados comportamentais como horas de estudo, assiduidade e qualidade do sono, construímos modelos de Machine Learning que:

- **Preveem a nota final** de cada aluno (Regressão)
- **Classificam o aluno** numa categoria A–F (Classificação)
- **Agrupam alunos** em perfis comportamentais (Clustering)

---

## Estrutura do Projeto

```
Grupo4_PL1_PAA_P1/
|
|-- projetopaa.py               # Pessoa 1 — Data Preparation
|-- modelacao_p2.py             # Pessoa 2 — Classificacao & Clustering
|-- modelacao_p3.py             # Pessoa 3 — Regressao & Avaliacao Final
|-- graficosr.R                 # Graficos em R (Ridgeline + Facet Wrap)
|
|-- train.csv                   # Dataset principal (630.000 registos)
|-- test.csv                    # Dataset de teste
|
|-- df_regressao_minmax.csv
|-- df_classificacao_minmax.csv
|-- df_regressao_standard.csv
|
|-- modelo_dt.pkl               # Arvore de Decisao
|-- modelo_rf_clf.pkl           # Random Forest (classificacao)
|-- modelo_xgb_clf.pkl          # XGBoost (classificacao)
|-- modelo_kmeans.pkl           # K-Means
|-- pca_model.pkl               # PCA
|-- pca_kmeans.pkl              # K-Means com PCA
|-- modelo_hierarquico.pkl      # Hierarchical Clustering
|-- label_encoder.pkl           # LabelEncoder (classes A-F)
|
|-- fig1_distribuicao_exam_score.png
|-- fig2_matriz_correlacao.png
|-- fig3_boxplot_genero.png
|-- fig4_violin_dificuldade.png
```

---

## Dataset

**Kaggle Playground Series S6E1** — Student Exam Score Prediction

| | |
|---|---|
| Fonte | [kaggle.com/competitions/playground-series-s6e1](https://www.kaggle.com/competitions/playground-series-s6e1/data) |
| Dimensao | 630.000 registos x 13 colunas |
| Variavel alvo | `exam_score` (0-100) / `classe` (A-F) |
| Valores em falta | 0 |
| Duplicados | 0 |

---

## Instalacao

```bash
# Clonar o repositorio
git clone https://github.com/seugrupo/Grupo4_PL1_PAA_P1.git
cd Grupo4_PL1_PAA_P1

# Instalar dependencias Python
pip install pandas numpy scikit-learn xgboost matplotlib seaborn imbalanced-learn

# Para R (graficos)
install.packages(c("ggplot2", "dplyr", "ggridges"))
```

---

## Como Correr

```bash
# 1. Data Preparation (Pessoa 1)
python projetopaa.py

# 2. Classificacao & Clustering (Pessoa 2)
python modelacao_p2.py

# 3. Regressao & Avaliacao Final (Pessoa 3)
python modelacao_p3.py

# 4. Graficos em R
Rscript graficosr.R
```

---

## Modelos Desenvolvidos

### Classificacao (A-F)

| Modelo | Accuracy | Recall F | F1-Macro |
|--------|----------|----------|----------|
| Arvore de Decisao | 0.4720 | **0.7171** | 0.4841 |
| Random Forest | 0.4998 | 0.7103 | 0.5126 |
| XGBoost | **0.5365** | 0.6369 | **0.5436** |

O Recall da classe F e a metrica prioritaria — queremos detetar o maximo de alunos em risco de reprovar.

### Clustering (k=4)

| Cluster | Perfil | Alerta SmartAlert |
|---------|--------|-------------------|
| 0 | Alto Desempenho | Sem alerta |
| 1 | Desempenho Moderado | Monitorizar |
| 2 | Em Atencao | Alerta amarelo |
| 3 | Em Risco de Reprovar | Alerta vermelho |

---

## Tecnologias

```python
# Python
pandas           # manipulacao de dados
numpy            # operacoes numericas
scikit-learn     # modelos ML, metricas, pre-processamento
xgboost          # gradient boosting
imbalanced-learn # SMOTE
matplotlib       # visualizacao
seaborn          # visualizacao estatistica
pickle           # guardar modelos

# R
ggplot2          # graficos
ggridges         # ridgeline plots
dplyr            # manipulacao de dados
```

---

## Metodologia

Este projeto segue a metodologia **CRISP-DM**:

```
Business Understanding -> Data Understanding -> Data Preparation
        -> Modeling -> Evaluation -> Deployment
```

---

<div align="center">
  <sub>Universidade do Minho · 2025/2026</sub>
</div>
