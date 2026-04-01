# Carregar o dataset
df <- read.csv("C:/Users/ruian/Downloads/playground-series-s6e1/train.csv")

# Ver as primeiras linhas
head(df)

library(ggplot2)
library(dplyr)
library(ggridges)
library(viridis)

#Distibuicao de notas por curso
ggplot(df, aes(x = exam_score, y = course, fill = course)) +
  geom_density_ridges(
    alpha = 0.75,
    scale = 1.8,
    bandwidth = 3,
    color = "white",
    linewidth = 0.6
  ) +
  scale_fill_brewer(palette = "Set2") +
  scale_x_continuous(breaks = seq(20, 100, by = 10)) +
  labs(
    title    = "Distribuição de Notas de Exame por Curso",
    subtitle = "Cada faixa representa a densidade de alunos para aquela nota",
    x        = "Nota de Exame",
    y        = "Curso"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position  = "none",
    plot.title       = element_text(face = "bold", size = 16),
    plot.subtitle    = element_text(color = "gray50", size = 11),
    panel.grid.minor = element_blank()
  )


#Horas de estudo vs Nota por metodo de estudo
df %>% sample_n(5000) %>%
  ggplot(aes(x = study_hours, y = exam_score, color = sleep_quality)) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 1.2, alpha = 0.15) +
  facet_wrap(~ study_method, ncol = 3) +
  scale_color_manual(
    values = c("good"    = "#70AD47",
               "average" = "#ED7D31",
               "poor"    = "#C00000"),
    name   = "Qualidade de Sono"
  ) +
  scale_x_continuous(breaks = seq(0, 8, by = 2)) +
  labs(
      title    = "Linhas de Tendência: Horas de Estudo vs Nota por Método de Estudo",
      subtitle = "Linha de tendência com intervalo de confiança por qualidade de sono",
      x        = "Horas de Estudo por Dia",
      y        = "Nota de Exame"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 16),
    plot.subtitle    = element_text(color = "gray50", size = 11),
    legend.position  = "bottom",
    panel.grid.minor = element_blank(),
    strip.text       = element_text(face = "bold", size = 11)
  )



