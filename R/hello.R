#' This code is designed to perform a machine learning analysis using the k-Nearest Neighbors (KNN) algorithm to predict the binary variable "DiabetesBinary". Here is a step-by-step description of the code:
# Load necessary libraries
install.packages("caret")
library(tidyverse)
library(caret)
library(devtools)
library(roxygen2)

# Part 0: GitHub Repository Setup (must be done manually)
# Part 1: Data Exploration and Manipulation
data <- read.csv("diabetes_012_health_indicators_BRFSS2015.csv")
data$DiabetesBinary <- make.names(factor(ifelse(data$Diabetes_012 > 0, 1, 0)))

# Convert the DiabetesBinary variable into a two-level factor
data$DiabetesBinary <- factor(data$DiabetesBinary, levels = c("X0", "X1"))

# Part 2: KNN
# Split the data into training and test sets (80% - 20%)
set.seed(123)  # Set seed for reproducibility
train_index <- createDataPartition(data$DiabetesBinary, p = 0.8,
                                   list = FALSE,
                                   times = 1)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Definir el control de entrenamiento usando 10-fold cross-validation y métrica de exactitud (Accuracy)
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE,
                     summaryFunction = twoClassSummary, metric = "Accuracy")

# Entrenar el modelo KNN
set.seed(20)  # Establecer semilla para reproducibilidad
knn_model <- train(DiabetesBinary ~ ., data = train_data,
                   method = "knn",
                   trControl = ctrl,
                   tuneLength = 10)  # Probar valores de k de 1 a 10

# Mostrar resultados del tuning para seleccionar el mejor valor de 'k'
print(knn_model)

# Realizar predicciones en el conjunto de prueba
predictions <- predict(knn_model, newdata = test_data)

# Evaluar el modelo utilizando la métrica de exactitud (Accuracy)
confusionMatrix(predictions, test_data$DiabetesBinary)

