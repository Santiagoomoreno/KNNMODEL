# Load necessary libraries
install.packages(c("caret", "MASS", "glmnet", "boot"))
library(tidyverse)
library(caret)
library(MASS)
library(glmnet)
library(boot)

# Part 0: GitHub Repository Setup (must be done manually)
# Part 1: Data Exploration and Manipulation
data <- read.csv("diabetes_012_health_indicators_BRFSS2015.csv")

# Take 1% of the data randomly for each task
set.seed(123)
sampled_data <- data %>% sample_frac(0.01)
sampled_data$DiabetesBinary <- make.names(factor(ifelse(sampled_data$Diabetes_012 > 0, 1, 0)))
sampled_data$DiabetesBinary <- factor(sampled_data$DiabetesBinary, levels = c("X0", "X1"))

# Subsetting data for each regression task
sampled_data_bmi <- sampled_data
sampled_data_menthlth <- sampled_data
sampled_data_physhtlth <- sampled_data

# Part 2: KNN
# Split the sampled data into training and test sets (80% - 20%)
set.seed(123)  # Set seed for reproducibility
train_index <- createDataPartition(sampled_data$DiabetesBinary, p = 0.8, list = FALSE, times = 1)
train_data <- sampled_data[train_index, ]
test_data <- sampled_data[-train_index, ]

# Define training control using 10-fold cross-validation and class probabilities
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)

# Train the KNN model
set.seed(20)  # Set seed for reproducibility
knn_model <- train(DiabetesBinary ~ ., data = train_data, method = "knn", trControl = ctrl, tuneLength = 10)  # Test k values from 1 to 10

# Show tuning results to select the best value of 'k'
print(knn_model)

# Make predictions on the test set
predictions_knn <- predict(knn_model, newdata = test_data)

# Evaluate the KNN model using the class probabilities
confusionMatrix(predictions_knn, test_data$DiabetesBinary)

# Part 3: Linear and Multilinear Regression (as shown in the previous response)

# BMI Regression
model_bmi <- lm(BMI ~ ., data = sampled_data_bmi)
cv_results_bmi <- cv.glm(data = sampled_data_bmi, glmfit = model_bmi, K = 10)
print(cv_results_bmi)

# MentHlth Regression
model_menthlth <- lm(MentHlth ~ ., data = sampled_data_menthlth)
cv_results_menthlth <- cv.glm(data = sampled_data_menthlth, glmfit = model_menthlth, K = 10)
print(cv_results_menthlth)

# PhysHlth Regression
model_physhtlth <- lm(PhysHlth ~ ., data = sampled_data_physhtlth)
cv_results_physhtlth <- cv.glm(data = sampled_data_physhtlth, glmfit = model_physhtlth, K = 10)
print(cv_results_physhtlth)



