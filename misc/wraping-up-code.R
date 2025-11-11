# https://www.datacamp.com/tutorial/neural-network-models-r

# using Neuralnet package

library(neuralnet)
library(magrittr)
library(tidyverse)
library(NeuralNetTools)

iris <- iris %>% mutate_if(is.character, as.factor)

summary(iris)

set.seed(245)
data_rows <- floor(0.80 * nrow(iris))
train_indices <- sample(c(1:nrow(iris)), data_rows)
train_data <- iris[train_indices,]
test_data <- iris[-train_indices,]

model = neuralnet(
  Species~Sepal.Length+Sepal.Width+Petal.Length+Petal.Width,
  data=train_data,
  hidden=c(4,2), algorithm = "backprop", learningrate = 0.01,
  linear.output = FALSE, lifesign = "minimal"
)

plot(model,rep = "best")
plotnet(model)

pred <- predict(model, test_data)
labels <- c("setosa", "versicolor", "virginca")
prediction_label <- data.frame(max.col(pred)) %>%     
  mutate(pred=labels[max.col.pred.]) %>%
  select(2) %>%
  unlist()

table(test_data$Species, prediction_label)

check = as.numeric(test_data$Species) == max.col(pred)
accuracy = (sum(check)/nrow(test_data))*100
print(accuracy)

plotnet(model)

# using the nnet package
library(nnet)
data(iris)
library(NeuralNetTools)

# create training and testing sets
set.seed(123) # for reproducibility
train_index <- sample(1:nrow(iris), 0.7 * nrow(iris)) # 70% for training
iris_train <- iris[train_index, ]
iris_test <- iris[-train_index, ]

# create model
model_nnet <- nnet(Species ~ ., data = iris_train, size = 4, decay = 0.0001, maxit = 500)

# summary of model
summary(model_nnet)

# test model
predictions <- predict(model_nnet, iris_test, type = "class")
confusion_matrix <- table(Actual = iris_test$Species, Predicted = predictions)
print(confusion_matrix)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", round(accuracy, 4)))
plotnet(model_nnet)
plotnet(model_nnet, nid = FALSE, circle_cex = 3, pos_col = "blue", neg_col = "red", var_labs = TRUE)

#
# from https://rpubs.com/Mentors_Ubiqum/Neural_Networks

# https://cran.r-project.org/web/packages/aweSOM/vignettes/aweSOM.html (self-organizing maps)


data(iris)

# One-hot encode the Species variable
iris_nn <- iris %>%
  mutate(setosa = as.numeric(Species == "setosa"),
         versicolor = as.numeric(Species == "versicolor"),
         virginica = as.numeric(Species == "virginica")) %>%
  select(-Species)

# Optional: Normalize numerical features (e.g., min-max scaling)
# For simplicity, we'll skip normalization here, but it's often recommended.
# You can create a custom function for this if desired.

# use the RSNNS package
# Install and load the RSNNS package if you haven't already
install.packages("RSNNS")
library(RSNNS)

# Prepare some sample data for classification
data(iris)
iris <- iris[sample(1:nrow(iris), nrow(iris)),] # Randomize the data
irisValues <- iris[, 1:4]
irisTargets <- decodeClassLabels(iris[, 5])

# Train the MLP model
# This uses standard backpropagation with 5 hidden nodes
mlp_model <- mlp(x = irisValues,
                 y = irisTargets,
                 size = c(5),
                 learnFuncParams = c(0.1),
                 maxit = 100)

# Make predictions
predictions <- predict(mlp_model, irisValues)

# Evaluate the results
confusion_matrix <- table(iris[, 5], getDecodedClassLabels(predictions))
print(confusion_matrix)


model = neuralnet(
  Species~Sepal.Length+Sepal.Width+Petal.Length+Petal.Width,
  data=train_data,
  hidden=c(2,2,2), algorithm = "backprop", learningrate = 0.01,
  linear.output = FALSE, lifesign = "minimal")
plotnet(model)

pred <- predict(model, test_data)
labels <- c("setosa", "versicolor", "virginca")
prediction_label <- data.frame(max.col(pred)) %>%     
  mutate(pred=labels[max.col.pred.]) %>%
  select(2) %>%
  unlist()
table(test_data$Species, prediction_label)
check = as.numeric(test_data$Species) == max.col(pred)
accuracy = (sum(check)/nrow(test_data))*100
print(accuracy)

# xor problem using [2,2,1] and nnet

# Input data for XOR
xor_input <- matrix(c(0,0, 0,1, 1,0, 1,1), ncol=2, byrow=TRUE)

# Output data for XOR
xor_output <- c(0, 1, 1, 0)

# Combine into a data frame
xor_data <- data.frame(x1 = xor_input[,1], x2 = xor_input[,2], y = xor_output)

library(nnet)

# Train the neural network
# size = 2 for the hidden layer, maxit for iterations, linout=FALSE for classification
xor_model <- nnet(y ~ x1 + x2, data = xor_data, size = 2, maxit = 1000, linout = FALSE)

# Predict on the training data
predictions <- predict(xor_model, newdata = xor_data, type = "class")

# View the predictions (will be probabilities, so round them for binary output)
rounded_predictions <- round(predictions)
print(rounded_predictions)

# xor using neuralnet
xor_data <- data.frame(
  x1 = c(0, 0, 1, 1),
  x2 = c(0, 1, 0, 1),
  y = c(0, 1, 1, 0)
)    
# Train the neural network
# y ~ x1 + x2 specifies the formula
# hidden = c(2) means one hidden layer with 2 neurons
# linear.output = FALSE for classification (XOR is a classification problem)
xor_net <- neuralnet(y ~ x1 + x2, data = xor_data, hidden = c(2), linear.output = FALSE)
plot(xor_net)
plotnet(xor_net)
# Create new data for prediction (same as training data for demonstration)
test_data <- data.frame(x1 = c(0, 0, 1, 1), x2 = c(0, 1, 0, 1))

# Compute predictions
predictions <- compute(xor_net, test_data)

# View the raw output (probabilities)
print(predictions$net.result)

# Convert probabilities to binary output (e.g., threshold at 0.5)
predicted_y <- ifelse(predictions$net.result > 0.5, 1, 0)
print(predicted_y)

predictions <- predict(xor_net, test_data, type = "class")
