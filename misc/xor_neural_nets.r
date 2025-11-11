# ============================================================================
# XOR PROBLEM USING nnet AND neuralnet PACKAGES
# Network Architecture: [2,2,1]
# ============================================================================

# Load required packages
library(nnet)
library(neuralnet)

# ============================================================================
# PART 1: PREPARE XOR DATA
# ============================================================================

# Create XOR dataset
xor_data <- data.frame(
  x1 = c(0, 0, 1, 1),
  x2 = c(0, 1, 0, 1),
  y = c(0, 1, 1, 0)
)

print("XOR Training Data:")
print(xor_data)

# ============================================================================
# PART 2: TRAIN USING nnet PACKAGE
# ============================================================================

cat("\n=== TRAINING WITH nnet PACKAGE ===\n")

# Set seed for reproducibility
set.seed(123)

# Train the neural network
# size = 2 means 2 hidden nodes
# maxit = 1000 increases max iterations (default is 100)
# linout = FALSE for binary classification (uses logistic output)
# trace = FALSE suppresses iteration output
nnet_model <- nnet(y ~ x1 + x2, 
                   data = xor_data,
                   size = 2,
                   linout = FALSE,
                   maxit = 1000,
                   trace = FALSE)

# Display model summary
cat("\nnnet Model Summary:\n")
print(summary(nnet_model))

# Make predictions
nnet_predictions <- predict(nnet_model, xor_data)
nnet_predictions_binary <- ifelse(nnet_predictions > 0.5, 1, 0)

# Calculate accuracy
nnet_accuracy <- mean(nnet_predictions_binary == xor_data$y) * 100

# Display results
cat("\nnnet Predictions:\n")
results_nnet <- data.frame(
  x1 = xor_data$x1,
  x2 = xor_data$x2,
  actual = xor_data$y,
  predicted_prob = round(nnet_predictions, 4),
  predicted_class = nnet_predictions_binary
)
print(results_nnet)
cat(sprintf("\nnnet Accuracy: %.1f%%\n", nnet_accuracy))

# ============================================================================
# PART 3: TRAIN USING neuralnet PACKAGE
# ============================================================================

cat("\n=== TRAINING WITH neuralnet PACKAGE ===\n")

# Set seed for reproducibility
set.seed(123)

# Train the neural network
# hidden = 2 means 2 hidden nodes
# linear.output = FALSE for classification (uses logistic activation)
# threshold = 0.01 is the stopping criterion for partial derivatives
neuralnet_model <- neuralnet(y ~ x1 + x2,
                              data = xor_data,
                              hidden = 2,
                              linear.output = FALSE,
                              threshold = 0.01)

# Display model information
cat("\nneuralnet Model Information:\n")
cat(sprintf("Steps to convergence: %d\n", neuralnet_model$result.matrix[1]))
cat(sprintf("Error: %.6f\n", neuralnet_model$result.matrix[2]))

# Make predictions
# predict() returns predictions as a matrix
neuralnet_predictions <- predict(neuralnet_model, xor_data[, c("x1", "x2")])
neuralnet_pred_values <- as.vector(neuralnet_predictions)
neuralnet_predictions_binary <- ifelse(neuralnet_pred_values > 0.5, 1, 0)

# Calculate accuracy
neuralnet_accuracy <- mean(neuralnet_predictions_binary == xor_data$y) * 100

# Display results
cat("\nneuralnet Predictions:\n")
results_neuralnet <- data.frame(
  x1 = xor_data$x1,
  x2 = xor_data$x2,
  actual = xor_data$y,
  predicted_prob = round(neuralnet_pred_values, 4),
  predicted_class = neuralnet_predictions_binary
)
print(results_neuralnet)
cat(sprintf("\nneuralnet Accuracy: %.1f%%\n", neuralnet_accuracy))

# ============================================================================
# PART 4: VISUALIZE NETWORKS (optional)
# ============================================================================

cat("\n=== VISUALIZATION ===\n")
cat("To visualize the nnet model, install NeuralNetTools:\n")
cat("  install.packages('NeuralNetTools')\n")
cat("  library(NeuralNetTools)\n")
cat("  plotnet(nnet_model)\n\n")



xor_data <- data.frame(
  x1 = c(0, 0, 1, 1),
  x2 = c(0, 1, 0, 1),
  y = c(0, 1, 1, 0)
)

nn2 = neuralnet(y ~ x1 + x2,
                data = xor_data,
                hidden = 2,
                linear.output = FALSE,
                threshold = 0.001,
                rep = 1,
                learningrate = 0.01,
                algorithm = "backprop",
                lifesign = "minimal",
                err.fct = "sse",
                act.fct = "logistic")




# Generate some sample data for a regression task
set.seed(123)
data <- data.frame(
  x1 = runif(100, 0, 10),
  x2 = runif(100, 0, 10)
)
data$y <- with(data, 2 * x1 + 3 * x2 + rnorm(100, 0, 1))

# Train the neural network
# The 'err.fct' argument can be set to 'sse' (sum of squared errors) or 'ce' (cross-entropy)
# The 'linear.output' is TRUE for regression, FALSE for classification
nn_model <- neuralnet(y ~ x1 + x2, data = data, hidden = c(5, 3), 
                      linear.output = TRUE, err.fct = "sse",
                      rep = 1) # rep=1 for a single training run

# Extract the error history
# The 'result.matrix' contains the error for each iteration
error_history <- nn_model$result.matrix["error", ]

# You can also get other metrics like steps (iterations)
steps_history <- nn_model$result.matrix["steps", ]

# Plot the error over time
plot(steps_history, error_history, type = "l", 
     xlab = "Iterations", ylab = "Error (SSE)",
     main = "Error Decrease During Neural Network Training")
