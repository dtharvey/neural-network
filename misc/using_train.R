# testing
# Train the network
trained <- train(training_samples = 4, eta = 0.001, epochs = 10000)

# Plot per-step error (oscillating pattern)
plot(1:500, trained$E[1:500], type = "l", 
     xlab = "Step", ylab = "Error", 
     main = "Per-Step Error", col = "gray60")

# Plot epoch-averaged error (smooth curve)
plot(1:500, trained$E_epoch[1:500], type = "l", 
     xlab = "Epoch", ylab = "Average Error", 
     main = "Epoch-Averaged Error", col = "blue", lwd = 2)

# Or overlay both (note different x-axes)
plot(1:500, trained$E[1:500], type = "l", 
     xlab = "Step", ylab = "Error", col = "gray70",
     main = "Error: Per-Step vs Epoch Average")
# Convert epoch indices to step indices for overlay
epoch_steps <- seq(2, 500, by = 4)  # Middle of each epoch's 4 steps
lines(epoch_steps, trained$E_epoch[1:125], col = "red", lwd = 2)
legend("topright", legend = c("Per-step", "Epoch average"), 
       col = c("gray70", "red"), lwd = c(1, 2))

