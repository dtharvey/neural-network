train_xor_nn = function(I1 = 1, I2 = 0, 
                        B1 = 1, B2 = 1,
                        t = 1,
                        eta = 0.01,
                        epochs = 1){
  
  # make data frame for xor problems
  I1 = c(1,1,0,0)
  I2 = c(1,0,1,0)
  t = c(0,1,1,0)
  xor_data = data.frame(I1,I2,t)
  
  # define initial weights between inputs and hidden nodes
  w11 = -0.030
  w12 = -0.463
  w21 = -0.568
  w22 = 0.033
  
  # define initial weights between hidden nodes and output node
  v11 = 0.156
  v21 = 0.446
  
  # define initial weights between bias terms and hidden/output nodes
  u11 = 0.213
  u12 = -0.565
  u21 = -0.168
  
  # set-up vectors to hold calculated values
  errors = rep(0,4*epochs)
  predicted_o1 = rep(0,4*epochs)
  w11_values = rep(0,4*epochs)
  w12_values = rep(0,4*epochs)
  w21_values = rep(0,4*epochs)
  w22_values = rep(0,4*epochs)
  v11_values = rep(0,4*epochs)
  v21_values = rep(0,4*epochs)
  u11_values = rep(0,4*epochs)
  u12_values = rep(0,4*epochs)
  u21_values = rep(0,4*epochs)
  
  # training loop
  for (j in 1:4){
    # set values for I1, I2, and t
    I1 = xor_data$I1[j]
    I2 = xor_data$I2[j]
    t = xor_data$t[j]
    # feed-forward and backpropagate
  for (i in 1:(4*epochs)){

  # calculate weighted sum for the hidden nodes
  net_h1 = I1 * w11 + I2 * w21 + B1 * u11
  net_h2 = I1 * w12 + I2 * w22 + B1 * u12
  
  # calculate activation of hidden nodes using sigmoidal function
  h1 = 1/(1 + exp(-net_h1))
  h2 = 1/(1 + exp(-net_h2))
  
  # calculate weighted sum for the output node
  net_o1 = v11 * h1 + v21 * h2 + B2 * u21
  
  # calculate activation of output nodes using sigmoidal function
  o1 = 1/(1 + exp(-net_o1))
  
  # calculate error between predicted output and target output
  E = 0.5 * (t - o1)^2
  errors[i] = E
  predicted_o1[i] = o1
  w11_values[i] = w11
  w12_values[i] = w12
  w21_values[i] = w21
  w22_values[i] = w22
  v11_values[i] = v11
  v21_values[i] = v21
  u11_values[i] = u11
  u12_values[i] = u12
  u21_values[i] = u21
  
  # <----- feed-forward process ends and backpropogation begins ----->
  
  # calculate output layer error signal
  delta_o1 = -(t - o1) * o1 * (1 - o1)
  
  # calculate how much hidden-to-output weight contributed to error
  delta_v11 = delta_o1 * h1
  delta_v21 = delta_o1 * h2
  
  # calculate error in hidden layer nodes
  delta_h1 = h1 * (1 - h1) * delta_o1 * v11
  delta_h2 = h2 * (1 - h2) * delta_o1 * v21
  
  #calculate how much input-to-hidden weights contributed to error
  delta_w11 = delta_h1 * I1
  delta_w12 = delta_h2 * I1
  delta_w21 = delta_h1 * I2
  delta_w22 = delta_h2 * I2
  
  # update the hidden-to-output weights to reduce the error
  v11 = v11 - (eta * delta_v11)
  v21 = v21 - (eta * delta_v21)
  
  # update the input-to-hidden weights to reduce the error
  w11 = w11 - (eta * delta_w11)
  w12 = w12 - (eta * delta_w12)
  w21 = w21 - (eta * delta_w21)
  w22 = w22 - (eta * delta_w22)
  
  }}
  
  output = list("errors" = errors,
                "epochs" = epochs,
                "predicted_o1" = predicted_o1,
                "w11_values" = w11_values,
                "w12_values" = w12_values,
                "w21_values" = w21_values,
                "w22_values" = w22_values,
                "v11_values" = v11_values,
                "v21_values" = v21_values,
                "u11_values" = u11_values,
                "u12_values" = u12_values,
                "u21_values" = u21_values,
                "learning" = eta,
                "w11" = w11,
                "w12" = w12,
                "w21" = w21,
                "w22" = w22,
                "delta_w11" = delta_w11,
                "delta_w12" = delta_w12,
                "delta_w21" = delta_w21,
                "delta_w22" = delta_w22,
                "net_h1" = net_h1,
                "net_h2" = net_h2,
                "h1" = h1,
                "h2" = h2,
                "v11" = v11,
                "v21" = v21,
                "delta_o1" = delta_o1,
                "delta_v11" = delta_v11,
                "delta_v21" = delta_v21,
                "delta_h1" = delta_h1,
                "delta_h2" = delta_h2,
                "net_o1" = net_o1,
                "o1" = o1)

  }
