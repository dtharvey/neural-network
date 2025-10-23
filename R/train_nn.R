train_nn = function(I1 = 0.6, 
                    I2 = 0.4,
                    t = 0.2, 
                    eta = 0.01, 
                    epochs = 10){
  # define initial weights: input nodes to hidden nodes
  w11 = 0.251
  w12 = 0.103
  w13 = -0.562
  w21 = 0.565
  w22 = -0.359
  w23 = -0.648
  
  # define initial weights: hidden nodes to output node
  v1 = -0.199
  v2 = 0.525
  v3 = 0.450
  
  # set-up vectors to hold calculated values
  errors = rep(0,epochs)
  predicted_o1 = rep(0,epochs)
  w11_values = rep(0,epochs)
  w12_values = rep(0,epochs)
  w13_values = rep(0,epochs)
  w21_values = rep(0,epochs)
  w22_values = rep(0,epochs)
  w23_values = rep(0,epochs)
  v1_values = rep(0,epochs)
  v2_values = rep(0,epochs)
  v3_values = rep(0,epochs)
  
  # training loop
  for (i in 1:epochs){
  
  # calculate weighted sum for the hidden nodes
  net_h1 = I1 * w11 + I2 * w21
  net_h2 = I1 * w12 + I2 * w22
  net_h3 = I1 * w13 + I2 * w23
  
  # calculate activation of hidden nodes using sigmoidal function
  h1 = 1/(1 + exp(-net_h1))
  h2 = 1/(1 + exp(-net_h2))
  h3 = 1/(1 + exp(-net_h3))
  
  # calculate weighted sum for the output node
  net_o1 = v1 * h1 + v2 * h2 + v3 * h3
  
  # calculate activation of output nodes using sigmoidal function
  o1 = 1/(1 + exp(-net_o1))
  
  # calculate error between predicted output and target output
  E = 0.5 * (t - o1)^2
  errors[i] = E
  predicted_o1[i] = o1
  w11_values[i] = w11
  w12_values[i] = w12
  w13_values[i] = w13
  w21_values[i] = w21
  w22_values[i] = w22
  w23_values[i] = w23
  v1_values[i] = v1
  v2_values[i] = v2
  v3_values[i] = v3
  
  # <----- feed-forward process ends and backpropogation begins ----->
  
  # calculate output layer error signal
  delta_o1 = -(t - o1) * o1 * (1 - o1)
  
  # calculate how much hidden-to-output weight contributed to error
  delta_v1 = delta_o1 * h1
  delta_v2 = delta_o1 * h2
  delta_v3 = delta_o1 * h3
  
  # calculate error in hidden layer nodes
  delta_h1 = h1 * (1 - h1) * delta_o1 * v1
  delta_h2 = h2 * (1 - h2) * delta_o1 * v2
  delta_h3 = h3 * (1 - h3) * delta_o1 * v3
  
  #calculate how much input-to-hidden weights contributed to error
  delta_w11 = delta_h1 * I1
  delta_w12 = delta_h2 * I1
  delta_w13 = delta_h3 * I1
  delta_w21 = delta_h1 * I2
  delta_w22 = delta_h2 * I2
  delta_w23 = delta_h3 * I2
  
  # update the hidden-to-output weights to reduce the error
  v1 = v1 - (eta * delta_v1)
  v2 = v2 - (eta * delta_v2)
  v3 = v3 - (eta * delta_v3)
  
  # update the input-to-hidden weights to reduce the error
  w11 = w11 - (eta * delta_w11)
  w12 = w12 - (eta * delta_w12)
  w13 = w13 - (eta * delta_w13)
  w21 = w21 - (eta * delta_w21)
  w22 = w22 - (eta * delta_w22)
  w23 = w23 - (eta * delta_w23)
  
  }
  
  output = list("errors" = errors,
                "epochs" = epochs,
                "predicted_o1" = predicted_o1,
                "w11_values" = w11_values,
                "w12_values" = w12_values,
                "w13_values" = w13_values,
                "w21_values" = w21_values,
                "w22_values" = w22_values,
                "w23_values" = w23_values,
                "v1_values" = v1_values,
                "v2_values" = v2_values,
                "v3_values" = v3_values,
                "I1" = I1,
                "I2" = I2,
                "target" = t,
                "learning" = eta,
                "w11" = w11,
                "w12" = w12,
                "w13" = w13,
                "w21" = w21,
                "w22" = w22,
                "w23" = w23,
                "w11_values" = w11_values,
                "delta_w11" = delta_w11,
                "delta_w12" = delta_w12,
                "delta_w13" = delta_w13,
                "delta_w21" = delta_w21,
                "delta_w22" = delta_w22,
                "delta_w23" = delta_w23,
                "net_h1" = net_h1,
                "net_h2" = net_h2,
                "net_h3" = net_h3,
                "h1" = h1,
                "h2" = h2,
                "h3" = h3,
                "v1" = v1,
                "v2" = v2,
                "v3" = v3,
                "delta_o1" = delta_o1,
                "delta_v1" = delta_v1,
                "delta_v2" = delta_v2,
                "delta_v3" = delta_v3,
                "delta_h1" = delta_h1,
                "delta_h2" = delta_h2,
                "delta_h3" = delta_h3,
                "net_o1" = net_o1,
                "o1" = o1)
}
