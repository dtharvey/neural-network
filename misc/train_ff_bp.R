# r code for training a neural network through feed-forward and backpropagation

# define function to train network
train_ff_bp = function(I1 = 1, I2 = 0, t = 1, training_samples = 1, 
                       eta = 0.05, epochs = 10){
  
  # bias values (constant value)
  B1 = 1
  B2 = 2
  
  
  
  # total steps
  steps = training_samples * epochs
  
  # vectors to store weights, error, and other values of interest
  w11 = rep(0, steps)
  w12 = rep(0, steps)
  w21 = rep(0, steps)
  w22 = rep(0, steps)
  
  v11 = rep(0,steps)
  v21 = rep(0,steps)
  
  u11 = rep(0,steps)
  u12 = rep(0,steps)
  u21 = rep(0,steps)
  
  E = rep(0,steps)
  net_o1 = rep(0,steps)
  o1 = rep(0,steps)
  
  # define initial weights: input nodes to hidden nodes
  w11[1] = -0.030
  w12[1] = -0.463
  w21[1] = -0.568
  w22[1] = +0.033
  
  # define initial weights: hidden nodes to output node
  v11[1] = +0.156
  v21[1] = +0.446
  
  # define initial weights: bias nodes to hidden nodes and output node
  u11[1] = +0.213
  u12[1] = -0.565
  u21[1] = -0.168
  
  # training loop begins here
  
  for(i in 1:steps){
    
    # calculate weighted sum for the hidden nodes
    net_h1 = I1 * w11[i] + I2 * w21[i] + B1 * u11[i]
    net_h2 = I1 * w12[i] + I2 * w22[i] + B1 * u12[i]
    
    # calculate activation of hidden nodes using sigmoidal function
    h1 = 1/(1 + exp(-net_h1))
    h2 = 1/(1 + exp(-net_h2))
    
    # calculate weighted sum for the output node
    net_o1[i] = v11[i] * h1 + v21[i] * h2 + B2 * u21[i]
    
    # calculate activation of output nodes using sigmoidal function
    o1[i] = 1/(1 + exp(-net_o1[i]))
    
    # calculate error between predicted output (value of actout) and target output
    E[i] = 0.5 * (t - o1[i])^2
    
    # <----- feed-forward process ends and backpropogation begins ----->
    
    # calculate output layer error signal
    delta_o1 = -(t - o1[i]) * o1[i] * (1 - o1[i])
    
    # calculate how much hidden-to-output weight contributed to error
    delta_v11 = delta_o1 * h1
    delta_v21 = delta_o1 * h2
    
    # calculate error in hidden layer nodes
    delta_h1 = h1 * (1 - h1) * delta_o1 * v11[i]
    delta_h2 = h2 * (1 - h2) * delta_o1 * v21[i]
    
    #calculate how much input-to-hidden weights contributed to error
    delta_w11 = delta_h1 * I1
    delta_w12 = delta_h2 * I1
    delta_w21 = delta_h1 * I2
    delta_w22 = delta_h2 * I2
    
    # update the hidden-to-output weights to reduce the error
    v11[i+1] = v11[i] - (eta * delta_v11)
    v21[i+1] = v21[i] - (eta * delta_v21)
    
    # update the input-to-hidden weights to reduce the error
    w11[i+1] = w11[i] - (eta * delta_w11)
    w12[i+1] = w12[i] - (eta * delta_w12)
    w21[i+1] = w21[i] - (eta * delta_w21)
    w22[i+1] = w22[i] - (eta * delta_w22)
    
    # update the bias weights
    u11[i+1] = u11[i] - (eta * delta_h1 * B1)
    u12[i+1] = u12[i] - (eta * delta_h2 * B1)
    u21[i+1] = u21[i] - (eta * delta_o1 * B2)
    
  }
  out = list(w11 = w11, w12 = w12, w21 = w21, w22 = w22,
             v11 = v11, v21 = v21, 
             u11 = u11, u12 = u12, u21 = u21,
             o1 = o1, net_o1 = net_o1, E = E, 
             epochs = epochs, steps = steps)
}
