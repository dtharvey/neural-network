# r code to test a neural network following feed-forward and backpropagation training; code is for the xor problem using a [2,2,1] network; testing can use any of the four possible training samples

# define function to test trained network
test = function(model, I1, I2, t) {
  B1 = 1
  B2 = 1
  
  w11 = model$w11[model$steps]
  w12 = model$w12[model$steps]
  w21 = model$w21[model$steps]
  w22 = model$w22[model$steps]
  
  v11 = model$v11[model$steps]
  v21 = model$v21[model$steps]
  
  u11 = model$u11[model$steps]
  u12 = model$u12[model$steps]
  u21 = model$u21[model$steps]
  
  net_h1 = I1 * w11 + I2 * w21 + B1 * u11
  net_h2 = I1 * w12 + I2 * w22 + B1 * u12
  
  h1 = 1 / (1 + exp(-net_h1))
  h2 = 1 / (1 + exp(-net_h2))
  
  net_o1 = v11 * h1 + v21 * h2 + B2 * u21
  o1 = 1 / (1 + exp(-net_o1))
  
  E = 0.5 * (t - o1)^2
  
  return(list(prediction = o1, error = E))
}
