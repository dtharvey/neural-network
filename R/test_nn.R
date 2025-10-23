test_nn = function(model, I1, I2, t) {
  w11 = model$w11; w12 = model$w12; w13 = model$w13
  w21 = model$w21; w22 = model$w22; w23 = model$w23
  v1 = model$v1; v2 = model$v2; v3 = model$v3
  
  net_h1 = I1 * w11 + I2 * w21
  net_h2 = I1 * w12 + I2 * w22
  net_h3 = I1 * w13 + I2 * w23
  
  h1 = 1 / (1 + exp(-net_h1))
  h2 = 1 / (1 + exp(-net_h2))
  h3 = 1 / (1 + exp(-net_h3))
  
  net_o1 = v1 * h1 + v2 * h2 + v3 * h3
  o1 = 1 / (1 + exp(-net_o1))
  
  E = 0.5 * (t - o1)^2
  
  return(list(prediction = o1, error = E))
}
