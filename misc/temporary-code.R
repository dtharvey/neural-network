# r code written to accompany solution from Claude

# define neural network structure
input_nodes = 2
hidden_nodes = 3
output_nodes = 1

# define inputs, target output, and learning rate
I1 = 0.6
I2 = 0.4
t = 0.2
eta = 0.01

# define weights: input nodes to hidden nodes
w11 = 0.251
w12 = 0.103
w13 = -0.562
w21 = 0.565
w22 = -0.359
w23 = -0.648

# define weights: hidden nodes to output node
v1 = -0.199
v2 = 0.525
v3 = 0.450

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

# calculate error between predicted output (value of actout) and target output
E = 0.5 * (t - o1)^2

# <----- feed-forward process ends and backpropogation begins ----->

# calculate output layer error signal
delta_o1 = -(t - o1) * o1 * (1 - o1)

# calculate how much hidden-to-output weight contributed to error
grad_v1 = delta_o1 * h1
grad_v2 = delta_o1 * h2
grad_v3 = delta_o1 * h3

# calculate error in hidden layer nodes
delta_h1 = h1 * (1 - h1) * delta_o1 * v1
delta_h2 = h2 * (1 - h2) * delta_o1 * v2
delta_h3 = h3 * (1 - h3) * delta_o1 * v3

#calculate how much input-to-hidden weights contributed to error
delta_w11 = delta_h1 * 11
delta_w12 = delta_h2 * 11
delta_w13 = delta_h3 * 11
delta_w21 = delta_h1 * 12
delta_w22 = delta_h2 * 12
delta_w23 = delta_h3 * 12

# update the hidden-to-output weights to reduce the error
new_v1 = v1 - (eta * grad_v1)
new_v2 = v2 - (eta * grad_v2)
new_v3 = v3 - (eta * grad_v3)

# update the input-to-hidden weights to reduce the error
new_w11 = w11 - (eta * delta_w11)
new_w12 = w12 - (eta * delta_w12)
new_w13 = w13 - (eta * delta_w13)
new_w21 = w21 - (eta * delta_w21)
new_w22 = w22 - (eta * delta_w22)
new_w23 = w23 - (eta * delta_w23)


