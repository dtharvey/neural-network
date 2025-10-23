# vector with integers from 1 to 5
base_numbers = 1:5

# vector of 10,000 values with base numbers randomly shuffled in 2000 groups of five
index_values = unlist(replicate(2000, sample(base_numbers, size = 5, replace = FALSE)))

# data,frame with five testing samples
input1 = c(0.6, 1.0, 0.6, 0.2, 0.6)
input2 = c(0.4, 0.4, 0.2, 0.2, 0.6)
t = c(0.2, 0.0, 0.0, 0.2, 0.4)
df = data.frame(input1, input2, t)

I1 = df$input1[index_values]
I2 = df$input2[index_values]
t = df$t[index_values]
