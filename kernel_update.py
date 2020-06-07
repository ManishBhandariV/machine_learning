import numpy as np
a = [[-0.5,1],[-1,-1.5],[-1.5,1.5],[1.5,-0.5],[0.5,-0.5]]
b = [[-0.5,-1,-1.5,1.5,0.5],[1,-1.5,1.5,-0.5,-0.5]]
c = np.dot(a, b)
kernel = np.square(c)
y = [1,-1,-1,-1,1]
o = [1,1,1,1,1]
alpha = [0,0,0,0,0]
bias = 0
score = [0,0,0,0,0]
bias_val = [0,0,0,0,0]
while(min(score)<= 0):
 score.clear()
 bias_val.clear()
 for i in range(5):
    sum = 0
    for j in range(5):
      v = alpha[j]*y[j]*kernel[i][j]
      sum = sum + v
    value = y[i]*(sum + bias)
    score.append(value)
    #print(value)
    if value <= 0:
        alpha[i] = alpha[i] + 1
        bias = bias + y[i]*4.5
        bias_val.append(bias)
 print('value =',score)
 print('bias = ',bias_val)
 print("alpha = ",alpha)






