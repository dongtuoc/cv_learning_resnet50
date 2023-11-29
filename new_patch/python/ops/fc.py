import numpy as np

def FullyConnect(img, weight, bias):
  '''
  fc compute [2048] * [1000, 2048] = [1000]
  img : [1, 1, 2048] from last layer
  weight: need reshpe to [1000, 2048]
  bias: [1000]
  '''
  img_new = img.reshape(2048)
  weight_new = np.array(weight).reshape([1000, 2048])
  bias_new = np.array(bias).reshape(1000)
  out = np.zeros(1000)
  for i in range(1000):
    sum_x = float(0)
    for j in range(2048):
      l = img_new[j]
      r = weight_new[i][j]
      sum_x = sum_x + l * r
    out[i] = sum_x + bias_new[i]
  return out

def FullyConnectOpt(img, weight, bias):
  '''
  fc compute [2048] * [1000, 2048] = [1000]
  img : [1, 1, 2048] from last layer
  weight: need reshpe to [1000, 2048]
  bias: [1000]
  '''
  img_new = img.reshape(2048)
  weight_new = np.array(weight).reshape([1000, 2048])
  bias_new = np.array(bias).reshape(1000)
  out = np.zeros(1000)

  for i in range(1000):
    # use vdot to optimize MAC operation
    sum_x = np.vdot(img_new, weight_new[i])
    out[i] = sum_x + bias_new[i]
  return out

