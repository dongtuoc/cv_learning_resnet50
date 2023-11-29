
def BatchNorm(img, mean, var, gamma, bias):
  h = img.shape[0]
  w = img.shape[1]
  c = img.shape[2]

  for c_ in range(c):
    data = img[:, :, c_]
    data_ = (data - mean[c_]) / (pow(var[c_] + 1e-5, 0.5))
    data_ = data_ * gamma[c_]
    data_ = data_ + bias[c_]
    img[:, :, c_] = data_
  return img
