import numpy as np

def MaxPool(img):
  hi  = img.shape[0]
  wi = img.shape[1]
  channel = img.shape[2]
  pad = 1
  stride = 2
  kernel = 3
  ho = (hi + 2 * pad - kernel) // stride + 1
  wo = (wi + 2 * pad - kernel) // stride + 1
  
  img_pad = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), 'constant')
  img_out = np.zeros((ho, wo, channel))

  for c_ in range(channel):
    for ho_ in range(ho):
      in_h_origin = ho_ * stride - pad
      for wo_ in range(wo):
        in_w_origin = wo_ * stride - pad
        filter_h_start = max(0, -in_h_origin)
        filter_w_start = max(0, -in_w_origin)
        filter_h_end = min(kernel, hi - in_h_origin)
        filter_w_end = min(kernel, wi - in_w_origin)
        max_x = float(0)
        for kh_ in range(filter_h_start, filter_h_end):
          hi_index = in_h_origin + kh_
          for kw_ in range(filter_w_start, filter_w_end):
            wi_index = in_w_origin + kw_
            in_data = img[hi_index][wi_index][c_]
            max_x = max(in_data, max_x)
        img_out[ho_][wo_][c_] = max_x 
  return img_out

def AvgPool(img):
  hi  = img.shape[0]
  wi = img.shape[1]
  channel = img.shape[2]
  pad = 0
  stride = 1
  kernel = 7
  ho = (hi + 2 * pad - kernel) // stride + 1
  wo = (wi + 2 * pad - kernel) // stride + 1

  img_pad = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), 'constant')
  img_out = np.zeros((ho, wo, channel))

  for c_ in range(channel):
    for ho_ in range(ho):
      in_h_origin = ho_ * stride - pad
      for wo_ in range(wo):
        in_w_origin = wo_ * stride - pad
        filter_h_start = max(0, -in_h_origin)
        filter_w_start = max(0, -in_w_origin)
        filter_h_end = min(kernel, hi - in_h_origin)
        filter_w_end = min(kernel, wi - in_w_origin)
        sum_x = float(0)
        for kh_ in range(filter_h_start, filter_h_end):
          hi_index = in_h_origin + kh_
          for kw_ in range(filter_w_start, filter_w_end):
            wi_index = in_w_origin + kw_
            in_data = img[hi_index][wi_index][c_]
            sum_x = sum_x + in_data
        img_out[ho_][wo_][c_] = sum_x / (kernel * kernel)
  return img_out

