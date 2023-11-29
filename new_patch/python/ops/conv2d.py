import numpy as np

# we just use NHWC to calculate
# no dilation
def Conv2d(img, weight, hi, wi, ci, co, kernel, stride, pad):
  ho = (hi + 2 * pad - kernel) // stride + 1
  wo = (wi + 2 * pad - kernel) // stride + 1
  
  weight = np.array(weight).reshape(co, kernel, kernel, ci)
  img_pad = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), 'constant')
  img_out = np.zeros((ho, wo, co))

  for co_ in range(co):
    for ho_ in range(ho):
      in_h_origin = ho_ * stride - pad
      for wo_ in range(wo):
        in_w_origin = wo_ * stride - pad
        filter_h_start = max(0, -in_h_origin)
        filter_w_start = max(0, -in_w_origin)
        filter_h_end = min(kernel, hi - in_h_origin)
        filter_w_end = min(kernel, wi - in_w_origin)
        acc = float(0)
        for kh_ in range(filter_h_start, filter_h_end):
          hi_index = in_h_origin + kh_
          for kw_ in range(filter_w_start, filter_w_end):
            wi_index = in_w_origin + kw_
            for ci_ in range(ci):
              in_data = img[hi_index][wi_index][ci_]
              weight_data = weight[co_][kh_][kw_][ci_]
              acc = acc + in_data * weight_data
        img_out[ho_][wo_][co_] = acc
  return img_out

def Conv2dOpt(img, weight, hi, wi, ci, co, kernel, stride, pad):
  ho = (hi + 2 * pad - kernel) // stride + 1
  wo = (wi + 2 * pad - kernel) // stride + 1
  
  weight = np.array(weight).reshape(co, kernel, kernel, ci)
  img_pad = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), 'constant')
  img_out = np.zeros((ho, wo, co))

  for co_ in range(co):
    for ho_ in range(ho):
      in_h_origin = ho_ * stride - pad
      for wo_ in range(wo):
        in_w_origin = wo_ * stride - pad
        filter_h_start = max(0, -in_h_origin)
        filter_w_start = max(0, -in_w_origin)
        filter_h_end = min(kernel, hi - in_h_origin)
        filter_w_end = min(kernel, wi - in_w_origin)
        acc = float(0)
        for kh_ in range(filter_h_start, filter_h_end):
          hi_index = in_h_origin + kh_
          for kw_ in range(filter_w_start, filter_w_end):
            wi_index = in_w_origin + kw_
            # use vdot to optimize MAC operation
            acc += np.vdot(img[hi_index][wi_index], weight[co_][kh_][kw_])
        img_out[ho_][wo_][co_] = acc
  return img_out

