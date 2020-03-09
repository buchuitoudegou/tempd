import cv2
import numpy as np
from tqdm import tqdm
from skimage.transform import rescale

MAX_NUM = 0xffffff

def bilinear_interpolate(src, i, j):
  src_w = src.shape[0]
  src_h = src.shape[1]
  src_x = j
  src_y = i
  src_x_0 = int(np.floor(j))
  src_y_0 = int(np.floor(i))
  src_x_1 = min(src_x_0 + 1, src_w - 1)
  src_y_1 = min(src_y_0 + 1, src_h - 1)
  value0 = (src_x_1 - src_x) * src[src_y_0][src_x_0] + (src_x - src_x_0) * src[src_y_0][src_x_1]
  value1 = (src_x_1 - src_x) * src[src_y_1, src_x_0] + (src_x - src_x_0) * src[src_y_1, src_x_1]
  return ((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1).astype(int)

def mls_affine_deformation(image, p, q, alpha=1.0):
  # Change (x, y) to (row, col)
  q = q[:, [1, 0]]
  p = p[:, [1, 0]]
  plen = p.shape[0]
  def mls(pt):
    w = 1.0 / np.sum((p - pt) ** 2, axis=1) ** alpha

    p_star = np.sum(p.T * w) / np.sum(w)
    q_star = np.sum(q.T * w) / np.sum(w)

    p_hat = p - p_star
    q_hat = q - q_star

    p_hat1 = p_hat.reshape((plen, 1, 2))
    p_hat2 = p_hat.reshape((plen, 2, 1))
    # q_hat = q_hat.reshape((plen, 1, 2))
    w = w.reshape((plen, 1, 1))

    pTwp = np.sum(p_hat2 * w * p_hat1, axis=0)
    try:
      M = np.linalg.inv(pTwp)
    except np.linalg.linalg.LinAlgError:
      if np.linalg.det(pTwp) < 1e-8:
        new_v = pt + q_star - p_star
        return new_v
      else:
        raise
    mul_left = (pt - p_star).reshape((1, 2))
    mul_right = np.sum(p_hat2 * w * q_hat[:, np.newaxis, :], axis=0)
    target = np.dot(np.dot(mul_left, M), mul_right) + q_star
    return target
  
  transform_img = np.zeros(image.shape)
  for i in range(0, image.shape[0]):
    for j in range(0, image.shape[1]):
      target = mls(np.array([i, j]))
      target = target.reshape((2,))
      if target[0] in range(0, image.shape[0]) and target[1] in range(0, image.shape[1]):
        # transform_img[i][j] = image[int(round(target[0]))][int(round(target[1]))]
        transform_img[i][j] = bilinear_interpolate(image, target[0], target[1])
  cv2.imwrite('test.jpg', transform_img)
  return transform_img

def mls_affine_deformation_inv(image, p, q, alpha=1.0, density=1.0):
  width = image.shape[1]
  height = image.shape[0]
  # Change (x, y) to (row, col)
  q = q[:, [1, 0]]
  p = p[:, [1, 0]]

  # Make grids on the original image
  gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
  gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
  vy, vx = np.meshgrid(gridX, gridY)
  grow = vx.shape[0]  # grid rows
  gcol = vx.shape[1]  # grid cols
  ctrls = p.shape[0]  # control points

  # Compute
  reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
  reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
  reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
  
  w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
  w[w == np.inf] = 2**31 - 1
  pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
  phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
  qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
  qhat = reshaped_q - qstar                                                           # [ctrls, 2, grow, gcol]

  reshaped_phat = phat.reshape(ctrls, 2, 1, grow, gcol)                               # [ctrls, 2, 1, grow, gcol]
  reshaped_phat2 = phat.reshape(ctrls, 1, 2, grow, gcol)                              # [ctrls, 2, 1, grow, gcol]
  reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol)                               # [ctrls, 1, 2, grow, gcol]
  reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]
  pTwq = np.sum(reshaped_phat * reshaped_w * reshaped_qhat, axis=0)                   # [2, 2, grow, gcol]
  try:
    inv_pTwq = np.linalg.inv(pTwq.transpose(2, 3, 0, 1))                            # [grow, gcol, 2, 2]
    flag = False
  except np.linalg.linalg.LinAlgError:
    flag = True
    det = np.linalg.det(pTwq.transpose(2, 3, 0, 1))                                 # [grow, gcol]
    det[det < 1e-8] = np.inf
    reshaped_det = det.reshape(1, 1, grow, gcol)                                    # [1, 1, grow, gcol]
    adjoint = pTwq[[[1, 0], [1, 0]], [[1, 1], [0, 0]], :, :]                        # [2, 2, grow, gcol]
    adjoint[[0, 1], [1, 0], :, :] = -adjoint[[0, 1], [1, 0], :, :]                  # [2, 2, grow, gcol]
    inv_pTwq = (adjoint / reshaped_det).transpose(2, 3, 0, 1)                       # [grow, gcol, 2, 2]
  mul_left = reshaped_v - qstar                                                       # [2, grow, gcol]
  reshaped_mul_left = mul_left.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)        # [grow, gcol, 1, 2]
  mul_right = np.sum(reshaped_phat * reshaped_w * reshaped_phat2, axis=0)             # [2, 2, grow, gcol]
  reshaped_mul_right =mul_right.transpose(2, 3, 0, 1)                                 # [grow, gcol, 2, 2]
  temp = np.matmul(np.matmul(reshaped_mul_left, inv_pTwq), reshaped_mul_right)        # [grow, gcol, 1, 2]
  reshaped_temp = temp.reshape(grow, gcol, 2).transpose(2, 0, 1)                      # [2, grow, gcol]

  # Get final image transfomer -- 3-D array
  transformers = reshaped_temp + pstar                                                # [2, grow, gcol]

  # Correct the points where pTwp is singular
  if flag:
    blidx = det == np.inf    # bool index
    transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
    transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

  # Removed the points outside the border
  transformers[transformers < 0] = 0
  transformers[0][transformers[0] > height - 1] = 0
  transformers[1][transformers[1] > width - 1] = 0

  # Mapping original image
  transformed_image = image[tuple(transformers.astype(np.int16))]    # [grow, gcol]

  # Rescale image
  transformed_image = rescale(transformed_image, scale=1.0 / density, mode='reflect')

  return transformed_image
