import cv2
from scipy.interpolate import spline
import numpy as np
import matplotlib.pyplot as plt

def draw_spline_curve(img, points):
  x = [p[0] for p in points]
  y = [p[1] for p in points]
  x_new = np.linspace(min(x), max(x), 70)
  y_new = spline(x, y, x_new)
  coord = []
  for i in range(0, len(x_new)):
    idx = (int(round(x_new[i])), int(round(y_new[i])))
    coord.append(idx)
    img[idx[1]][idx[0]] = [0, 0, 255]
  return img, coord

def bfs_dye(img, points, color):
  bfs_queue = []
  direction = [(-1, 0), (1, 0), (0, 1), (0, -1)]
  xs = [p[0] for p in points]
  ys = [p[1] for p in points]
  center = ((max(xs) + min(xs)) // 2, (max(ys) + min(ys)) // 2)
  print(center)
  bfs_queue.append(center)
  while len(bfs_queue) > 0:
    cur_pt = bfs_queue[0]
    bfs_queue = bfs_queue[1:]
    for cur_dir in direction:
      new_pt = (cur_pt[0] + cur_dir[0], cur_pt[1] + cur_dir[1])
      if not new_pt in points and not (img[new_pt[1]][new_pt[0]] == color).all():
        # reach the boundary
        bfs_queue.append(new_pt)
    img[cur_pt[1]][cur_pt[0]] = color
  print('dye finished')
  return img

def scatter(img, points, color):
  for pt in points:
    img[pt[1]][pt[0]] = color
  return img