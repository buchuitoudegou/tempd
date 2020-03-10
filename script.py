import os
import cv2

all_input = os.listdir('./input')
print(all_input)

for img_pth in all_input:
  img = cv2.imread(f'./input/{img_pth}')
  img = cv2.resize(img, (256, 256))
  cv2.imwrite('./input/{}-resize.jpg'.format(img_pth.split('.')[0]), img)
