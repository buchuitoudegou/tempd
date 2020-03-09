import base64
import requests
from config import api_key, api_secret, right_eye_top_keys
import json
import cv2
import numpy as np
import warp
import util

example_pth = './Paul-phase-1.png'
detect_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'

def get_image_landmark(img_pth):
  with open(img_pth, 'rb') as img:
    base64_img = base64.b64encode(img.read())
    body = {
      "api_key": api_key,
      "api_secret": api_secret,
      "image_base64": base64_img,
      "return_landmark": 2,
    }
    resp = requests.post(detect_url, body)
    if resp.status_code == 200:
      return json.loads(str(resp.content, encoding='utf-8'))["faces"][0]["landmark"]
    return None

# image_landmark = get_image_landmark(example_pth)
# eye_top_landmark = [[image_landmark[key]["x"], image_landmark[key]["y"]] for key in right_eye_top_keys]

img = cv2.imread(example_pth)

p = []
q = []

for row in range(0, img.shape[0]):
  for col in range(0, img.shape[1]):
    if (img[row][col] == [7, 2, 251]).all():
      p.append([col, row])
    elif (img[row][col] == [255, 0, 0]).all():
      q.append([col, row])
      # print(img[row][col + 1])

p = np.array(p)
q = np.array(q)
print(p, q)
img = warp.mls_affine_deformation_inv(img, p, q)
print('warp finished')

cv2.imshow('test', img)
cv2.waitKey()

# cv2.imwrite('./output/Paul-phase-1.jpg', img)