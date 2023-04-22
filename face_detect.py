import base64
import requests
from config import \
  api_key,\
  api_secret,\
  mouth_ext_size,\
  nose_ext_size,\
  chin_ext_size,\
  eyes_ext_size,\
  mouth_key,\
  nose_key,\
  chin_key,\
  eyes_key,\
  src_pth,\
  detect_url,\
  dst_pth
import json
import cv2
import numpy as np
import warp
import util

nose_landmark = []
mouth_landmark = []
chin_landmark = []
eyes_landmark = []


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

def warp_img():
  img = cv2.imread(src_pth)
  p = np.array(mouth_landmark + nose_landmark + chin_landmark)
  p_eyes = np.array(eyes_landmark + chin_landmark)
  # warp nose
  nose_keys = list(nose_key.keys())
  for i in range(0, len(nose_landmark)):
    nose_landmark[i] = [
      nose_landmark[i][0] + nose_ext_size * nose_key[nose_keys[i]][1],\
      nose_landmark[i][1] + nose_ext_size * nose_key[nose_keys[i]][0]\
    ]

  # warp mouth
  mouth_keys = list(mouth_key.keys())
  for i in range(0, len(mouth_landmark)):
    mouth_landmark[i] = [\
      mouth_landmark[i][0] + mouth_ext_size * mouth_key[mouth_keys[i]][1],\
      mouth_landmark[i][1] + mouth_ext_size * mouth_key[mouth_keys[i]][0]\
    ]

  # warp chin
  chin_keys = list(chin_key.keys())
  for i in range(0, len(chin_landmark)):
    chin_landmark[i] = [\
      chin_landmark[i][0] + chin_ext_size * chin_key[chin_keys[i]][1],\
      chin_landmark[i][1] + chin_ext_size * chin_key[chin_keys[i]][0]\
    ]
  
  # warp eyes
  eyes_keys = list(eyes_key.keys())
  for i in range(0, len(eyes_landmark)):
    eyes_landmark[i] = [\
      eyes_landmark[i][0] + eyes_ext_size * eyes_key[eyes_keys[i]][1],\
      eyes_landmark[i][1] + eyes_ext_size * eyes_key[eyes_keys[i]][0]\
    ]

  q = np.array(mouth_landmark + nose_landmark + chin_landmark)
  q_eyes = np.array(eyes_landmark + chin_landmark)
  
  # img = util.scatter(img, p, [0, 0, 255])
  # img = util.scatter(img, q, [255, 0, 0])

  img = warp.mls_affine_deformation_inv(img, p, q)
  print('warp 1 finished')

  img = warp.mls_affine_deformation_inv(img, p_eyes, q_eyes)
  print('warp 2 finished')
  return img

if __name__ == "__main__":
  image_landmark = get_image_landmark(src_pth)
  print('getting facial landmark finished')
  # get landmark
  nose_landmark = [[image_landmark[key]["x"], image_landmark[key]["y"]] for key in nose_key]
  mouth_landmark = [[image_landmark[key]["x"], image_landmark[key]["y"]] for key in mouth_key]
  chin_landmark = [[image_landmark[key]["x"], image_landmark[key]["y"]] for key in chin_key]
  eyes_landmark = [[image_landmark[key]["x"], image_landmark[key]["y"]] for key in eyes_key]

  img = warp_img()

  cv2.imshow('test', img)
  cv2.waitKey()

  img = (img * 255).astype(np.uint)
  img = img.reshape((256, 256, 3))
  cv2.imwrite(dst_pth.split('.')[0] + str(mouth_ext_size) + '_'\
  + str(nose_ext_size) + '_'\
  + str(chin_ext_size) + '_'\
  + str(eyes_ext_size) + '.jpg', img)
