api_key = 'VnmZ14C7QxpkyfJW6t683YTlM2w-CdWw'
api_secret = 'DzWtkHwCw52gfW8mkFAIU8pz8KtX3diI'

example_pth = './Paul3-phase-1.png'
detect_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
src_pth = './mls_samples/white_p.png'
dst_pth = 'mls_samples/white_p_.png'

mouth_ext_size = 7
nose_ext_size = 6
chin_ext_size = 10
eyes_ext_size = -3

# right_eye_top_keys = [
#   'right_eye_left_corner',
#   'right_eye_upper_left_quarter',
#   'right_eye_top',
#   'right_eye_upper_right_quarter',
#   'right_eye_right_corner'
# ]

# right_eye_bottom_keys = [
#   'right_eye_left_corner',
#   'right_eye_bottom',
#   'right_eye_lower_right_quarter',
#   'right_eye_right_corner'
# ]

# (row, col)
mouth_key = {
  'mouth_left_corner': (0, -1),\
  'mouth_lower_lip_left_contour3': (1, -1),\
  'mouth_lower_lip_right_contour3': (1, 1),\
  'mouth_right_corner': (0, 1),\
  'mouth_upper_lip_right_contour1': (-1, 1),\
  'mouth_upper_lip_left_contour1': (-1, -1),\
}

# (row, col)
nose_key = {\
  'nose_tip': (1, 0),\
  'nose_left_contour3': (0, -2),\
  'nose_right_contour3': (0, 2),\
}

# (row, col)
chin_key = {
  'contour_chin': (1, 0),\
  'contour_left8': (0, -1),\
  'contour_right8': (0, 1),\
}

# (row, col)
eyes_key = {\
  # left eye
  'left_eye_left_corner': (0, -1),
  'left_eye_upper_left_quarter': (-1, -1),
  'left_eye_top': (-1, 0),
  'left_eye_upper_right_quarter': (-1, 1),
  'left_eye_right_corner': (0, 1),
  'left_eye_lower_right_quarter': (1, 1),
  'left_eye_lower_left_quarter': (1, -1),
  # right eye
  'right_eye_right_corner': (0, 1),
  'right_eye_upper_right_quarter': (-1, 1),
  'right_eye_top': (-1, 0),
  'right_eye_upper_left_quarter': (-1, -1),
  'right_eye_left_corner': (0, -1),
  'right_eye_lower_right_quarter': (1, 1),
  'right_eye_lower_left_quarter': (1, -1)
}