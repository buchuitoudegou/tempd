import os

all_images = os.listdir('./result/input')

if not os.path.exists('./result/output'):
  os.mkdir('./result/output')

for image in all_images:
  os.system(f'python3 test.py pretrained/warpgan_pretrained result/input/{image} result/output/{image.split('.')[0]} --num_styles 5')
