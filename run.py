import os

all_images = os.listdir('./result/input')

if not os.path.exists('./result/output'):
  os.mkdir('./result/output')

for image in all_images:
  os.system('python3 test.py pretrained/warpgan_pretrained result/input/{} result/output/{} --num_styles 5'.format(image, image.split('.')[0]))
