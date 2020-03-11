import os

cari_imgs = os.listdir('./style')
test_imgs = os.listdir('./test_img')

if not os.path.exists('style_result'):
  os.mkdir('./style_result')

for cari_img in cari_imgs:
  for test_img in test_imgs:
    os.system('python neural_style.py --content {} --styles {} --output {}'\
      .format('./test_img/' + test_img, './style/' + cari_img, \
        './style_result/' + test_img.split('.')[0] + '_' + cari_img.split('.')[0] + '.jpg'))