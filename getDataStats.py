import os
import numpy as np
from PIL import Image

dirs = []

gray_scale_mean = []

gray_scale_std = []

color_mean = []
color_std = []

for sub1 in [os.path.join('sign_data',i) for i in ['train','test']]:
    for sub2 in os.listdir(sub1):
        for sub3 in os.listdir(os.path.join(sub1,sub2)):
            img = Image.open(os.path.join(os.path.join(sub1,os.path.join(sub2,sub3))))

            color_mean.append(np.asarray(img).mean(axis=(0,1)))
            color_std.append(np.asarray(img).std(axis=(0,1)))

            img = np.asarray(img.convert('L'))

            gray_scale_mean.append(img.mean())
            gray_scale_std.append(img.std())



gray_scale_mean = np.array(gray_scale_mean).mean()
gray_scale_std = np.array(gray_scale_std).mean()

color_mean = np.array(color_mean).mean(axis=0)
color_std = np.array(color_std).mean(axis=0)

print(f'GRAYSCALE MEAN | {gray_scale_mean:.1f}')
print('---------------+ ---------------------')
print(f'GRAYSCALE STD  | {gray_scale_mean:.1f}')
print('--------------------------------------')
print(f'COLOR MEAN     | {color_mean[0]:.1f} | {color_mean[1]:.1f} | {color_mean[2]:.1f}')
print('---------------+ ---------------------')
print(f'COLOR STD      | {color_mean[0]:.1f} | {color_mean[1]:.1f} | {color_mean[2]:.1f}')