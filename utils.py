import torch
from PIL import Image
import numpy as np

def read_and_process_file(f):
    im = Image.open(f)
    im = im.convert('L')
    im = im.resize((512,512))
    im = np.asarray(im)

    im[im>239.8] = 1 #239.8 is the mean of the dataset
    im[im!=1] = 0

    return im

def process_pil(im):

    im = im.convert('L')
    im = im.resize((512,512))
    im = np.asarray(im)

    im[im>=239.8] = 1 #239.8 is the mean of the dataset
    im[im!=1] = 0

    im = torch.from_numpy(im).unsqueeze(0).unsqueeze(0).float()

    return im
