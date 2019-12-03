import torch
import numpy
from torchvision.transforms import functional as F
import numpy as np

Img_path_target='Image/Target/'
import scipy.io
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import random
Msk_path = "/Fashion_Project/clothing-co-parsing-master/annotations/pixel-level/"
import os

def get_Seg_Msk(Filename):
  #Filename ='0536.mat'

  Mat_img = scipy.io.loadmat("%s%s" %(Msk_path,Filename))
  img = Mat_img['groundtruth']
  #Skin, hair, glass
  Seg_body = ((np.logical_or(np.logical_or(img==19,img==41),img==47))*1)
  try:
      rescaled = (255.0 / Seg_body.max() * (Seg_body - Seg_body.min())).astype(np.uint8)
      temp_Seg_Body = Image.fromarray(rescaled)
  except:
      print("No body segmentation for the  image %s" % Filename)
  #print(np.unique(Seg_body))
  Seg_Clth = np.multiply((1-Seg_body),(1-(img==0)))
  try:
      rescaled = (255.0 / Seg_Clth.max() * (Seg_Clth - Seg_Clth.min())).astype(np.uint8)
      Seg_body = temp_Seg_Body
      Seg_Clth = Image.fromarray(rescaled)
  except:
      print("No cloth segmentation for the image %s" % Filename)
  return Seg_body,Seg_Clth

def Get_Aug(img,Seg_body,Seg_Clth):
    Seg_body = np.array(Seg_body)
    Seg_Clth = np.array(Seg_Clth)
    img = np.array(img)
    Aug_Cth = np.copy(img)
    # Change person's cloth color
    Aug_Cth[Seg_Clth > 0] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    # Change Person's skin color
    Aug_body = np.copy(img)
    Aug_body[Seg_body > 0] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    return Aug_Cth,Aug_body



def random_crop(img,bbox,crop_size):
    ww = img.width
    hh = img.height
    crop_x, crop_y = np.random.randint(0,ww-crop_size+1),\
                     np.random.randint(0,hh-crop_size+1)
    img_cropped = img.crop((crop_x,crop_y,crop_x+crop_size,crop_y+crop_size))
    bbox_new = (max(0,bbox[0]))

def to_tensor(img):
    return F.to_tensor(img)
