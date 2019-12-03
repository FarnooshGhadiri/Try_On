import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from Transformer import get_Seg_Msk,Get_Aug,to_tensor
import numpy as np
import random
import scipy
import torchvision.transforms as Transform

class Try_On_dataset(Dataset):

  def __init__(self,root,bboxes, dataAug=False, Img_size=256,Crop_size=224, mean=0.5, std=0.5):
      super(GetData,self).__init__()
      self.root_Img = ("%sImg") % root
      self.root_Clt_seg = ("%sCloth_Seg") % root
      self.root_Prt_seg = ("%sPart_Seg") % root
      self.get_file_name()
      self.dataAug = dataAug
      self.Img_size = Img_size
      self.Crop_size = Crop_size
      self.mean = mean
      self.std = std

  def get_file_name(self):
      self.filenames = [f for f in os.listdir(self.root_Img) if os.path.splitext(f)[-1] == '.jpg']

  def __getitem__(self, index):
      filepath_Img = os.path.join(self.root_Img, self.filenames[int(index)])
      img = Image.open(filepath_Img)
      img = img.convert("RGB")
      if self.dataAug:
         Seg_Bdy, Seg_Clth = get_Seg_Msk(self.filenames[int(index)])
         # Source image will be wear the target cloth
         Source_img, Target_img = Get_Aug(img, Seg_Bdy, Seg_Clth)
         Source_img = Source_img.convert("RGB")
         Target_img = Target_img.convert("RGB")
      else:
          index = np.roll(index,1)
          filepath_Img = os.path.join(self.root_Img, self.filenames[int(index)])
          Source_img = img
          Target_img = Image.open(filepath_Img)
          Target_img = Target_img.convert("RGB")

      Source_img = (to_tensor(Source_img) - self.mean) / self.std
      Target_img = (to_tensor(Target_img) - self.mean) / self.std

      sample = {'Source_img': Source_img, 'Target_img': Target_img}
      return sample




  def __len__(self):
      return len(self.filenames)