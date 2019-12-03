Img_path_target='Image/Target/'
import scipy.io
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
Msk_path = "/Fashion_Project/clothing-co-parsing-master/annotations/pixel-level/"
import os

def get_annotation(Filename):
  Filename ='0537.mat'

  Mat_img = scipy.io.loadmat("%s%s" %(Msk_path,Filename))
  img = Mat_img['groundtruth']
  #print(np.unique(img))
  #Skin, hair, glass
  Seg_body = ((np.logical_or(np.logical_or(img==19,img==41),img==47))*1)
  plt.imshow(img)
  plt.show()
  try:
      rescaled = (255.0 / Seg_body.max() * (Seg_body - Seg_body.min())).astype(np.uint8)
      im = Image.fromarray(rescaled)
      im.save("/Fashion_Project/clothing-co-parsing-master/Body_Segment/%s%s" % (Filename[:-4],'.jpg'))
  except:
      print("No body segmentation for the  image %s" % Filename)
  #print(np.unique(Seg_body))
  Seg_Clth = np.multiply((1-Seg_body),(1-(img==0)))
  plt.show(Seg_Clth)
  try:
      rescaled = (255.0 / Seg_Clth.max() * (Seg_Clth - Seg_Clth.min())).astype(np.uint8)
      im = Image.fromarray(rescaled)
      im.save("/Fashion_Project/clothing-co-parsing-master/Cloth_Segment/%s%s" % (Filename[:-4], '.jpg'))
  except:
      print("No cloth segmentation for the image %s" % Filename)
  #Seg_Clth.save("/Fashion_Project/clothing-co-parsing-master/annotations/pixel-level/%s" % Filename)
  #print(np.unique(Seg_Clth))
  return Seg_body,Seg_Clth

#plt.imshow(Seg_body)
#plt.show()
#print(type(Lable_img['tags']))
#print(Lable_img['tags'].shape)
#print('done')
#Mat_img['truths']
#Mat_img

import os

def preprocessing():
    scipy.io.loadmat('/Fashion_Project/clothing-co-parsing-master/label_list.mat')
    if os.path.isfile(Img_path_source):
        print("%s source Img exists." % Img_path_source)
        return

    if os.path.isfile(Img_path_target):
        print("%s Target Img exists." % Img_path_target)
        return

def main():
    for filename in os.listdir(Msk_path):
        get_annotation(filename)


if __name__ == '__main__':

    main()


