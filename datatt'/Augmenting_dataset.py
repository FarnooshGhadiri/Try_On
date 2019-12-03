# Produce different Cloth invariant and identity invariant Image
import numpy as np
import os
import random
img_path = "/Fashion_Project/clothing-co-parsing-master/photos/"
Cloth_msk = "/Fashion_Project/clothing-co-parsing-master/Cloth_Segment"
Body_msk = "/Fashion_Project/clothing-co-parsing-master/Body_Segment"
from PIL import Image
import matplotlib.pyplot as plt

Color = [(1,0,0),(0,2,1)]

for filename in os.listdir(img_path):
    Filename = os.path.join(img_path,filename)
    File_mask = os.path.join(Cloth_msk,filename)
    File_BSeg = os.path.join(Body_msk,filename)
    BSeg_msk = Image.open(File_BSeg)
    BSeg_img = np.array(BSeg_msk)
    Clth_Msk = Image.open(File_mask)
    Clth_Msk = np.array(Clth_Msk)
    img = Image.open(Filename)
    img = np.array(img)
    Aug_Cth = np.copy(img)
    # Change person's cloth color
    Aug_Cth[Clth_Msk>0] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    # Change Person's skin color
    Aug_body = np.copy(img)
    Aug_body[BSeg_img > 0] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

    plt.imshow(SC_img)
    plt.show()
    print('is okay')



