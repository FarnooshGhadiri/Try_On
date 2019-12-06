
from torch.utils.data import dataloader
from data.DeepCloth import Try_On_dataset
from options import options
import numpy as np
import logging
import os
from util import util
def forward_batch():

def forward_database():

def main():

def save_model():

def load_model():

def train():

def validate():

def main():

   op = options
   opt = op.parse()
   # initialize train or test working directory
   opt.model_dir = os.path.join("results",opt.name)
   logging.info = ("model directory %s" % opt.model_dir)
   if not os.path.exists(opt.model_dir):
       os.makedirs(opt.model_dir)
   log_dir = opt.model_dir
   log_path = log_dir + "/train.log"
   util.opt2file(opt, log_dir + "/opt.txt")
   #log setting
   log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   formatter = logging.Formatter(log_format)
   fh = logging.FileHandler(log_path, 'a')
   fh.setFormatter(formatter)
   ch = logging.StreamHandler()
   ch.setFormatter(formatter)
   logging.getLogger().addHandler(fh)
   logging.getLogger().addHandler(ch)
   log_level = logging.INFO
   logging.getLogger().setLevel(log_level)
    #define database
   indices = list(range(opt.num_example))
   rand_indices = np.random.RandomState(0)
   rand_indices.shuffle(indices)
   train_idx = indices[0:0.9*len(indices)]
   valid_idx = indices[0.9*len(indices)::]
   ds_train = Try_On_dataset(root=opt.data_dir,
                                  indices=train_idx,
                                  data_aug=opt.data_aug,
                                  img_size=opt.img_size,
                                  crop_size=opt.crop_size)
   ds_valid = Try_On_dataset(root=opt.data_dir,
                                  indices=valid_idx,
                                  data_aug=opt.data_aug,
                                  img_size=opt.img_size,
                                  crop_size=opt.crop_size)
   loader_train = dataloader(ds_train,shuffel=True,batch_size=opt.batch_size,num_workers=opt.num_wokers)
   loader_valid = dataloader(ds_valid,shuffel=True,batch_size=opt.batch_size,num_workers=opt.num_wokers)
   #load model













if __name__ =="__main__()":
    main()