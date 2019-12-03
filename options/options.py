import os
import argparse
import torch
class Options():

   def __init__(self):
       self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
       self.parser.add_argument('--data_dir', default='/data/', help="path to the test/train data")
       self.parser.add_argument('--batch_size', type='int',default='64')
       self.parser.add_argument('--mode',default='train',choises=['train','test','validate'])
       self.parser.add_argument('--name',default='my_experiment',help="name of your experiment")
       self.parser.add_argument('--model',default='resnet18',help="what type of network should be used")
       self.parser.add_argument('--model_dir',default='')
       self.parser.add_argument('--gpu_ids',type='str',default='0',help="""Comma-separated numbers denoting gpu ids
                                                                          e.g.0  0,1,2, 0,2. use -1 for CPU """)

   def parse(self):
       opt = self.parser.parse_args()
       gpu_ids = opt.gpu_ids.split(',')
       opt.devices = []
       for id in gpu_ids:
           if eval(id) >= 0:
               opt.devices.append(eval(id))

       # cuda
       opt.cuda = False
       if len(opt.devices) > 0 and torch.cuda.is_available():
            opt.cuda = True



if __name__ == "__main__":
   op = Options()
   op.parse()


