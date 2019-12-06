import torch
import functools
from torch import nn
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()

class Generator(nn.Module):
    def __init__(self, sourceImg_nc,targetImg_nc, output_nc=3,
                        dropout=0.0, norm_layer=nn.BatchNorm2d, fuse_mode='cat', connect_layers=0):
        super(Generator,self).__init__()
        assert(connect_layers>=0 and connect_layers<=5)
        ngf=64
        self.output_nc = output_nc
        self.fuse_mode = fuse_mode
        self.norm_layer = norm_layer
        self.dropout = dropout
        if type(norm_layer)==functools.partial:
            self.use_bias = norm_layer.func = nn.InstanceNorm2d
        else:
            self.use_bias = norm_layer = nn.InstanceNorm2d
        input_channel = [[8,8,4,2,1],
                        [16,8,4,2,1],
                        [16,16,4,2,1],
                        [16,16,8,2,1],
                        [16,16,8,4,1],
                        [16,16,8,4,2]]

    ################## Decoder####################

        if fuse_mode =='cat':
            dc_avg = [nn.LeakyReLU(True),
                      nn.ConvTranspose2d(sourceImg_nc+targetImg_nc,ngf*8,
                                         kernel_size=[8,4],bias=self.use_bias),
                                         norm_layer(ngf*8),nn.Dropout(dropout)]
        elif fuse_mode =='add':
            nc = max(sourceImg_nc,targetImg_nc)
            self.w_target_feature = nn.linear(sourceImg_nc,nc,bias=False)
            self.w_source_feature = nn.linear(targetImg_nc,nc,bias=False)
            dc_avg =[nn.LeakyReLU(True),
                     nn.ConvTranspose2d(nc,ngf*8,
                                        kenel_size=[8,4],bias=self.use_bias),
                                        norm_layer(ngf*8),nn.Dropout(dropout)]
        else:
            raise("Wronge fuse_mode")
        self.dc_avg = nn.sequential(*dc_avg)
        # N*512*8,4
        self.dc_conv5 = self._make_layer_decode(ngf*input_channel[connect_layers][0],ngf*8)
        #N*512*16*8
        self.dc_conv4 = self._make_layer_decode(ngf*input_channel[connect_layers][1],ngf*4)
        #N*256*32*16
        self.dc_conv3 = self._make_layer_decode(ngf*input_channel[connect_layers][2],ngf*2)
        #N*128*64*32
        self.dc_conv2 = self._make_layer_decode(ngf*input_channel[connect_layers][3],ngf)
        #N*64*128*64
        dc_conv1 = [nn.LeakyReLU(True),
                         nn.ConvTranspose2d(ngf*input_channel[connect_layers][0][4],output_nc,
                         kernel_size=4,stride=2,
                         padding=1,bias=self.use_bias),
                         nn.tanh()]
        self.dc_conv1 = nn.sequential(*dc_conv1)


    def _make_layer_decode(self,in_nc,out_nc):
        block = [nn.LeakyReLU(True),
                nn.ConvTranspose2d(in_nc,out_nc,
                        kernel_size=4,stride=2),
                        self.norm_layer(out_nc),
                        nn.Dropout(self.dropout)]
        return nn.Sequential(*block)
    #If we have our own Encoder inseat of resnet, then we will use this
    '''
        def decode(self,model,fake_features,en_features cnlayers):
        # if we have a symmetrical encoder instead of resnet, we use this
        if cnlayers>0:
            return model(torch.cat(fake_features,en_features)),cnlayers
        else:
            return  model(fake_features),cnlayers
            
    '''


    def forward(self,source_features,target_features):
        batch_size = source_features.data.size(0)
        if self.fuse_mode == 'cat':
            features = torch.cat(source_features,target_features,dim=1)
        elif self.fuse_mode =='add':
            features = self.w_source_feature(source_features.view(batch_size,-1))+\
                       self.w_target_feature(target_features.view(batch_size,-1))
            features = features.view(batch_size,-1,1,1)

        fake_features = self. dc_avg(features)
        cnlayers = self.connect_layers
        ######if we have our own Encoder instead of resnet
        ###fake_feature_5, cnlayers = self.decode(self.de_conv5, fake_feature, en_feature_5, cnlayers)
        fake_features_5 = self.dc_conv5(fake_features)
        fake_features_4 = self.dc_conv4(fake_features_5)
        fake_features_3 = self.dc_conv3(fake_features_4)
        fake_features_2 = self.dc_conv2(fake_features_3)
        fake_features_1 = self.dc_conv1(fake_features_2)
        fake_img =fake_features_1
        return fake_img







