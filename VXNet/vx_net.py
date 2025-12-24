#==============================================================================================================
#Citation: A Novel Automated System for Pathological Lung Segmentation Using Modified Local Binary Patterns and Hierarchical Transformers
# A. Sharafeldeen, F. Taher, M. Ghazal, A. Khalil, A. Mahmoud, S. Contractor, and A. El-Baz, “A Novel Automated System for Pathological Lung Segmentation Using Modified Local Binary Patterns and Hierarchical Transformers,” 2025 IEEE International Conference on Image Processing (ICIP). IEEE, pp. 1642–1647, 14-Sept-2025.
#==============================================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from VXNet.vx_encoder import vxnet_encoder

def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self,in_chans, outChans, elu):
        super(InputTransition, self).__init__()
        self.outChans=outChans
        self.conv1 = nn.Conv3d(in_chans, outChans, kernel_size=5, padding=2)#16
        self.bn1 = ContBatchNorm3d(outChans)#16
        self.relu1 = ELUCons(elu, outChans)#16

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        x16 = torch.cat([x]*math.ceil(self.outChans/1), 1)
        
        out = self.relu1(torch.add(out, x16[:,0:self.outChans,:,:,:]))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False, depth_reduction=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans

        if depth_reduction:
            self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        else:
            self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=(3,2,2), stride=(1,2,2), padding=(1,0,0))

        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False, depth_reduction=False):
        super(UpTransition, self).__init__()
        if depth_reduction:
            self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)#2, stride=2
        else:
            self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=(3,2,2), stride=(1,2,2), padding=(1,0,0))#2, stride=2
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, num_classes, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, num_classes, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(num_classes)
        self.conv2 = nn.Conv3d(num_classes, num_classes, kernel_size=1)
        self.relu1 = ELUCons(elu, num_classes)
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out


class VXNet(nn.Module):
    """
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes. Default: 2
        dims (int): Feature dimension at each stage. Must contain three values. Default: [32, 64, 128]
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        encoder_nConv (int): Number of covolution at each stage of encoder block. Default: [2, 3, 2]
        decoder_nConv (int): Number of covolution at each stage of decoder layer in a reverse direction. Default: [1, 1, 2]
        dropout (boolean): dropout setting at each stage. Default: [False,True,True]
        depth_reduction (boolean): This flag applies downsampling to the depth dimension. Default: False
        elu (boolean): This flag applies either ELU (True) or PReLU (False). Default: True
        nll (boolean): This flag applies either log softmax (True) or sofmax (False). Default: False
    """
    def __init__(self,in_chans=3, num_classes=2, dims=[32, 64, 128],depths=[2, 2, 2], encoder_nConv=[2, 3, 2], decoder_nConv=[1, 1, 2], dropout=[False,True,True], depth_reduction=False, elu=True, nll=False):
        super(VXNet, self).__init__()
        # dims=[32, 64, 128]#[384, 192, 96, 48]
        
        if len(dims) < 2:
            raise ValueError('The length of dims list should be greater than 1.')

        if  not (len(dims) == len(depths) == len(encoder_nConv) == len(decoder_nConv) == len(dropout)):
            raise ValueError('The lengths of dims, depths, encoder_nConv, decoder_nConv, and dropout should be the same.')
        
        

        #Input Layer
        self.in_tr = InputTransition(in_chans, dims[0]//2, elu)

        # Encoder Layer

        # VX-Block
        self.vxnet_3d = vxnet_encoder(
            in_chans= dims[0]//2,
            depths=depths,
            dims=dims,
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            out_indices= list(range(len(dims))),#[0, 1, 2]
            dropout=dropout,
            depth_reduction=depth_reduction
        )
        
        # Encoder-Block
        self.encoder_blocks = nn.ModuleList()
        
        if not self.is_power_of_two(dims[0]) or dims[0] == 2:
            raise ValueError('All values in the dims list should be represented as a power of 2 and greater than 2.')

        self.encoder_blocks.append(DownTransition(dims[0]//2, 1, elu,depth_reduction=depth_reduction))

        for i in range(len(dims)):
            if not self.is_power_of_two(dims[i]) or dims[i] == 2:
                raise ValueError('All values in the dims list should be represented as a power of 2 and greater than 2.')
            self.encoder_blocks.append(DownTransition(dims[i], encoder_nConv[i], elu,depth_reduction=depth_reduction, dropout=dropout[i]))



        # Decoder Layer
        self.decoder_layers = nn.ModuleList()
        self.decoder_layers.append(UpTransition(dims[-1]*2, dims[-1]*2, 2, elu, dropout=True,depth_reduction=depth_reduction))

        for i in range(len(dims)-1,-1,-1):
            if i==len(dims)-1:
                self.decoder_layers.append(UpTransition(dims[i]*2, dims[i], decoder_nConv[i], elu, dropout=True,depth_reduction=depth_reduction))  
            else:
                self.decoder_layers.append(UpTransition(dims[i+1], dims[i], decoder_nConv[i], elu, dropout=True,depth_reduction=depth_reduction))  

        

        #Output Layer
        self.out_tr = OutputTransition(dims[0], num_classes, elu, nll)
        

    def is_power_of_two(self, n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0
    def forward(self, x):
        # x = self.CAs(x) * x
        # x = self.SAs(x) * x

        # Input layer
        out16 = self.in_tr(x)
        
        # VX block
        vx_outs = self.vxnet_3d(out16)

        
        # Encoder block
        enc_outs = []
        for i in range(len(self.encoder_blocks)):
            if i==0:
                enc_outs.append(self.encoder_blocks[i](out16))
            else:
                enc_outs.append(self.encoder_blocks[i](vx_outs[i-1]))
        

        # Decoder layer
        for i in range(len(self.decoder_layers)-1):
            if i==0:
                out=self.decoder_layers[i](enc_outs[-1],enc_outs[-2])
            else:
                out=self.decoder_layers[i](out,enc_outs[-i-2])
        out=self.decoder_layers[-1](out,out16)
        

        # Output Layer
        out = self.out_tr(out)


        return out

