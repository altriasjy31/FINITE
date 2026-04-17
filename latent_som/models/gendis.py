## generator and discriminator
import typing as T
import functools as ft
from argparse import ArgumentParser, Namespace
import pickle
import torch
from torch import nn
from torch import Tensor
from torch.autograd import Variable
import torch.nn.functional as F
import torch.autograd as autograd
import torch.nn.parallel as nn_parallel
import numpy as np

import os
import sys

prj_dir = os.path.dirname(os.path.dirname(__file__))
if prj_dir not in sys.path:
    sys.path.append(prj_dir)

from experiments.msa import MSAEncoder

import models.architecture as arch
from models.architecture import get_norm_layer, init_net
from models.architecture import ResnetGenerator
from resnet import resnet18, resnet34, resnet50,resnet101, resnet152
from resnet import ResNet
from resnet import resnext50_32x4d, resnext101_32x8d
from resnet import wide_resnet50_2, wide_resnet101_2
from timm_models import timm_create_model
from timm_models import TimmResNet
from timm_models import TimmMobileNetV3
from timm_models import TimmSwinTransformerV2
from timm_models import TimmEva
from timm_models import TimmVisionTransformer
from timm_models import TimmNaFlexVit

from models.utils import parsing

import helper_functions.aug as aug

cuda = True if torch.cuda.is_available() else False

class Arch(nn.Module):
    def __init__(self, opt : Namespace):
        """
        in_channels: int
        out_channels_G: int
        num_classes: int
        top_k: int
        max_len: int
        msa_cutoff: float
        msa_penalty: float
        msa_embedding_dim: int
        msa_encoding_strategy: str
        ngf: int
        netG: str
        no_antialias: bool
        no_antialias_up: bool
        load_gen: Optional[str]
        freeze_gen: Optional[str]
        ndf: int
        netD: str
        replace_stride_with_dilation: List[bool]
        no_dropout: bool
        normG: str
        normD: str
        init_type: str
        init_gain: float
        gpu_ids: List[int]
        no_jit: bool
        """
        super(Arch, self).__init__()

        self.in_channels = opt.in_channels
        self.out_channels_G = opt.out_channels_G
        self.num_classes = opt.num_classes
        self.gpu_ids = opt.gpu_ids

        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        need_init = True

        pre_model = MSAEncoder(opt.msa_embedding_dim, 
                                encoding_strategy=opt.msa_encoding_strategy)
        # pre_model = nn.Embedding(21, embedding_dim)

        # the weight of pre_model should also be loaded
        self.pre_model = init_net(pre_model, init_type=opt.init_type, init_gain=opt.init_gain,
                                  gpu_ids=opt.gpu_ids, initialize_weights=need_init)

        # nets = []
        # input: b * 21 * seqLen * topK
        # update
        # input: b * 441 * seqLen(patched) * seqLen(patched)
        # defaults is b * 441 * 128 * 128
        
        self.gnet = define_G(self.in_channels, self.out_channels_G,ngf=opt.ngf, netG=opt.netG,
                             norm=opt.normG, use_dropout=not opt.no_dropout,
                             init_type=opt.init_type, init_gain=opt.init_gain,
                             no_antialias=opt.no_antialias, no_antialias_up=opt.no_antialias_up,
                             gpu_ids=opt.gpu_ids,
                             need_init_weights=need_init)

        params = opt.__dict__
        ks = get_init_keys(name2model[opt.netD])
        ropt = {"params": {k: params[k] for k in ks},
                "head_params": {}}
        self.rnet = define_D(self.out_channels_G, self.num_classes, netD=opt.netD,
                             init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids,
                             **ropt)
    
    def get_fc(self):
        model = self.rnet.module if isinstance(self.rnet, nn_parallel.DataParallel) \
            else self.rnet
        if hasattr(model, "fc"):
            fc = model.fc
        elif hasattr(model, "head"):
            fc = model.head
        else:
            raise NotImplementedError("only support fc or head")
        assert isinstance(fc, nn.Module)
        return fc
    
    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False
    
    def freeze_all_except_fc(self):
        self.freeze_all()
        fc = self.get_fc()
        for p in fc.parameters():
            p.requires_grad = True
            
    def forward(self, x : Tensor, 
                permute_dims: T.Tuple[int, int, int, int] = (0, 3, 2, 1),
                aug_params: T.Optional[T.Dict] = None):
        if aug_params is None:
            x = self.pre_model(x)
            # x = x.permute(0, 3, 2, 1) # b c h w -> b w h c
            x = x.permute(*permute_dims)
            if self.gnet is not None:
                x = self.gnet(x)
            x = self.rnet(x)
            return x
        else:
            x = self.pre_model(x)
            assert aug_params.get("y") is not None, "Not contain target data"
            y = aug_params["y"]
            assert isinstance(y, Tensor), "Y must be a Tensor"
            x_mix, y_mix, lam = aug.mixup_msa_data(x, y, 
                                                   alpha=aug_params.get("mixup_alpha", 0.2))
            x_mix = x_mix.permute(*permute_dims)
            if self.gnet is not None:
                x_mix = self.gnet(x_mix)
            x_mix = self.rnet(x_mix)
            return x_mix, y_mix


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, no_antialias=False, no_antialias_up=False, gpu_ids=[], opt=None,
             need_init_weights = True):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2.
        need_init_weights (bool) -- control whether to initialize the net weights
    """

    norm_layer = get_norm_layer(norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, 
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up, 
                              n_blocks=9, opt=opt)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, 
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up, 
                              n_blocks=6, opt=opt)
    elif netG == 'resnet_4blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, 
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up, 
                              n_blocks=4, opt=opt)
    elif netG == "resnet_2blocks":
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up,
                              n_blocks=2, opt=opt)
    elif netG == "resnet_oneblock":
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up,
                              n_blocks=1, opt=opt)
    elif netG is None or netG == "none":
        net = None
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_net(net, init_type, init_gain, gpu_ids, initialize_weights=need_init_weights) \
            if net is not None else net

name2net = {
    "resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50,"resnet101": resnet101, "resnet152" : resnet152,
    "resnext50_32x4d": resnext50_32x4d, "resnext101_32x8d": resnext101_32x8d,
    "wide_resnet50_2":wide_resnet50_2, "wide_resnet101_2": wide_resnet101_2,
    "timm_resnet50d": ft.partial(timm_create_model, variant="resnet50d"),
    "timm_resnetaa50d": ft.partial(timm_create_model, variant="resnetaa50d"),
    "timm_ecaresnet50d": ft.partial(timm_create_model, variant="ecaresnet50d"),
    "timm_ecaresnet101d": ft.partial(timm_create_model, variant="ecaresnet101d"),
    "timm_ecaresnet152d": ft.partial(timm_create_model, variant="ecaresnet152d"),
    "timm_ecaresnet101t": ft.partial(timm_create_model, variant="ecaresnet101t"),
    "timm_ecaresnetblur101d": ft.partial(timm_create_model, variant="ecaresnetblur101d"),
    "timm_ecaresnetblur152d": ft.partial(timm_create_model, variant="ecaresnetblur152d"),
    "timm_ecaresnetaa101d": ft.partial(timm_create_model, variant="ecaresnetaa101d"),
    "timm_mobilenetv4_conv_blur_medium": ft.partial(timm_create_model, variant="mobilenetv4_conv_blur_medium"),
    "timm_mobilenetv4_conv_blur_large": ft.partial(timm_create_model, variant="mobilenetv4_conv_blur_large"),
    "timm_mobilenetv4_hybrid_large": ft.partial(timm_create_model, variant="mobilenetv4_hybrid_large"),
    "timm_swinv2_base_window16": ft.partial(timm_create_model, variant="swinv2_base_window16"),
    "timm_swinv2_base_window8": ft.partial(timm_create_model, variant="swinv2_base_window8"),
    "timm_swinv2_small_window16": ft.partial(timm_create_model, variant="swinv2_small_window16"),
    "timm_vit_base_patch16_rope_reg1_gap": ft.partial(timm_create_model, variant="vit_base_patch16_rope_reg1_gap"),
    "timm_vit_base_patch32": ft.partial(timm_create_model, variant="vit_base_patch32"),
    "timm_vit_base_patch16": ft.partial(timm_create_model, variant="vit_base_patch16"),
    "timm_vit_base_patch8": ft.partial(timm_create_model, variant="vit_base_patch8"),
    "timm_vit_base_patch16_plus": ft.partial(timm_create_model, variant="vit_base_patch16_plus"),
    "timm_vit_base_patch16_rpn": ft.partial(timm_create_model, variant="vit_base_patch16_rpn"),
    "timm_vit_base_patch16_rope": ft.partial(timm_create_model, variant="vit_base_patch16_rope"),
    "timm_vit_base_patch16_rope_mixed": ft.partial(timm_create_model, variant="vit_base_patch16_rope_mixed"),
    "timm_vit_base_patch16_siglip": ft.partial(timm_create_model, variant="vit_base_patch16_siglip"),
    "timm_vit_intern_patch16": ft.partial(timm_create_model, variant="vit_intern_patch16"),
    "timm_vit_large_patch16": ft.partial(timm_create_model, variant="vit_large_patch16"),
    "timm_vit_base_patch14": ft.partial(timm_create_model, variant="vit_base_patch14"),
    "timm_naflexvit_base_patch16": ft.partial(timm_create_model, variant="naflexvit_base_patch16"),
}

name2model = {
    "resnet18": ResNet, "resnet34": ResNet, "resnet50": ResNet,"resnet101": ResNet, "resnet152" : ResNet,
    "resnext50_32x4d": ResNet, "resnext101_32x8d": ResNet,
    "wide_resnet50_2":ResNet, "wide_resnet101_2": ResNet,
    "timm_resnet50d": TimmResNet, "timm_resnetaa50d": TimmResNet,
    "timm_ecaresnet50d": TimmResNet, 
    "timm_ecaresnet101d": TimmResNet,
    "timm_ecaresnet152d": TimmResNet,
    "timm_ecaresnet101t": TimmResNet,
    "timm_ecaresnetblur101d": TimmResNet,
    "timm_ecaresnetblur152d": TimmResNet,
    "timm_ecaresnetaa101d": TimmResNet,
    "timm_mobilenetv4_conv_blur_medium": TimmMobileNetV3,
    "timm_mobilenetv4_conv_blur_large": TimmMobileNetV3,
    "timm_swinv2_base_window16": TimmSwinTransformerV2,
    "timm_swinv2_base_window8": TimmSwinTransformerV2,
    "timm_swinv2_small_window16": TimmSwinTransformerV2,
    "timm_vit_base_patch16_rope_reg1_gap": TimmEva,
    "timm_vit_base_patch32": TimmVisionTransformer,
    "timm_vit_base_patch16": TimmVisionTransformer,
    "timm_vit_base_patch8": TimmVisionTransformer,
    "timm_vit_base_patch16_plus": TimmVisionTransformer,
    "timm_vit_base_patch16_rpn": TimmVisionTransformer,
    "timm_vit_base_patch16_rope": TimmEva,
    "timm_vit_base_patch16_rope_mixed": TimmEva,
    "timm_vit_base_patch16_siglip": TimmVisionTransformer,
    "timm_vit_intern_patch16": TimmVisionTransformer,
    "timm_vit_large_patch16": TimmVisionTransformer,
    "timm_vit_base_patch14": TimmVisionTransformer,
    "timm_naflexvit_base_patch16": TimmNaFlexVit,
}

def get_init_keys(net):
    if net == ResNet:
        init_keys = ["replace_stride_with_dilation", "normD"]
    elif net == TimmSwinTransformerV2 or \
         net == TimmVisionTransformer or \
         net == TimmEva or \
         net == TimmNaFlexVit:
        init_keys = ["top_k", "max_len"]
    else:
        init_keys = []

    return init_keys

def define_D(in_channels : int, num_classes : int, netD : str,
             init_type='normal', init_gain=0.02, gpu_ids = [], need_init_weights = True,
             **opt):
        assert  name2net.get(netD) is not None, \
            f"{netD} is not implemented"
        
        params = opt["params"]
        norm_type = params.get("normD", None)
        if norm_type is not None:
            del params["normD"]
            params["norm_layer"] = get_norm_layer(norm_type)
        
        rnet =  name2net[netD](in_channels=in_channels, num_classes=num_classes, **params)
        if need_init_weights:
            rnet = init_net(rnet, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)

        return rnet

def main():
    """
    """
    parser = ArgumentParser()

    parser.add_argument("--in-channels", dest="in_channels", type=int, default=21)
    parser.add_argument("--out-channels-G", dest="out_channels_G", type=int, default=42)
    parser.add_argument("--num-classes",dest="num_classes", type=int, default=19939)

    parser.add_argument("--top-k",dest="top_k",type=int,default=100,
                        help="select the top k sequence ib msa")
    parser.add_argument("--max-len",dest="max_len", type=int, default=1000,
                        help="the max lenght of sequence for using")

    parser.add_argument("--msa-cutoff", dest="msa_cutoff", type=float,default=0.8,
                        help="parameters for encoding the msa file")
    parser.add_argument("--msa-penalty", dest="msa_penalty", type=float,default=4.5,
                        help="parameters for encoding the msa file")
    parser.add_argument("--msa-embedding-dim", dest="msa_embedding_dim", type=int, default=21,
                        help="parameters for encoding the msa file")
    parser.add_argument("--msa-encoding-strategy", dest="msa_encoding_strategy", type=str, 
                        choices=["one_hot", "emb", "emb_plus_one_hot", "emb_plus_pssm","fast_dca"], 
                        default="emb_plus_one_hot",
                        help="parameters for encoding the msa file")

    parser.add_argument('--ngf', type=int, default=64, 
                        help='# of gen filters in the last conv layer')
    parser.add_argument('--netG', type=str, default='resnet_4blocks', 
                        help="specify generator architecture " +
                        "[resnet_9blocks | resnet_6blocks | resnet_4blocks]")
    parser.add_argument("--no-antialias", dest="no_antialias", action="store_true",
                        help="use dilated convolution blocks in generator")
    parser.add_argument("--no-antialias-up", dest="no_antialias_up", action="store_true",
                        help="use dilated convolution_transposed blocks in generator")
    parser.add_argument("--load-gen", dest="load_gen", type=str,
                        help="the path of pre-trained generator model")
    parser.add_argument("--freeze-gen", dest="freeze_gen", action="store_true",
                        help="control whether to freeze the generator model when training")
    parser.add_argument('--ndf', type=int, default=64, 
                            help='# of dis filters in the last conv layer')
    parser.add_argument('--netD', type=str, default='resnet50v2', 
                        help='specify discriminator architecture '+
                        '[resnet18 | resnet34 | resnet50 | resnet101 | resnet152] or ' + 
                        '[resnext50_32x4d | resnext101_32x8d]' +
                        '[wide_resnet50_2 | wide_resnet101_2]'+
                        '[basic]')

    parser.add_argument("--dilation", nargs=3, dest="replace_stride_with_dilation", 
                        default=[False, False, False], type=bool,
                        help="using dilation to replace the stride in resnet")
    parser.add_argument('--no-dropout', dest="no_dropout",action='store_true', 
                        help='no dropout for the generator', default=False)
    parser.add_argument('--normG', type=str, default='instance', 
                        help='for generator, instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--normD', type=str, default='batch', 
                            help='for discriminator, instance normalization or batch normalization [instance | batch | none]')    
     
    parser.add_argument('--init-type', dest="init_type", type=str, default='xavier', 
                        help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init-gain', dest="init_gain", type=float, default=0.02, 
                        help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--gpu-ids', dest="gpu_ids",type=str, default='0,1', 
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument("--no-jit", dest="no_jit", action="store_true",
                        help="not use torch.jit.script")
    opt : Namespace
    opt = parsing(parser)

    print(opt)

    model_arch = Arch(opt).cuda()
    print(model_arch)
    x = torch.randint(0, 21,(12, 40, 2000)).cuda()
    x = model_arch(x)

if __name__ == "__main__":
    main()
