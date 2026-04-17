from functools import partial
import torch.nn as nn
import timm.models as tmodels

import timm.models.resnet as R
import timm.models.mobilenetv3 as m3
import timm.models.swin_transformer_v2 as swinv2
from timm.models import register_model
from timm.models.resnet import ResNet as TimmResNet
from timm.models.mobilenetv3 import MobileNetV3 as TimmMobileNetV3
from timm.models.swin_transformer_v2 import SwinTransformerV2 as TimmSwinTransformerV2
from timm.models.eva import Eva as TimmEva
import timm.models.eva as eva
from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer
from timm.models.naflexvit import NaFlexVit as TimmNaFlexVit
import timm.models.vision_transformer as vt

@register_model
def ecaresnet101t(pretrained: bool = False, **kwargs) -> R.ResNet:
    """Constructs an ECA-ResNet-101-T model.
    Like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels in the deep stem and ECA attn.
    """
    model_args = dict(
        block=R.Bottleneck, layers=[3, 4, 23, 3], stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'))
    return R._create_resnet('ecaresnet101t', pretrained, **dict(model_args, **kwargs))

@register_model
def ecaresnetblur101d(pretrained: bool = False, **kwargs) -> R.ResNet:
    """Constructs a ECA-ResNet-101-D model with blur anti-aliasing
    """
    model_args = dict(
        block=R.Bottleneck, layers=[3, 4, 23, 3], 
        aa_layer=R.BlurPool2d,
        stem_width=32, 
        stem_type='deep', avg_down=True,
        # drop_rate = 0.2,
        # drop_path_rate = 0.1,
        block_args=dict(attn_layer='eca'))
    return R._create_resnet('ecaresnetblur101d', pretrained, **dict(model_args, **kwargs))

@register_model
def ecaresnetblur152d(pretrained: bool = False, **kwargs) -> R.ResNet:
    """Constructs a ResNet-152-D model with eca.
    """
    model_args = dict(
        block=R.Bottleneck, layers=[3, 8, 36, 3], 
        aa_layer=R.BlurPool2d,
        stem_width=32, 
        stem_type='deep', 
        avg_down=True,
        block_args=dict(attn_layer='eca'))
    return R._create_resnet('ecaresnetblur152d', pretrained, **dict(model_args, **kwargs))

@register_model
def ecaresnetblur200d(pretrained: bool = False, **kwargs) -> R.ResNet:
    """Constructs a ResNet-152-D model with eca.
    """
    model_args = dict(
        block=R.Bottleneck, layers=[3, 24, 36, 3], 
        aa_layer=R.BlurPool2d,
        stem_width=32, 
        stem_type='deep', 
        avg_down=True,
        block_args=dict(attn_layer='eca'))
    return R._create_resnet('ecaresnetblur152d', pretrained, **dict(model_args, **kwargs))

@register_model
def ecaresnetaa101d(pretrained: bool = False, **kwargs) -> R.ResNet:
    """Constructs a ECA-ResNet-101-D model w/ avgpool anti-aliasing
    """
    model_args = dict(
        block=R.Bottleneck, layers=[3, 4, 23, 3], 
        aa_layer=nn.AvgPool2d,
        stem_width=32, 
        stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return R._create_resnet('ecaresnetaa101d', pretrained, **dict(model_args, **kwargs))

@register_model
def ecaresnet152d(pretrained: bool = False, **kwargs) -> R.ResNet:
    """Constructs a ResNet-152-D model with eca.
    """
    model_args = dict(
        block=R.Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return R._create_resnet('ecaresnet152d', pretrained, **dict(model_args, **kwargs))

# @register_model
# def mobilenetv4_conv_blur_large(pretrained: bool = False, **kwargs):
#     model = m3._gen_mobilenet_v4('mobilenetv4_conv_blur_large', 1.0, 
#                                  pretrained=pretrained, 
#                                  aa_layer='blurpc', 
#                                  **kwargs)
#     return model

# @register_model
# def mobilenetv4_conv_blur_medium(pretrained: bool = False, **kwargs):
#     model = m3._gen_mobilenet_v4('mobilenetv4_conv_blur_medium', 1.0, 
#                                  pretrained=pretrained, 
#                                  aa_layer='blurpc', 
#                                  **kwargs)
#     return model

# @register_model
# def mobilenetv4_hybrid_large(pretrained: bool = False, **kwargs):
#     model = m3._gen_mobilenet_v4('mobilenetv4_hybrid_large', 1.0, 
#                                  pretrained=pretrained, 
#                                  **kwargs)
#     return model

@register_model
def swinv2_base_window16(pretrained: bool = False, **kwargs):
    model_args = dict(window_size=16, 
                      embed_dim=128, 
                      depths=(2, 2, 18, 2), 
                      num_heads=(4, 8, 16, 32))
    model = swinv2._create_swin_transformer_v2(
        'swinv2_base_window16_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def swinv2_base_window8(pretrained=False, **kwargs):
    """
    """
    model_args = dict(window_size=8, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    return swinv2._create_swin_transformer_v2(
        'swinv2_base_window8_256', pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def swinv2_small_window16(pretrained=False, **kwargs):
    """
    """
    model_args = dict(window_size=16, 
                      embed_dim=96, 
                      depths=(2, 2, 18, 2), 
                      num_heads=(3, 6, 12, 24))
    return swinv2._create_swin_transformer_v2(
        'swinv2_small_window16_256', pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def vit_base_patch16_rope_reg1_gap(pretrained: bool = False, **kwargs) -> TimmEva:
    model_args = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=True,
        qkv_bias=True,
        init_values=1e-5,
        class_token=False,
        num_reg_tokens=1,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = eva._create_eva('vit_base_patch16_rope_reg1_gap', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch32(pretrained: bool = False, **kwargs) -> TimmVisionTransformer:
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12)
    model = vt._create_vision_transformer('vit_base_patch32', 
                                          pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16(pretrained: bool = False, **kwargs) -> TimmVisionTransformer:
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=(32, 16), embed_dim=768, depth=12, num_heads=12,
                      init_values=1e-5,
                    #   mlp_ratio=2.66667 * 2,
                    #   mlp_layer=vt.SwiGLUPacked, act_layer=nn.SiLU,
                      reg_tokens=4, no_embed_class=True,
                    # do not use this in multi cards setting
                    #   class_token=False, global_pool='map',
                    #   drop_path_rate=0.15,
                      )
    model = vt._create_vision_transformer('vit_base_patch16', 
                                          pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch14(pretrained: bool = False, **kwargs) -> TimmVisionTransformer:
    """ ViT-B/14 for DINOv2
    """
    model_args = dict(patch_size=14, embed_dim=768, depth=12, num_heads=12, init_values=1e-5)
    model = vt._create_vision_transformer(
        'vit_base_patch14', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch8(pretrained: bool = False, **kwargs) -> TimmVisionTransformer:
    """ ViT-Base (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12)
    model = vt._create_vision_transformer('vit_base_patch8', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_plus(pretrained: bool = False, **kwargs) -> TimmVisionTransformer:
    """ ViT-Base (ViT-B/16+)
    """
    model_args = dict(patch_size=16, embed_dim=896, depth=12, num_heads=14, init_values=1e-5)
    model = vt._create_vision_transformer(
        'vit_base_patch16_plus', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_rpn(pretrained: bool = False, **kwargs) -> TimmVisionTransformer:
    """ ViT-Base (ViT-B/16) w/ residual post-norm
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, init_values=1e-5,
        class_token=False, block_fn=vt.ResPostBlock, global_pool='avg')
    model = vt._create_vision_transformer(
        'vit_base_patch16_rpn', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_siglip(pretrained: bool = False, **kwargs) -> TimmVisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
    )
    model = vt._create_vision_transformer(
        'vit_base_patch16_siglip', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_rope(pretrained: bool = False, **kwargs) -> TimmEva:
    """RoPE-Axial ViT-B/16 from https://github.com/naver-ai/rope-vit"""
    model_args = dict(
        patch_size=(64, 8),
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        attn_type='rope',
        use_fc_norm=False,
        qkv_bias=True,
        init_values=1e-5,
        class_token=True,
        global_pool='token',
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        rope_grid_indexing='xy',
        rope_temperature=100.0,
    )
    model = eva._create_eva('vit_base_patch16_rope', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_rope_mixed(pretrained: bool = False, **kwargs) -> TimmEva:
    """RoPE-Mixed ViT-B/16 from https://github.com/naver-ai/rope-vit"""
    model_args = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        attn_type='rope',
        init_values=1e-5,
        class_token=True,
        global_pool='token',
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        rope_grid_indexing='xy',
        rope_temperature=10.0,
        rope_mixed_mode=True,
    )
    model = eva._create_eva('vit_base_patch16_rope_mixed', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_intern_patch16(pretrained: bool = False, **kwargs) -> TimmVisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        init_values=0.1, final_norm=False, dynamic_img_size=True,
    )
    model = vt._create_vision_transformer(
        'vit_intern_patch16', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_large_patch16(pretrained: bool = False, **kwargs) -> TimmVisionTransformer:
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
    model = vt._create_vision_transformer('vit_large_patch16', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def naflexvit_base_patch16(pretrained: bool = False, **kwargs) -> TimmNaFlexVit:
    """ NaFlex-ViT-Base model (NaFlex-ViT-B/16)
    """
    model_args = dict(patch_size=(32, 16), embed_dim=768, depth=12, num_heads=12,
                      init_values=1e-5,
                    #   mlp_ratio=2.66667 * 2,
                    #   mlp_layer=vt.SwiGLUPacked, act_layer=nn.SiLU,
                      reg_tokens=4, 
                      no_embed_class=True,
                    # do not use this in multi cards setting
                    #   class_token=False, 
                    #   global_pool='map',
                    #   drop_path_rate=0.15,
                      )
    model = vt._create_vision_transformer('naflexvit_base_patch16', pretrained=pretrained, 
                                          use_naflex=True,
                                          **dict(model_args, **kwargs))
    return model

def timm_create_model(in_channels, num_classes,
                variant="resnet50d",
                **kwargs,
                ):
  part_create = partial(tmodels.create_model, variant,
                            in_chans=in_channels,
                            num_classes=num_classes,)
  if variant.find("swin") != -1 or variant.find("vit") != -1:
    assert kwargs.get("top_k") is not None, "Must contain top_k params"
    assert kwargs.get("max_len") is not None, "Must contain max_len params"

    img_size = (kwargs["max_len"], kwargs["top_k"])
    
    return part_create(img_size=img_size)
  else:
    return part_create()