# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'



from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from util.pos_embed import interpolate_pos_embed
from allocator import Allocator


class LongContextViT(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, expert_nums=64, expert_ratio = 0.25,**kwargs):
        super(LongContextViT, self).__init__(**kwargs)


        patchs_num = int(kwargs['img_size'] /kwargs['patch_size'])
        self.dist = Allocator(num_experts=expert_nums, top_k=1, expert_capacity=128, 
        patchs_num = patchs_num, input_dim= kwargs['embed_dim'], expert_ratio = expert_ratio)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # ([1, 197, 1024])
        x = x + self.pos_embed[:,1:,:]
        x = self.pos_drop(x)

        # x, aux_loss = self.dist(x)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x, aux_loss = self.dist(x)


        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome, aux_loss

    def forward(self, x):
        x, aux_loss = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x, aux_loss



def lct_vit_base_patch16(**kwargs):
    expert_nums = 64
    expert_ratio = 0.25
    print("expert_nums: {} expert_ratio: {}".format(expert_nums, expert_ratio))
    model = LongContextViT(
        img_size = 224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        expert_nums = expert_nums, expert_ratio = expert_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def lct_vit_large_patch16(**kwargs):
    expert_nums = 64
    expert_ratio = 0.25
    print("expert_nums: {} expert_ratio: {}".format(expert_nums, expert_ratio))
    model = LongContextViT(
        img_size = 224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        expert_nums = expert_nums, expert_ratio = expert_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def lct_vit_huge_patch14(**kwargs):
    expert_nums = 64
    expert_ratio = 0.25
    print("expert_nums: {} expert_ratio: {}".format(expert_nums, expert_ratio))
    model = LongContextViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        expert_nums = expert_nums, expert_ratio = expert_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



