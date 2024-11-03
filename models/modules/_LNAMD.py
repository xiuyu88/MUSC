"""
    PatchMaker, Preprocessing and MeanMapper are copied from https://github.com/amazon-science/patchcore-inspection.
"""

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import math


# 将图像分割成patch
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """
        Args:
            features (_type_): 特征张量
            return_spatial_info (bool, optional): 决定是否返回空间信息。. Defaults to False.
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features


class Preprocessing(torch.nn.Module):
    def __init__(self, input_layers, output_dim):
        super(Preprocessing, self).__init__()
        self.output_dim = output_dim
        self.preprocessing_modules = torch.nn.ModuleList()
        for input_layer in input_layers:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1) # 自适应平均池化


class LNAMD(torch.nn.Module):
    def __init__(self, device, feature_dim=1024, feature_layer=[1,2,3,4], r=3, patchstride=1):
        """
        Args:
            device (_type_): CPU或GPU
            feature_dim (int, optional): 特征维度. Defaults to 1024.
            feature_layer (list, optional): 用于提取特征的模型层级列表. Defaults to [1,2,3,4].
            r (int, optional): 聚合度. Defaults to 3.
            patchstride (int, optional): 划分补丁时的步长. Defaults to 1.
        """
        super(LNAMD, self).__init__()
        self.device = device
        self.r = r
        self.patch_maker = PatchMaker(r, stride=patchstride) # 用于将特征图划分成补丁
        self.LNA = Preprocessing(feature_layer, feature_dim) # 自适应平均池化对特征进行降维

    def _embed(self, features):
        B = features[0].shape[0] # batch size

        features_layers = []
        for feature in features: # [B, N+1, C]
            # reshape and layer normalization
            feature = feature[:, 1:, :] # remove the cls token  -> [B, N, C]
            feature = feature.reshape(feature.shape[0],
                                      int(math.sqrt(feature.shape[1])),
                                      int(math.sqrt(feature.shape[1])),
                                      feature.shape[2]) # -> [B, √N, √N, C] 
            feature = feature.permute(0, 3, 1, 2) # -> [B, C, √N, √N]
            feature = torch.nn.LayerNorm([feature.shape[1], feature.shape[2],
                                          feature.shape[3]]).to(self.device)(feature)
            features_layers.append(feature)

        if self.r != 1:
            # divide into patches 将每个特征层划分为补丁
            features_layers = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features_layers]
            patch_shapes = [x[1] for x in features_layers]
            features_layers = [x[0] for x in features_layers]
        else: # r = 1
            patch_shapes = [f.shape[-2:] for f in features_layers]
            features_layers = [f.reshape(f.shape[0], f.shape[1], -1, 1, 1).permute(0, 2, 1, 3, 4) for f in features_layers]

        # 确保补丁尺寸统一
        ref_num_patches = patch_shapes[0]
        for i in range(1, len(features_layers)):
            patch_dims = patch_shapes[i]
            if patch_dims[0] == ref_num_patches[0] and patch_dims[1] == ref_num_patches[1]:
                continue
            _features = features_layers[i]
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features_layers[i] = _features
        features_layers = [x.reshape(-1, *x.shape[-3:]) for x in features_layers]
        
        # aggregation 聚合
        features_layers = self.LNA(features_layers)
        features_layers = features_layers.reshape(B, -1, *features_layers.shape[-2:])   # (B, L, layer, C)

        return features_layers.detach().cpu()


if __name__ == "__main__":
    import time
    device = 'cuda:0'
    LNAMD_r = LNAMD(device=device, r=3, feature_dim=1024, feature_layer=[1,2,3,4])
    B = 32
    patch_tokens = [torch.rand((B, 1370, 1024)), torch.rand((B, 1370, 1024)), torch.rand((B, 1370, 1024)), torch.rand((B, 1370, 1024))]
    patch_tokens = [f.to('cuda:0') for f in patch_tokens]
    s = time.time()
    features = LNAMD_r._embed(patch_tokens)
    e = time.time()
    print((e-s)*1000/32)