import torch
from torch import linalg as LA
import torch.nn.functional as F
import torch.nn as nn
from .Fusion import FusionBlock

from models.model_utils import *

class BPDSurv_Surv(nn.Module):
    def __init__(self, fusion='concat', omic_sizes=[94, 334, 521, 468, 1496, 479], n_classes=4,
                 model_size_wsi: str = 'small', model_size_omic: str = 'small'):
        super(BPDSurv_Surv, self).__init__()
        self.fusion = fusion
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = {
            "small": [512, 256, 256],
            "big": [1024, 512, 384]}
        self.size_dict_omic = {
            'small': [256, 256],
            'big': [1024, 1024, 1024, 256]}

        size_wsi = self.size_dict_WSI[model_size_wsi]
        size_omic = self.size_dict_omic[model_size_omic]

        fc = [
            nn.Linear(size_wsi[0],  size_wsi[1]),
            nn.GELU(),
        ]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)

        ### Constructing Genomic SNN
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=size_omic[0])]
            for i, _ in enumerate(size_omic[1:]):
                fc_omic.append(SNN_Block(dim1=size_omic[i], dim2=size_omic[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

        self.MultimodalFusion = FusionBlock(dim=256, num_heads=8, qkv_bias=True,drop_path=0.25)


        ### Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(
                size_wsi[1]+size_omic[-1]
                # 256
                , size_wsi[2]), nn.ReLU(), nn.Linear(size_wsi[2], size_wsi[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            self.mm = None

        ### Classifier
        self.classifier = nn.Linear(size_wsi[2], n_classes)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, 7)]

        h_path_bag = self.wsi_net(x_path).unsqueeze(0)

        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in
                  enumerate(x_omic)]
        h_omic_bag = torch.stack(h_omic).unsqueeze(0)


        h_fused = self.MultimodalFusion(h_omic_bag, h_path_bag)
        h_fused = self.mm(h_fused)


        logits = self.classifier(h_fused)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        attention_scores = {
        }
        return (hazards, S, Y_hat
                ,attention_scores
                )
