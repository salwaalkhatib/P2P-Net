import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from anchors import generate_default_anchor_maps, hard_nms
from clustering import PartsResort
from collections import OrderedDict


class ResidualAttentionBlock(nn.Module):
    '''
    Residual block of each attention layer with MLP
    '''
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor):
        attn_output, attn_output_weights = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x))
            
        x = x + attn_output
        x_ffn = self.mlp(self.ln_2(x))
        x = x + x_ffn
        return x
    

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Transformer(nn.Module):
    '''
    Builds the transformer based on number of layers and dimension of embeddings
    '''
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        for i,block in enumerate(self.resblocks):
            x = block(x)    
        return x


class PMG(nn.Module):
    def __init__(self, model, feature_size, num_ftrs, classes_num, topn, attn_width):
        super(PMG, self).__init__()

        self.backbone = model
        self.num_ftrs = num_ftrs
        self.topn = topn
        self.im_sz = 448
        self.pad_side = 224
        self.num_heads = 8
        self.attn_width = attn_width
        # self.PR = PartsResort(self.topn, self.num_ftrs//2)

        self.proposal_net = ProposalNet(self.num_ftrs) # object detection head
        _, edge_anchors, _ = generate_default_anchor_maps()
        self.edge_anchors = (edge_anchors+self.pad_side).astype(np.int)
        
        # mlp for regularization
        # self.self_attn_1 = nn.MultiheadAttention(self.attn_width, self.num_heads)
        # # Inverted bottleneck MLP
        # self.reg_mlp1 = nn.Sequential(
        #     nn.Linear(self.attn_width * self.topn, self.attn_width * self.topn * 4),
        #     nn.ELU(inplace=True),
        #     nn.Linear(self.attn_width * self.topn * 4, self.num_ftrs//2)
        # )
        # self.self_attn_2 = nn.MultiheadAttention(self.attn_width, self.num_heads)
        # # Inverted bottleneck MLP
        # self.reg_mlp2 = nn.Sequential(
        #     nn.Linear(self.attn_width * self.topn, self.attn_width  * self.topn * 4),
        #     nn.ELU(inplace=True),
        #     nn.Linear(self.attn_width  * self.topn * 4, self.num_ftrs//2)
        # )
        # self.self_attn_3 = nn.MultiheadAttention(self.attn_width, self.num_heads)
        # # Inverted bottleneck MLP
        # self.reg_mlp3 = nn.Sequential(
        #     nn.Linear(self.attn_width * self.topn, self.attn_width * self.topn * 4),
        #     nn.ELU(inplace=True),
        #     nn.Linear(self.attn_width * self.topn * 4, self.num_ftrs//2)
        # )
        
        # Projection layer from self.num_ftrs//2 to 256 (so concat of all three will give 768)
        # self.projection = nn.Parameter(torch.empty(self.num_ftrs//2, self.attn_width))
        # nn.init.normal_(self.projection, std=self.attn_width ** -0.5)

        # Transformer and reproject to 1024 dim for KL divergence
        self.transformer1 = Transformer(self.attn_width, 3, self.num_heads)
        self.transformer2 = Transformer(self.attn_width, 3, self.num_heads)
        self.transformer3 = Transformer(self.attn_width, 3, self.num_heads)
        self.reproj = nn.Sequential(
            nn.Linear(self.attn_width, self.attn_width * 4),
            nn.ELU(inplace=True),
            nn.Linear(self.attn_width * 4, self.num_ftrs//2)
        )

        # stage 1
        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs//4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )

        # stage 2
        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs//2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )

        # stage 3
        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )
        
        # concat features from different stages
        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2 * 3),
            nn.Linear(self.num_ftrs//2 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

    def forward(self, x, is_train=True):
        _, _, f1, f2, f3 = self.backbone(x) #part2pose regualrization uses these as input

        batch = x.shape[0]
        rpn_score = self.proposal_net(f3.detach()) # passing through fpn
        # top n proposals
        all_cdds = [np.concatenate((x.reshape(-1, 1), 
                    self.edge_anchors.copy(),
                    np.arange(0, len(x)).reshape(-1, 1)), 
                    axis=1) for x in rpn_score.data.cpu().numpy()]
        top_n_cdds = np.array([hard_nms(x, self.topn, iou_thresh=0.25) for x in all_cdds])
        top_n_index = top_n_cdds[:, :, -1].astype(np.int)
        top_n_index = torch.from_numpy(top_n_index).long().to(x.device)
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)
        
        # re-input salient parts
        part_imgs = torch.zeros([batch, self.topn, 3, 224, 224]).to(x.device)
        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        for i in range(batch):
            for j in range(self.topn):
                [y0, x0, y1, x1] = top_n_cdds[i, j, 1:5].astype(np.int)
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], 
                                                        size=(224, 224), mode='bilinear',
                                                        align_corners=True)
        
        part_imgs = part_imgs.view(batch*self.topn, 3, 224, 224)
        _, _, f1_part, f2_part, f3_part = self.backbone(part_imgs.detach())
        f1_part = self.conv_block1(f1_part).view(batch*self.topn, -1)
        f2_part = self.conv_block2(f2_part).view(batch*self.topn, -1)
        f3_part = self.conv_block3(f3_part).view(batch*self.topn, -1)
        yp1 = self.classifier1(f1_part)
        yp2 = self.classifier2(f2_part)
        yp3 = self.classifier3(f3_part)
        yp4 = self.classifier_concat(torch.cat((f1_part, f2_part, f3_part), -1))

        f1_points = f1_part.view(batch, self.topn, -1)
        f2_points = f2_part.view(batch, self.topn, -1)
        f3_points = f3_part.view(batch, self.topn, -1)
        f1_points = f1_points.permute(1, 0, 2).contiguous()
        f2_points = f2_points.permute(1, 0, 2).contiguous()
        f3_points = f3_points.permute(1, 0, 2).contiguous()
        f1_att = self.transformer1(f1_points)
        f2_att = self.transformer2(f2_points)
        f3_att = self.transformer3(f3_points)
        f1_att = f1_att.permute(1, 0, 2).contiguous()
        f2_att = f2_att.permute(1, 0, 2).contiguous()
        f3_att = f3_att.permute(1, 0, 2).contiguous()
        f1_gap = f1_att.mean(dim=1)
        f2_gap = f2_att.mean(dim=1)
        f3_gap = f3_att.mean(dim=1)
        f1_m = self.reproj(f1_gap)
        f2_m = self.reproj(f2_gap)
        f3_m = self.reproj(f3_gap)

        # f1_points = (f1_part @ self.projection).view(batch, self.topn, -1)
        # f1_attn , _ = self.self_attn_1(f1_points.permute(1,0,2), f1_points.permute(1,0,2), f1_points.permute(1,0,2))
        # f1_attn = f1_attn.permute(1,0,2).reshape(batch, -1)
        # f1_m = self.reg_mlp1(f1_attn) #part2pose regualrization uses these as input
        # f2_points = (f2_part @ self.projection).view(batch, self.topn, -1)
        # f2_attn , _= self.self_attn_2(f2_points.permute(1,0,2), f2_points.permute(1,0,2), f2_points.permute(1,0,2))
        # f2_attn = f2_attn.permute(1,0,2).reshape(batch, -1)
        # f2_m = self.reg_mlp2(f2_attn) #part2pose regualrization uses these as input
        # f3_points = (f3_part @ self.projection).view(batch, self.topn, -1)
        # f3_attn, _ = self.self_attn_3(f3_points.permute(1,0,2), f3_points.permute(1,0,2), f3_points.permute(1,0,2))
        # f3_attn = f3_attn.permute(1,0,2).reshape(batch, -1)
        # f3_m = self.reg_mlp3(f3_attn) #part2pose regualrization uses these as input
 
        # stage-wise classification
        f1 = self.conv_block1(f1).view(batch, -1)
        f2 = self.conv_block2(f2).view(batch, -1)
        f3 = self.conv_block3(f3).view(batch, -1)
        y1 = self.classifier1(f1)
        y2 = self.classifier2(f2)
        y3 = self.classifier3(f3)
        y4 = self.classifier_concat(torch.cat((f1, f2, f3), -1))

        return y1, y2, y3, y4, yp1, yp2, yp3, yp4, top_n_prob, f1_m, f1, f2_m, f2, f3_m, f3       
    
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ProposalNet(nn.Module):
    def __init__(self, depth):
        super(ProposalNet, self).__init__()
        self.down1 = nn.Conv2d(depth, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)
        # proposals: 14x14x6, 7x7x6, 4x4x9

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1).view(batch_size, -1)
        t2 = self.tidy2(d2).view(batch_size, -1)
        t3 = self.tidy3(d3).view(batch_size, -1)
        return torch.cat((t1, t2, t3), dim=1)
