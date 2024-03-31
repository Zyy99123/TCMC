import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import normalize
import torch
from torch import nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class ClusterLoss(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 1e4

    def __init__(self, tau=1.0, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed

    def forward(self, c_i,c_j, get_map=False):
        # n = c.shape[0]
        # assert n % self.multiplier == 0

        # c = c / np.sqrt(self.tau)

        # if self.distributed:
        #     c_list = [torch.zeros_like(c) for _ in range(dist.get_world_size())]
        #     # all_gather fills the list as [<proc0>, <proc1>, ...]
        #     c_list = diffdist.functional.all_gather(c_list, c)
        #     # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
        #     c_list = [chunk for x in c_list for chunk in x.chunk(self.multiplier)]
        #     # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
        #     c_sorted = []
        #     for m in range(self.multiplier):
        #         for i in range(dist.get_world_size()):
        #             c_sorted.append(c_list[i * self.multiplier + m])
        #     c_aug0 = torch.cat(
        #         c_sorted[: int(self.multiplier * dist.get_world_size() / 2)], dim=0
        #     )
        #     c_aug1 = torch.cat(
        #         c_sorted[int(self.multiplier * dist.get_world_size() / 2) :], dim=0
        #     )

        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        en_i = np.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        en_j = np.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        en_loss = en_i + en_j

        c = torch.cat((c_i.t(), c_j.t()), dim=0)
        n = c.shape[0]

        c = F.normalize(c, p=2, dim=1) / np.sqrt(self.tau)

        logits = c @ c.t()   #相似度矩阵
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER              #LARGE_NUMBER = 1e4

        logprob = F.log_softmax(logits, dim=1)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1)

        return loss + en_loss
class InstanceLossBoost(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 1e4

    def __init__(
        self,
        tau=0.5,
        multiplier=2,
        distributed=False,
        alpha=0.99,
        gamma=0.5,
        cluster_num=10,
    ):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.alpha = alpha
        self.gamma = gamma
        self.cluster_num = cluster_num



    def forward(self, z_i,z_j, pseudo_label):
        n = z_i.shape[0]       #n更新为一个bachsize

        invalid_index = pseudo_label == -1
        #mask 为预测是否相等矩阵
        mask = torch.eq(pseudo_label.view(-1, 1), pseudo_label.view(1, -1)).to(
            z_i.device
        )
        mask[invalid_index, :] = False
        mask[:, invalid_index] = False
        mask_eye = torch.eye(n).float().to(z_i.device)        #对角线为1的对角矩阵
        mask &= ~(mask_eye.bool())
        mask = mask.float()       #保留了除了对角线以外所有预测相同的对

        contrast_count = self.multiplier
        contrast_feature = torch.cat((z_i, z_j), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits        相似度矩阵   大小：2n*2n
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.tau
        )         #div 除法

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()          #相似度矩阵每列减去该列最大值？

        # tile mask
        # mask_with_eye = mask | mask_eye.bool()
        # mask = torch.cat(mask)
        #mask 大小:n*n
        mask = mask.repeat(anchor_count, contrast_count)            #复制（2，2）   现在mask 2n*2n
        mask_eye = mask_eye.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        #构建一个对角线全0,其余全1的2n*2n矩阵
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(n * anchor_count).view(-1, 1).to(z_i.device),
            0,
        )

        logits_mask *= 1 - mask         #2n*2n矩阵，求出伪标签不同的位置
        mask_eye = mask_eye * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask_eye * log_prob).sum(1) / mask_eye.sum(1)

        # loss
        instance_loss = -mean_log_prob_pos
        instance_loss = instance_loss.view(anchor_count, n).mean()

        return instance_loss

class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
            # Varying the number of layers of W can obtain the representations with different shapes.
        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, class_num),
            nn.Softmax(dim=1)
        )
        self.view = view
    def forward(self, xs):
        hs = []
        qs = []
        xrs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.feature_contrastive_module(z), dim=1)
            q = self.label_contrastive_module(z)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            qs.append(q)
            xrs.append(xr)
        return hs, qs, xrs, zs

    def forward_plot(self, xs):
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
            h = self.feature_contrastive_module(z)
            hs.append(h)
        return zs, hs

    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            q = self.label_contrastive_module(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds
