import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.functional import normalize
import numpy as np

class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward_feature(self, h_i, h_j):
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def forward_label(self, q_i, q_j):
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss + entropy
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

    @torch.no_grad()
    def generate_pseudo_labels(self, c, pseudo_label_cur, index):
        batch_size = c.shape[0]
        device = c.device
        pseudo_label_nxt = -torch.ones(batch_size, dtype=torch.long).to(device)
        tmp = torch.arange(0, batch_size).to(device)

        prediction = c.argmax(dim=1)
        confidence = c.max(dim=1).values
        unconfident_pred_index = confidence < self.alpha
        pseudo_per_class = np.ceil(batch_size / self.cluster_num * self.gamma).astype(
            int
        )
        for i in range(self.cluster_num):
            class_idx = prediction == i
            if class_idx.sum() == 0:
                continue
            confidence_class = confidence[class_idx]
            num = min(confidence_class.shape[0], pseudo_per_class)
            confident_idx = torch.argsort(-confidence_class)  # 降序排列只返回下标index
            for j in range(num):
                idx = tmp[class_idx][confident_idx[j]]
                pseudo_label_nxt[idx] = i

        todo_index = pseudo_label_cur == -1  # cur伪标签中需要更新的
        pseudo_label_cur[todo_index] = pseudo_label_nxt[todo_index]  # cur伪标签中需要更新的更新到这一次计算的
        pseudo_label_nxt = pseudo_label_cur
        pseudo_label_nxt[unconfident_pred_index] = -1  # 再将不满足的变为待更新
        return pseudo_label_nxt.cpu(), index

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

class ClusterLossBoost(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 1e4

    def __init__(self, multiplier=1, distributed=False, cluster_num=10):
        super().__init__()
        self.multiplier = multiplier
        self.distributed = distributed
        self.cluster_num = cluster_num

    def forward(self, c, pseudo_label):

        pesudo_label_all = pseudo_label
        pseudo_index = pesudo_label_all != -1
        pesudo_label_all = pesudo_label_all[pseudo_index]
        idx, counts = torch.unique(pesudo_label_all, return_counts=True)
        freq = pesudo_label_all.shape[0] / counts.float()
        weight = torch.ones(self.cluster_num).to(c.device)
        weight[idx] = freq
        pseudo_index = pseudo_label != -1
        if pseudo_index.sum() > 0:
            criterion = nn.CrossEntropyLoss(weight=weight).to(c.device)
            loss_ce = criterion(
                c[pseudo_index], pseudo_label[pseudo_index].to(c.device)
            )
        else:
            loss_ce = torch.tensor(0.0, requires_grad=True).to(c.device)
        return loss_ce
