import pandas as pd
import torch
from network import Network
from metric import valid, eval_dataframe
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss, ClusterLossBoost
from dataloader import load_data
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from loss import ClusterLoss, InstanceLossBoost

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# MNIST-USPS
# BDGP
# CCV
# Fashion
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
Dataname = 'Fashion'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.0001)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--tune_epochs", default=50)
parser.add_argument("--feature_dim", default=1024)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument("--load_pretrain", default=0)
parser.add_argument("--alpha", default=0.9)
parser.add_argument("--gamma", default=0.3)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The code has been optimized.
# The seed was fixed for the performance reproduction, which was higher than the values shown in the paper.
if args.dataset == "MNIST-USPS":
    args.load_pretrain = 1
    args.con_epochs = 50
    args.tune_epochs = 12
    args.batch_size = 128
    args.high_feature_dim = 128
    args.alpha = 0.9
    args.gamma = 0.9
    seed = 10
if args.dataset == "BDGP":
    args.load_pretrain = 1
    args.con_epochs = 7
    args.tune_epochs = 18
    args.high_feature_dim = 64
    seed = 10
if args.dataset == "CCV":
    args.batch_size = 256
    args.con_epochs = 7
    args.tune_epochs = 27
    args.load_pretrain = 1
    args.weight_decay = 0
    seed = 3
if args.dataset == "Fashion":
    args.con_epochs = 96
    args.tune_epochs = 8
    args.batch_size = 256
    args.feature_dim = 64
    args.weight_decay = 0.0001
    args.load_pretrain = 1
    seed = 10
if args.dataset == "Caltech-2V":
    args.con_epochs = 39
    args.weight_decay = 0
    args.load_pretrain = 1
    seed = 10
if args.dataset == "Caltech-3V":
    # args.feature_dim = 64
    args.load_pretrain = 0
    args.con_epochs = 33
    args.weight_decay = 0
    seed = 10
if args.dataset == "Caltech-4V":
    args.feature_dim = 64
    args.weight_decay = 0
    args.load_pretrain = 1
    args.con_epochs = 34
    seed = 10
if args.dataset == "Caltech-5V":
    args.con_epochs = 77
    args.feature_dim = 64
    args.weight_decay = 0
    args.load_pretrain = 1
    seed = 5


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)


def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
    return tot_loss / len(data_loader)


def contrastive_train_2(epoch):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, xrs, zs = model(xs)
        loss_list = []
        for v in range(view):
            for w in range(v + 1, view):
                loss_list.append(criterion.forward_feature(hs[v], hs[w]))
                loss_list.append(criterion_clu(qs[v], qs[w]))
                # 一致性预测loss：
                loss_list.append(mes(qs[v], qs[w]))

            loss_list.append(mes(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
    return tot_loss / len(data_loader)


def fine_tuning_2(epoch, model, device, criterion_ins, criterion_clu, pseudo_labels):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    tot_loss = 0.
    mes = torch.nn.MSELoss()

    for batch_idx, (xs, _, index) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)

        model.eval()
        with torch.no_grad():
            _, qs, _, _ = model.forward(xs)
        pseudo_nums = 0
        for v in range(view):
            pseudo_label = pseudo_labels[v]
            pseudo_label_cur, index_cur = criterion_ins.generate_pseudo_labels(
                qs[v], pseudo_label.to(qs[v].device), index.to(qs[v].device)
            )
            pseudo_label[index_cur] = pseudo_label_cur
            pseudo_nums += (pseudo_label != -1).float().sum().numpy()
        print("pseudo_nums:" + str(pseudo_nums))

        model.train()
        optimizer.zero_grad()
        hs, qs, xrs, zs = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(mes(xs[v], xrs[v]))
            for w in range(v + 1, view):
            #     loss_ins = criterion_ins(hs[v], hs[w], pseudo_labels[v][index].to(hs[v].device))
            #     loss_list.append(loss_ins)
            #     # loss_clu = criterion_clu(qs[w], pseudo_labels[v][index].to(qs[v].device))
            #     # loss_list.append(loss_clu)
            #     # 一致性预测loss：
                loss_list.append(mes(qs[v], qs[w]))

            for w in range(view):
                if v == w:
                    continue
                # loss_ins = criterion_ins(hs[v],hs[w], pseudo_labels[v][index].to(hs[v].device))
                # loss_list.append(loss_ins)
                loss_clu = criterion_clu(qs[w], pseudo_labels[v][index].to(qs[v].device))
                loss_list.append(loss_clu)
                # 一致性预测loss：
                # loss_list.append(mes(qs[v], qs[w]))

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
    return tot_loss / len(data_loader)


evaluate_data = eval_dataframe(eval_conf=False)

conf_evaluate_data = eval_dataframe(eval_conf=True)
if not os.path.exists('./models'):
    os.makedirs('./models')
if not os.path.exists('./pretrain'):
    os.makedirs('./pretrain')
T = 1
for i in range(T):
    print("ROUND:{}".format(i + 1))
    setup_seed(seed)
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
    print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)
    epoch = 1
    # ------------------step1------------------------------------------
    if args.load_pretrain:
        checkpoint = torch.load('./pretrain/' + args.dataset + '.pth')
        model.load_state_dict(checkpoint)
    else:
        model.train()
        while epoch <= args.mse_epochs:
            pretrain(epoch)
            epoch += 1
        state = model.state_dict()
        torch.save(state, './pretrain/' + args.dataset + '.pth')
        print('pretrain Saving..')

    # ------------------step2------------------------------------------
    criterion_clu = ClusterLoss()
    model.train()
    epoch = 1
    while epoch <= args.con_epochs:
        loss = contrastive_train_2(epoch)
        acc, nmi, ari,pur = valid(model, device, dataset, view, data_size, class_num,alpha=args.alpha, eval_h=False,eval_conf=False)
        evaluate_data=evaluate_data._append(
            {'Epoch':epoch,'ACC': acc, 'NMI': nmi, 'ARI': ari, 'PUR': pur,'loss':loss},ignore_index=True)
        # conf_acc, conf_nmi, conf_ari, conf_pur, sum_allconf = valid(model, device, dataset, view, data_size, class_num,
        #                                                             alpha=args.alpha, eval_h=False, eval_conf=True)
        # conf_evaluate_data=conf_evaluate_data._append(
        #     {'Epoch':epoch,'conf_ACC': conf_acc, 'conf_NMI': conf_nmi, 'conf_ARI': conf_ari, 'conf_PUR': conf_pur, 'conf_NUM': sum_allconf},ignore_index=True)
        epoch += 1

    # ------------------step3------------------------------------------
    criterion_ins = InstanceLossBoost(
        tau=0.5, distributed=True, alpha=args.alpha, gamma=args.gamma, cluster_num=class_num
    )
    criterion_clu = ClusterLossBoost(cluster_num=class_num)
    pseudo_labels = [-torch.ones(data_size, dtype=torch.long) for v in range(view)]
    epoch = 1
    while epoch <= args.tune_epochs:
        loss=fine_tuning_2(epoch, model, device, criterion_ins, criterion_clu, pseudo_labels)
        acc, nmi, ari,pur = valid(model, device, dataset, view, data_size, class_num,alpha=args.alpha, eval_h=False,eval_conf=False)
        evaluate_data=evaluate_data._append(
            {'Epoch':epoch,'ACC': acc, 'NMI': nmi, 'ARI': ari, 'PUR': pur,'loss':loss},ignore_index=True)
        epoch += 1
    state = model.state_dict()
    torch.save(state, './models/' + args.dataset + '.pth')
    print('Saving..')
evaluate_data.to_csv('./evaluate_data/'+ args.dataset+'.csv', index=True)
# conf_evaluate_data.to_csv('./evaluate_data/'+ args.dataset+'_conf.csv', index=True)