'''
Assignment 08-23: watch prediction
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from torch import nn

DATA_DIR = '../../../data/'


def get_kuairec_data(data_dir=DATA_DIR):
    # adapted from https://github.com/chongminggao/KuaiRec/blob/main/loaddata.py
    csv_dir = os.path.join(data_dir, "KuaiRec 2.0", "data")

    try:
        # big_matrix = pd.read_csv(os.path.join(csv_dir,"big_matrix.csv"))
        small_matrix = pd.read_csv(os.path.join(csv_dir, "small_matrix.csv"))
        # social_network = pd.read_csv(os.path.join(csv_dir,"social_network.csv"))
        # social_network["friend_list"] = social_network["friend_list"].map(eval)
        # item_categories = pd.read_csv(os.path.join(csv_dir,"item_categories.csv"))
        # item_categories["feat"] = item_categories["feat"].map(eval)
        user_features = pd.read_csv(os.path.join(csv_dir, "user_features.csv"))
        item_daily_feat = pd.read_csv(
            os.path.join(csv_dir, "item_daily_features.csv"))
    except FileNotFoundError as e:
        print("Data file not found at", csv_dir)
        print("Have you downloaded the data? See https://kuairec.com/#download-the-data")
    return small_matrix, item_daily_feat, user_features


def extract_features_labels_groups(small_matrix, item_daily_feat, user_features, group_num):
    # Joining user and item features
    merged_user_matrix = small_matrix.merge(
        user_features, how='left', on='user_id')
    merged_matrix = merged_user_matrix.merge(
        item_daily_feat, how='left', on=['video_id', 'date'])

    # Cleaning and subsampling
    merged_matrix.dropna(inplace=True)

    # EDIT: subset by group
    if group_num == 1:
        merged_matrix = merged_matrix[(merged_matrix['is_live_streamer'] > 0) | (merged_matrix['is_video_author'] > 0)]
    elif group_num == 2:
        merged_matrix = merged_matrix[(merged_matrix['is_live_streamer'] == 0) | (merged_matrix['is_video_author'] == 0)]
    elif group_num == 3:
        merged_matrix = merged_matrix[merged_matrix['video_duration_x'] > 13005]
    elif group_num == 4:
        merged_matrix = merged_matrix[merged_matrix['video_duration_x'] <= 13005]

    print('GROUP: ', group_num)

    subsampled_matrix = merged_matrix.sample(n=10000, random_state=0)
    label_name = ['watch_ratio']  # target
    features_name = [
        'like_cnt', 'comment_cnt',  # about video, remove music_id
        'play_cnt', 'video_duration_x',           # EDIT: added video feats
        'follow_user_num_x', 'friend_user_num',  # about user
        'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3', 'onehot_feat4'  # EDIT: added user feats
    ]
    data_matrix = subsampled_matrix[label_name + features_name]

    return label_name, features_name, data_matrix


def preprocess_features(X):
    # EDIT: normalize along feature dim
    row_sums = X.sum(axis=1)
    new_x = X / row_sums[:, np.newaxis]

    # change to tensor
    return new_x


def to_torch(X):
    return torch.from_numpy(X).type(torch.float32)


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, h_dim):
        super(MLP, self).__init__()
        self.L1 = nn.Linear(in_dim, h_dim)
        self.L2 = nn.Linear(h_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, X):
        inter = self.L1(X)
        return self.L2(self.relu(inter))


def train(X_train, y_train, model, optimizer, loss_fn):
    optimizer.zero_grad()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(X_eval, y_eval, model, loss_fn, classify):
    model.eval()
    pred = model(X_eval)
    loss = loss_fn(pred, y_eval)
    if classify:
        bin_pred = (pred.detach().numpy() > np.log(1 + 0.5)).flatten()
        bin_label = (y_eval.detach().numpy() > np.log(1 + 0.5)).flatten()
        total = bin_pred.size
        acc = np.sum(bin_pred == bin_label) / total
        pr = np.sum(bin_pred) / total
        fpr = np.sum(bin_pred[bin_label == 0]) / bin_pred[bin_label == 0].size
        fnr = 1 - (np.sum(bin_pred[bin_label == 1]) / bin_pred[bin_label == 1].size)
        ppv = np.sum(bin_label[bin_pred == 1]) / bin_label[bin_pred == 1].size
        npv = 1 - (np.sum(bin_label[bin_pred == 0]) / bin_label[bin_pred == 0].size)
        return acc, pr, fpr, fnr, ppv, npv

    return loss.item()


# Loading data
small_matrix, item_daily_feat, user_features = get_kuairec_data()
# Merging and extracting features and lavels
label_name, features_name, data_matrix = extract_features_labels_groups(
    small_matrix, item_daily_feat, user_features, 4)
print('LOADING DATA DONE')

# splitting into test and evaluation
# log transform on watch ratio
y_train = np.log(1+np.array(data_matrix[:9000][label_name]))
X_train = np.array(data_matrix[:9000][features_name])
# EDIT: calling preprocess features
X_train = preprocess_features(X_train)
# X_train, y_train = to_torch(X_train), to_torch(y_train)

y_eval = np.log(1+np.array(data_matrix[9000:][label_name]))
X_eval = np.array(data_matrix[9000:][features_name])
X_eval = preprocess_features(X_eval)
# X_eval, y_eval = to_torch(X_eval), to_torch(y_eval)

# EDIT: init model, assoc components
# h = 128
# model = MLP(X_train.shape[1], y_train.shape[1], h)
# loss_fn = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# epochs = 100

# for e in range(epochs):

    # training a linear predictor
    # train_loss = train(X_train, y_train, model, optimizer, loss_fn)
lam = 0  # regularization
d = X_train.shape[1]
theta_hat = np.linalg.inv(X_train.T @ X_train + lam *
                          np.eye(d)) @ X_train.T @ y_train
y_pred = X_train @ theta_hat
train_loss = np.mean((y_pred-y_train)**2)
# print('training loss', train_loss)

# evaluating the linear predictor
    # test_loss = evaluate(X_eval, y_eval, model, loss_fn, True)

y_pred = X_eval @ theta_hat
test_loss = np.mean((y_pred-y_eval)**2)
# print('testing loss', test_loss)

bin_pred = (y_pred > np.log(1 + 0.5)).flatten()
bin_label = (y_eval > np.log(1 + 0.5)).flatten()
total = bin_pred.size
acc = np.sum(bin_pred == bin_label) / total
pr = np.sum(bin_pred) / total
fpr = np.sum(bin_pred[bin_label == 0]) / bin_pred[bin_label == 0].size
fnr = 1 - (np.sum(bin_pred[bin_label == 1]) / bin_pred[bin_label == 1].size)
ppv = np.sum(bin_label[bin_pred == 1]) / bin_label[bin_pred == 1].size
npv = 1 - (np.sum(bin_label[bin_pred == 0]) / bin_label[bin_pred == 0].size)
print(acc, pr, fpr, fnr, ppv, npv)

# plt.plot(y_eval, y_pred, '.')
# plt.xlabel('actual log-watch ratio')
# plt.ylabel('predicted log-watch ratio')
# plt.show()

# orig training loss 0.16562870130674018 testing loss 0.15952662257017505
# added features training loss 0.1639176726147408 testing loss 0.1573221811080986
# normalize + add features training loss 0.09736843152442684 testing loss 0.08497483248963952
# remove music_id training loss 0.09652752599628554 testing loss 0.08422849921329123
# MLP training loss 0.09758633375167847 testing loss 0.08541905134916306

# produces content 253512 training loss 0.08773595094680786 testing loss 0.08449804037809372
    # with onehot training loss 0.0912407785654068 testing loss 0.0876886323094368
# does not produce 2138674 training loss 0.1006687581539154 testing loss 0.1057567298412323
    # with onehot 2138674 training loss 0.10038013011217117 testing loss 0.10542500764131546
# below avg video dur 355063 training loss 0.048338521271944046 testing loss 0.04873631149530411
# above avg video dur 1785164 training loss 0.0827745869755745 testing loss 0.07607518136501312
