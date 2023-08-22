'''
Assignment 08-23: watch prediction
'''

import numpy as np
import os
import pandas as pd

DATA_DIR = '../../../data/'

def get_kuairec_data(data_dir=DATA_DIR):
	# adapted from https://github.com/chongminggao/KuaiRec/blob/main/loaddata.py
    csv_dir = os.path.join(data_dir, "KuaiRec 2.0", "data")

    try:
        # big_matrix = pd.read_csv(os.path.join(csv_dir,"big_matrix.csv"))
        small_matrix = pd.read_csv(os.path.join(csv_dir,"small_matrix.csv"))
        # social_network = pd.read_csv(os.path.join(csv_dir,"social_network.csv"))
        # social_network["friend_list"] = social_network["friend_list"].map(eval)
        # item_categories = pd.read_csv(os.path.join(csv_dir,"item_categories.csv"))
        # item_categories["feat"] = item_categories["feat"].map(eval)
        user_features = pd.read_csv(os.path.join(csv_dir,"user_features.csv"))
        item_daily_feat = pd.read_csv(os.path.join(csv_dir,"item_daily_features.csv"))
    except FileNotFoundError as e:
        print("Data file not found at", csv_dir)
        print("Have you downloaded the data? See https://kuairec.com/#download-the-data")
    return small_matrix, item_daily_feat, user_features

def extract_features_labels(small_matrix, item_daily_feat, user_features):
	# Joining user and item features
	merged_user_matrix = small_matrix.merge(user_features, how='left', on='user_id')
	merged_matrix = merged_user_matrix.merge(item_daily_feat, how='left', on=['video_id', 'date'])

	# Cleaning and subsampling
	merged_matrix.dropna(inplace=True)
	subsampled_matrix = merged_matrix.sample(n=10000, random_state=0)
	label_name = ['watch_ratio']  # target
	features_name = ['like_cnt', 'comment_cnt', 'music_id', # about video
	                                'follow_user_num_x', 'friend_user_num'] # about user
	data_matrix = subsampled_matrix[label_name + features_name]

	return label_name, features_name, data_matrix                             


# Loading data
small_matrix, item_daily_feat, user_features = get_kuairec_data()
# Merging and extracting features and lavels
label_name, features_name, data_matrix = extract_features_labels(small_matrix, item_daily_feat, user_features)

# splitting into test and evaluation
# log transform on watch ratio
y_train = np.log(1+np.array(data_matrix[:9000][label_name]))
X_train = np.array(data_matrix[:9000][features_name])

y_eval = np.log(1+np.array(data_matrix[9000:][label_name]))
X_eval = np.array(data_matrix[9000:][features_name])

# training a linear predictor

lam = 0 # regularization
d = X_train.shape[1]
theta_hat = np.linalg.inv(X_train.T @ X_train + lam * np.eye(d)) @ X_train.T @ y_train
y_pred = X_train @ theta_hat
loss = np.mean((y_pred-y_train)**2)
print('training loss', loss)

# evaluating the linear predictor

y_pred = X_eval @ theta_hat
loss = np.mean((y_pred-y_eval)**2)
print('testing loss', loss)

import matplotlib.pyplot as plt
plt.plot(y_eval, y_pred, '.')
plt.xlabel('actual log-watch ratio')
plt.ylabel('predicted log-watch ratio')
plt.show()