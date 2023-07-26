import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision.models import inception
from torchsummary import summary
import matplotlib.pyplot as plt
import pandas as pd
import collections
from model.model_pooling import EmbeddingNet
import math

from numpy.random import rand
import matplotlib

import matplotlib.pyplot as plt


my_str = "c02_20210201185459_9_64"

print(my_str.rsplit('_', 1)[0])

# x = np.array([-0.5, -0.3, -0.1, -0.5, -0.7, -0.9])
# x_max = np.max(x)
# x_min = np.min(x)

# x = (2*(x - x_min)/(x_max - x_min) ) - 1 

# print(x)


'''
need to check
'''
# eps = 1e-6
# def cos(A, B): 
#     return (A*B).sum(axis=1) / (A*A).sum(axis=1) ** .5 / (B*B).sum(axis=1) ** .5

# output1 = torch.rand(2,2048)
# output1 = torch.tensor([[0., 1.], [0., 1.]])
# output1 = output1 / (torch.norm(output1, p=2, dim=1, keepdim=True) + eps).expand_as(output1)
# output2 = torch.rand(2,2048)
# output2 = torch.tensor([[1., 0.], [0., 1.]])
# output2 = output2 / (torch.norm(output2, p=2, dim=1, keepdim=True) + eps).expand_as(output2)
# # e_dim = F.pairwise_distance(output1, output2)
# # print(e_dim.shape)
# # print(output1.shape, output1)
# # print(output2.shape, output2)
# print(torch.mul(output1, output2).sum(1))
# c_sim = torch.matmul(output1, output2.T).sum(0)
# print(c_sim.shape)

# print(e_dim)
# print(c_sim)

# margin = 1.
# eps = 1e-6
# size_average = False
# output1 = torch.tensor([[0., 1.]])
# print(output1.shape)
# output1 = output1 / (torch.norm(output1, p=2, dim=1, keepdim=True) + eps).expand_as(output1)
# output2 = torch.tensor([[0., 1.]])
# output2 = output2 / (torch.norm(output2, p=2, dim=1, keepdim=True) + eps).expand_as(output2)
# target = 1
# print(output1)
# print(output2)
# dif = output2 - output1

# distances = torch.pow(dif + eps, 2).sum(dim=1).sqrt()
# print("square distance:", distances)
# cosine_sim = torch.matmul(output1, output2.T)
# print("cosine:", cosine_sim)
# losses = 0.5 * target*torch.pow(cosine_sim, 2) + (1 + -1*target) * torch.pow(torch.clamp(margin - cosine_sim, min=0), 2)
# # losses = 0.5 * target*torch.pow(distances, 2) + 0.5 * (1 - target) * torch.pow(torch.clamp(margin - distances, min=0), 2)
# losses_2 = target*(1 - cosine_sim) + (1-target)*(margin - 1 + cosine_sim)
# print("cosine loss:", losses_2, "square loss:", losses)
# rs = losses.mean() if size_average else losses.sum()
# # print(rs)



# train_labels = np.array([1, 2, 1, 2, 1, 2, 3])
# labels_set = set(train_labels)

# label_to_indices = {label: np.where(train_labels == label)[0] for label in labels_set}
# print(label_to_indices)

# RES_RESULT = r"KF_one_result.npz"
# rs_data = np.load(RES_RESULT)
# rs_threshold = rs_data['threshold']
# rs_precision = rs_data['precision']
# rs_recall = rs_data['recall']
# rs_f1 = rs_data['F1_score']



# x = 3
# print(x**2)

# # create all axes we need
# ax0 = plt.subplot(211)
# ax1 = ax0.twinx()
# ax2 = plt.subplot(212)
# ax3 = ax2.twinx()

# # share the secondary axes
# # ax1.get_shared_y_axes().join(ax1, ax3)

# ax0.plot(rand(1) * rand(10),'r')
# ax1.plot(10*rand(1) * rand(10),'b')
# ax2.plot(3*rand(1) * rand(10),'g')
# ax3.plot(10*rand(1) * rand(10),'y')
# plt.show()

# q_class_ids = torch.from_numpy(np.arange(10))
# a = torch.rand(10)
# c = torch.zeros(10)
# y = torch.where(q_class_ids != q_class_ids[2], a, c)
# print(q_class_ids[2], q_class_ids, y)
# OUTPUT_FILE = r'output\peaked_train_matching_2.csv'
# df = pd.read_csv(OUTPUT_FILE)

# df['period'] = df.classIDx.astype(str).str.cat(df.video_name.astype(str))

# data = df['period'].to_list()
# print(len(set(data)))




# x = np.arange(-0.5, -0.1, 0.05)
# x2 = np.arange(-0.1, 0., 0.005)
# x = np.append(x, x2)
# x = np.append(x, [0.])
# print(x)

# POS_DESC = r'iciap_data\KF_pos_test_descriptor.npz'
# query = np.load(POS_DESC)
# rows = len(query['image_ids'])
# q_image_ids = query['image_ids']
# q_class_ids = query['class_ids']
# q_descriptors = query['descriptors']

# image_ids = []
# class_ids = []
# descriptors = []


# for i in range(rows):
#     image_ids.append(q_image_ids[i])
#     class_ids.append(q_class_ids[i])
#     descriptors.append(q_descriptors[i])

# NEG_DESC = r'output\neg_test_descriptor.npz'

# query = np.load(NEG_DESC)
# rows = len(query['image_ids'])
# q_image_ids = query['image_ids']
# q_class_ids = query['class_ids']
# q_descriptors = query['descriptors']

# for i in range(rows):
#     image_ids.append(q_image_ids[i])
#     class_ids.append(q_class_ids[i])
#     descriptors.append(q_descriptors[i])

# NEW_FILE = 'iciap_data\KF_FS_test_descriptor.npz'

# np.savez(
#         NEW_FILE,
#         image_ids=image_ids,
#         class_ids=class_ids,
#         descriptors=descriptors,
#     )

# KF_FS_test_descriptor


# for key in query.keys():
#     print(key)
#     print(query[key].shape)


# count = 0
# count_2 = 0
# for i in range(rows):
#     if (q_class_ids[i] < 243):
#         count += 1
#     if (q_class_ids[i] >= 243):
#         count_2 += 1
# print(count, count_2)

# image_ids = []
# class_ids = []
# descriptors = []

# for i in range(5000):
#     # print(image_ids[i], class_ids[i], descriptors[i].shape)   
#     image_ids.append(q_image_ids[i])
#     class_ids.append(q_class_ids[i])
#     descriptors.append(q_descriptors[i])

# NEW_FILE = r'output\pos_train_descriptor.npz'
# np.savez(
#     NEW_FILE,
#     image_ids=image_ids,
#     class_ids=class_ids,
#     descriptors=descriptors,
# )


# NUM_THREADS = 3
# frame_lst = np.arange(10)
# splits = list(split(frame_lst, NUM_THREADS))

# print(splits)





# eps = 1e-6
# eps = torch.tensor(eps)
# ovr = 0.4 # desired overlap of neighboring regions
# steps = torch.tensor([2, 3, 4, 5, 6, 7]) # possible regions for the long dimension

# x = torch.rand(1, 2048, 7, 7)
# W = x.size(3)
# H = x.size(2)

# w = min(W, H)
# w2 = math.floor(w/2.0 - 1)

# b = (max(H, W)-w)/(steps-1)

# # print(b)
# (tmp, idx) = torch.min(torch.abs(((w**2 - w*b)/w**2)-ovr), 0) 

# # print(idx)
# # steps(idx) regions for long dimension region overplus per dimension

# Wd = 0;
# Hd = 0;
# if H < W:
#     Wd = idx.item() + 1
# elif H > W:
#     Hd = idx.item() + 1

# v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
# v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

# # print(Wd, Hd) 0, 0
# L = 3


# for l in range(1, L+1):
    
#     wl = math.floor(2*w/(l+1))
#     wl2 = math.floor(wl/2 - 1)
       
#     if l+Wd == 1:
#         b = 0
#     else:
#         b = (W-wl)/(l+Wd-1)
#     cenW = torch.floor(wl2 + torch.Tensor(range(l-1+Wd+1))*b) - wl2 # center coordinates
#     if l+Hd == 1:
#         b = 0
#     else:
#         b = (H-wl)/(l+Hd-1)
#     cenH = torch.floor(wl2 + torch.Tensor(range(l-1+Hd+1))*b) - wl2 # center coordinates
    
#     print(cenW, cenH)
    
#     for i_ in cenH.tolist():
#         for j_ in cenW.tolist():
#             if wl == 0:
#                 continue
#             R = x[:,:,(int(i_)+torch.Tensor(range(wl)).long()).tolist(),:]
#             # print(R.shape)
#             R = R[:,:,:,(int(j_)+torch.Tensor(range(wl)).long()).tolist()]
#             print(R.shape)
#             vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
#             vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
#             v += vt



# pool_names = ['mac', 'spoc', 'rmac', 'gem']
# model = EmbeddingNet(pool_names[0])
# model.to('cuda:0')
# summary(model, (3, 224, 224), batch_size=1)

# eps = 1e-9
# x = torch.rand((1,5,1,1))
# print(x)
# x = x / (torch.norm(x, p=2, dim=0, keepdim=True) + eps).expand_as(x)

# print(x)
# x = x.squeeze(-1).squeeze(-1)
# print(x)

# root_path = r"data\positive_frame.csv"
# df = pd.read_csv(root_path)

# data = {}
# min = 99999
# classIDx = 0
# for idx, item in df.iterrows():
#     parent_dir = int(item["classIDx"])
#     if (parent_dir != 243):
#         if (parent_dir in data):
#             data[parent_dir] += 1 
#         else:
#             data[parent_dir] = 1
        

# for key, value in data.items():
#        if value < min:
#             min = value
#             classIDx = key

# print(classIDx, min)

# data_sorted = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))

# x = np.arange(0, 243)
# y = []

# for key, value in data_sorted.items():
#    if value < min:
#         min = value
#    y.append(value)

# print(min)
# x, y = zip(*lists) # unpack a list of pairs into two tuples

# plt.plot(x, y)
# plt.title('The mininum frames per a class: 8 - The total frames: 27 K')
# plt.show()


# thresholds = np.arange(start=0., stop=1.0, step=0.1)
# high_threshold = np.arange(start=0.91, stop=1., step=0.001)
# thresholds = np.append(thresholds, high_threshold)
# print("Thresolds:", thresholds)
# from torch.nn.functional import normalize

# a = torch.tensor([[1., 2.], [3., 4.]])
# # a = normalize(a, p=2.0, dim = 1)
# a = a / (a.pow(2).sum(1, keepdim=True).sqrt())
# b = torch.tensor([[1., 2.], [3., 4.]])
# # b = normalize(b, p=2.0, dim = 1)
# b = b / (b.pow(2).sum(1, keepdim=True).sqrt())
# print(a)
# print(b)
# c = torch.matmul(a,b.T)
# print(c)

# OUTPUT_FILE = r'output\neg_train_descriptor.npz'
# data = np.load(OUTPUT_FILE)


# i_s = data['image_ids']
# c_ids = data['class_ids']
# d_cl =  data['descriptors']

# num_record = len(i_s)
# count = 0
# for key in data.keys():
#     print(key)
#     print(data[key].shape)

# # image_ids = []
# # class_ids = []
# # descriptors = []

# for i in range(num_record):
#     if c_ids[i] < 243:
#         count += 1

# print("Positive", count)
#     # print(image_ids[i], class_ids[i], descriptors[i].shape)    
#     if (c_ids[i] != 243):
#         image_ids.append(i_s[i])
#         class_ids.append(c_ids[i])
#         descriptors.append(d_cl[i])

# NEW_FILE = r'output\rs50_train_descriptor.npz'
# np.savez(
#     NEW_FILE,
#     image_ids=image_ids,
#     class_ids=class_ids,
#     descriptors=descriptors,
# )

# QUERY_DESC = r'output\vgg16_test_descriptor.npz'
# query = np.load(QUERY_DESC)
# rows = len(query['image_ids'])
# q_image_ids = query['image_ids']
# q_class_ids = query['class_ids']
# q_descriptors = query['descriptors']


# REF_DESC = r'output\vgg16_train_descriptor.npz'
# reference = np.load(REF_DESC)
# cols = len(reference['image_ids'])
# r_image_ids = reference['image_ids']
# r_class_ids = reference['class_ids']
# r_descriptors = reference['descriptors']

# device = "cuda:1"

# tensor_r_descriptors = torch.from_numpy(r_descriptors)
# tensor_r_descriptors = tensor_r_descriptors.to(device)

# if __name__ == '__main__': 
    

#     with torch.no_grad():    

#         for i in range(rows):    
#             tensor1 = torch.from_numpy(q_descriptors[i]).to(device) 
#             result_mx = torch.matmul(tensor_r_descriptors, tensor1)
#             # print(tensor1.shape, tensor_r_descriptors.shape, result_mx.shape)   

#             # Get the maximum along dim = 0 (axis = 0)
#             # max_indx = torch.argmin(result_mx, dim=0)
#             # max_value = result_mx[max_indx]

#             max_value, max_indx = torch.max(result_mx, dim=0)
#             # print(max_value.cpu().numpy(), max_indx)

#             case = {
#                 'q_image': q_image_ids[i],
#                 'q_class': q_class_ids[i],
#                 'ref_image': r_image_ids[max_indx],
#                 'ref_class': r_class_ids[max_indx],
#                 'score': max_value
#             }
#             print(case)

#             if i > 10:
#                 break




# path = r"E:\STVD_DL\data\train\243"
# os.remove(path)