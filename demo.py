import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# thresholds = np.arange(start=0., stop=1.0, step=0.1)
# high_threshold = np.arange(start=0.91, stop=1., step=0.001)
# thresholds = np.append(thresholds, high_threshold)
# print("Thresolds:", thresholds)


OUTPUT_FILE = r'output\rs50_train_descriptor_1.npz'
data = np.load(OUTPUT_FILE)


i_s = data['image_ids']
c_ids = data['class_ids']
d_cl =  data['descriptors']

num_record = len(i_s)

# for key in data.keys():
#     # print(key)
#     print(data[key].shape)

image_ids = []
class_ids = []
descriptors = []

for i in range(num_record):
    # print(image_ids[i], class_ids[i], descriptors[i].shape)    
    if (c_ids[i] != 243):
        image_ids.append(i_s[i])
        class_ids.append(c_ids[i])
        descriptors.append(d_cl[i])

NEW_FILE = r'output\rs50_train_descriptor.npz'
np.savez(
    NEW_FILE,
    image_ids=image_ids,
    class_ids=class_ids,
    descriptors=descriptors,
)

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