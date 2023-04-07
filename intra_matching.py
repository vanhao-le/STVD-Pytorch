import numpy as np
import torch
import time
import pandas as pd
import os


POS_DESC = r'output\pos_train_descriptor.npz'
query = np.load(POS_DESC)
rows = len(query['image_ids'])
q_image_ids = query['image_ids']
q_class_ids = query['class_ids']
q_descriptors = query['descriptors']

frame_idx = []
for i in range(rows):
    fr_idx = int(q_image_ids[i].split('_')[-1])
    frame_idx.append(fr_idx)

device = "cuda:1"

tensor_descriptors = torch.from_numpy(q_descriptors)
tensor_descriptors = tensor_descriptors.to(device)

ouput_csv = r"output\train_intra_matching.csv"

tensor_class_ids = torch.from_numpy(q_class_ids)
tensor_class_ids = tensor_class_ids.to(device)

tensor_frame_idx = torch.from_numpy(np.array(frame_idx))
tensor_frame_idx = tensor_frame_idx.to(device)

if __name__ == '__main__':
    print("[INFO] starting .........")
    since = time.time()     
    data = []
    count = 0
    for i in range(rows):               
        min_score = torch.tensor(1.)
        vec_a = tensor_descriptors[i]        

        result_mx = torch.matmul(tensor_descriptors, vec_a)  
        result_mx = torch.nan_to_num(result_mx)
        mask = torch.ones(rows)*32.
        mask = mask.to(device)

        rs_mx = torch.where((tensor_class_ids == tensor_class_ids[i]) & (tensor_frame_idx == tensor_frame_idx[i]), result_mx, mask)
        min_indx = torch.argmin(rs_mx, dim=0)
        min_score = result_mx[min_indx]

       
        case = {
            'image_name': q_image_ids[i],                           
            'classIDx': q_class_ids[i],            
            'intra_score': min_score.cpu().numpy() 
        }
        data.append(case)
        # print(case)
        # count += 1
        # if count > 10: 
        #     break
   
    
    df = pd.DataFrame(data) 
    df.to_csv(ouput_csv, index=False, header=True)

    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))  
    


