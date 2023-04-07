import numpy as np
import torch
import time
import pandas as pd
import os
from multiprocessing import Pool


POS_DESC = r'output\pos_test_descriptor.npz'
query = np.load(POS_DESC)
rows = len(query['image_ids'])
q_image_ids = query['image_ids']
q_class_ids = query['class_ids']
q_descriptors = query['descriptors']


device = "cuda:1"

tensor_descriptors = torch.from_numpy(q_descriptors)
tensor_descriptors = tensor_descriptors.to(device)

ouput_csv = r"output\test_pos_inter_matching.csv"

tensor_class_ids = torch.from_numpy(q_class_ids)
tensor_class_ids = tensor_class_ids.to(device)

if __name__ == '__main__':
    print("[INFO] starting .........")
    since = time.time()     
    data = []
    count = 0
    for i in range(rows):               
        max_score = torch.tensor(0.)
        vec_a = tensor_descriptors[i]

        result_mx = torch.matmul(tensor_descriptors, vec_a)  
        result_mx = torch.nan_to_num(result_mx)
        mask = torch.zeros(rows)
        mask = mask.to(device)

        rs_mx = torch.where(tensor_class_ids != tensor_class_ids[i], result_mx, mask)
        max_indx = torch.argmax(rs_mx, dim=0)
        max_score = result_mx[max_indx]

        # for j in range(rows):
        #     if result_mx[j] > max_score and q_class_ids[i] != q_class_ids[j]:
        #         max_score = result_mx[j]
                # print(q_image_ids[i], q_image_ids[j], result_mx[j])
        
        case = {
            'image_name': q_image_ids[i],                           
            'classIDx': q_class_ids[i],
            'inter_score': max_score.cpu().numpy() 
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
    


