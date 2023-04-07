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

ouput_csv = r"output\test_intra_matching.csv"

def split(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

NUM_THREADS = 6
frame_lst = np.arange(rows)

splits = list(split(frame_lst, NUM_THREADS))

def target(sub_list):
    data = []
    for i in sub_list:
        image_name_a = str(q_image_ids[i])
        order_frame = int(q_image_ids[i].split('_')[-1])
        classIDx = int(q_class_ids[i])
        min_score = 1.
        vec_a = tensor_descriptors[i]

        result_mx = torch.matmul(tensor_descriptors, vec_a)  
        result_mx = torch.nan_to_num(result_mx)
        
        for j in range(rows):
            b_filename = str(q_image_ids[j])
            b_order_frame = int(q_image_ids[j].split('_')[-1])
            b_classIDx = int(q_class_ids[j])   
            if image_name_a != b_filename and classIDx == b_classIDx and order_frame == b_order_frame:               
                sm_score = result_mx[j].cpu().numpy()
                if sm_score < min_score:
                    min_score = sm_score
        
        case = [q_image_ids[i], q_class_ids[i], min_score]
       
        data.append(case)
    
    return data
       

if __name__ == '__main__':
    print("[INFO] starting .........")
    since = time.time()
    data = []

    p = Pool(NUM_THREADS)
    results = p.map(target, splits)
    p.close()
    p.join()

    for i in range(len(results)):
        for j in range(len(results[i])):
            case = {
                'image_name': results[i][j][0],
                'classIDx': results[i][j][1],
                'intra_score': results[i][j][2]
            }
            data.append(case)

    
    
    # count = 0
    # for i in range(rows):
    #     image_name_a = str(q_image_ids[i])
    #     order_frame = int(q_image_ids[i].split('_')[-1])
    #     classIDx = int(q_class_ids[i])
    #     min_score = 1.        
    #     vec_a = tensor_descriptors[i]

    #     result_mx = torch.matmul(tensor_descriptors, vec_a)  
    #     result_mx = torch.nan_to_num(result_mx)
        
    #     for j in range(rows):
    #         b_filename = str(q_image_ids[j])
    #         b_order_frame = int(q_image_ids[j].split('_')[-1])
    #         b_classIDx = int(q_class_ids[j])   
    #         if image_name_a != b_filename and classIDx == b_classIDx and order_frame == b_order_frame:               
    #             sm_score = result_mx[j].cpu().numpy()
    #             if sm_score < min_score:
    #                 min_score = sm_score
        
    #     case = {
    #         'image_name': q_image_ids[i],                           
    #         'classIDx': q_class_ids[i],
    #         'intra_score': min_score
    #     }
    #     data.append(case)
    #     # print(case)
    #     count += 1
    #     if count > 1000: 
    #         break

    
    
    df = pd.DataFrame(data) 
    df.to_csv(ouput_csv, index=False, header=True)

    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))  
    


