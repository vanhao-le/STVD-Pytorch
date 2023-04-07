import numpy as np
import torch
import time
import pandas as pd
import os
from multiprocessing import Pool


POS_DESC = r'output\pos_train_descriptor.npz'
query = np.load(POS_DESC)
rows = len(query['image_ids'])
q_image_ids = query['image_ids']
q_class_ids = query['class_ids']
q_descriptors = query['descriptors']


device = "cuda:1"

tensor_descriptors = torch.from_numpy(q_descriptors)
tensor_descriptors = tensor_descriptors.to(device)

ouput_csv = r"output\train_pos_inter_matching_pa.csv"

def split(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

NUM_THREADS = 3
frame_lst = np.arange(rows)

splits = list(split(frame_lst, NUM_THREADS))

def target(sub_list):
    data = []
    for i in sub_list:        
        max_score = 0.
        # max_score = torch.tensor(0.)
        vec_a = tensor_descriptors[i]

        result_mx = torch.matmul(tensor_descriptors, vec_a)  
        result_mx = torch.nan_to_num(result_mx)
        
        for j in range(rows):           
            if int(q_class_ids[i]) != int(q_class_ids[j]) and result_mx[j] > max_score:
                max_score = result_mx[j]
        
        case = [q_image_ids[i], q_class_ids[i], max_score]
       
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
                'inter_score': results[i][j][2]
            }
            data.append(case)

    df = pd.DataFrame(data) 
    df.to_csv(ouput_csv, index=False, header=True)

    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))  
    


