import numpy as np
import torch
import time
import pandas as pd


QUERY_DESC = r'output\pos_test_descriptor.npz'
query = np.load(QUERY_DESC)
rows = len(query['image_ids'])
q_image_ids = query['image_ids']
q_class_ids = query['class_ids']
q_descriptors = query['descriptors']


REF_DESC = r'output\neg_test_descriptor.npz'
reference = np.load(REF_DESC)
cols = len(reference['image_ids'])
r_image_ids = reference['image_ids']
r_class_ids = reference['class_ids']
r_descriptors = reference['descriptors']


device = "cuda:0"

tensor_r_descriptors = torch.from_numpy(r_descriptors)
tensor_r_descriptors = tensor_r_descriptors.to(device)

ouput_csv = r"output\test_neg_inter_matching.csv"

if __name__ == '__main__':
    print("[INFO] starting .........")
    since = time.time()
    data = []

    for i in range(rows):    
        tensor1 = torch.from_numpy(q_descriptors[i]).to(device) 
        result_mx = torch.matmul(tensor_r_descriptors, tensor1)  
        result_mx = torch.nan_to_num(result_mx)
        # print(tensor1.shape, tensor_r_descriptors.shape, result_mx.shape)    
        # Get the maximum along dim = 0 (axis = 0)    
        max_indx = torch.argmax(result_mx, dim=0)
        max_value = result_mx[max_indx]
        max_value = max_value.cpu().numpy()
        
        case = {
            'image_name': q_image_ids[i],
            'classIDx': q_class_ids[i],           
            'inter_score': max_value
        }
        data.append(case)

        # # print(max_value, max_indx)
        # if i > 10:
        #     break
    
    
    df = pd.DataFrame(data) 
    df.to_csv(ouput_csv, index=False, header=True)

    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))  
    

