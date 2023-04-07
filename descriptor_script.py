import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from pathlib import Path, PurePath
import io
import pandas as pd


OUTPUT_PATH = Path("output")
OUTPUT_FILE = OUTPUT_PATH / "pos_train_descriptor.npz"
DATA_PATH = r"data"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# DEVICE = torch.device("cpu")

def generate_query_descriptors(image_ids):  

    image_ids = []
    class_ids = []
    descriptors = []

    # for image_id in image_ids:
    for image_id in range(5000):
        
        class_id = np.random.randint(242, size=1).squeeze()       
        embedding = torch.rand(2048)
        # embedding = embedding / (embedding.pow(2).sum(0, keepdim=True).sqrt())
        embedding = embedding.numpy().squeeze()
        # print(embedding)

       
        image_ids.append(image_id)        
        class_ids.append(class_id)
        descriptors.append(embedding)

   
    # print(descriptors)
    # image_ids = np.concatenate(image_ids)
    # class_ids = np.concatenate(class_ids).astype(np.int32)
    # descriptors = np.concatenate(descriptors).astype(np.float32)
    # print(descriptors)
    return image_ids, class_ids, descriptors

def main():

    
    # Loading subset of query images    
    csv_path = os.path.join(DATA_PATH, "positive_metadata.csv")
    # print(csv_path)
    query_subset = pd.read_csv(csv_path)
    query_image_ids = query_subset.classIDx.values.astype("U")

    # Generation of query descriptors happens here
    image_ids, class_ids, descriptors = generate_query_descriptors(
        query_image_ids
    )


    np.savez(
        OUTPUT_FILE,
        image_ids=image_ids,
        class_ids=class_ids,
        descriptors=descriptors,
    )

    # data = np.load(OUTPUT_FILE)

    # image_ids = data['image_ids']
    # class_ids = data['class_ids']
    # descriptors = data['descriptors']

    # num_record = len(image_ids)
    
    # # for i in range(num_record):
    # #     print(image_ids[i], class_ids[i], descriptors[i].shape)
   
    # print(num_record)
    return



if __name__ == '__main__':
    main()
    