import os
import pandas as pd


def main():
    root_path = r"E:\STVD_DL\data\test"
    output_file = r'data\test_negative_metadata.csv'
    data = []
    for r, d, f in os.walk(root_path):
        for file in f:
            if file.endswith(".jpg"):
                parent_dir = str(r).split('\\')[-1]
                if(parent_dir == "243"):
                    case = {
                        'image_name': str(file),
                        'classIDx': parent_dir
                    }
                    data.append(case)

    print("Dataset lenght:", len(data))    
    df = pd.DataFrame(data)
    df.to_csv(output_file, header=True, index=False)
    
if __name__ == '__main__':
    main()
    