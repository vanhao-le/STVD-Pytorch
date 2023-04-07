import pandas as pd


INPUT_FILE = r'output\train_matching.csv'

OUTPUT_FILE = r'output\sorted_train_matching.csv'

data = []

df = pd.read_csv(INPUT_FILE)

for idx, item in df.iterrows():
    str_lst = str(item['image_name']).split('_')
    frame_idxs = int(str_lst[-1])
    x = str_lst[:-1]
    video_name = '_'.join(x)
    classIDx = int(item['classIDx'])
    score = round(float(item['intra_score']) - max(float(item['inter_neg_score']), float(item['inter_pos_score'])), 5)
    case = {
        'classIDx': classIDx,
        'video_name': video_name,
        'image_name': item['image_name'],
        'frame_idx': frame_idxs,        
        'inter_neg_score': item['inter_neg_score'],
        'inter_pos_score': item['inter_pos_score'],
        'intra_score': item['intra_score'],
        'score': score,
    }
    data.append(case)

df_out = pd.DataFrame(data)

df_out.sort_values(['classIDx', 'video_name', 'frame_idx'], ascending=[True, True, True], inplace=True)

df_out.to_csv(OUTPUT_FILE, index=False, header=True)