import pandas as pd



INTRA_FILE = r"output\test_intra_matching.csv"
INTER_NEG_FILE = r"output\test_neg_inter_matching.csv"
INTER_POS_FILE = r"output\test_pos_inter_matching.csv"

OUTPUT_FILE = r'output\test_matching.csv'

intra_df = pd.read_csv(INTRA_FILE)

inter_neg_df= pd.read_csv(INTER_NEG_FILE)

inter_pos_df = pd.read_csv(INTER_POS_FILE)


df_merged = pd.merge(pd.merge(inter_neg_df, inter_pos_df, how='inner', on=['image_name', 'classIDx']), intra_df, how='inner', on=['image_name', 'classIDx'])

df_merged.to_csv(OUTPUT_FILE, index=False, header=True)

# data = []
# inter_neg_num = inter_neg_df.to_numpy()

# for idx, item in intra_df.iterrows():
#     # image_name,classIDx,intra_score
#     image_name = 




