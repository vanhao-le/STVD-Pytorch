import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


root_path = r"data\positive_frame.csv"
df = pd.read_csv(root_path)

data = {}
min = 99999
classIDx = 0
for idx, item in df.iterrows():
    parent_dir = int(item["classIDx"])
    if (parent_dir != 243):
        if (parent_dir in data):
            data[parent_dir] += 1 
        else:
            data[parent_dir] = 1
        

# for key, value in data.items():
#        if value < min:
#             min = value
#             classIDx = key

# print(classIDx, min)

data_sorted = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))

x = np.arange(0, 243)
y = []

for key, value in data_sorted.items():
   if value < min:
        min = value
   y.append(value)

# print(min)
# x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)
plt.title('The mininum frames per a class: 8 - The total frames: 27 K')
plt.show()