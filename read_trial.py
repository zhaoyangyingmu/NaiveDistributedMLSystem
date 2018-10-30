import numpy as np
path = ".//data//student_data.csv"
file = open(path, encoding='utf-8-sig')
first_line_flag = True
tags = []
datas = []
for line in file:
    if first_line_flag:
        first_line_flag = False
        continue
    line = line.strip()
    parts = line.split(",")
    tag = int(parts[0])
    tags.append(tag)
    data = [float(parts[1]), float(parts[2]), float(parts[3])]
    datas.append(data)

file.close()
tags = np.array(tags) * 1.0
tags = tags.reshape(tags.shape[0], 1)
datas = np.array(datas) * 1.0

## 做归一化处理
for i in range(datas.shape[1]):
    column_max = np.max(datas[:, i])
    column_min = np.min(datas[:, i])
    datas[:, i] = datas[:, i] * 1.0 / (column_max - column_min)

np.save(".//data//tags.npy", tags)
np.save(".//data//datas.npy", datas)

tags = np.load(".//data//tags.npy")
datas = np.load(".//data//datas.npy")
m = tags.shape[0]
train_idx = int(m * 0.8)
x_train = datas[:train_idx, :]
y_train = tags[:train_idx, :]
x_test = datas[train_idx:, :]
y_test = tags[train_idx:, :]

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
