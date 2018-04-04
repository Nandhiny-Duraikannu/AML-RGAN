import csv
import numpy as np
import matplotlib.pyplot as plt
import data_utils

wait_time = 60
seq_len = 60
num_windows = 5
activity2id = {"Walking": 0, "Jogging": 1, "Stairs": 2, "Sitting": 3, "Standing": 4, "LyingDown": 5}

with open('D:\Documents\AML\RGAN\experiments\data\WISDM_at_v2.0_raw.txt', 'rt') as csvfile:
    raw = csv.reader(csvfile, delimiter=',')
    raw_data = [row for row in raw]
    raw_data = np.array(raw_data)

# print(raw_data[1])

cur_id = 0
cur_count = 0
num_windows = 1

labels = []
samples = []
pdf = None

for entry in raw_data:
    if len(entry) != 6:
        continue
    # if entry[1] != "Walking" and entry[1] != "Sitting":
    #     continue
    if cur_id != entry[0]:
        cur_id = entry[0]
        labels.append(activity2id[entry[1]])
        samples.append([[], [], []])
        cur_count = 1
        num_windows = 1
    else:
        if num_windows > 5:
            continue
        if cur_count >= seq_len:
            labels.append(activity2id[entry[1]])
            samples.append([[], [], []])
            cur_count = 0
            num_windows += 1
        cur_count += 1
        # if cur_count <= wait_time+1:
        #     continue
    samples[-1][0].append(np.float(entry[3]))
    samples[-1][1].append(np.float(entry[4]))
    samples[-1][2].append(np.float(entry[5][:-1]))

mask = []
vars = []
for data_point in range(len(samples)):
    samples[data_point] = np.array(samples[data_point])

    if len(samples[data_point][0]) < seq_len:
        mask.append(data_point)

    elif labels[data_point] == 4:
        vars.append(np.var(samples[data_point]))
        if np.var(samples[data_point]) > 0.5:
            mask.append(data_point)

# Plot variance
# n, bins, patches = plt.hist(np.array(vars), 20, normed=1, facecolor='green', alpha=0.75)
#
# print(np.mean(vars))
# plt.show()

for i in sorted(mask, reverse=True):
    del samples[i]
    del labels[i]

# TODO: Normalize? Cond_dim? Limit types like just walking/standing
# TODO: Sample data at time intervals instead of just taking the first n data points?
# Average length, incomplete data,

print(labels[0])
print(samples[0])

samples = np.dstack(samples)
samples = np.swapaxes(samples, 0, 2)
samples = samples/20

labels = np.array(labels)

print(samples.shape)
print(labels.shape)

train, vali, test, labels_list = data_utils.split(samples, [0.6, 0.2, 0.2], normalise=False, labels=labels)
train_labels, vali_labels, test_labels = labels_list

labels = dict()
train_labels = np.reshape(train_labels, (train_labels.shape[0], 1))
vali_labels = np.reshape(vali_labels, (vali_labels.shape[0], 1))
test_labels = np.reshape(test_labels, (test_labels.shape[0], 1))

labels['train'], labels['vali'], labels['test'] = train_labels, vali_labels, test_labels

# train, vali, test = data_utils.normalise_data(train, vali, test)

samples = dict()
samples['train'], samples['vali'], samples['test'] = train, vali, test

data_path = './experiments/data/' + 'wisdm' + '.data.npy'
np.save(data_path, {'samples': samples, 'labels': labels, 'pdf':pdf})
