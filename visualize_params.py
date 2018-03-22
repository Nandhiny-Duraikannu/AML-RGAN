import pickle
import data_utils
import numpy as np
import matplotlib.pyplot as plt

activity2id = {"Walking": 0, "Jogging": 1, "Stairs": 2, "Sitting": 3, "Standing": 4, "LyingDown": 5}

with open('D:\Documents\AML\RGAN\synthetic_wisdm_datasets\samples_wisdm_synthetic_dataset_r2_1000.pk', 'rb') as f:
    gen_samples = pickle.load(f)

with open('D:\Documents\AML\RGAN\synthetic_wisdm_datasets\labels_wisdm_synthetic_dataset_r2_1000.pk', 'rb') as f:
    gen_labels = pickle.load(f)

print(gen_samples.shape)
print(gen_labels.shape)
print(gen_labels)

data_path = './experiments/data/' + 'wisdm' + '.data.npy'
samples, pdf, labels = data_utils.get_data('load', data_path)

print(samples['test'].shape)
print(labels['test'].shape)
print(labels['test'])

activity_type = activity2id['Standing']

sample_id = np.where(labels['vali']==activity_type)[0][0]
gen_id = np.where(gen_labels==activity_type)[0][0]

sample1 = samples['vali'][sample_id]
gen1 = gen_samples[gen_id]

plt.figure(1)
plt.subplot(321)
plt.plot(range(sample1[:, 0].shape[0]), sample1[:, 0], 'b--')
plt.subplot(323)
plt.plot(range(sample1[:, 1].shape[0]), sample1[:, 1], 'b--')
plt.subplot(325)
plt.plot(range(sample1[:, 2].shape[0]), sample1[:, 2], 'b--')

plt.subplot(322)
plt.plot(range(gen1[:,0].shape[0]), gen1[:, 0], 'r--')
plt.subplot(324)
plt.plot(range(gen1[:,1].shape[0]), gen1[:, 1], 'r--')
plt.subplot(326)
plt.plot(range(gen1[:,2].shape[0]), gen1[:, 2], 'r--')
plt.show()

