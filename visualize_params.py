import pickle
import data_utils
import numpy as np
import matplotlib.pyplot as plt

activity2id = {"Walking": 0, "Jogging": 1, "Stairs": 2, "Sitting": 3, "Standing": 4, "LyingDown": 5}

activity = "LyingDown"

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

activity_type = activity2id[activity]

sample_id = np.where(labels['test']==activity_type)[0][0]
gen_id = np.where(gen_labels==activity_type)[0][0]

sample1 = samples['test'][sample_id]
gen1 = gen_samples[gen_id]

plt.figure(1)
ax1 = plt.subplot(321)
plt.plot(range(sample1[:, 0].shape[0]), sample1[:, 0], 'b--')
ax1.set_title('Sample')
ax1.set_ylabel('x', rotation=0, size='large')

ax2 = plt.subplot(323, sharey=ax1)
plt.plot(range(sample1[:, 1].shape[0]), sample1[:, 1], 'b--')
ax2.set_ylabel('y', rotation=0, size='large')

ax3 = plt.subplot(325, sharey=ax1)
plt.plot(range(sample1[:, 2].shape[0]), sample1[:, 2], 'b--')
ax3.set_ylabel('z', rotation=0, size='large')


ax4 = plt.subplot(322, sharey=ax1)
plt.plot(range(gen1[:,0].shape[0]), gen1[:, 0], 'r--')
ax4.set_title('Generated')
ax4.set_ylabel('x', rotation=0, size='large')

ax5 = plt.subplot(324, sharey=ax1)
plt.plot(range(gen1[:,1].shape[0]), gen1[:, 1], 'r--')
ax5.set_ylabel('y', rotation=0, size='large')

ax6 = plt.subplot(326, sharey=ax1)
plt.plot(range(gen1[:,2].shape[0]), gen1[:, 2], 'r--')
ax6.set_ylabel('z', rotation=0, size='large')

ax1.set_ylim([-1,1])
plt.autoscale(False)
plt.suptitle(activity)

plt.show()

