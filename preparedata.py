import json
import h5py
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

data = json.load(open('raw-data.json'))

train_data_list = []
train_label_list = []
test_data_list = []
test_label_list = []
i = 0
for vector, imageName in data:
    image = load_img(
        'data/' + imageName,
        target_size=(176, 176),
        interpolation='bilinear'
    )

    image = img_to_array(image)
    if i % 5 != 0:
        train_data_list.append(image)
        train_label_list.append(vector)
    else:
        test_data_list.append(image)
        test_label_list.append(vector)
    i += 1

train_data = np.array(train_data_list).astype('uint8')
train_label = np.array(train_label_list).astype('uint8')
test_data = np.array(test_data_list).astype('uint8')
test_label = np.array(test_label_list).astype('uint8')
with h5py.File('dataset.h5', 'w') as hf:
    hf.create_dataset('train_data', data=train_data)
    hf.create_dataset('train_label', data=train_label)
    hf.create_dataset('test_data', data=test_data)
    hf.create_dataset('test_label', data=test_label)

print('done')
