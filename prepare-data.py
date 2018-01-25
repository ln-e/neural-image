import h5py
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

data = (
    ((1, 0, 1, 0), '10007_big.png'),
    ((0, 1, 1, 0), '10007_grey.png'),
    ((1, 0, 0, 0), '10007_small.png'),
)

imList = []
for vector, imageName in data:
    image = load_img('data/' + imageName).resize((120, 120))
    image = img_to_array(image)
    imList.append(image)


imTensor = np.array(imList).astype('uint8')
with h5py.File('dataset.h5', 'w') as hf:
    hf.create_dataset('dataset',  data=imTensor)

print(imTensor.shape)
