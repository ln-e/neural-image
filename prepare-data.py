import numpy as np
import imageio
from scipy import ndimage, misc
import pickle

data = (
    ((1, 0, 1, 0), '10007_big.png'),
    ((0, 1, 1, 0), '10007_grey.png'),
    ((1, 0, 0, 0), '10007_small.png'),
)

imList = []
for vector, imageName in data:
    image = ndimage.imread('data/' + imageName, mode='RGB')
    image = misc.imresize(image, (150, 150))
    # imageio.imwrite('data_out/' + imageName, image, 'png')
    imList.append(image)


imTensor = np.array(imList).astype('uint8')
with open('data.pkl', 'wb') as f:
    pickle.dump(imTensor, f)
print(imTensor.shape)
