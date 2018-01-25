import imageio
from keras.preprocessing.image import ImageDataGenerator
import h5py

with h5py.File('dataset.h5', 'r') as hf:
    imTensor = hf['dataset'][:]

i = 0
for image in imTensor:
    i = i + 1
    imageio.imwrite('data_out/' + str(i) + '.png', image, 'png')


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


# пример того как будет работать аугументация
i = 0
for batch in datagen.flow(imTensor,
                          batch_size=1,
                          save_to_dir='preview',
                          save_prefix='example_',
                          save_format='jpeg'):
    i += 1
    if i > 20:
        break

print(imTensor.shape)
