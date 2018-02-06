import h5py
from keras.preprocessing.image import ImageDataGenerator


def get_dataset():
    with h5py.File('dataset.h5', 'r') as hf:
        train_data = hf['train_data'][:]
        train_label = hf['train_label'][:]
        test_data = hf['test_data'][:]
        test_label = hf['test_label'][:]
    return (train_data, train_label), (test_data, test_label)


def get_datagen():
    return ImageDataGenerator(
        rescale=1. / 255,
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
