import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing import image

model = load_model('saved_models/keras_fabrikant_trained_model.h5')

sgd = SGD(lr=0.005)
model.compile(
    loss='binary_crossentropy',
    optimizer=sgd,
    metrics=['accuracy']
)

img = image.load_img('use/img.jpg', target_size=(144, 144))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

prediction_result = model.predict(x=x)

print('>>> PREDICTION RESULT', prediction_result)
