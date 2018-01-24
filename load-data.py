import pickle
import imageio

with open('data.pkl', 'rb') as f:
    imTensor = pickle.load(f)

i = 0
for image in imTensor:
    i = i + 1
    imageio.imwrite('data_out/' + str(i) + '.png', image, 'png')

print(imTensor.shape)
