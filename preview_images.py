import imageio
from fabrikant_dataset import get_dataset, get_datagen

(train_image, train_vector), (test_image, test_vector) = get_dataset()

# i = 0
# for image in train_image:
#     i = i + 1
#     imageio.imwrite('data_out/' + str(i) + '.png', image, 'png')


datagen = get_datagen()


# пример того как будет работать аугументация
i = 0
for batch in datagen.flow(train_image,
                          y=train_vector,
                          batch_size=1,
                          save_to_dir='preview',
                          save_prefix='example_',
                          save_format='jpeg'):
    i += 1
    if i > 50:
        break

print(train_image.shape, train_vector.shape)
print('done')
