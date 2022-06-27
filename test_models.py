import pickle
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import ResNet50, preprocess_input
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow import keras


def similar_images(indices):
    plt.figure(figsize=(15,10), facecolor='white')
    plotnumber = 1
    for index in indices:
        if plotnumber<=len(indices) :
            ax = plt.subplot(2,4,plotnumber)
            plt.imshow(mpimg.imread(filenames[index]), interpolation='lanczos')
            plotnumber+=1
    plt.tight_layout()

img_size = 256
# knnPickle = open('knn_models/model1', 'wb')
# model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3), pooling='max')
model = keras.models.load_model('saved_models/My_Train_Data/ResNet  20v  2_model.005.h5')
loaded_model = pickle.load(open('knn_models/model3', 'rb'))

root_dir = 'Train_Data'

batch_size = 64
img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
datagen = img_gen.flow_from_directory(root_dir,
                                          target_size=(img_size, img_size),
                                          batch_size=batch_size,
                                          class_mode=None,
                                          shuffle=False)
filenames = [root_dir + '/' + s for s in datagen.filenames]


img_path = 'pants.jpg'
input_shape = (img_size, img_size, 3)
img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)

#result = loaded_model.predict(X_test)
test_img_features = model.predict(preprocessed_img, batch_size=512)

_, indices = loaded_model.kneighbors(test_img_features)

plt.imshow(mpimg.imread(img_path), interpolation='lanczos')
plt.xlabel(img_path.split('.')[0] + '_Original Image', fontsize=20)
plt.show()
print('********* Predictions ***********')
similar_images(indices[0])
plt.figure(figsize=(15, 10), facecolor='white')
plotnumber = 1
for index in indices[0]:
    if plotnumber <= len(indices[0]):
        ax = plt.subplot(2, 4, plotnumber)
        name = filenames[index].split('\\')
        plt.imshow(mpimg.imread(name[0] + '/' + name[1]), interpolation='lanczos')
        plotnumber += 1
plt.show()