import os
import pickle
import numpy as np
from numpy.linalg import norm
from tensorflow import keras
from keras import Model, Sequential
from keras.preprocessing import image
from keras.applications.resnet_v2 import preprocess_input
from keras.applications.vgg16 import preprocess_input
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tqdm import notebook


def extract_features(img_path, model):
    input_shape = (256, 256, 3)
    img = image.load_img(img_path, target_size=(
        input_shape[0], input_shape[1]))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)

    return normalized_features


extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']


def get_file_list(root_dir):
    file_list = []
    counter = 1
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
                counter += 1
    return file_list


if __name__ == '__main__':

    img_size = 256
    model = keras.models.load_model('saved_models/Train_Data/ResNet50V2Final.ep031.h5')
    model = Sequential(Model(inputs=model.get_layer('resnet50v2').input, outputs=model.get_layer('resnet50v2').output))

    model.summary()
    batch_size = 64
    root_dir = 'Test_Data'

    img_path = 'blouse.jpg'

    features = extract_features(img_path, model)
    print(len(features))

    filenames = sorted(get_file_list(root_dir))

    feature_list = []
    for i in notebook.tqdm(range(len(filenames))):
        print(i)
        feature_list.append(extract_features(filenames[i], model))

    num_feature_dimensions = 100
    pca = PCA(n_components=num_feature_dimensions)
    pca.fit(feature_list)
    feature_list_compressed = pca.transform(feature_list)

    neighbors = NearestNeighbors(n_neighbors=5,
                                 metric='euclidean',
                                 algorithm='brute').fit(feature_list_compressed)

    features_compressed = pca.transform(features.reshape(1, -1))
    distances, indices = neighbors.kneighbors(features_compressed)

    print(indices.shape)
    print(indices)

    plt.figure(figsize=(15, 10), facecolor='white')
    plt.subplot(3, 5, 3)

    plt.imshow(mpimg.imread(img_path), interpolation='lanczos')
    plt.axis('off')
    plotnumber = 6
    for i in range(len(indices[0])):
        if plotnumber <= len(indices[0]) + 6:
            ax = plt.subplot(3, 5, plotnumber)
            name = filenames[indices[0][i]]
            plt.imshow(mpimg.imread(name), interpolation='lanczos')
            plt.axis('off')
            plotnumber += 1
    plt.tight_layout()
    plt.show()

    # source, destination
    knnPickle = open('knn_models/model7', 'wb')
    pickle.dump(neighbors, knnPickle)
    pickle.dump(feature_list, open('data/features-outfits-resnet.pickle', 'wb'))
    pickle.dump(filenames, open('data/filenames-outfits.pickle', 'wb'))
    plt.imshow(mpimg.imread(img_path), interpolation='lanczos')
