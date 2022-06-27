import pickle
from flask import request
from flask import Flask, render_template
from tensorflow import keras
from keras import Model, Sequential
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input

import numpy as np
from numpy.dual import norm
from sklearn.decomposition import PCA
from werkzeug.utils import secure_filename
import os
from os.path import join, dirname, realpath

app = Flask(__name__)

UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'static\\uploads')
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['MAX_CONTENT_PATH'] = 30000
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route('/')
def my_form():
    return render_template('home.html')


@app.route('/', methods=['POST'])
def my_form_post():
    if request.method == "POST":
        image1 = request.files['image']
        image_name = secure_filename(image1.filename)

        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))

        image1.save(os.path.join(app.config['UPLOAD_FOLDER'], image_name))

        model = keras.models.load_model('saved_models/Train_Data/ResNet50V2Final.ep031.h5')
        model = Sequential(Model(inputs=model.get_layer('resnet50v2').input,
                                 outputs=model.get_layer('resnet50v2').output))

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)

        features = extract_features(img_path, model)

        filenames = pickle.load(open('data/filenames-outfits.pickle', 'rb'))
        feature_list = pickle.load(open('data/features-outfits-resnet.pickle', 'rb'))
        num_feature_dimensions = 100
        pca = PCA(n_components=num_feature_dimensions)
        pca.fit(feature_list)

        neighbors = pickle.load(open('knn_models/model7', 'rb'))

        features_compressed = pca.transform(features.reshape(1, -1))
        distances, indices = neighbors.kneighbors(features_compressed)

        images = []

        for index in indices[0]:
            name = filenames[index].split("\\")
            print(name)
            images.append(name[0]+'/'+name[1]+'/'+name[2])

    return render_template('home.html', input=["uploads/"+image_name], images=images, show=True, showImage=image_name)


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


if __name__ == "__app__":
    app.run(port=8088, threaded=False)
