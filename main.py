import os
import streamlit as st
import pickle
import cv2
import numpy as np
from skimage.feature import graycomatrix
from skimage.util import img_as_ubyte
from PIL import Image

# import io

# st.set_option('deprecation.showfileUploaderEncoding', False)


# @st.cache_data
# def load_train_data():
#     x = pickle.load(open('finalized_train_data.pkl', 'rb'))
#     return x

@st.cache_data
def load_train_data():
    # Define the directory where the segmented files are stored
    # current_directory = os.getcwd()
    # directory = os.path.join(current_directory, 'Model_Files')
    #
    # # Initialize an empty list to store the data
    # x = []
    #
    # # Loop over the segmented files
    # for i in range(1, 31):  # You mentioned you have 31 files
    #     # Create the file path
    #     # file_path = os.path.join(directory, 'final_model' + str(i))
    #     file_path = directory + '/X_train_split_' + str(i) + '.pkl'
    #
    #     # Load the data from the file and append it to the list
    #     with open(file_path, 'rb') as file:
    #         x.append(pickle.load(file))

    X_train_loaded = []

    # Load each part and append it to the list
    for i in range(1, 31):  # You mentioned you have 30 files
        with open(f'C:\\Users\\toyes\\PycharmProjects\\imagePre\\Model_Files\\X_train_split_{i}.pkl', 'rb') as f:
            X_train_loaded.append(pickle.load(f))

    # Concatenate the loaded parts to get the original data
    X_train = np.concatenate(X_train_loaded)

    print(type(X_train))

    return X_train


@st.cache_data
def out_weight():
    o = pickle.load(open('finalized_model.pkl', 'rb'))
    return o


x_train = load_train_data()
out_weight = out_weight()

st.write("""
# ToMV Leaf Classification
""")

uploaded_file = st.file_uploader("Drag and Drop File here", type=["jpg", "png", "jpeg", "JPG"])


def extract_texture_features(img):
    hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])

    # Normalize the histograms
    hist_r = cv2.normalize(hist_r, hist_r)
    hist_g = cv2.normalize(hist_g, hist_g)
    hist_b = cv2.normalize(hist_b, hist_b)

    # Flatten the histograms to create feature vectors
    features_hist_r = hist_r.flatten()
    features_hist_g = hist_g.flatten()
    features_hist_b = hist_b.flatten()

    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    g_img = cv2.resize(g_img, (256, 256))

    g_img = img_as_ubyte(g_img)

    distances = [1, 3, 5, 7]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4]

    glcm = graycomatrix(g_img, distances, angles, symmetric=False, normed=True)

    glcm = np.mean(glcm, axis=(2, 3))

    features_glcm = glcm.flatten()

    features = np.concatenate((features_glcm, features_hist_r, features_hist_g, features_hist_b))

    return features


def kernel_matrix(Xtrain, kernel_type, kernel_pars, Xt=None):
    nb_data = Xtrain.shape[0]

    if kernel_type == 'RBF_kernel':
        if Xt is None:
            XXh = np.sum(Xtrain ** 2, axis=1).reshape(-1, 1) * np.ones((1, nb_data))
            omega = XXh + XXh.T - 2 * (Xtrain @ Xtrain.T)
            omega = np.exp(-omega / kernel_pars[0])
        else:
            XXh1 = np.sum(Xtrain ** 2, axis=1).reshape(-1, 1) * np.ones((1, Xt.shape[0]))
            XXh2 = np.sum(Xt ** 2, axis=1).reshape(-1, 1) * np.ones((1, nb_data))
            print(str(Xtrain.shape) + ", " + str(Xt.T.shape))
            omega = XXh1 + XXh2.T - 2 * (Xtrain @ Xt.T)
            omega = np.exp(-omega / kernel_pars[0])

    elif kernel_type == 'lin_kernel':
        if Xt is None:
            omega = Xtrain @ Xtrain.T
        else:
            omega = Xtrain @ Xt.T

    elif kernel_type == 'poly_kernel':
        if Xt is None:
            omega = (Xtrain @ Xtrain.T + kernel_pars[0]) ** kernel_pars[1]
        else:
            omega = (Xtrain @ Xt.T + kernel_pars[0]) ** kernel_pars[1]

    elif kernel_type == 'fpp':
        if Xt is None:
            omega = np.sign(Xtrain @ Xtrain.T + kernel_pars[0]) * (
                    (np.abs(Xtrain @ Xtrain.T + kernel_pars[0])) ** kernel_pars[1])
        else:
            omega = np.sign(Xtrain @ Xt.T + kernel_pars[0]) * (
                    (np.abs(Xtrain @ Xt.T + kernel_pars[0])) ** kernel_pars[1])

    elif kernel_type == 'wav_kernel':
        if Xt is None:
            XXh = np.sum(Xtrain ** 2, axis=1).reshape(-1, 1) * np.ones((1, nb_data))
            omega = XXh + XXh.T - 2 * (Xtrain @ Xtrain.T)

            XXh1 = np.sum(Xtrain, axis=1).reshape(-1, 1) * np.ones((1, nb_data))
            omega1 = XXh1 - XXh1.T
            omega = np.cos(kernel_pars[2] * omega1 / kernel_pars[1]) * np.exp(-omega / kernel_pars[0])
        else:
            XXh1 = np.sum(Xtrain ** 2, axis=1).reshape(-1, 1) * np.ones((1, Xt.shape[0]))
            XXh2 = np.sum(Xt ** 2, axis=1).reshape(-1, 1) * np.ones((1, nb_data))
            omega = XXh1 + XXh2.T - 2 * (Xtrain @ Xt.T)

            XXh11 = np.sum(Xtrain, axis=1).reshape(-1, 1) * np.ones((1, Xt.shape[0]))
            XXh22 = np.sum(Xt, axis=1).reshape(-1, 1) * np.ones((1, nb_data))
            omega1 = XXh11 - XXh22.T

            omega = np.cos(kernel_pars[2] * omega1 / kernel_pars[1]) * np.exp(-omega / kernel_pars[0])
    else:
        raise ValueError(f"Unexpected kernel_type: {kernel_type}")

    return omega


def predict_health_status(image, x_train, out_weight, kernel_type, kernel_para):
    texture_features = extract_texture_features(image)

    X_train = x_train

    texture_features = texture_features.reshape(-1, 1)

    Omega_test = kernel_matrix(X_train.T, kernel_type, kernel_para, texture_features.T)
    TY = ((Omega_test.T) @ out_weight).T
    print(f"The value of TY is: {TY}")
    # The predicted label is the sign of the output
    predicted_label = np.sign(TY)

    return predicted_label


classify_button = st.button("Get Classification")
classified = False

if uploaded_file is None:

    st.text("Please upload an image file.")

else:

    uploaded_image = Image.open(uploaded_file)
    uploaded_image = np.array(uploaded_image)
    g_img_out = cv2.resize(uploaded_image, (256, 256))
    st.image(g_img_out, caption="Uploaded Image", use_column_width=True)

    if classify_button:
        predicted_label = predict_health_status(uploaded_image, x_train, out_weight, 'RBF_kernel', [0.5])
        # cv2_image = cv2.imread(image)
        final_class = ""
        if (predicted_label > 0):
            final_class = "Healthy"
        else:
            final_class = "Diseased"
        classified = True

        if classified:
            string_out = f"The predicted label for the image is: {final_class}"
            st.success(string_out)
    else:
        st.text("Click 'Get Classification' to analyze the image.")