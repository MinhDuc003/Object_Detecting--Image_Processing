import streamlit as st
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import model_from_json, Sequential
import numpy as np
import random
import cv2
from PIL import Image

st.title("0️⃣Nhận dạng chữ số MNIST1️⃣")

def tao_anh_ngau_nhien():
    image = np.zeros((10 * 28, 10 * 28), np.uint8)
    data = np.zeros((100, 28, 28, 1), np.uint8)

    for i in range(0, 100):
        n = random.randint(0, 9999)
        sample = st.session_state.X_test[n]
        data[i] = st.session_state.X_test[n]
        x = i // 10
        y = i % 10
        image[x * 28:(x + 1) * 28, y * 28:(y + 1) * 28] = sample[:, :, 0]
    return image, data

def load_model():
    model_architecture = r'NhanDangChuSo/digit_config.json'
    model_weights = r'NhanDangChuSo/digit_weight.h5'
    with open(model_architecture, 'r') as json_file:
        model = model_from_json(json_file.read(), custom_objects={"Sequential": Sequential})
    model.load_weights(model_weights)

    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

if 'is_load' not in st.session_state:
    model = load_model()
    st.session_state.model = model

    # Load data
    (_, _), (X_test, y_test) = datasets.mnist.load_data()
    X_test = X_test.reshape((10000, 28, 28, 1))
    st.session_state.X_test = X_test

if st.button('Tạo ảnh'):
    image, data = tao_anh_ngau_nhien()
    st.session_state.image = image
    st.session_state.data = data
    st.write('Đã tạo ảnh!')

if 'image' in st.session_state:
    image = st.session_state.image
    st.image(image, caption='Ảnh tạo ngẫu nhiên')

    if st.button('Nhận dạng'):
        data = st.session_state.data
        data = data / 255.0
        data = data.astype('float32')
        data = data.reshape((data.shape[0], -1))  # Reshape to (100, 784)
        ket_qua = st.session_state.model.predict(data)
        dem = 0
        s = ''
        for x in ket_qua:
            s = s + '%d ' % (np.argmax(x))
            dem = dem + 1
            if (dem % 10 == 0) and (dem < 100):
                s = s + '\n'
        st.text(s)
        st.write('Đã nhận dạng xong!')
