import numpy as np
import streamlit as st
import cv2

# Ignore warnings in output
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Nhận Dạng Chữ Cái Viết Tay",
    page_icon="🔠",
    )

word_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'}
model = load_model(r'NhanDangChuViet/modelHandWritten.h5')

def classify(img):
    img = cv2.GaussianBlur(img, (7, 7), 0)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
    img_resized = cv2.resize(img_thresh, (28, 28))
    img_final = np.reshape(img_resized, (1, 28, 28, 1))
    prediction = model.predict(img_final).flatten()
    top_indices = np.argsort(prediction)[::-1][:3]
    top_results = {word_dict[i]: float(prediction[i]) for i in top_indices}
    return top_results

st.title("📝Nhận diện chữ cái viết tay")
uploaded_file = st.file_uploader("Tải ảnh lên", type=["png", "jpg"])
if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption='Ảnh đã tải lên', use_column_width=True)
    results = classify(image)
    st.write("<h2 style='font-size: 30px;'> 3 chữ cái nhận diện được và phần trăm cao nhất</h2>", unsafe_allow_html=True)
    for char, probability in results.items():
        st.write(f"<h3 style='font-size: 24px;'>{char}: {probability*100:.2f}%</h3>", unsafe_allow_html=True)

