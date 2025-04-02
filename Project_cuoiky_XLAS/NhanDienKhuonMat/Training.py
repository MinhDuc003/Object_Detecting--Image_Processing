#%% Nạp thư viện
import numpy as np
import os.path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

#%% Lớp chỉ định dữ liệu nhận dạng
class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 
    
#%% Hàm Load tất cả IdentityMetadata
def load_metadata(path):
    metadata = []
    for i in sorted(os.listdir(path)):
        for f in sorted(os.listdir(os.path.join(path, i))):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg' or ext == '.bmp':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

#%% Tạo detector với model yunet 2023
score_threshold = 0.9
nms_threshold = 0.3
top_k = 5000
# Download at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet
detector = cv2.FaceDetectorYN.create(
    "./face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    score_threshold,
    nms_threshold,
    top_k
)

#%% Tạo recognizer nhận dạng khuôn mặt với model sface 2021
#Download at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface
recognizer = cv2.FaceRecognizerSF.create(
            "./face_recognition_sface_2021dec.onnx","")


#%% Load, tiền xử lý hình ảnh và chia dữ liệu
metadata = load_metadata('./image')

embedded = np.zeros((metadata.shape[0], 128))

for i, m in enumerate(metadata):
    print(m.image_path())
    img = cv2.imread(m.image_path(), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error reading image at {m.image_path()}")
        continue  
    face_feature = recognizer.feature(img)
    embedded[i] = face_feature

targets = np.array([m.name for m in metadata])

encoder = LabelEncoder()
encoder.fit(targets)

# Numerical encoding of identities
y = encoder.transform(targets)

train_idx = np.arange(metadata.shape[0]) % 5 != 0
test_idx = np.arange(metadata.shape[0]) % 5 == 0
X_train = embedded[train_idx]
X_test = embedded[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

#%% Train model và lưu model
svc = LinearSVC()                                                                           
svc.fit(X_train, y_train)
acc_svc = accuracy_score(y_test, svc.predict(X_test))
print('SVM accuracy: %.6f' % acc_svc)
joblib.dump(svc,'svm_model.pkl')

# %%
