import streamlit as st
import numpy as np
import cv2 as cv
import joblib
from PIL import Image
import time

st.set_page_config(
    page_title="Nhận dạng khuôn mặt",
    page_icon="😀",
)

st.subheader('Nhận dạng khuôn mặt')
st.image(Image.open('images/recognition_face.jpg'), width=700)
FRAME_WINDOW = st.image('images/video_notfound.jpg')
cap = cv.VideoCapture(0)

if 'begining' not in st.session_state:
    st.session_state.begining = True

start_btn, stop_btn = st.columns(2)
with start_btn:
    start_press = st.button('Start')
with stop_btn:
    stop_press = st.button('Stop')

if start_press:
    st.session_state.begining = False

if stop_press:
    FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')
    cap.release()


if 'frame_stop' not in st.session_state:
    frame_stop = cv.imread('images/video_notfound.jpg')
    st.session_state.frame_stop = frame_stop

if st.session_state.begining == True:
    cap.release()


svc = joblib.load('NhanDienKhuonMat/svm_model.pkl')
mydict = ['AnhKhoa', 'DangKhoa', 'MinhDuc', 'PhucCong', 'ThanhLoi']

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            #print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


if __name__ == '__main__':
    detector = cv.FaceDetectorYN.create(
        'NhanDienKhuonMat/face_detection_yunet_2023mar.onnx',
        "",
        (320, 320),
        0.9,
        0.3,
        5000)
    
    recognizer = cv.FaceRecognizerSF.create(
    'NhanDienKhuonMat/face_recognition_sface_2021dec.onnx',"")

    tm = cv.TickMeter()

    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    dem = 0
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Inference
        tm.start()
        faces = detector.detect(frame) # faces is a tuple
        print(faces) 
        tm.stop()
       
      
        if faces[1] is not None:
            quantity = len(faces[1])
            for i in range(0, quantity):
                coords = faces[1][i][:-1].astype(np.int32)
                face_align = recognizer.alignCrop(frame, faces[1][i])
                face_feature = recognizer.feature(face_align)
                test_predict = svc.predict(face_feature)
                result = mydict[test_predict[0]]
                cv.putText(frame,result,(coords[0]+10, coords[1]-10),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
           

        # Draw results on the input image
        visualize(frame, faces, tm.getFPS())

        # Visualize results
        FRAME_WINDOW.image(frame, channels='BGR')
    cv.destroyAllWindows()
