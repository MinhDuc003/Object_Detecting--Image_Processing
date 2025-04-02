import argparse

import numpy as np
import cv2 as cv

score_threshold = 0.9
nms_threshold = 0.3
top_k = 5000

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
    # Download at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet
    detector = cv.FaceDetectorYN.create(
        "./face_detection_yunet_2023mar.onnx",
        "",
        (320, 320),
        score_threshold,
        nms_threshold,
        top_k
    )
    
    #Download at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface
    recognizer = cv.FaceRecognizerSF.create(
    "./face_recognition_sface_2021dec.onnx","")

    tm = cv.TickMeter()

    cap = cv.VideoCapture(0)
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
        tm.stop()
        
        key = cv.waitKey(1)
        if key == 27:
            break
        #print(faces)
        if key == ord('s') or key == ord('S'):
            if faces[1] is not None:
                face_align = recognizer.alignCrop(frame, faces[1][0])
                file_name = './image/ThanhLoi/ThanhLoi_%04d.bmp' % dem
                cv.imwrite(file_name, face_align)
                dem = dem + 1
        # Draw results on the input image
        visualize(frame, faces, tm.getFPS())

        # Visualize results
        cv.imshow('Live', frame)
    cv.destroyAllWindows()
