import cv2
import mediapipe as mp
import math
import numpy as np
def get_angle  (y,x):
    # print(y,x)
    angle1 = math.atan2(y, x)
    # print(angle1)
    angle1 = math.degrees(angle1)
    # print(angle1)
    return angle1


def rotate_bound(image, angle): #图像旋转
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH),borderValue=(255, 255, 255))
    image = cv2.flip(image,-1)
    return image
def add_glasses(face_data,frame,glass):
    frame_size = frame.shape
    # 保存双眼中间点，以用于放置墨镜
    eyes_x = int((face_data.relative_keypoints[0].x + face_data.relative_keypoints[1].x) * frame_size[1] // 2)
    eyes_y = int((face_data.relative_keypoints[0].y + face_data.relative_keypoints[1].y) * frame_size[0] // 2)
    eyes_disdance_x = (face_data.relative_keypoints[0].x - face_data.relative_keypoints[1].x) * frame_size[1]
    eyes_disdance_y = (face_data.relative_keypoints[0].y - face_data.relative_keypoints[1].y) * frame_size[0]
    eyes_disdance = math.sqrt((eyes_disdance_x) ** 2 + (eyes_disdance_y) ** 2)
    angle = np.rad2deg(np.arctan2(eyes_disdance_y,eyes_disdance_x))
    x_size = int(abs(face_data.relative_keypoints[1].x - face_data.relative_keypoints[0].x + 95))
    glass_size = (int(x_size * 2), x_size, 3)

    glass = cv2.resize(glass, (glass_size[0], glass_size[1]))
    glass = rotate_bound(glass, angle)

    glass_size = (glass.shape[1]//2 , glass.shape[0]//2 , 3)
    for x, i in enumerate(range(eyes_x - glass_size[0], eyes_x + glass_size[0])):
        for y, j in enumerate(range(eyes_y - glass_size[1], eyes_y + glass_size[1])):
            try:
                t = list(glass[y, x, :])
                if t != [255, 255, 255] and j > 0 and i > 0:
                    frame[j, i, :] = glass[y, x, :]
            except:
                continue
    return  frame
    return  frame
def run (frame):
  mp_face_detection = mp.solutions.face_detection
  mp_drawing = mp.solutions.drawing_utils
  # For webcam input:
  with mp_face_detection.FaceDetection(
      model_selection=0, min_detection_confidence=0.5) as face_detection:
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      frame.flags.writeable = False
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = face_detection.process(frame)
      # Draw the face detection annotations on the image.
      frame.flags.writeable = True
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      if results.detections:
        for detection in results.detections:
          face_data = detection.location_data
            #特征点验证
          # for i in range(2):
          #   print(f'{mp_face_detection.FaceKeyPoint(i).value}')
          #   print(f'{mp_face_detection.FaceKeyPoint(i).name}:')
          #   print(f'{face_data.relative_keypoints[mp_face_detection.FaceKeyPoint(i).value]}')
          glass = cv2.imread('./data/image/glasses.jpg')
          add_glasses(face_data,frame,glass)
          # print("坐标为")
          # print(eyes_x,eyes_y)
          # print(f"x:{face_data.relative_keypoints[0].x*frame.shape[0], face_data.relative_keypoints[1].x*frame.shape[0]}")
          # print(f"y:{face_data.relative_keypoints[0].y*frame.shape[1], face_data.relative_keypoints[1].y*frame.shape[1]}")
          # print(eyes_x - glass_size[0],eyes_y - glass_size[1],eyes_x + glass_size[0],eyes_y + glass_size[1])


      return frame


if __name__ == '__main__':
    glass = cv2.imread('data/image/glasses.jpg')
    print(glass)
    glasst = rotate_bound(glass,45)
    glasse = rotate_bound(glass,-45)
    while (1):
        cv2.imshow('start', glass)
        cv2.imshow('shuchu', glasst)
        cv2.imshow('glasse', glasse)
        if cv2.waitKey(1) & 0xFF == 27:
            break
