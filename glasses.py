import cv2
import mediapipe as mp
import math
#获取角度值
def get_angle  (y,x):
    # print(y,x)
    angle1 = math.atan2(y, x)
    # print(angle1)
    angle1 = math.degrees(angle1)
    # print(angle1)
    return angle1
#两角合的sin值计算
def add_rad (rad_a,rad_b):
    angles = math.sin(rad_a)*math.cos(rad_b)+math.sin(rad_b)*math.cos(rad_a)
    return angles
#添加墨镜
def add_glasses(face_data,frame,glass):
    frame_size = frame.shape
    # 保存双眼中间点，以用于放置墨镜
    eyes_x = int((face_data.relative_keypoints[0].x + face_data.relative_keypoints[1].x) * frame_size[1] // 2)
    eyes_y = int((face_data.relative_keypoints[0].y + face_data.relative_keypoints[1].y) * frame_size[0] // 2)
    eyes_disdance_x = (face_data.relative_keypoints[0].x - face_data.relative_keypoints[1].x) * frame_size[1]
    eyes_disdance_y = (face_data.relative_keypoints[0].y - face_data.relative_keypoints[1].y) * frame_size[0]
    eyes_disdance = math.sqrt((eyes_disdance_x) ** 2 + (eyes_disdance_y) ** 2)
    angle = get_angle(eyes_disdance_y,eyes_disdance_x)
    angle1 = math.atan2(50, eyes_disdance_x/2)
    eyes_angle = add_rad(angle1,angle)

    ts = math.sqrt(50**2 +  eyes_disdance_x/2**2)
    point_x = face_data.relative_keypoints[0].x + math.cos(eyes_angle) * ts
    point_y = face_data.relative_keypoints[0].y + math.sin(eyes_angle) * ts

    # eyes_sin = eyes_disdance_y / eyes_disdance
    # eyes_cos = eyes_disdance_x / eyes_disdance


    x_size = int(abs(face_data.relative_keypoints[1].x - face_data.relative_keypoints[0].x + 95))
    glass_size = (int(x_size * 2), x_size, 3)

    glass = cv2.resize(glass, (glass_size[0], glass_size[1]))
    glass_size = (glass_size[0] // 2, glass_size[1] // 2, 3)
    for x, i in enumerate(range(eyes_x - glass_size[0], eyes_x + glass_size[0])):
        for y, j in enumerate(range(eyes_y - glass_size[1], eyes_y + glass_size[1])):
            t = list(glass[y, x, :])
            try:
                if t != [255, 255, 255] and j > 0 and i > 0:
                    frame[j, i, :] = glass[y, x, :]
            except:
                continue
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
          glass = cv2.imread('glasses.jpg')
          add_glasses(face_data,frame,glass)
          # print("坐标为")
          # print(eyes_x,eyes_y)
          # print(f"x:{face_data.relative_keypoints[0].x*frame.shape[0], face_data.relative_keypoints[1].x*frame.shape[0]}")
          # print(f"y:{face_data.relative_keypoints[0].y*frame.shape[1], face_data.relative_keypoints[1].y*frame.shape[1]}")
          # print(eyes_x - glass_size[0],eyes_y - glass_size[1],eyes_x + glass_size[0],eyes_y + glass_size[1])


      return frame

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    run(cap)