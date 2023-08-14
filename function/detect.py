import cv2
import mediapipe as mp
import math
import json
import os

import  function.glass as glass
import  function.painting_style as painting_style
#彩色转灰色
def BGR2GRAY(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#墨镜特效
def glasses(frame):
    return glass.run(frame)

#漫画风格转化
def toCarttonStyle(frame):
    return painting_style.toCarttonStyle(frame)

#计算两向量间夹角
def vector_2d_angle(v1,v2):
    '''
        求解二维向量的角度
    '''
    v1_x=v1[0]
    v1_y=v1[1]
    v2_x=v2[0]
    v2_y=v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ =65535.
    if angle_ > 180.:
        angle_ = 65535.
    return angle_

#手指弯曲程度检测，
def hand_angle(hand_):
    '''
        获取对应手相关向量的二维角度,根据角度确定手势
    '''
    angle_list = []
    #---------------------------- thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    #---------------------------- index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    #---------------------------- middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    #---------------------------- ring 无名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    #---------------------------- pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list

#特效区
def specual_effects(frame,gesture):
    a = ["1","2","3","4","5","6" "three","thumbup","yeah"]
    b = [BGR2GRAY,glasses,toCarttonStyle]
    if gesture:
        if gesture == a[2]:
            frame = b[1](frame)
        elif gesture == a[1]:
            frame = b[0](frame)
        elif gesture == a[0]:
            frame = b[2](frame)
    return frame
#自定义手势特效区
def specual_effects_zdy(frame,gesture):
    a = ["1","2","3","4","5"]
    b = [BGR2GRAY,glasses,toCarttonStyle]
    if gesture:
        if gesture == a[2]:
            frame = b[1](frame)
        elif gesture == a[1]:
            frame = b[0](frame)
        elif gesture == a[0]:
            frame = b[2](frame)
    return frame


#保存数据点至hand_n.json,将第n个动作替换
def save(data,n):
    filename = f'data/json/hand_{n}.json'
    with open(filename,"w") as file_obj:
        json.dump(data,file_obj)

#读取hand_n.json
def load(n):
    filename=f'data/json/hand_{n}.json'
    try:
        with open(filename) as file_obj:
            data = json.load(file_obj)
        return data
    except:
        print(f"{filename}文件不存在")

#采样,Hz为频率，n为替换的姿态
def sampling(n):
    t=[]
    nums = 0
    min_x,min_y = 999 ,999
    max_x,max_y = 0,0
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75)
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame= cv2.flip(frame,1)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks: #如果识别到手
            for hand_landmarks in results.multi_hand_landmarks: #遍历识别到的所有手
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_local = []
                #21个姿态点，转化为适配输出格式的形式
                for i in range(21):
                    x = hand_landmarks.landmark[i].x*frame.shape[1]
                    y = hand_landmarks.landmark[i].y*frame.shape[0]
                    min_x = min(x,min_x)
                    min_y = min(y,min_y)
                    max_x = max(x,max_x)
                    max_y = max(y,max_y)
                    hand_local.append((x,y))
                if hand_local:
                    t.append((hand_local))
                    nums+=1

        # cv2.startWindowThread()  # 加在这个位置
        cv2.imshow('loading,just a moment', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if  nums == 150:
            min_x, min_y,max_x, max_y = int( min_x),int(min_y),int(max_x),int(max_y)
            save(t,n)
            frame = frame[min_y-10:max_y+10,min_x-10:max_x+10]
            frame =cv2.resize(frame, (200, 300))
            cv2.imwrite(f'data/gesture/gesture{n}.jpg', frame)
            break
    cap.release()
    k_means(6)
    # cv2.destroyallwindows()

#距离检测（欧式距离）
def distance(list1,list2):
    distances = 0
    for i,j in zip(list1,list2):
        distances += (i-j) ** 2
    distances = math.sqrt(distances)
    return  distances

#对自定义手势的识别
def classify(angle_list):
    model = load(0) #训练完的数据保存至0
    dis = 999
    t = -1
    # print()
    for index,x in enumerate(model):
        g = distance(x,angle_list)

        if  g < dis  and g <= 100:
            dis =  g
            t = index+1
            # print(t)
    # print()
    return str(t)

#计算聚类中心，以聚类中心与手势的欧氏距离为分类标准
def k_means(n):
    list = os.listdir("data/json")
    data_list = []
    model = [[0 for _ in range(10)]for _ in range(n)]

    for i in range(1,n+1):
        if f'hand_{i}.json' in list :
            data_list.append([load(i),i])
            # print(i)

    for finger_data in data_list:
        # print(1)
        for hand_ in finger_data[0]:
            t = data_get(hand_)
            # print(model[finger_data[1] - 1])
            for i in range(10):
                model[finger_data[1] - 1][i] += t[i]/150
    save(model,0)
    print("训练完成")
    # for i in model:
    #     print(i)

#获取手指弯曲角度，两指间夹角，手掌旋转角度，共十维数据
def data_get(hand_):
    angle_list = []
    #---------------------------- thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    # ---------------------------- index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[6][0])), (int(hand_[0][1]) - int(hand_[6][1]))),
        ((int(hand_[7][0]) - int(hand_[8][0])), (int(hand_[7][1]) - int(hand_[8][1])))
    )
    angle_list.append(angle_)
    # ---------------------------- middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[10][0])), (int(hand_[0][1]) - int(hand_[10][1]))),
        ((int(hand_[11][0]) - int(hand_[12][0])), (int(hand_[11][1]) - int(hand_[12][1])))
    )
    angle_list.append(angle_)
    # ---------------------------- ring 无名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[14][0])), (int(hand_[0][1]) - int(hand_[14][1]))),
        ((int(hand_[15][0]) - int(hand_[16][0])), (int(hand_[15][1]) - int(hand_[16][1])))
    )
    angle_list.append(angle_)
    # ---------------------------- pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[18][0])), (int(hand_[0][1]) - int(hand_[18][1]))),
        ((int(hand_[19][0]) - int(hand_[20][0])), (int(hand_[19][1]) - int(hand_[20][1])))
    )
    angle_list.append(angle_)
    # ----------------------------  大拇指与食指夹角
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[2][0])), (int(hand_[0][1]) - int(hand_[2][1]))),
        ((int(hand_[0][0]) - int(hand_[6][0])), (int(hand_[0][1]) - int(hand_[6][1])))
    )
    angle_list.append(angle_)
    # ----------------------------  食指与中指夹角
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[6][0])), (int(hand_[0][1]) - int(hand_[6][1]))),
        ((int(hand_[0][0]) - int(hand_[10][0])), (int(hand_[0][1]) - int(hand_[10][1])))
    )
    angle_list.append(angle_)
    # ----------------------------  中指与无名指夹角
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[10][0])), (int(hand_[0][1]) - int(hand_[10][1]))),
        ((int(hand_[0][0]) - int(hand_[14][0])), (int(hand_[0][1]) - int(hand_[14][1])))
    )
    angle_list.append(angle_)
    # ----------------------------  无名指与小拇指夹角
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[14][0])), (int(hand_[0][1]) - int(hand_[14][1]))),
        ((int(hand_[0][0]) - int(hand_[18][0])), (int(hand_[0][1]) - int(hand_[18][1])))
    )
    angle_list.append(angle_)
    # ----------------------------  手掌旋转夹角
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[2][0])), (int(hand_[0][1]) - int(hand_[2][1]))),
        ((int(hand_[0][0]) - int(hand_[18][0])), (int(hand_[0][1]) - int(hand_[18][1])))
    )
    angle_list.append(angle_)
    return angle_list

#手势检测，
def h_gesture(angle_list):
    '''
        # 二维约束的方法定义手势
        # fist five gun love one six three thumbup yeah
    '''
    thr_angle = 90.      #其余手指弯曲状态角度
    thr_angle_thumb = 45.# 大拇指弯曲角度分界线
    thr_angle_s = 49.    #其余手指伸直状态角度
    gesture_str = None
    # print(angle_list)
    if 65535. not in angle_list:
        if (angle_list[0]>thr_angle_thumb) and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "fist"
        elif (angle_list[0]<thr_angle_s) and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]<thr_angle_s):
            gesture_str = "five"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "gun"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]<thr_angle_s):
            gesture_str = "love"
        elif (angle_list[0]>5)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "one"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]<thr_angle_s):
            gesture_str = "six"
        elif (angle_list[0]>thr_angle_thumb)  and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]>thr_angle):
            gesture_str = "three"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "thumbUp"
        elif (angle_list[0]>thr_angle_thumb)  and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "two"
    return gesture_str

#识别程序主体
def detect_zdy():

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75)
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame= cv2.flip(frame,1)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks: #如果识别到手
            for hand_landmarks in results.multi_hand_landmarks: #遍历识别到的所有手
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_local = []
                #21个姿态点，转化为适配输出格式的形式
                for i in range(21):
                    x = hand_landmarks.landmark[i].x*frame.shape[1]
                    y = hand_landmarks.landmark[i].y*frame.shape[0]
                    hand_local.append((x,y))
                if hand_local:
                    angle_list = data_get(hand_local) #手指角度获取
                    gesture_str = classify(angle_list) #手势判断

                    cv2.putText(frame,gesture_str,(0,100),0,1.3,(0,0,0),3) #输出文字

                    # frame = specual_effects(frame,gesture_str)
        cv2.imshow('camera', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    # cv2.destroyallwindows()

if __name__ == '__main__':
    sampling(1)


