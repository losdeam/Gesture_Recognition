import json
import  cv2
import mediapipe as mp
import  os
import math
#求解二维向量的角度
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
        angle_ = 360.-angle_
    return angle_

def run
if __name__ == '__main__':
    model = load(0)
    print(model)





