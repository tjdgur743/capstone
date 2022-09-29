import cv2
import numpy as np

class Leaving_case():
    def __init__(self, _arduino, _parking_spaces, _parked_list):
        self.parking_spaces=_parking_spaces
        self.parked_list=_parked_list
        self.arduino=_arduino
        self.head_lights=np.array([[0 for i in range(4)] for i in range(5)]) # xxyy
        # 여러 프레임의 정보
        PREV_FRAMES=22
        self.prev_brightnesses=[[999 for i in range(PREV_FRAMES)] for i in range(5)]
        self.prev_shapes=[[] for i in range(5)]

        if self.arduino: # x_width는 60 고정
            self.head_lights[0] = np.array([_parking_spaces[0][0]-10, _parking_spaces[0][1]+40, _parking_spaces[0][2]+20, _parking_spaces[0][3]+20])
            self.head_lights[1] = np.array([_parking_spaces[1][0]-35, _parking_spaces[1][1]+15, _parking_spaces[1][2]+20, _parking_spaces[1][3]+20])
            self.head_lights[2] = np.array([_parking_spaces[2][0]-30, _parking_spaces[2][1]+30, _parking_spaces[2][2]+30, _parking_spaces[2][3]+30])
            self.head_lights[3] = np.array([_parking_spaces[3][0]-30, _parking_spaces[3][1]+30, _parking_spaces[3][2]+30, _parking_spaces[3][3]+30])
            if len(_parking_spaces)==5:
                self.head_lights[4] = np.array([_parking_spaces[4][0]-10, _parking_spaces[4][1]+50, _parking_spaces[4][2]+50, _parking_spaces[4][3]+30])
        else: # 영상
            self.head_lights[0] = np.array([_parking_spaces[0][0]-80, _parking_spaces[0][1]+30, _parking_spaces[0][2]+40, _parking_spaces[0][3]+10])
            self.head_lights[1] = np.array([_parking_spaces[1][0]-80, _parking_spaces[1][1]+20, _parking_spaces[1][2]+40, _parking_spaces[1][3]+20])
            self.head_lights[2] = np.array([_parking_spaces[2][0]-40, _parking_spaces[2][1]+50, _parking_spaces[2][2]+40, _parking_spaces[2][3]+20])
            self.head_lights[3] = np.array([_parking_spaces[3][0]-40, _parking_spaces[3][1]+70, _parking_spaces[3][2]+50, _parking_spaces[3][3]])
            if len(_parking_spaces)==5:
                self.head_lights[4]=np.array((_parking_spaces[4][0]-40, _parking_spaces[4][1]+60, _parking_spaces[4][2]+40, _parking_spaces[4][3]+30))
        for i in range(len(self.head_lights)):
            empty_headlight_area=np.zeros((self.head_lights[i][1]-self.head_lights[i][0]) * (self.head_lights[i][3]-self.head_lights[i][2]))
            self.prev_shapes[i]=[empty_headlight_area for i in range(PREV_FRAMES)]

    def cos_similarity(self, A, B):
        A=np.where(A==0,0,2)
        B=np.where(B==0,0,2)
        return np.dot(A, B) / ((A**2).sum()**0.5 * (B**2).sum()**0.5)

    def judge_brightness(self, img, i):
        # 여러 프레임 전과 비교
        output=False
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness=img.mean()
        d_brightness=brightness-self.prev_brightnesses[i][0]
        img=cv2.GaussianBlur(img, (5,5), 0)
        shape=cv2.Canny(img,100,100).flatten()
        similarity=len(np.where(shape==self.prev_shapes[i][0])[0])/len(shape)
        #similarity=self.cos_similarity(shape, self.prev_shapes[i][0])
        if d_brightness>10 and similarity>0.9:
            print(i, 'light', d_brightness, similarity)
            output=True
        return output

    def judge_leaving_car(self, ori_img):
        # 주차면들
        for i in range(len(self.parking_spaces)):
            if self.parked_list[i]<1: # 상태가 주차완료일때만 출차예정 판단
                continue
            # 주차면에 헤드라이트가 켜져있으면
            #if self.judge_brightness(ori_img[100:150,100:150], i): # PC캠 디버깅
            if self.judge_brightness(ori_img[self.head_lights[i][2]:self.head_lights[i][3], self.head_lights[i][0]:self.head_lights[i][1]], i):
                self.parked_list[i] *= -1
                print(self.parked_list,"going out")
                if self.arduino:
                    self.arduino.parking_state_determiner(self.parked_list)


    def judge_leaving_car_by_YOLO(self, ori_img, center, class_name):
        if not(class_name=='bright'):
            return
        # 헤드라이트가 있는 주차면 찾기
        space=99 # space_being_left
        for i in range(len(self.parked_list)):
            if self.head_lights[i][0]<=center[0]<=self.head_lights[i][1] and self.head_lights[i][2]<=center[1]<=self.head_lights[i][3]:
                space=i
                break
        # 헤드라이트 위치가 주차된 차에 있지 않거나 주차완료된 곳이 아니면 skip
        if space==99 or self.parked_list[space]<1:
            return
        # 헤드라이트가 어두웠다가 밝아졌으면 출차 알림
        img=ori_img[self.head_lights[space][2]:self.head_lights[space][3], self.head_lights[space][0]:self.head_lights[space][1]]
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness=img.mean()
        d_brightness=brightness-self.prev_brightnesses[space][0] # 여러 프레임 전의 밝기와 비교
        # 유사도 비교
        # img=cv2.GaussianBlur(img, (5,5), 0)
        # shape=cv2.Canny(img,100,100).flatten()
        # similarity=len(np.where(shape==self.prev_shapes[i][0])[0])/len(shape)
        # 여러 프레임 전보다 밝고 유사도가 높으면
        if d_brightness>10: #and similarity>0.89:
            self.parked_list[space] *= -1
            print(self.parked_list,"going out",d_brightness)
            if self.arduino:
                self.arduino.parking_state_determiner(self.parked_list)

    def store_prev_brightnesses(self, ori_img):
        for i in range(len(self.parked_list)):
            cv2.rectangle(ori_img, (self.head_lights[i][0], self.head_lights[i][2]), (self.head_lights[i][1], self.head_lights[i][3]), (255,0,0))
            img=ori_img[self.head_lights[i][2]:self.head_lights[i][3], self.head_lights[i][0]:self.head_lights[i][1]]
            img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness=img.mean()
            del self.prev_brightnesses[i][0]
            self.prev_brightnesses[i].append(brightness)
            # 유사도
            img=cv2.GaussianBlur(img, (5,5), 0)
            shape=cv2.Canny(img,100,100).flatten()
            del self.prev_shapes[i][0]
            self.prev_shapes[i].append(shape)