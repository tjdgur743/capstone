import sys
sys.path.insert(0, './yolov5')

#import base64, socketio, requests
#import time, math, keyboard
#from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import operator, threading
from leaving_case import Leaving_case
import LED_controller, raspberry, car

# from yolov5.models.experimental import attempt_load
# from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, MyStream, VID_FORMATS
from yolov5.utils.general import (check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

# Communication
# requests.post('http://localhost:4000/process/python_login', json={'id': 'police', 'password':'112'})
#socket_io = socketio.Client()
# socket_io.connect('http://localhost:4000')

#import timer_alarm
#timer_alarm.socket_io=socket_io

cam=True
with_arduino = True
led_controller=LED_controller.Bluetooth() if with_arduino else False
with_car=True
car_controller=car.Bluetooth() if with_car and cam else False
if car_controller:
    threading.Thread(target=car_controller.drive).start()
with_socket_streaming=True
address='192.168.137.177'
#address='192.168.137.248'
socket_streaming=raspberry.Socket(address) if with_socket_streaming else False
pi_ip=False # 웹 스트리밍
pi_ip = address if pi_ip else ''

# Yolo
yolo_model='yolov5/weights/yolov5s.onnx' if not cam else 'yolov5/weights/headlight.onnx'
device = select_device('')
model = DetectMultiBackend(yolo_model, device=device, dnn='')
stride, names, pt = model.stride, model.names, model.pt
img_size = check_img_size([640, 640], s=stride)  # check image size
names = model.module.names if hasattr(model, 'module') else model.names # Get names and colors
classes=[0,1] # car, bright, dark
#classes=[0,2]

# Colors
np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Dataloader
source="resource/outcase_4.mp4"
parking_space_center=[[90,200],[200,200],[320,200],[430,190],[520,200]] # 중앙xy
width=40
    
if cam: #165 290 410 510
    top_y=220
    bottom_y=270
    parking_space=[[61,92,top_y,bottom_y],[160,208,top_y,bottom_y],[280,326,top_y,bottom_y],[400,438,top_y,bottom_y],[512,552,top_y,bottom_y]]
    #top_y=275
    #bottom_y=335
    #parking_space = [[85,122,top_y,bottom_y],[214,261,top_y,bottom_y],[354,399,top_y,bottom_y],[496,531,top_y,bottom_y],[630,666,top_y,bottom_y]]
    parked_list = [0, 0, 0, 0, 0]
    least = [0, 0, 0, 0, 0]
    disappeared = [0, 0, 0, 0, 0]
    cudnn.benchmark = True  # set True to speed up constant image size inference
    #input = LoadStreams(source, img_size=img_size, stride=stride, auto=pt)
    input = MyStream(img_size=img_size, stride=stride, auto=pt, socket=socket_streaming, pi_ip=pi_ip) # Turn on webcam
elif source == "resource/outcase_4.mp4" or source == "resource/incase_4.mp4":
    parking_space = [[220, 320, 210, 310], [490, 590, 210, 310], [700, 800, 200, 300], [960, 1060, 200, 300]]
    parked_list = [0, 0, 0, 0]
    least = [0, 0, 0, 0]
    disappeared = [0, 0, 0, 0]
    input = LoadImages(source, img_size=img_size, stride=stride, auto=pt)
else:
    parking_space = [[270,300,150,250],[450,480,150,250],[640,670,150,250],[820,850,150,250],[1000,1030,150,250]]
    parked_list = [0, 0, 0, 0, 0]
    least = [0, 0, 0, 0, 0]
    disappeared = [0, 0, 0, 0, 0]
    input = LoadImages(source, img_size=img_size, stride=stride, auto=pt)
# else:
#     input = LoadImages(source, img_size=img_size, stride=stride, auto=pt)

waiting_time = 0
# initialize deepsort
cfg = get_config()
config_deepsort="deep_sort/configs/deep_sort.yaml"
cfg.merge_from_file(config_deepsort)
deepsort=DeepSort(
                'osnet_x0_50', #cfg.DEEPSORT.MODEL_TYPE,
                device,
                max_dist=0.6,#cfg.DEEPSORT.MAX_DIST,
                max_iou_distance=0.7,#cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=30,#cfg.DEEPSORT.MAX_AGE,
                n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET)

# Run tracking
outputs = []
led_counter = 0
dic_least = {}
min_index = 0

leaving_case=Leaving_case(led_controller, parking_space, parked_list)
model.warmup(imgsz=(1, 3, *img_size))
for frame_idx, (path, image, image0s, vid_cap, s) in enumerate(input): # Frames
    if cam:
        image0= image0s[0] # A frame
    else:
        image0, _ = image0s, getattr(input, 'frame', 0) # A frame
    #leaving_case.judge_leaving_car(image0) # only 밝기변화로 헤드라이트 빛 감지
    leaving_case.store_prev_brightnesses(image0) # YOLO로 헤드라이트 빛 감지 (output 밑에 한 줄 더있음)
    if car_controller:
        car_controller.target_drawer(image0)

    # Preprocessing for YOLO
    image = torch.from_numpy(image).to(device)
    image = image.float()  # uint8 to fp16/32
    image /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(image.shape) == 3: # If video
        image = image[None]  # expand for batch dim

    # Inference
    pred = model.forward(image, augment=False, visualize=False) # prediction
    pred = non_max_suppression(pred, 0.6, 0.5, classes, False, max_det=1000) # Apply NMS

    # Process detections
    for i, detection in enumerate(pred):  # Deepsort on detections in this frame
        #start_time=time.time() # To measure FPS

        #s += '%gx%g ' % image.shape[2:]  # print string
        annotator = Annotator(image0, line_width=2, pil=not ascii)

        if detection is not None and len(detection):
            # Rescale boxes from img_size to im0 size
            detection[:, :4] = scale_coords(image.shape[2:], detection[:, :4], image0.shape).round()

            xywhs = xyxy2xywh(detection[:, 0:4])
            confs = detection[:, 4] # confidence
            clss = detection[:, 5] # class

            # Pass a detection to deepsort
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), image0, parked_list, cam, led_controller)

            # Draw boxes for visualization
            for j, (output) in enumerate(outputs):
                bbox = output[0:4]
                id = output[4]
                cls = output[5] # class
                conf = output[6] # confidencce
                elapsed_time=output[7]
                slow='slow' if output[8] else ''
                c = int(cls)  # integer class
                label = f'{id:.0f}/{elapsed_time:0.0f}/{output[10]}/{output[11]}'
                #label = f'{names[c]}/{100*conf:0.0f}%/{output[10]}/{output[11]}'
                #label = f'{100*conf:0.0f}%/{elapsed_time:0.0f}/{output[10]}/{output[11]}'
                #label = f'{output[10]}/{output[11]}'
                #label = f'{elapsed_time:0.0f}/{output[10]}/{output[11]}'
                color=[1,1,1] if output[9] else [ int(c) for c in COLORS[c%len(classes)] ] # Parked, not parked
                annotator.box_label(bbox, label, color=color)
                cv2.circle(image0, (int(output[10]), int(output[11])), 5, (255,255,0), -1) # 객체의 중앙 좌표
                leaving_case.judge_leaving_car_by_YOLO(image0, (output[10], output[11]), names[c]) # YOLO로 헤드라이트 빛 감지
                if car_controller:
                    car_controller.update(id, (output[10], output[11]))

                for k in range(len(parked_list)):
                    if names[c] == 'car' and parking_space[k][0] <= output[10] <= parking_space[k][1] and  parking_space[k][2] <= output[11] <= parking_space[k][3] and parked_list[k] >= 0:
                        if int(id) not in parked_list:
                            parked_list[k] = int(id)
                            print(f'{parked_list}parked')
                            if led_controller:
                                led_controller.parking_state_determiner(parked_list)
                    if names[c] == 'car' and (not (parking_space[k][0] <= output[10] <= parking_space[k][1]) or not( parking_space[k][2] <= output[11] <= parking_space[k][3])):
                        if parked_list[k] == -int(id) or parked_list[k] == int(id):
                            parked_list[k] = 0
                            print(f'{parked_list}parked')
                            if led_controller:
                                led_controller.parking_state_determiner(parked_list)
                    if parked_list[k] < 0 and names[c] == 'car' and parking_space[k][0] <= output[10] <= parking_space[k][1] and  parking_space[k][2] <= output[11] <= parking_space[k][3]:
                        if waiting_time == 0:
                            waiting_time = elapsed_time   
                        elif elapsed_time - waiting_time > 5: # 주차가능 예정 해제 시간
                            waiting_time = 0
                            parked_list[k] = -parked_list[k] 
                            print(f'{parked_list}parked')
                            if led_controller:
                                led_controller.parking_state_determiner(parked_list)                                          
                             
                if  names[c] == 'car' and int(id) not in parked_list and 2 < int(elapsed_time) < 20:
                    for i in range(len(parked_list)):
                        least[i] = abs(((parking_space[i][0] + parking_space[i][1])/2) - output[10])
                    min_index_1 = least.index(min(least))
                    # print(min_index_1)
                    for i, num in enumerate(least):
                        dic_least.setdefault(i,num)
                    sort_least = dict(sorted(dic_least.items(), key=operator.itemgetter(1)))

                    list_sort = list(sort_least.values())
                    for key, value in sort_least.items():
                        if value == list_sort[0] and parked_list[key] == 0:
                            min_index = key
                    if parked_list.count(0.5) > 1:
                        if min_index > 0 and parked_list[min_index - 1] == 0.5:
                            parked_list[min_index - 1] = 0
                            print(f'{parked_list}parked')
                            if led_controller:
                                led_controller.parking_state_determiner(parked_list)
                        elif min_index < len(parked_list) - 1 and parked_list[min_index + 1] == 0.5:
                            parked_list[min_index + 1] = 0
                            print(f'{parked_list}parked')
                            if led_controller:
                                led_controller.parking_state_determiner(parked_list)                         
                    elif parked_list[min_index] == 0:
                        parked_list[min_index] = 0.5
                        print(f'{parked_list}parked')
                        if led_controller:
                            led_controller.parking_state_determiner(parked_list)                     
                dic_least.clear()

                if parked_list.count(0.5) == 1:
                    # if min_index_1 != parked_list.index(0.5):
                    disappeared[parked_list.index(0.5)] += 1
                    # print(disappeared)
                    if sum(disappeared) >= 100:
                        parked_list[parked_list.index(0.5)] = 0
                        for i in range(len(disappeared)):
                            disappeared[i] = 0
                        print(f'{parked_list}parked')
                        if led_controller:
                            led_controller.parking_state_determiner(parked_list)                         


        else: # No detection
            deepsort.increment_ages()
            #LOGGER.info('No detections')

        if led_counter > 20:
            led_counter = 0
        else : led_counter+=1
        
        if cam:
            circle_y = 90
            for i in range(5): 
                cv2.rectangle(image0, (parking_space[i][0], parking_space[i][2]), (parking_space[i][1], parking_space[i][3]), (0,0,255))
                if i == 0:
                    circle_x = parking_space_center[0][0]
                elif i == 1:
                    circle_x = parking_space_center[1][0]
                elif i == 2:
                    circle_x = parking_space_center[2][0]
                elif i == 3:
                    circle_x = parking_space_center[3][0]
                elif i == 4:
                    circle_x = parking_space_center[4][0]
                if parked_list[i] == 0:
                    cv2.circle(image0, (circle_x, circle_y), 10, (0,255,0), -1)
                elif parked_list[i] < 0:
                    cv2.circle(image0, (circle_x, circle_y), 10, (0,127,255), -1)
                elif parked_list[i] >= 1:
                    cv2.circle(image0, (circle_x, circle_y), 10, (0,0,255), -1)
                elif parked_list[i] == 0.5 and led_counter % 2 == 0:
                    cv2.circle(image0, (circle_x, circle_y), 10, (0,255,0), -1)
        elif source == "resource/outcase_4.mp4" or source == "resource/incase_4.mp4":
            circle_y = 100
            for i in range(4):
                cv2.rectangle(image0, (parking_space[i][0], parking_space[i][2]), (parking_space[i][1], parking_space[i][3]), (0,0,255))
                if i == 0:
                    circle_x = 320
                elif i == 1:
                    circle_x = 560
                elif i == 2:
                    circle_x = 770
                elif i == 3:
                    circle_x = 1000
                if parked_list[i] == 0:
                    cv2.circle(image0, (circle_x, circle_y), 10, (0,255,0), -1)
                elif parked_list[i] < 0:
                    cv2.circle(image0, (circle_x, circle_y), 10, (0,127,255), -1)
                elif parked_list[i] >= 1:
                    cv2.circle(image0, (circle_x, circle_y), 10, (0,0,255), -1)
                elif parked_list[i] == 0.5 and led_counter % 2 == 0:
                    cv2.circle(image0, (circle_x, circle_y), 10, (0,255,0), -1)
        else:
            circle_y = 70
            for i in range(5):
                if i == 0:
                    circle_x = 286
                elif i == 1:
                    circle_x = 462
                elif i == 2:
                    circle_x = 655
                elif i == 3:
                    circle_x = 834
                elif i == 4:
                    circle_x = 1017
                if parked_list[i] == 0:
                    cv2.circle(image0, (circle_x, circle_y), 10, (0,255,0), -1)
                elif parked_list[i] < 0:
                    cv2.circle(image0, (circle_x, circle_y), 10, (0,127,255), -1)
                elif parked_list[i] >= 1:
                    cv2.circle(image0, (circle_x, circle_y), 10, (0,0,255), -1)
                elif parked_list[i] == 0.5 and led_counter % 2 == 0:
                    cv2.circle(image0, (circle_x, circle_y), 10, (0,255,0), -1)        
        
        image0 = annotator.result()
        
        if cv2.waitKey(1)==ord('q'):
            x_pos,y_pos,width,height = cv2.selectROI("location", image0, False)
            print(f'{x_pos}, {x_pos+width}, {y_pos}, {y_pos+height}')
        # image0=cv2.GaussianBlur(image0, (5,5), 0)
        # image0=cv2.Canny(image0,100,100)
        cv2.imshow('capstone', image0)
        if cv2.waitKey(1)==27:
            exit()

        # Streaming on webpage
        '''result, encoded_frame = cv2.imencode('.jpg', image0)
        image_as_text = base64.b64encode(encoded_frame)#.decode('utf-8')
        socket_io.emit('frame from python', image_as_text)'''
                
        '''if keyboard.is_pressed('etc'):
                exit()'''
        '''if keyboard.is_pressed('space'): # Enter to reset
            deepsort=DeepSort(
                'osnet_x0_50',#cfg.DEEPSORT.MODEL_TYPE,
                device,
                max_dist=0.4,#cfg.DEEPSORT.MAX_DIST,
                max_iou_distance=0.7,#cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=30,#cfg.DEEPSORT.MAX_AGE,
                n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET)'''