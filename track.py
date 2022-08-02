from dis import dis
from http import client
import resource
import sys
sys.path.insert(0, './yolov5')

#import base64, socketio, requests
#import time, math, keyboard
import socket
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import operator

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams, MyStream, VID_FORMATS
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

# Yolo
parking_space_coordinates=[]
yolo_model='yolov5/weights/yolov5s.onnx'
device = select_device('')
model = DetectMultiBackend(yolo_model, device=device, dnn='')
stride, names, pt = model.stride, model.names, model.pt
img_size = check_img_size([640, 640], s=stride)  # check image size
names = model.module.names if hasattr(model, 'module') else model.names # Get names and colors
classes=[0,2,7,67] # car:2, truck:7, 67: phone

# Colors
np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

client_socket = 0
def connect_to_raspberry():
    global client_socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ip = "192.168.137.226"
    client_socket.connect((ip, 9000))
    print(f"Bound to port 9000")
raspberry = True
if raspberry: connect_to_raspberry()

# Dataloader
cam=True
#source='0'
# source="resource/other_case.mp4"
source="resource/incase_5_1.mp4"
# source="resource/outcase_5.mp4"
# source="resource/outcase_4.mp4"
# source="resource/incase_4.mp4"
if cam:
    parking_space = [[138, 178, 72, 112], [296, 336, 77, 117], [461, 501, 87, 127], [615, 655, 78, 118]]
    parked_list = [0, 0, 0, 0]
    least = [0, 0, 0, 0]
    disappeared = [0, 0, 0, 0]
    cudnn.benchmark = True  # set True to speed up constant image size inference
    #input = LoadStreams(source, img_size=img_size, stride=stride, auto=pt)
    # input = MyStream(img_size=img_size, stride=stride, auto=pt, raspberry=False) # Turn on webcam
    input = MyStream(img_size=img_size, stride=stride, auto=pt, raspberry=raspberry) # Turn on webcam
elif source == "resource/outcase_4.mp4" or source == "resource/incase_4.mp4":
    parking_space = [[250, 350, 200, 300], [490, 560, 210, 290], [730, 780, 220, 270], [980, 1050, 220, 300]]
    parked_list = [0, 0, 0, 0]
    least = [0, 0, 0, 0]
    disappeared = [0, 0, 0, 0]
else:
    parking_space = [[270,300,150,250],[450,480,150,250],[640,670,150,250],[820,850,150,250],[1000,1030,150,250]]
    parked_list = [0, 0, 0, 0, 0]
    least = [0, 0, 0, 0, 0]
    disappeared = [0, 0, 0, 0, 0]
# else:
#     input = LoadImages(source, img_size=img_size, stride=stride, auto=pt)

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
disappear_count = 0

model.warmup(imgsz=(1, 3, *img_size))
for frame_idx, (path, image, image0s, vid_cap, s) in enumerate(input): # Frames
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
        if cam:
            image0= image0s[i] # A frame
        else:
            image0, _ = image0s, getattr(input, 'frame', 0) # A frame

        #s += '%gx%g ' % image.shape[2:]  # print string
        annotator = Annotator(image0, line_width=2, pil=not ascii)

        if detection is not None and len(detection):
            # Rescale boxes from img_size to im0 size
            detection[:, :4] = scale_coords(image.shape[2:], detection[:, :4], image0.shape).round()

            xywhs = xyxy2xywh(detection[:, 0:4])
            confs = detection[:, 4] # confidence
            clss = detection[:, 5] # class

            # Pass a detection to deepsort
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), image0, parked_list, client_socket)

            # Draw boxes for visualization
            for j, (output) in enumerate(outputs):
                bbox = output[0:4]
                id = output[4]
                cls = output[5] # class
                conf = output[6] # confidencce
                elapsed_time=output[7]
                slow='slow' if output[8] else ''
                c = int(cls)  # integer class
                # label = f'{id:.0f} {names[c]}/{elapsed_time:0.0f}/{output[10]}/{output[11]}'
                label = f'id:{id:.0f} {names[c]}'
                color=[1,1,1] if output[9] else [ int(c) for c in COLORS[c%len(classes)] ] # Parked, not parked
                annotator.box_label(bbox, label, color=color)
                for k in range(len(parked_list)):
                    if names[c] == 'car' and parking_space[k][0] <= output[10] <= parking_space[k][1] and  parking_space[k][2] <= output[11] <= parking_space[k][3] and parked_list[k] >= 0:
                        if int(id) not in parked_list:
                            parked_list[k] = int(id)
                            print(f'{parked_list}parked')
                            if raspberry:
                                client_socket.sendall(str(parked_list).encode())
                                client_socket.recv(99)
                    elif names[c] == 'car' and (not (parking_space[k][0] <= output[10] <= parking_space[k][1]) and not( parking_space[k][2] <= output[11] <= parking_space[k][3])):
                        if (parked_list[k] == -int(id)):
                            parked_list[k] = 0
                            print(f'{parked_list}parked')
                            if raspberry:
                                client_socket.sendall(str(parked_list).encode())
                                client_socket.recv(99)                            
                if  names[c] == 'car' and int(id) not in parked_list and 0 < int(elapsed_time) < 30:
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
                            if raspberry:
                                client_socket.sendall(str(parked_list).encode())
                                client_socket.recv(99)
                        elif min_index < len(parked_list) - 1 and parked_list[min_index + 1] == 0.5:
                            parked_list[min_index + 1] = 0
                            print(f'{parked_list}parked')
                            if raspberry:
                                client_socket.sendall(str(parked_list).encode())
                                client_socket.recv(99)                            
                    elif parked_list[min_index] == 0:
                        parked_list[min_index] = 0.5
                        print(f'{parked_list}parked')
                        if raspberry:
                            client_socket.sendall(str(parked_list).encode())
                            client_socket.recv(99)                        
                dic_least.clear()

                if parked_list.count(0.5) == 1:
                    if min_index_1 != parked_list.index(0.5):
                        disappeared[parked_list.index(0.5)] += 1
                        print(disappeared)
                        if sum(disappeared) >= 100:
                            disappear_count += 1
                            parked_list[parked_list.index(0.5)] = 0
                            for i in range(len(disappeared)):
                                disappeared[i] = 0
                            print(f'{parked_list}parked')
                            if raspberry:
                                client_socket.sendall(str(parked_list).encode())
                                client_socket.recv(99)                            


        else: # No detection
            deepsort.increment_ages()
            #LOGGER.info('No detections')

        if led_counter > 20:
            led_counter = 0
        else : led_counter+=1
        
        if source == "resource/outcase_4.mp4" or source == "resource/incase_4.mp4" or cam == True:
            circle_y = 100
            for i in range(4):
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
        
            # FPS
        '''period=time.time()-start_time
        fps=math.ceil(1/period if period>0.01 else 0.01)
        cv2.putText(image0, 'FPS: {}'.format(str(fps)), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)'''
        # x_pos,y_pos,width,height = cv2.selectROI("location", image0, False)
        # parking_space_coordinates.append([int(x_pos*(640/len(image0[0]))), int(y_pos*(640/len(image0)))])
        # print(parking_space_coordinates)
        # cv2.rectangle(image0,(250,350), (300, 400), (255, 255, 255), 10)
        # parking_space = [[270,300,150,250],[450,480,150,250],[640,670,150,250],[820,850,150,250],[1000,1030,150,250]]
        # parked_list = [0, 0, 0, 0, 0]
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
