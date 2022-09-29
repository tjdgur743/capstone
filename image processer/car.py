import socket, cv2, time, threading

class Bluetooth:
    def __init__(self) -> None:
        self.arduino = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        MAC = '98:d3:31:f6:0b:4b'
        self.arduino.connect((MAC,1))
        self.target=(190,330)
        self.difference=(0,0)
        self.car_id=9999

    def target_drawer(self, ori_img):
        cv2.circle(ori_img, self.target, 7, (255,255,255), -1) # 객체의 중앙 좌표

    def update(self, id, center):
        if id==self.car_id:
            self.difference=(self.target[0]-center[0],self.target[1]-center[1])

    def send_signal(self, signal):
        while 1:
            self.arduino.send(f'{signal}'.encode())
            if self.arduino.recv(1).decode()=='O':
                break

    def drive(self):
        self.car_id=int(input())
        time.sleep(0.2)
        #while self.difference==(0,0):pass
        threading.Thread(target=self.test).start()

        # step1: 빠르게 직진
        print('step1')
        while 1:
            if self.difference[1]>25:
                self.send_signal('lza')
            elif self.difference[1]<-25:
                self.send_signal('rza')

            if self.difference[0]<-25:
                self.send_signal('wge')
            else:
                self.send_signal('paa')
                break
            #time.sleep(0.3)
        # 미세조정
        print('step1 미세조정')
        reached=0
        while 1:
            if self.difference[0]<-5:
                reached=0
                self.send_signal('wga')
            elif self.difference[0]>5:
                reached=0
                self.send_signal('sga')
            else:
                reached+=1
                print('reached:',reached)
                if reached>=5:
                    self.send_signal('paa')
                    break

        #step2: 좌회전 전진
        print('step2')
        self.target=(100,400)
        while 1:
            if self.difference[1]>25:
                self.send_signal('lza')
                self.send_signal('wke')
        # 미세조정
        print('step2 미세조정')
        reached=0
        while 1:
            if self.difference[1]>5:
                reached=0
                self.send_signal('lza')
                self.send_signal('wka')
            elif self.difference[1]<-5:
                reached=0
                self.send_signal('lza')
                self.send_signal('ska')
            else:
                reached+=1
                print('reached:',reached)
                if reached>=5:
                    self.send_signal('paa')
                    break
        #step3: 우회전 후진
        print('step3')
        self.target(100,270)
        while 1:
            if self.difference[1]<-25:
                self.send_signal('rza')
                self.send_signal('ske')
            
    # Q: LED on / W: LED off
    def test(self):
        while 1:
            while 1:
                msg=input()
                self.arduino.send(f'{msg}'.encode())
                if self.arduino.recv(1).decode()=='O':
                    break
#bt=Bluetooth()
#bt.test()