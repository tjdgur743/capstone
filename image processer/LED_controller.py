import socket, time, threading

class Bluetooth:
    def __init__(self) -> None:
        self.arduino = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        MAC = '98:D3:21:FC:94:10'
        self.arduino.connect((MAC,1))
        for i in range(5):
            self.arduino.send(f'{i}{3}'.encode())

    # def blink_LED(self, i):
    #     # Not needed for turning on real LED
    #     #time.sleep(0.1)
    #     LED_state=0
    #     self.is_blink_LED[i]=True
    #     while 1:
    #         if self.is_blink_LED[i]==False:
    #             return
    #         while 1:
    #             self.arduino.send(f'{i}{LED_state}'.encode())
    #             if self.arduino.recv(1).decode()=='O':
    #                 break
    #         LED_state=0 if LED_state==1 else 1
    #         time.sleep(0.5)

    def parking_state_determiner(self, parked_list):
        for i, LED_state in enumerate(parked_list):
            if -0.0001<LED_state<0.0001: # Green
                while 1:
                    self.arduino.send(f'{i}1'.encode())
                    if self.arduino.recv(1).decode()=='O':
                        break
            elif LED_state>=1: # Red
                while 1:
                    self.arduino.send(f'{i}2'.encode())
                    if self.arduino.recv(1).decode()=='O':
                        break
            elif LED_state<0: # Orange
                while 1:
                    self.arduino.send(f'{i}3'.encode())
                    if self.arduino.recv(1).decode()=='O':
                        break
            elif LED_state==0.5: # Blink
                while 1:
                    self.arduino.send(f'{i}4'.encode())
                    if self.arduino.recv(1).decode()=='O':
                        break
    def test(self):
        while 1:
            while 1:
                msg=input()
                self.arduino.send(f'{msg}'.encode())
                if self.arduino.recv(1).decode()=='O':
                    break
bt=Bluetooth()
bt.test()