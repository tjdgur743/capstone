# 라즈베리파이4에서 gpu 앱을 우분투와 연결 방법

![Screenshot from 2022-04-27 23-55-34](https://user-images.githubusercontent.com/87261213/165548487-7462b762-4627-4d0e-ae0f-1afcf736e1e3.png)

 
 /etc/ssh/ssh_config 파일 수정
 sudo vi ssh_config 명령을 실행

![Screenshot from 2022-04-27 23-56-54](https://user-images.githubusercontent.com/87261213/165548534-dd22ab4c-e359-4b0d-8b85-ead66ead880f.png)


 ForwardX11 yes 의 주석 # 제거 후 저장
 
 ![Screenshot from 2022-04-27 23-59-46](https://user-images.githubusercontent.com/87261213/165548721-bed86ae0-0cff-43f4-8b1e-4c75326eab51.png)
