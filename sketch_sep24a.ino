void setup() {
  // put your setup code here, to run once:
  pinMode(6,OUTPUT);
  pinMode(7,OUTPUT);
  pinMode(9,OUTPUT);
  pinMode(10,OUTPUT);
}
void left(int speed)
{
  digitalWrite(6, HIGH); 
  digitalWrite(7, LOW); 
  analogWrite(5,speed); // 좌우 speed 
}

void right(int speed)
{
  digitalWrite(6, LOW); 
  digitalWrite(7, HIGH); 
  analogWrite(5,speed); // 좌우 speed 
}
void RL_stop()
{
  analogWrite(5,0); // 좌우 speed 
}
void forward(int speed)
{
  digitalWrite(9, LOW); 
  digitalWrite(10, HIGH); 
  analogWrite(8,speed); // 앞뒤 speed 
}
void back(int speed)
{
  digitalWrite(9, HIGH); 
  digitalWrite(10, LOW); 
  analogWrite(8,speed); // 앞뒤 speed 
}
void FB_stop() 
{
  analogWrite(8,0); 
}

void loop() {


  left(200);
  forward(200);
  delay(1000);
  right(200);
  back(200);
  delay(1000);
  RL_stop();
  FB_stop();
  delay(1000);


}
