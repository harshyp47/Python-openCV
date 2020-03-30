import cv2

cap = cv2.VideoCapture('traffic.mp4')  #Path to footage
car_cascade = cv2.CascadeClassifier('cars.xml')  #Path to cars.xml
bicycle_cascade= cv2.CascadeClassifier('bicycle.xml') #Path to bicycle.xml

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cars=car_cascade.detectMultiScale(gray,1.8,2)
    bicycle = bicycle_cascade.detectMultiScale(gray, 1.8, 2)

    #Drawing rectangles on detected cars
    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0),2)

    # Drawing rectangles on detected bicycles
    for (x,y,w,h) in bicycle:
        cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0),2)


    cv2.imshow('img',img) #Shows the frame



    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()