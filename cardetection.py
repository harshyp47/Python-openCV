import cv2
cap = cv2.VideoCapture('E:\\CarDetection\\carvid.mp4')

car_cascade = cv2.CascadeClassifier('E:\\CarDetection\\cars.xml')


while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cars=car_cascade.detectMultiScale(gray,1.8,2)

    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0),2)

    cv2.imshow('frame',img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()