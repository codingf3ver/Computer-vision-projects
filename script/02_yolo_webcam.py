from ultralytics import YOLO
import cv2
import cvzone
import math


cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

model=YOLO('../yolo_weights/yolov8l.pt')

classNames = model.names
print(classNames)

while True:
    success, img = cap.read()
    results=model(img,stream=True)
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,200),3)

            # x1,y1,w,h=box.xywh[0]
            
            w,h=x2-x1,y2-y1
            bbox=(int(x1),int(y1),int(w),int(h))
            cvzone.cornerRect(img,bbox)

            conf=math.ceil(box.conf[0]*100)/100
            cls=int(box.cls[0])
            if cls in classNames:
                label = f"{classNames[cls]} {conf}"
            else:
                label = f"Unknown {conf}"
            cvzone.putTextRect(img, label, (max(0, x1), max(30, y1)), scale=0.7, thickness=1)

                       
    if not success:
        break

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
