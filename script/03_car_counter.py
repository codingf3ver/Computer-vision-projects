from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


cap=cv2.VideoCapture('../videos/cars2.mp4')

model=YOLO('../yolo_weights/yolov8l.pt')
classNames = model.names

mask=cv2.imread('../images/mask.png')
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)

while True:
    success, img = cap.read()
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
    image_region=cv2.bitwise_and(img,mask_resized)

    results=model(image_region,stream=True)

    detections=np.empty((0,5))
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
           
            w,h=x2-x1,y2-y1
            bbox=(int(x1),int(y1),int(w),int(h))
            

            confidenece_score=math.ceil(box.conf[0]*100)/100
            cls=int(box.cls[0])
            current_class=classNames[cls]

            if current_class=='car' or current_class=='truck' or current_class=='motorbike' or current_class=='bus'\
                and confidenece_score>0.3:
                cvzone.putTextRect(img, f'{current_class} {confidenece_score}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1,offset=3)
                cvzone.cornerRect(img,bbox,l=9)
                current_array=np.array([x1,y1,x2,y2,confidenece_score])
                detections=np.vstack((detections,current_array))

                       
    if not success:
        break
    
    result_trackers=tracker.update(detections)
    for result in result_trackers:
        x1,y1,x2,y2,id=result
        print(result)

    cv2.imshow("Image", img)
    cv2.imshow('imgaeRegion',image_region)
    cv2.waitKey(0)
#     if cv2.waitKey(0) & 0xFF == 27:  
#         break

# cap.release()
# cv2.destroyAllWindows()
