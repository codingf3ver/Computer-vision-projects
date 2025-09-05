from ultralytics import YOLO
import cv2

model=YOLO('../yolo_weights/yolov8l.pt')
results=model('../images/1.jpg',show=True)

annotated_frame = results[0].plot()

cv2.imshow("Detections", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()