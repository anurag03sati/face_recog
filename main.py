from ultralytics import YOLO
import cv2
import time

model =YOLO("best (1).pt")

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise("No camera")

while True:
    ret, image = cam.read()
    if not ret:
        break

    _time_mulai = time.time()
    result = model.predict(image,show=True)
    print("time", time.time()- _time_mulai)

    _key = cv2.waitKey(1)
    if _key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()