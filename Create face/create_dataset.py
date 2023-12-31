import cv2
import os
import shutil

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Cannot open camera")
    exit()

cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

face_name = input('\nEnter user name and press Enter: ')
print("\n[INFO] Initializing face capture. Look at the camera and wait...")

count = 0
path = "dataset/" + str(face_name)
if os.path.exists(path):
    shutil.rmtree(path)
os.makedirs(path)

num_data = 100

while True:
    ret, img = cam.read()

    if not ret:
        print("Failed to grab a frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(10, 10)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        crop = gray[y:y + h, x:x + w]
        crop = cv2.resize(crop, (100, 100), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(path + "/" + "User" + '.' + str(face_name) + '.' + str(count) + ".jpg", crop)
        cv2.imshow('image', img)

    k = cv2.waitKey(100)  # Press 'ESC' for exiting video
    if k == 27 or count >= num_data:
        break

# Clean up
print("\n[INFO] Exiting Program and cleaning up")
cam.release()
cv2.destroyAllWindows()
