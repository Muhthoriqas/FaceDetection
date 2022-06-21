import pathlib
import cv2
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / 'data' / 'haarcascade_frontalface_default.xml'

clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture(0) # 0 is the default camera
while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
