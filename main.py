import pathlib
import cv2

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

clf = cv2.CascadeClassifier(str(cascade_path))

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(10,10),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    i = 0
    for(x, y, width, height) in faces:
        cv2.rectangle(frame,(x, y), (x+width, y+height), (255,255,0),2)
        i = i+1
        # Display the box and faces
        cv2.putText(frame, 'face num'+str(i), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print(i)
    
    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()