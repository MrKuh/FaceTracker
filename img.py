import pathlib
import cv2

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

#clf = cv2.CascadeClassifier(str(cascade_path))
clf = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
	
img = cv2.imread('C:/Users/micha/Downloads/IMG_2949.png')

print(img.shape)


grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = clf.detectMultiScale(
        grayImage,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(10,10),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
i = 0
for(x, y, width, height) in faces:
    cv2.rectangle(img,(x, y), (x+width, y+height), (255,255,0),2)
    i = i+1
    # Display the box and faces
    cv2.putText(img, 'face num'+str(i), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    print(i)
    
    
scale_percent = 35 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Faces", image)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
        