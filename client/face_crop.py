import cv2
from threading import Thread

def identify_face(args):
    print("verifying")
    # resImg = cv2.resize(roi_color, (256, 256), interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite('face.jpg', resImg)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml') 
verifying_thread = Thread(target = identify_face, args = (10, ))
# faces  = face_cascade.detectMultiScale(gray, 1.3, 5) 
print("")
last_faces_count = 0
def detect(gray, frame): 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    print("Faces count : " ,len(faces), end ='\r')
    for (x, y, w, h) in faces: 
        roi_color = frame[y:y + h, x:x + w] 
        if(last_faces_count!=len(faces)):
            if(not verifying_thread.is_alive()):
                verifying_thread.start()
                # pass
            
        roi_gray = gray[y:y + h, x:x + w] 
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2) 
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
  
        for (sx, sy, sw, sh) in smiles: 
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2) 
    return frame


video_capture = cv2.VideoCapture(0) 
while True: 
   # Captures video_capture frame by frame 
    _, frame = video_capture.read()  
  
    # To capture image in monochrome                     
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
      
    # calls the detect() function     
    canvas = detect(gray, frame)    
  
    # Displays the result on camera feed                      
    cv2.imshow('Video', canvas)  
  
    # The control breaks once q key is pressed                         
    if cv2.waitKey(1) & 0xff == ord('q'):                
        break
  
# Release the capture once all the processing is done. 
video_capture.release()                                  
cv2.destroyAllWindows() 
verifying_thread.join()
print("App exit")