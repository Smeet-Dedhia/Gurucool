import cv2
import numpy as np
import face_recognition

imgdhoni = face_recognition.load_image_file('img_main/dhoni.jpg')
imgdhoni = cv2.cvtColor(imgdhoni, cv2.COLOR_BGR2RGB)
imgdhonitest = face_recognition.load_image_file('img_main/dhoni_test.jpg')
imgdhonitest = cv2.cvtColor(imgdhonitest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgdhoni)[0]
encodedhoni = face_recognition.face_encodings(imgdhoni)[0]
cv2.rectangle(imgdhoni, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255),2)

faceLocTest = face_recognition.face_locations(imgdhonitest)[0]
encodedhonitest = face_recognition.face_encodings(imgdhonitest)[0]
cv2.rectangle(imgdhonitest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255),2)

results = face_recognition.compare_faces([encodedhoni], encodedhonitest)
faceDis = face_recognition.face_distance([encodedhoni], encodedhonitest)
cv2.putText(imgdhonitest, f'{results}{round(faceDis[0], 2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL,1 ,(0, 0, 255), 1)
print(results, faceDis)

print(faceLoc)

cv2.imshow('dhoni', imgdhoni)
cv2.imshow('dhoni_test', imgdhonitest)
cv2.waitKey(0)