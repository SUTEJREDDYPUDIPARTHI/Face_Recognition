import cv2
import face_recognition

# load the image
imgSutej = face_recognition.load_image_file(r"C:\Users\LENOVO\Downloads\Face-Recognition-main\Face-Recognition-main\images\Photo.JPG")
imgSutej = cv2.cvtColor(imgSutej, cv2.COLOR_BGR2RGB)#changing bgr to rgb
imgTest = face_recognition.load_image_file(r"C:\Users\LENOVO\Downloads\Face-Recognition-main\Face-Recognition-main\images\my pic.jpg") #load the test image
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgSutej)[0]
encodeSutej = face_recognition.face_encodings(imgSutej)[0] #encode the image
cv2.rectangle(imgSutej,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,0),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,0),2)

#compare faces
results = face_recognition.compare_faces([encodeSutej], encodeTest)
#print the result
if results[0]:
	print('Faces matched both are same person')
else:
	print('Faces not matched both are different person')

cv2.imshow("Sutej", imgSutej)
cv2.imshow("Sutej test", imgTest)
cv2.waitKey(0)
