import cv2
import face_recognition

# here we load the image
known_image = face_recognition.load_image_file(r'C:\Users\LENOVO\Downloads\Face-Recognition-main\Photo.JPG')
known_image = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)


face_locations = face_recognition.face_locations(known_image)
if face_locations:
	faceLoc = face_locations[0]
	encodeImage = face_recognition.face_encodings(known_image)[0]
	cv2.rectangle(known_image, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 255, 255), 2)

# start the video capture
captureImage = cv2.VideoCapture(0)
prev_match_status = None

while True:
	success, frame = captureImage.read()
	if not success:
		break
	
	rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	liveFaceLocs = face_recognition.face_locations(rgb_image)
	
	if liveFaceLocs:
		liveFaceLoc = liveFaceLocs[0]
		encodeFace = face_recognition.face_encodings(rgb_image)
		
		if encodeFace:
			encodeFace = encodeFace[0]
			cv2.rectangle(frame, (liveFaceLoc[3], liveFaceLoc[0]), (liveFaceLoc[1], liveFaceLoc[2]), (0, 255, 255),2)
			
			#here we are comparing faces
			results = face_recognition.compare_faces([encodeImage], encodeFace)
			match_status = results[0]
			if match_status != prev_match_status:
				if match_status:
					print('Faces Matched')
				else:
					print('No Faces Matched')
				prev_match_status = match_status
	
	# display the image and live video
	cv2.imshow('Live Face', frame)
	cv2.imshow('Known Image', known_image)
	
	# here we set a key to terminate the program
	key = cv2.waitKey(1)
	if key == 27: #press esc to exit
		break

