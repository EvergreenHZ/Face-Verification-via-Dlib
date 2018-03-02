import cv2
import numpy as np
import face_recognition
import sys

def displayImage(image, location):
    # Display the results
    print location
    for (top, right, bottom, left) in location:
        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

    # Display the resulting image
    cv2.namedWindow('image')
    cv2.imshow('image',image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def verification(img1Path, img2Path2):
    # load images
    img1 = face_recognition.load_image_file(img1Path)
    img2 = face_recognition.load_image_file(img2Path)

    # detect faces
    img1FaceLocation = face_recognition.face_locations(img1)  # list
    img2FaceLocation = face_recognition.face_locations(img2)

    print img1FaceLocation, img2FaceLocation

    if len(img1FaceLocation) != 1 or len(img2FaceLocation) != 1:
        print "Face Detection Error"
        exit()

    # encode the faces
    img1_face_encoding = face_recognition.face_encodings(img1, img1FaceLocation)
    img2_face_encoding = face_recognition.face_encodings(img2, img2FaceLocation)

    # displayImage(img1, img1FaceLocation)

    # compare
    for face_encoding in img2_face_encoding:
        match = face_recognition.compare_faces([img1_face_encoding[0]], face_encoding)
        #print match
        if match[0]:
            print "same people"
        else:
            print "different people"


if len(sys.argv) > 3 :
    print "wrong arguments!"
    print "usage:\
                python verif.py img1 img2"
    exit()

img1Path = sys.argv[1]
img2Path = sys.argv[2]
verification(img1Path, img2Path)
