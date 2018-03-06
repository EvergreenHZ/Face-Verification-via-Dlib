import cv2
import numpy as np
import face_recognition
import sys
import argparse

def detect_face(img1Path, img2Path):
    # load images
    img1 = face_recognition.load_image_file(img1Path)
    img2 = face_recognition.load_image_file(img2Path)

    # detect faces
    img1FaceLocation = face_recognition.face_locations(img1)  # list
    img2FaceLocation = face_recognition.face_locations(img2)

    return img1, img2, img1FaceLocation, img2FaceLocation

def verification(img1, img2, img1_face_location, img2_face_location):

    # encode the faces  always one face
    img1_face_encoding = face_recognition.face_encodings(img1, img1_face_location)[0]
    img2_face_encoding = face_recognition.face_encodings(img2, img2_face_location)[0]

    # compare
    match = face_recognition.compare_faces([img1_face_encoding], img2_face_encoding)
    if match[0]:
        return True
    else:
        return False

def read_pair_list(pair_list_path):
    pairs = []
    with open(pair_list_path) as f:
        pair_list = f.read().splitlines()

        for pair in pair_list:
            pair = pair.split()
            pairs.append(pair)

        return pairs

def test_identity(pair_list_path):

    # read all the image names
    pair_list = read_pair_list(pair_list_path)

    # record the verification result
    same = 0
    diff = 0
    num_pairs = len(pair_list)
    i = 1


    for pair in pair_list:
        img1, img2, loc1, loc2 = detect_face(pair[0], pair[1])
        if len(loc1) != 1 or len(loc2) != 1:
            print("in the %dth pair, img1 detected %d faces, img2 detected %d faces" %(i, len(loc1), len(loc2)))
        else:
            if verification(img1, img2, loc1, loc2):
                same = same + 1
            else:
                diff = diff + 1

        i = i + 1

    return num_pairs, same, diff


def parse_args():
    parser = argparse.ArgumentParser(description='Test Verification',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pairlist', dest='pairlist', help='input the path of pairlist.txt',
                        default='./pairlist.txt', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    num_pairs, same, diff = test_identity(args.pairlist)

    precision = float(same) / num_pairs

    print('precision of verification: %.4f' %(precision))
