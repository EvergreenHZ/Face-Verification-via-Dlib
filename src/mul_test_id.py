import cv2
import numpy as np
import face_recognition
import sys
import argparse
from multiprocessing.dummy import Pool as ThreadPool


def read_pair_list(pair_list_path):
    pairs = []
    with open(pair_list_path) as f:
        pair_list = f.read().splitlines()

        for pair in pair_list:
            pair = pair.split()
            pairs.append(pair)

        return pairs

def multithreading_verification(pair):

    # get image name
    img1Path = pair[0]
    img2Path = pair[1]

    # load images
    img1 = face_recognition.load_image_file(img1Path)
    img2 = face_recognition.load_image_file(img2Path)

    # detect faces
    img1FaceLocation = face_recognition.face_locations(img1)  # list
    img2FaceLocation = face_recognition.face_locations(img2)

    if len(img1FaceLocation) != 1 or len(img2FaceLocation) != 1:
        print("img1 detected %d faces, img2 detected %d faces" %(len(img1FaceLocation), len(img2FaceLocation)))
        return -1

    # encode the faces
    img1_face_encoding = face_recognition.face_encodings(img1, img1FaceLocation)
    img2_face_encoding = face_recognition.face_encodings(img2, img2FaceLocation)

    # compare
    for face_encoding in img2_face_encoding:
        match = face_recognition.compare_faces([img1_face_encoding[0]], face_encoding)
        #print match
        if match[0]:
            print("same people")
            return 1
        else:
            print("different people")
            return 0

def multithreading_test(pair_list_path):
    pair_list = read_pair_list(pair_list_path)
    pool = ThreadPool(128)
    results = pool.map(multithreading_verification, pair_list)

    same = [i for i, x in enumerate(results) if x == 1]
    diff = [i for i, x in enumerate(results) if x == 0]
    miss = [i for i, x in enumerate(results) if x == -1]

    return results, same, diff, miss


def parse_args():
    parser = argparse.ArgumentParser(description='Test Verification',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pairlist', dest='pairlist', help='input the path of pairlist.txt',
                        default='./pairlist.txt', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    results, same, diff, miss = multithreading_test(args.pairlist)
    print('same: %4d\ndiff: %4d\nmiss: %4d' %(len(same), len(diff), len(miss)))
