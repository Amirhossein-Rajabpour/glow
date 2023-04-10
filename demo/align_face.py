# OLD USAGE
# python align_faces.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from PIL import Image
import numpy as np
# import argparse
import imutils
import dlib
import cv2

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("--shape-predictor", help="path to facial landmark predictor", default='shape_predictor_68_face_landmarks.dat')
# ap.add_argument("--input", help="path to input images", default='input_raw')
# ap.add_argument("--output", help="path to input images", default='input_aligned')
# args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256,
                 desiredLeftEye=(0.371, 0.480))


# Input: numpy array for image with RGB channels
# Output: (numpy array, face_found)
def align_face(img):
    img = img[:, :, ::-1]  # Convert from RGB to BGR format
    img = imutils.resize(img, width=800)

    # img = imutils.resize(img, width=800, inter=cv2.INTER_CUBIC)

    # detect faces in the grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)
    # detect faces in the grayscale image
    # rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


    if len(rects) > 0:
        # align the face using facial landmarks
        # align_img = fa.align(img, gray, rects[0])[:, :, ::-1]
        # align_img = fa.align(img, gray, rects[0])

        # align_img = np.array(Image.fromarray(align_img).convert('RGB'))

        # aligned_faces = []

        # # loop over the face detections
        # for (x, y, w, h) in rects:
        #     # get the facial landmarks
        #     rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        #     shape = predictor(gray, rect)
        #     face_aligned = fa.align(image, gray, rect)
        #     aligned_faces.append(face_aligned)

        # Another method:
	    # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rects[0])
        faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
        align_img = fa.align(image, gray, rects[0])
        align_img = np.array(Image.fromarray(align_img).convert('RGB'))

        return align_img, True
        # return aligned_faces[0], True

    else:
        # No face found
        return None, False


# Input: img_path
# Output: aligned_img if face_found, else None
def align(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB')  # if image is RGBA or Grayscale etc
    img = np.array(img)
    x, face_found = align_face(img)
    return x