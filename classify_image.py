import cv2
from face_detection import findFace

# set file root
file_root = "C:\\GitHub\\emotion_to_emoji"

# load face finder
face_finder = findFace()
face_finder.save_face(file_path = f"{file_root}\\intake\\test.jpg", root = file_root)