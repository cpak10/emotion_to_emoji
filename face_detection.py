import cv2
import matplotlib.pyplot as plt

class findFace:
    # create class for identifying faces and cropping them out of images

    def save_face(self, file_path, root):

        # load the image
        image = cv2.imread(file_path)

        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # load the cascade classifier
        face_cascade = cv2.CascadeClassifier(f"{root}\\intake\\haarcascade_frontalface_default.xml")

        # detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5, flags = cv2.CASCADE_SCALE_IMAGE | cv2.CASCADE_FIND_BIGGEST_OBJECT)

        # crop the face from the image
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

        # save the cropped face
        try:
            cv2.imwrite(f"{root}\\working\\cropped_face.jpg", face)

            # print the found face
            plt.imshow(face, cmap = "gray")
            plt.title("Found Face")
            plt.axis("off")
            plt.show()

        except: 
            print("\nERROR: No face identified.")        