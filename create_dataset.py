import cv2
import glob

input_path = "raw_images/*.jpg"
output_path = "faces"

i=0
for imagePath in glob.glob(input_path):
  print(imagePath)
  image = cv2.imread(imagePath)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
  faces = faceCascade.detectMultiScale(
      gray,
      scaleFactor=1.3,
      minNeighbors=3,
      minSize=(60, 60)
  )

  print("[INFO] Found {0} Faces!".format(len(faces)))

  for (x, y, w, h) in faces:
      sub_face = image[y:y+h, x:x+w]
      face_file_name = output_path+"/facean_" + str(i) + ".jpg"
      sub_face=cv2.resize(sub_face, (256,256))
      cv2.imwrite(face_file_name, sub_face)
      i+=1


