import face_recognition
from PIL import Image, ImageDraw
import cv2
import numpy as np

def main():
  print("Meme machine now online")
  image = face_recognition.load_image_file("people.jpg")

  #frame = image[:, :, ::-1]
  face_locations = face_recognition.face_locations(image)

  #im = Image.open("people.jpg")
  #draw = ImageDraw.Draw(im)
  
  #for loc in face_locations:
  #  draw.rectangle(loc, fill=None, outline=128)

  #del draw

  #im.show()
  #face_locations = face_recognition.face_locations(rgb_frame)
  face_encodings = face_recognition.face_encodings(image, face_locations)

  # Load a sample picture and learn how to recognize it.
  obama_image = face_recognition.load_image_file("obama.jpg")
  obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
  biden_image = face_recognition.load_image_file("biden.jpg")
  biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Load a second sample picture and learn how to recognize it.
  noah_image = face_recognition.load_image_file("noah.jpg")
  noah_face_encoding = face_recognition.face_encodings(noah_image)[0]

# Load a second sample picture and learn how to recognize it.
  mitch_image = face_recognition.load_image_file("mitch.jpg")
  mitch_face_encoding = face_recognition.face_encodings(mitch_image)[0]

  known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    noah_face_encoding,
    mitch_face_encoding
  ]
  known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "Noah",
    "Mitch"
  ]

  for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

      name = "Unknown"

      # use the known face with the smallest distance to the new face
      face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
          name = known_face_names[best_match_index]

      # Draw a box around the face
      cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

      # Draw a label with a name below the face
      cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
      font = cv2.FONT_HERSHEY_DUPLEX
      cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

  # Display the resulting image
  #cv2.imshow('Image', frame)
  pil_image = Image.fromarray(image)
  pil_image.show()

if __name__ == "__main__":
  main()