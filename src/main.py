import face_recognition
from PIL import Image, ImageDraw

def main():
  print("Meme machine now online")
  image = face_recognition.load_image_file("src/people.jpg")
  face_locations = face_recognition.face_locations(image)

  im = Image.open("src/people.jpg")
  draw = ImageDraw.Draw(im)
  
  for loc in face_locations:
    draw.rectangle(loc, fill=None, outline=128)

  del draw

  im.show()

if __name__ == "__main__":
  main()