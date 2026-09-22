"""
Simple Object Detection Proof of Concept
Uses OpenCV's built-in Haar Cascade classifier for face detection.
"""

import cv2
import urllib.request
import os

def main():
    # Download a sample test image if we don't already have one
    image_path = "face_sample.jpg"
    if not os.path.exists(image_path):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
        urllib.request.urlretrieve(url, image_path)
        print(f"Downloaded sample image to {image_path}")

    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load OpenCV's built-in pretrained face detector (ships with the library)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Run detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print(f"Detected {len(faces)} face(s)")

    # Draw bounding boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(f"  Box: x={x}, y={y}, width={w}, height={h}")

    # Save and show the result
    cv2.imwrite("output.jpg", img)
    print("Saved annotated image as output.jpg")

    cv2.imshow("Detection Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()