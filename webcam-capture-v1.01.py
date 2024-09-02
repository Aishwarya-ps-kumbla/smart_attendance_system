import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

def capture_and_save_images(num_of_students, resize_to):
    """Captures and saves multiple images of different persons from the webcam.

    Args:
        num_images (int): The number of images to capture.
        resize_to (tuple): The desired image size (width, height).
    """

    cap = cv2.VideoCapture(0)  # Open the default webcam
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Create a directory to store the captured images
    image_dir = "train_images"
    test_dir = "test_images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(test_dir):
      os.makedirs(test_dir)
        
    images = []
    labels = []

    for student_num in range(1, num_of_students + 1):
        person_name = input(f"Enter the name for student {student_num}: ")

        for image_num in range(5):
            ret, frame = cap.read()  # Capture a frame from the webcam
            if not ret:
                print("Error capturing frame")
                break

            # Resize the frame if needed
            if resize_to:
                frame = cv2.resize(frame, resize_to)

            # Normalize the image (e.g., to a range of 0-1)
            frame = frame / 255.0
            
            images.append(frame)
            labels.append(person_name)



            ret, frame = cap.read()  # Capture a frame from the webcam
            if not ret:
                print("Error capturing frame")
                break


            # Save the frame as an image with the person's name
            image_path = os.path.join(image_dir, f"{person_name}_{image_num}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Image saved to {image_path}")

     # Split the dataset into training, validation, and test sets (assuming sufficient data)
    if len(images) > 0:  # Check if any images were captured
      X_train, X_val, y_train, y_val = train_test_split(
          np.array(images),
          np.array(labels),
          test_size=0.2,
          random_state=42
      )

      # Save the datasets
      np.savez("train_data.npz", X_train=X_train, y_train=y_train)
      np.savez("val_data.npz", X_val=X_val, y_val=y_val)

      # Create test images
      test_images = np.array(images)[len(X_train):]
      test_labels = np.array(labels)[len(y_train):]

      # Save test images individually
      for i in range(len(test_images)):
          image_path = os.path.join(test_dir, f"test_image_{i+1}.jpg")
          cv2.imwrite(image_path, test_images[i])
          print(f"Test image {i+1} saved to {image_path}")

    


    cap.release()  # Release the webcam
    cv2.destroyAllWindows()

# Example usage:
capture_and_save_images(3, (224, 224))

