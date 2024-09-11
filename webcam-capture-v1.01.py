
    

import sqlite3
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import load
from numpy import asarray
from numpy import expand_dims
from numpy import savez_compressed
from numpy import reshape
from keras.models import load_model 
from datetime import date
from datetime import datetime
import joblib
from sklearn.neighbors import KNeighborsClassifier
#import db


#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")



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
    if not os.path.isdir('Attendance'):
        os.makedirs('Attendance')
    if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
        with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
            f.write('Name,Roll,Time')
            
    
        
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
            
    #### get a number of total registered users

    def totalreg():
        return len(os.listdir(r'C:\Users\HP\Desktop\webcam\train_images'))
    
    def extract_faces(img):
        if img!=[]:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_points = face_detector.detectMultiScale(gray, 1.3, 5)
            return face_points
        else:return []
        
    #### Identify face using ML model
    def identify_face(facearray):
        file_path = os.path.join('C:\\Users\\HP\\Desktop\\webcam', 'train_images')
        model = joblib.load(file_path, encoding='utf-8')
        return model.predict(facearray)
    
    #### A function which trains the model on all the faces available in faces folder
    def train_model():
        faces = []
        labels = []
        userlist = os.listdir('C:\\Users\\HP\\Desktop\\webcam', 'train_images')
        for user in userlist:
            for imgname in os.listdir(f'C:\\Users\\HP\\Desktop\\webcam', 'train_images/{user}'):
                img = cv2.imread(f'C:\\Users\\HP\\Desktop\\webcam', 'train_images/{user}/{imgname}')
                resized_face = cv2.resize(img, (50, 50))
                faces.append(resized_face.ravel())
                labels.append(user)
        faces = np.array(faces)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(faces,labels)
        file_path = os.path.join('C:\\Users\\HP\\Desktop\\webcam', 'face_recognition_model.pkl')
        joblib.dump(knn, file_path, encoding='utf-8')
        
    #### Extract info from today's attendance file in attendance folder
    def extract_attendance():
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        names = df['Name']
        rolls = df['Roll']
        times = df['Time']
        l = len(df)
        return names,rolls,times,l

    #### Add Attendance of a specific user
    def add_attendance(name):
        username = name.split('_')[0]
        userid = name.split('_')[1]
        current_time = datetime.now().strftime("%H:%M:%S")
    
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        if str(userid) not in list(df['Roll']):
            with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
                f.write(f'\n{username},{userid},{current_time}')
        else:
            print("this user has already marked attendence for the day , but still i am marking it ")
        # with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
        #     f.write(f'\n{username},{userid},{current_time}')



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

