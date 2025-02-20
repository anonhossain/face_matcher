# image_encoder_and_matcher.py
import face_recognition
import cv2
import os
import numpy as np
import sqlite3
import shutil

class ImageEncoderAndMatcher:
    def __init__(self, db_path="encoded.db", result_folder="result_pictures"):
        self.db_path = db_path
        self.result_folder = result_folder
        os.makedirs(result_folder, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        # Create the table for storing face encodings and image paths
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS face_data (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                image_path TEXT NOT NULL,
                                encoding BLOB NOT NULL)''')
        self.conn.commit()

    def encode_faces(self, image_folder):
        for filename in os.listdir(image_folder):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(image_folder, filename)

                # Load the image
                img = cv2.imread(image_path)

                # Convert to RGB
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Detect faces
                face_locations = face_recognition.face_locations(rgb_img)
                print(f"Detected {len(face_locations)} face(s) in {filename}")

                # Encode the faces
                face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

                # Save encodings and image path to the database
                for encoding in face_encodings:
                    encoding_blob = np.array(encoding).tobytes()
                    self.cursor.execute("INSERT INTO face_data (image_path, encoding) VALUES (?, ?)", (image_path, encoding_blob))
                    self.conn.commit()
                    print(f"Saved encoding for {filename}")

        print("Face encoding complete. Encoded faces are saved in the database.")

    def match_faces(self, target_folder, tolerance=0.6):
        result_images = []
        for filename in os.listdir(target_folder):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(target_folder, filename)

                # Load the target image
                target_img = cv2.imread(image_path)
                rgb_target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

                # Detect faces in the target image
                target_face_locations = face_recognition.face_locations(rgb_target_img)
                print(f"Detected {len(target_face_locations)} face(s) in {filename}")

                # Encode the detected faces
                target_face_encodings = face_recognition.face_encodings(rgb_target_img, target_face_locations)

                if not target_face_encodings:
                    print(f"No faces found in {filename}.")
                    continue

                for target_encoding in target_face_encodings:
                    # Retrieve all stored encodings from the database
                    self.cursor.execute("SELECT image_path, encoding FROM face_data")
                    rows = self.cursor.fetchall()

                    found_match = False
                    for row in rows:
                        stored_image_path = row[0]
                        stored_encoding_blob = row[1]
                        stored_encoding = np.frombuffer(stored_encoding_blob, dtype=np.float64)

                        # Compare the target encoding with stored encodings
                        matches = face_recognition.compare_faces([stored_encoding], target_encoding, tolerance=tolerance)

                        if True in matches:
                            print(f"Match found: {stored_image_path}")
                            found_match = True

                            # Copy the matched image to the result_pictures folder
                            if os.path.exists(stored_image_path):
                                dest_path = os.path.join(self.result_folder, os.path.basename(stored_image_path))
                                shutil.copy(stored_image_path, dest_path)
                                result_images.append(dest_path)
                                print(f"Copied {stored_image_path} to {self.result_folder}")

                    if not found_match:
                        print(f"No matches found for the face in {filename}.")
        return result_images

    def clean_up(self, image_folder):
        # Delete images from the 'all_pictures' folder and database
        for filename in os.listdir(image_folder):
            os.remove(os.path.join(image_folder, filename))
        os.rmdir(image_folder)

        # Drop the face_data table from the database
        self.cursor.execute("DROP TABLE IF EXISTS face_data")
        self.conn.commit()

    def close_connection(self):
        self.conn.close()
