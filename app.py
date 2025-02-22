from flask import Flask, render_template, Response, jsonify
import cv2
import face_recognition
import os
import numpy as np
import csv
from datetime import datetime

app = Flask(__name__)

data_path = 'training_images/'
attendance_file = 'Attendance.csv'

if not os.path.exists(attendance_file):
    with open(attendance_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Time'])

known_faces = []
known_names = []

for file in os.listdir(data_path):
    image = face_recognition.load_image_file(f"{data_path}/{file}")
    encodings = face_recognition.face_encodings(image)
    if encodings:
        encoding = encodings[0]
        known_faces.append(encoding)
        known_names.append(os.path.splitext(file)[0])

def mark_attendance(name):
    with open(attendance_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, datetime.now().strftime('%H:%M:%S')])

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for encoding, location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_faces, encoding)
                name = "Unknown"
                
                if True in matches:
                    match_index = np.argmin(face_recognition.face_distance(known_faces, encoding))
                    name = known_names[match_index]
                    mark_attendance(name)
                
                top, right, bottom, left = location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def get_attendance():
    with open(attendance_file, 'r') as file:
        data = file.readlines()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
