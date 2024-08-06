import cv2
import time
import pyttsx3
import numpy as np
import mediapipe as mp
import dlib
from num2words import num2words
from deepface import DeepFace

# Initialize the video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize the face cascade and DeepFace
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
detector = dlib.get_frontal_face_detector()

# Initialize text-to-speech engine
engine = pyttsx3.init()
def speak_number(number):
    word = num2words(number, to='ordinal')
    engine.say(word)
    engine.runAndWait()

# Initialize variables
vertical_area = [(320, 0), (320, 480)]
counter = 0
person_crossed = False
no_detection_start_time = None
detection_threshold = 5

# Function to detect faces using MediaPipe
def detect_faces(frame, face_detection):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)
    if results.detections:
        return True, results.detections
    else:
        return False, None

# Function to detect hands using MediaPipe
def detect_hands(frame, hand_detection):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_detection.process(frame_rgb)
    if results.multi_hand_landmarks:
        return True, results.multi_hand_landmarks
    else:
        return False, None

# Function to map emotion
def map_emotion(emotion):
    if emotion in ['happy']:
        return 'Happy'
    elif emotion in ['sad', 'angry', 'disgust', 'fear']:
        return 'Sad'
    elif emotion in ['neutral', 'surprise']:
        return 'Neutral'
    else:
        return 'Neutral'

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection with Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    detected_face, detections = detect_faces(frame, face_detection)
    detected_hand, hand_landmarks = detect_hands(frame, hands)

    # Count persons crossing the vertical line
    for (x, y, w, h) in faces:
        if x < vertical_area[0][0] < x + w:
            if not person_crossed:
                counter += 1
                speak_number(counter)
                person_crossed = True
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            person_crossed = False

    # Hand detection with MediaPipe
    if detected_face and not detected_hand:
        display_text = "Sitting"
        no_detection_start_time = None
    elif detected_hand:
        display_text = "Namastey"
        no_detection_start_time = None
    else:
        if no_detection_start_time is None:
            no_detection_start_time = time.time()
        elif time.time() - no_detection_start_time > detection_threshold:
            display_text = "Head Down"
        else:
            display_text = "Neutral"

    if display_text:
        cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Emotion detection with DeepFace
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        try:
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list):
                emotion = result[0]['dominant_emotion']
            else:
                emotion = result['dominant_emotion']

            mapped_emotion = map_emotion(emotion)

            if mapped_emotion == 'Happy':
                color = (0, 255, 0)
            elif mapped_emotion == 'Sad':
                color = (0, 0, 255)
            else:
                color = (255, 255, 0)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, mapped_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        except Exception as e:
            print("Error analyzing face:", e)

    # Drawing vertical area line and showing count
    count_word = num2words(counter, to='ordinal')
    cv2.putText(frame, f'Persons Crossing Vertical Area: {count_word}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.line(frame, vertical_area[0], vertical_area[1], (0, 0, 255), 2)

    # Detect and draw faces using dlib
    rects = detector(gray, 1)
    for i, face_rect in enumerate(rects):
        left = face_rect.left()
        top = face_rect.top()
        width = face_rect.right() - left
        height = face_rect.bottom() - top
        cv2.rectangle(frame, (left, top), (left+width, top+height), (0, 255, 0), 2)
        cv2.putText(frame, f"Face {i+1}", (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Show frame
    cv2.imshow('Frame', frame)

    if person_crossed:
        time.sleep(0.25)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
