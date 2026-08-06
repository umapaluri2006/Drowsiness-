import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import winsound

# --------------------------
# Alarm Function
# --------------------------

alarm_on = False

def sound_alarm():
    global alarm_on
    while alarm_on:
        winsound.Beep(1000, 500)
    

# --------------------------
# Eye Aspect Ratio Function
# --------------------------
def calculate_EAR(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --------------------------
# Initialize MediaPipe
# --------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# --------------------------
# Start Webcam (Laptop Camera)
# --------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("Drowsiness Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Drowsiness Detection",
                      cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)

EAR_THRESHOLD = 0.25
CLOSED_EYES_TIME = 2  # seconds
start_time = None
alarm_on = False

print("Drowsiness Detection Started... Press 'Q' to exit")
def adjust_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    if brightness < 60:   # Dark environment
        frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
        night_mode = True
    else:
        night_mode = False

    return frame, night_mode

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame, night_mode = adjust_brightness(frame)

    if night_mode:
        cv2.putText(frame, "🌙 NIGHT MODE ON",
                (30, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            h, w, _ = frame.shape

            # Left Eye Landmark Points
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            left_eye = []
            right_eye=[]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            right_eye_indices = [362, 385, 387, 263, 373, 380]

            for idx in left_eye_indices:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                left_eye.append(np.array([x, y]))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            for idx in right_eye_indices:
                
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                right_eye.append(np.array([x, y]))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)


            left_eye = np.array(left_eye)
            right_eye = np.array(right_eye)
            left_ear = calculate_EAR(left_eye)
            right_ear = calculate_EAR(right_eye)
            ear = (left_ear + right_ear) / 2.0
            

            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Check if eyes are closed
            if ear < EAR_THRESHOLD:
                
                if start_time is None:
                    
                    start_time = time.time()

                elapsed = time.time() - start_time

                if elapsed >= CLOSED_EYES_TIME:
                    cv2.putText(frame, "OPEN YOUR EYES", (200, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                    if not alarm_on:
                        alarm_on = True
                        t = threading.Thread(target=sound_alarm)
                        t.daemon = True
                        t.start()
            else:
                start_time = None
                alarm_on = False

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
