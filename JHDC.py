import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

min_y = float('inf')
max_y = float('-inf')
pixel_to_cm_ratio = None

KNOWN_HEIGHT_CM = 170

LANDMARK_INDEX = mp_pose.PoseLandmark.LEFT_HIP

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        h, w, _ = frame.shape

        landmark = results.pose_landmarks.landmark[LANDMARK_INDEX]
        y_coord = landmark.y * h
        
        min_y = min(min_y, y_coord)
        max_y = max(max_y, y_coord)

        cx, cy = int(landmark.x * w), int(y_coord)
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)

        if pixel_to_cm_ratio is None:
            shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
            foot_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * h
            height_in_pixels = abs(foot_y - shoulder_y)
            pixel_to_cm_ratio = KNOWN_HEIGHT_CM / height_in_pixels if height_in_pixels > 0 else None

        jump_height_px = max_y - min_y
        if pixel_to_cm_ratio:
            jump_height_cm = jump_height_px * pixel_to_cm_ratio
            cv2.putText(frame, f"Jump Height: {jump_height_cm:.2f} cm", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Jump Height Estimation", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
