import cv2
import mediapipe as mp
import csv
import math

# === Setup MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# === Webcam Feed ===
cap = cv2.VideoCapture(0)
data = []

# === Gesture Key Mapping ===
label_map = {
    ord('t'): 'takeoff',    # Open hand
    ord('l'): 'land',       # Fist
    ord('u'): 'up',
    ord('d'): 'down',
    ord('h'): 'happy',
    ord('s'): 'sad',      # Two hands
    ord('a'): 'left',
    ord('r'): 'right'
}

current_label = None
print("📷 Press: [t]akeoff, [l]and, [u]p, [d]own, [h]appy, [s]ad, [a]left, [r]ight — [q] to quit and save.")

# === Helper to calculate distance ===
def calc_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)

# === Recording Loop ===
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    all_landmarks = []

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            base = hand.landmark[0]
            ref = hand.landmark[12]
            scale = calc_distance(base, ref)
            if scale == 0:
                scale = 1e-6  # Avoid divide-by-zero

            for lm in hand.landmark:
                all_landmarks.extend([
                    (lm.x - base.x) / scale,
                    (lm.y - base.y) / scale,
                    (lm.z - base.z) / scale
                ])

    # Pad to always have 2 hands × 21 landmarks = 126 values
    while len(all_landmarks) < 126:
        all_landmarks.append(0.0)

    if current_label and any(all_landmarks):  # Only record if hand(s) detected
        data.append(all_landmarks + [current_label])

    # Display current label
    cv2.putText(frame, f"Label: {current_label or 'None'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recorder", frame)

    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    elif key in label_map:
        current_label = label_map[key]
        print(f"📝 Labeling frames as '{current_label}'")

# === Save to CSV ===
cap.release()
cv2.destroyAllWindows()

filename = "gesture_data.csv"
with open(filename, 'a', newline='') as f:
    writer = csv.writer(f)
    #writer.writerow([f"x{i}" for i in range(126)] + ["label"])  # 2 hands × 21 points × 3D
    writer.writerows(data)

print(f"✅ Saved {len(data)} samples to {filename}")
