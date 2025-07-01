import time
import cv2
import mediapipe as mp
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.high_level_commander import HighLevelCommander

# ========== CRAZYFLIE SETUP ==========
URI = 'radio://0/80/2M'
cflib.crtp.init_drivers()

# Start connection
print("🔗 Connecting to Crazyflie...")
cf = Crazyflie(rw_cache=None)
scf = SyncCrazyflie(URI, cf)
scf.__enter__()
commander = scf.cf.high_level_commander
print("✅ Connected!")
time.sleep(2)

# ========== GESTURE SETUP ==========
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

last_gesture_time = 0
cooldown = 3  # seconds
has_taken_off = False
current_pos = [0.0, 0.0, 0.5]

def fingers_extended(hand):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    extended = 0
    for tip, pip in zip(tips, pips):
        if hand.landmark[tip].y < hand.landmark[pip].y:
            extended += 1
    return extended

def classify_gesture(result):
    if not result.multi_hand_landmarks:
        return "none"

    hands_count = len(result.multi_hand_landmarks)
    landmarks = result.multi_hand_landmarks[0]

    extended = fingers_extended(landmarks)

    if hands_count >= 2:
        return "both hands"
    elif extended == 0:
        return "fist"
    elif extended >= 4:
        return "open"
    elif extended == 2:
        return "victory"
    else:
        return "none"

def handle_gesture(gesture):
    global has_taken_off, current_pos

    print(f"🤖 Recognized gesture: {gesture}")

    if gesture == "open" and not has_taken_off:
        print("✈️ Taking off...")
        commander.takeoff(current_pos[2], 2.0)
        time.sleep(3)
        has_taken_off = True

    elif gesture == "fist" and has_taken_off:
        print("🛬 Landing...")
        commander.land(0.0, 2.0)
        time.sleep(3)
        has_taken_off = False

    elif gesture == "victory" and has_taken_off:
        print("🎉 Excited jump!")
        commander.go_to(current_pos[0], current_pos[1], current_pos[2] + 0.4, 0.0, 1.0)
        time.sleep(1)
        commander.go_to(*current_pos, 0.0, 1.0)

    elif gesture == "both hands" and has_taken_off:
        print("😊 Happy wiggle!")
        commander.go_to(current_pos[0] - 0.1, current_pos[1], current_pos[2] + 0.2, 0.0, 1.0)
        time.sleep(1)
        commander.go_to(current_pos[0] + 0.2, current_pos[1], current_pos[2], 0.0, 1.0)
        time.sleep(1)
        commander.go_to(*current_pos, 0.0, 1.0)

# ========== MAIN LOOP ==========
print("🖐️ Show gestures to control the drone (open = takeoff, fist = land)...")

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture = classify_gesture(result)

        if gesture != "none" and (time.time() - last_gesture_time) > cooldown:
            handle_gesture(gesture)
            last_gesture_time = time.time()

        # Draw hands + gesture label
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Gesture Control", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("👋 Exiting...")
finally:
    cap.release()
    cv2.destroyAllWindows()
    commander.land(0.0, 2.0)
    time.sleep(3)
    scf.__exit__(None, None, None)
