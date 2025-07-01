import cv2
import joblib
import mediapipe as mp
import time
import math

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.high_level_commander import HighLevelCommander
from cflib.crazyflie.log import LogConfig  # Correct import for logging

# === SETUP ===
URI = 'radio://0/80/2M'
cflib.crtp.init_drivers()

# === Hand Landmarks Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# === Load Gesture Model ===
model = joblib.load("gesture_knn_model.pkl")

# === Flowdeck Stability Check ===
def wait_for_position_estimator(scf):
    print("📱 Waiting for Flow deck position estimator to stabilize...")
    log_config = LogConfig(name='Kalman', period_in_ms=500)
    log_config.add_variable('kalman.stateZ', 'float')
    log_config.add_variable('stabilizer.roll', 'float')

    with SyncLogger(scf, log_config) as logger:
        stable_count = 0
        for log_entry in logger:
            z = log_entry[1]['kalman.stateZ']
            roll = abs(log_entry[1]['stabilizer.roll'])
            print(f"kalman.stateZ: {z:.2f} | roll: {roll:.2f} | stable: {stable_count}")
            if 0.01 < z < 2.0 and roll < 20:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count > 5:
                break

# === Distance Reading (MultiRanger) ===
class Ranger:
    def __init__(self, cf):
        self.distances = {'front': None, 'back': None, 'left': None, 'right': None}
        self._setup_logger(cf)

    def _setup_logger(self, cf):
        log_conf = LogConfig(name='ranger', period_in_ms=100)
        for d in ['front', 'back', 'left', 'right']:
            log_conf.add_variable(f'range.{d}', 'float')
        self.cf = cf
        self.cf.log.add_config(log_conf)
        log_conf.data_received_cb.add_callback(self._ranger_callback)
        log_conf.start()

    def _ranger_callback(self, timestamp, data, logconf):
        for d in self.distances:
            self.distances[d] = data.get(f'range.{d}', None)

    def is_clear(self, direction, min_dist=300):
        d = self.distances.get(direction)
        return d is not None and d > min_dist

# === Utility ===
def calc_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def extract_landmarks(result):
    all_landmarks = []
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            base = hand.landmark[0]
            ref = hand.landmark[12]
            scale = calc_distance(base, ref) or 1e-6
            for lm in hand.landmark:
                all_landmarks.extend([
                    (lm.x - base.x) / scale,
                    (lm.y - base.y) / scale,
                    (lm.z - base.z) / scale
                ])
    while len(all_landmarks) < 126:
        all_landmarks.append(0.0)
    return all_landmarks

def clamp_position(pos):
    return [
        max(-1.5, min(1.5, pos[0])),
        max(-1.5, min(1.5, pos[1])),
        max(0.1, min(1.5, pos[2]))
    ]

# === MAIN ===
with SyncCrazyflie(URI, cf=Crazyflie(rw_cache=None)) as scf:
    commander = scf.cf.high_level_commander
    ranger = Ranger(scf.cf)

    wait_for_position_estimator(scf)
    print("✅ Estimator ready. Starting gesture control.")

    cap = cv2.VideoCapture(0)
    last_prediction = None
    last_action_time = 0
    cooldown = 3
    has_taken_off = False
    current_pos = [0.0, 0.0, 0.5]

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            landmarks = extract_landmarks(result)

            if sum(landmarks) == 0.0:
                cv2.putText(frame, "Gesture: None", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Gesture Control", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                continue

            prediction = model.predict([landmarks])[0]
            now = time.time()

            if prediction != last_prediction or (now - last_action_time > cooldown):
                print(f"🧠 Gesture: {prediction}")

                if prediction == "takeoff" and not has_taken_off:
                    print("✈️ Taking off...")
                    commander.takeoff(current_pos[2], 2.0)
                    time.sleep(3)
                    has_taken_off = True

                elif prediction == "land" and has_taken_off:
                    print("🛬 Landing...")
                    commander.land(0.0, 2.0)
                    time.sleep(3)
                    has_taken_off = False

                elif prediction == "spin" and has_taken_off:
                    print("🌀 Full 180° spin (Flow deck)")
                    commander.go_to(*current_pos, 180.0, 2.0)
                    time.sleep(3)
                    commander.go_to(*current_pos, -180.0, 2.0)
                    time.sleep(3)
                    commander.go_to(*current_pos, 0.0, 2.0)

                elif prediction == "excited" and has_taken_off:
                    commander.go_to(current_pos[0], current_pos[1], current_pos[2] + 0.3, 0.0, 1.0)
                    time.sleep(1)
                    commander.go_to(*current_pos, 0.0, 1.0)

                elif prediction == "stop" and has_taken_off:
                    commander.land(0.0, 2.0)
                    time.sleep(3)
                    has_taken_off = False

                elif has_taken_off:
                    if prediction == "forward" and ranger.is_clear('front'):
                        current_pos[1] += 0.3
                    elif prediction == "back" and ranger.is_clear('back'):
                        current_pos[1] -= 0.3
                    elif prediction == "left" and ranger.is_clear('left'):
                        current_pos[0] -= 0.3
                    elif prediction == "right" and ranger.is_clear('right'):
                        current_pos[0] += 0.3
                    elif prediction == "up":
                        current_pos[2] += 0.3
                    elif prediction == "down":
                        current_pos[2] = max(0.1, current_pos[2] - 0.3)

                    current_pos = clamp_position(current_pos)
                    commander.go_to(*current_pos, 0.0, 2.0)

                last_action_time = now
                last_prediction = prediction

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"Gesture: {prediction}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Gesture Control", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if has_taken_off:
            commander.land(0.0, 2.0)
            time.sleep(3)
        scf.__exit__(None, None, None)
        print("✅ Disconnected safely.")