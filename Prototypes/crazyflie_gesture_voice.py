import time, json, queue, cv2, math, joblib
import sounddevice as sd
import mediapipe as mp
import pandas as pd
from vosk import Model, KaldiRecognizer
from sentence_transformers import SentenceTransformer, util

from cflib.crtp import init_drivers
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.high_level_commander import HighLevelCommander
from cflib.crazyflie.log import LogConfig

# === Setup ===
URI = 'radio://0/80/2M'
init_drivers()

# === Voice ===
vosk_model = Model("model")
recognizer = KaldiRecognizer(vosk_model, 16000)
audio_q = queue.Queue()
intent_model = SentenceTransformer('all-MiniLM-L6-v2')

intent_examples = {
    "takeoff": ["take off", "please take off", "can you take off", "lift off", "start flying"],
    "land": ["land", "please land", "can you land", "stop flying", "touch down"],
    "forward": ["go forward", "move forward", "fly forward"],
    "back": ["go back", "move back", "fly backward"],
    "left": ["go left", "move left", "fly to the left"],
    "right": ["go right", "move right", "fly to the right"],
    "up": ["go up", "fly higher", "ascend", "climb up"],
    "down": ["go down", "fly lower", "descend"],
    "excited": ["get excited", "do a jump", "show excitement", "bounce"],
    "happy": ["be happy", "do a happy dance", "wiggle", "celebrate"],
    "sad": ["look sad", "be sad", "descend sadly"],
    "spin": ["spin", "spin around", "twirl"],
    "shake": ["shake", "shake your head", "wiggle head"],
    "come here": ["come here", "fly to me", "come closer", "approach me"],
    "stop": ["stop", "halt", "land now", "end movement"]
}
flat_examples, intent_labels = [], []
for k, v in intent_examples.items():
    flat_examples.extend(v)
    intent_labels.extend([k]*len(v))
example_embeddings = intent_model.encode(flat_examples)

# === Gesture ===
gesture_model = joblib.load("gesture_knn_model.pkl")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

def audio_callback(indata, frames, time, status):
    if status:
        print("Audio error:", status)
    audio_q.put(bytes(indata))

def wait_for_position_estimator(scf):
    log_conf = LogConfig(name='Kalman', period_in_ms=500)
    log_conf.add_variable('kalman.stateZ', 'float')
    log_conf.add_variable('stabilizer.roll', 'float')
    with SyncLogger(scf, log_conf) as logger:
        stable = 0
        for entry in logger:
            z = entry[1]['kalman.stateZ']
            roll = abs(entry[1]['stabilizer.roll'])
            print(f"Waiting... Z={z:.2f} | Roll={roll:.2f}")
            if 0.01 < z < 2.0 and roll < 20:
                stable += 1
            else:
                stable = 0
            if stable > 5:
                break

def extract_landmarks(result):
    landmarks = []
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            base = hand.landmark[0]
            ref = hand.landmark[12]
            scale = math.dist((base.x, base.y, base.z), (ref.x, ref.y, ref.z)) or 1e-6
            for lm in hand.landmark:
                landmarks.extend([
                    (lm.x - base.x) / scale,
                    (lm.y - base.y) / scale,
                    (lm.z - base.z) / scale
                ])
    while len(landmarks) < 126:
        landmarks.append(0.0)
    return landmarks

def local_ai_intent(text):
    embedding = intent_model.encode(text)
    sim = util.cos_sim(embedding, example_embeddings)[0]
    idx = int(sim.argmax())
    confidence = float(sim[idx])
    print(f"🧠 Intent match: {intent_labels[idx]} ({confidence:.2f})")
    return intent_labels[idx] if confidence > 0.35 else None

def clamp(pos):
    return [max(-1.5, min(1.5, pos[0])), max(-1.5, min(1.5, pos[1])), max(0.1, min(1.5, pos[2]))]

# === Main ===
with SyncCrazyflie(URI, cf=Crazyflie(rw_cache=None)) as scf:
    commander = scf.cf.high_level_commander
    wait_for_position_estimator(scf)
    print("✅ Ready!")

    cap = cv2.VideoCapture(0)
    last_time = 0
    cooldown = 3
    taken_off = False
    current_pos = [0.0, 0.0, 0.5]
    mood = "neutral"
    last_interaction = time.time()
    idle_check = time.time() + 5

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=audio_callback):
        try:
            while cap.isOpened():
                _, frame = cap.read()
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                landmarks = extract_landmarks(result)

                # Voice
                text, intent = None, None
                if not audio_q.empty():
                    data = audio_q.get()
                    if recognizer.AcceptWaveform(data):
                        text = json.loads(recognizer.Result()).get("text", "").lower()
                        if text:
                            intent = local_ai_intent(text)

                # Gesture
                gesture = None
                if sum(landmarks) != 0.0 and len(landmarks) == 126:
                    try:
                        cols = [f'x{i}' for i in range(126)]
                        X_input = pd.DataFrame([landmarks], columns=cols)
                        gesture = gesture_model.predict(X_input)[0]
                        print("🖐️ Gesture recognized:", gesture)
                    except Exception as e:
                        print("⚠️ Gesture prediction error:", e)

                command = intent or gesture
                now = time.time()

                # Execute command
                if command and now - last_time > cooldown:
                    last_time = now
                    last_interaction = now

                    if command in ["happy", "excited", "spin"]:
                        mood = "happy"
                    elif command == "land":
                        mood = "neutral"

                    if command == "takeoff" and not taken_off:
                        commander.takeoff(current_pos[2], 2.0)
                        time.sleep(3)
                        taken_off = True
                    elif command == "land" and taken_off:
                        commander.land(0.0, 2.0)
                        time.sleep(3)
                        taken_off = False
                    elif taken_off:
                        if command == "forward":
                            current_pos[1] += 0.3
                            commander.go_to(*current_pos, 0.0, 2.0)
                        elif command == "back":
                            current_pos[1] -= 0.3
                            commander.go_to(*current_pos, 0.0, 2.0)
                        elif command == "left":
                            current_pos[0] -= 0.3
                            commander.go_to(*current_pos, 0.0, 2.0)
                        elif command == "right":
                            current_pos[0] += 0.3
                            commander.go_to(*current_pos, 0.0, 2.0)
                        elif command == "up":
                            current_pos[2] += 0.3
                            commander.go_to(*current_pos, 0.0, 2.0)
                        elif command == "down":
                            current_pos[2] = max(0.1, current_pos[2] - 0.3)
                            commander.go_to(*current_pos, 0.0, 2.0)
                        elif command == "excited":
                            commander.go_to(current_pos[0], current_pos[1], current_pos[2] + 0.4, 0.0, 1.0)
                            time.sleep(1)
                            commander.go_to(*current_pos, 0.0, 1.0)
                        elif command == "happy":
                            commander.go_to(current_pos[0] - 0.1, current_pos[1], current_pos[2] + 0.2, 0.0, 1.0)
                            time.sleep(1)
                            commander.go_to(current_pos[0] + 0.2, current_pos[1], current_pos[2], 0.0, 1.0)
                            time.sleep(1)
                            commander.go_to(*current_pos, 0.0, 1.0)
                        elif command == "sad":
                            commander.go_to(current_pos[0], current_pos[1], max(0.2, current_pos[2] - 0.3), 0.0, 2.0)
                            time.sleep(2)
                            commander.go_to(*current_pos, 0.0, 2.0)
                        elif command == "spin":
                            commander.go_to(*current_pos, 180.0, 2.0)
                            time.sleep(3)
                            commander.go_to(*current_pos, -180.0, 2.0)
                            time.sleep(3)
                            commander.go_to(*current_pos, 0.0, 2.0)
                        elif command == "stop":
                            commander.land(0.0, 2.0)
                            time.sleep(3)
                            taken_off = False
                        else:
                            current_pos = clamp(current_pos)
                            commander.go_to(*current_pos, 0.0, 2.0)

                # Idle behavior
                if time.time() > idle_check:
                    idle_check = time.time() + 5
                    if time.time() - last_interaction > 20 and mood == "happy":
                        mood = "bored"
                        print("😐 Feeling a little bored...")
                    elif time.time() - last_interaction > 40 and mood == "bored":
                        mood = "sad"
                        print("😢 Feeling ignored...")

                    if taken_off:
                        if mood == "bored":
                            commander.go_to(*current_pos, 180.0, 2.0)
                            time.sleep(2)
                            commander.go_to(*current_pos, -180.0, 2.0)
                            time.sleep(2)
                            commander.go_to(*current_pos, 0.0, 1.0)
                        elif mood == "sad":
                            commander.go_to(current_pos[0], current_pos[1], max(0.1, current_pos[2] - 0.2), 0.0, 1.0)
                            time.sleep(2)
                            commander.go_to(*current_pos, 0.0, 1.0)

                # Draw + display
                if result.multi_hand_landmarks:
                    for hand in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, f"Gesture: {gesture or 'None'}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.imshow("Gesture + Voice", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if taken_off:
                commander.land(0.0, 2.0)
                time.sleep(3)
            scf.__exit__(None, None, None)
            print("✅ Landed & Disconnected")
