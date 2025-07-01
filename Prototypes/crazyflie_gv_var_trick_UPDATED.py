import time, json, queue, cv2, math, joblib, re
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
URI = 'radio://0/80/2M'
init_drivers()
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
    "excited": ["get excited", "do a jump", "show excitement", "bounce", "good"],
    "happy": ["be happy", "do a happy dance", "wiggle", "celebrate"],
    "sad": ["look sad", "be sad", "descend sadly"],
    "spin": ["spin", "spin around", "twirl"],
    "shake": ["shake", "shake your head", "wiggle head"],
    "come here": ["come here", "fly to me", "come closer", "approach me"],
    "stop": ["stop", "halt", "land now", "end movement"],
    "learn_trick": ["learn a new trick", "teach a new trick", "create a command"],
    "end_trick": ["end trick", "and trick", "finish trick", "done with trick", "save trick"]
}
flat_examples, intent_labels = [], []
for k, v in intent_examples.items():
    flat_examples.extend(v)
    intent_labels.extend([k]*len(v))
example_embeddings = intent_model.encode(flat_examples)
gesture_model = joblib.load("gesture_knn_model.pkl")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils
learning_mode = False
learned_trick_name = None
learned_trick_actions = []
saved_tricks = {}
def audio_callback(indata, frames, time, status):
    if status:
        print("Audio error:", status)
    audio_q.put(bytes(indata))
def extract_distance(text):
    word_to_number = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "half": 0.5, "quarter": 0.25
    }
    match = re.search(r'\b(\d+(\.\d+)?)\b', text)
    if match:
        return float(match.group(1))
    for word, number in word_to_number.items():
        if word in text.lower():
            return float(number)
    return None
def local_ai_intent(text):
    embedding = intent_model.encode(text)
    sim = util.cos_sim(embedding, example_embeddings)[0]
    idx = int(sim.argmax())
    confidence = float(sim[idx])
    best_intent = intent_labels[idx]
    print(f" Intent match: {best_intent} ({confidence:.2f})")
    keywords = ["forward", "back", "left", "right", "up", "down", "spin", "shake"]
    for kw in keywords:
        if kw in text:
            print(f" Keyword override: '{kw}' detected in text")
            return kw
    return best_intent if confidence > 0.50 else None
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
def perform_command(command, commander, current_pos, taken_off, move=0.3):
    if not move:
        move = 0.3
    if command == "takeoff" and not taken_off:
        print("️ Taking off...")
        commander.takeoff(current_pos[2], 2.0)
        time.sleep(3)
        taken_off = True
    elif command == "land" and taken_off:
        print(" Landing...")
        commander.land(0.0, 2.0)
        time.sleep(3)
        taken_off = False
    elif taken_off:
        if command == "forward":
            current_pos[1] += move
            print(f"⬆️ Moving forward {move:.2f}m")
        elif command == "back":
            current_pos[1] -= move
            print(f"⬇️ Moving back {move:.2f}m")
        elif command == "left":
            current_pos[0] -= move
            print(f"⬅️ Moving left {move:.2f}m")
        elif command == "right":
            current_pos[0] += move
            print(f"️ Moving right {move:.2f}m")
        elif command == "up":
            current_pos[2] += move
            print(f" Ascending {move:.2f} meters")
        elif command == "down":
            current_pos[2] = max(0.1, current_pos[2] - move)
            print(f" Descending {move:.2f} meters")
        elif command == "sad":
            commander.go_to(current_pos[0], current_pos[1], max(0.2, current_pos[2] - 0.3), 0.0, 2.0)
            time.sleep(2)
            commander.go_to(*current_pos, 0.0, 2.0)
        elif command == "shake":
            print(" Shaking head...")
            commander.go_to(*current_pos, -30.0, 0.5)
            time.sleep(0.5)
            commander.go_to(*current_pos, 30.0, 0.5)
            time.sleep(0.5)
            commander.go_to(*current_pos, -30.0, 0.5)
            time.sleep(0.5)
            commander.go_to(*current_pos, 0.0, 0.5)
        elif command == "spin":
            print(" Spinning...")
            commander.go_to(*current_pos, 180.0, 2.0)
            time.sleep(2)
            commander.go_to(*current_pos, -180.0, 2.0)
            time.sleep(2)
            commander.go_to(*current_pos, 0.0, 2.0)
        elif command == "happy":
            print(" Happy wiggle")
            commander.go_to(current_pos[0] - 0.1, current_pos[1], current_pos[2] + 0.2, 0.0, 1.0)
            time.sleep(1)
            commander.go_to(current_pos[0] + 0.2, current_pos[1], current_pos[2], 0.0, 1.0)
            time.sleep(1)
            commander.go_to(*current_pos, 0.0, 1.0)
        elif command == "excited":
            print(" Excited jump!")
            commander.go_to(current_pos[0], current_pos[1], current_pos[2] + 0.4, 0.0, 1.0)
            time.sleep(1)
            commander.go_to(*current_pos, 0.0, 1.0)
        else:
            print(f" Executing '{command}'")
        current_pos = clamp(current_pos)
        commander.go_to(*current_pos, 0.0, 2.0)
    return current_pos, taken_off
def clamp(pos):
    return [max(-1.5, min(1.5, pos[0])), max(-1.5, min(1.5, pos[1])), max(0.1, min(1.5, pos[2]))]
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
            if 0.00 < z < 2.0 and roll < 20:
                stable += 1
            else:
                stable = 0
            if stable > 5:
                break
with SyncCrazyflie(URI, cf=Crazyflie(rw_cache=None)) as scf:
    commander = scf.cf.high_level_commander
    wait_for_position_estimator(scf)
    print(" Ready!")
    cap = cv2.VideoCapture(0)
    current_pos = [0.0, 0.0, 0.5]
    taken_off = False
    cooldown = 3
    last_action = 0
    idle_check = time.time() + 5
    last_interaction = time.time()
    mood = "neutral"
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=audio_callback):
        try:
            while cap.isOpened():
                _, frame = cap.read()
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                landmarks = extract_landmarks(result)
                text, intent, distance = None, None, None
                if not audio_q.empty():
                    data = audio_q.get()
                    if recognizer.AcceptWaveform(data):
                        text = json.loads(recognizer.Result()).get("text", "").lower()
                        print(f" Heard: '{text}'")
                        if text:
                            intent = local_ai_intent(text)
                            distance = extract_distance(text)
                gesture = None
                if sum(landmarks) != 0.0 and len(landmarks) == 126:
                    try:
                        cols = [f'x{i}' for i in range(126)]
                        X_input = pd.DataFrame([landmarks], columns=cols)
                        gesture = gesture_model.predict(X_input)[0]
                        print("️ Gesture recognized:", gesture)
                    except Exception as e:
                        print("️ Gesture prediction error:", e)
                command = intent or gesture
                now = time.time()
                if intent == "learn_trick":
                    print(" Entering learning mode. Say the name of the new trick.")
                    learning_mode = True
                    learned_trick_name = None
                    learned_trick_actions = []
                    continue
                if learning_mode and not learned_trick_name and text:
                    learned_trick_name = text.strip().lower()
                    print(f" Trick will be saved as: '{learned_trick_name}'")
                    print("️ Now perform a series of commands. Say 'end trick' to finish.")
                    continue
                if learning_mode and intent == "end_trick":
                    if learned_trick_name and learned_trick_actions:
                        print(f" Trick '{learned_trick_name}' saved with {len(learned_trick_actions)} steps.")
                        saved_tricks[learned_trick_name] = learned_trick_actions.copy()
                        if learned_trick_name not in intent_examples:
                            intent_examples[learned_trick_name] = [learned_trick_name]
                            flat_examples.append(learned_trick_name)
                            intent_labels.append(learned_trick_name)
                            example_embeddings = intent_model.encode(flat_examples)
                    else:
                        print("️ No trick name or steps to save.")
                    learning_mode = False
                    learned_trick_name = None
                    learned_trick_actions = []
                    continue
                if learning_mode and intent and intent != "end_trick":
                    print(f" Saving step: '{intent}'")
                    learned_trick_actions.append(intent)
                    continue
                if command in saved_tricks:
                    print(f" Performing learned trick: '{command}'")
                    for step in saved_tricks[command]:
                        print(f"️ Executing: {step}")
                        current_pos, taken_off = perform_command(command, commander, current_pos, taken_off, move=distance)
                    continue
                    for step in saved_tricks[command]:
                        print(f"️ Executing: {step}")
                        intent = step
                        command = step
                if command and now - last_action > cooldown:
                    last_action = now
                    last_interaction = now
                    current_pos, taken_off = perform_command(command, commander, current_pos, taken_off, move=distance)
                if time.time() > idle_check:
                    idle_check = time.time() + 5
                    idle_time = time.time() - last_interaction
                    if idle_time > 20 and mood == "happy":
                        mood = "bored"
                        print(" Feeling bored...")
                    elif idle_time > 40 and mood == "bored":
                        mood = "sad"
                        print(" Feeling ignored...")
                    if taken_off:
                        if mood == "bored":
                            commander.go_to(*current_pos, 180.0, 2.0)
                            time.sleep(2)
                            commander.go_to(*current_pos, 0.0, 1.0)
                        elif mood == "sad":
                            commander.go_to(current_pos[0], current_pos[1], max(0.1, current_pos[2] - 0.2), 0.0, 1.0)
                            time.sleep(1)
                            commander.go_to(*current_pos, 0.0, 1.0)
                if result.multi_hand_landmarks:
                    for hand in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, f"Command: {command or 'None'}", (10, 40),
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
            print(" Landed & Disconnected")
