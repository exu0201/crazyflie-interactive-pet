import time
import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from sentence_transformers import SentenceTransformer, util
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.high_level_commander import HighLevelCommander
URI = 'radio://0/80/2M'
cflib.crtp.init_drivers()
model = Model("model")
recognizer = KaldiRecognizer(model, 16000)
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
for intent, phrases in intent_examples.items():
    for phrase in phrases:
        flat_examples.append(phrase)
        intent_labels.append(intent)
example_embeddings = intent_model.encode(flat_examples)
def audio_callback(indata, frames, time, status):
    if status:
        print("Audio error:", status)
    audio_q.put(bytes(indata))
def local_ai_intent(text):
    input_embedding = intent_model.encode(text)
    similarity = util.cos_sim(input_embedding, example_embeddings)[0]
    best_idx = int(similarity.argmax())
    confidence = float(similarity[best_idx])
    best_intent = intent_labels[best_idx]
    print(f" Intent match: {best_intent} (confidence: {confidence:.2f})")
    return best_intent if confidence > 0.35 else None
def voice_control_loop(commander: HighLevelCommander):
    print("️ Speak naturally! Start with: 'take off', 'please take off', etc.")
    current_pos = [0.0, 0.0, 0.5]
    step = 0.3
    has_taken_off = False
    mood = "neutral"
    last_interaction_time = time.time()
    idle_check_interval = 5
    next_idle_check = time.time() + idle_check_interval
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=audio_callback):
        while True:
            data = audio_q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower()
                if not text:
                    continue
                print(f"️ Heard: {text}")
                last_interaction_time = time.time()
                intent = local_ai_intent(text)
                if intent:
                    if intent in ["happy", "excited", "come here", "spin"]:
                        mood = "happy"
                    elif intent == "land":
                        mood = "neutral"
                print(f"️ Current mood: {mood}")
                if intent == "takeoff":
                    if not has_taken_off:
                        print("️ Taking off...")
                        commander.takeoff(current_pos[2], 2.0)
                        time.sleep(3)
                        has_taken_off = True
                    else:
                        print(" Already in air.")
                elif intent == "land":
                    if has_taken_off:
                        print(" Landing...")
                        commander.land(0.0, 2.0)
                        time.sleep(3)
                        has_taken_off = False
                        mood = "neutral"
                    else:
                        print(" Already on the ground.")
                elif not has_taken_off:
                    print("️ Please say 'take off' before giving other commands.")
                    continue
                elif intent == "forward":
                    current_pos[1] += step
                    print("⬆️ Moving forward...")
                    commander.go_to(*current_pos, 0.0, 2.0)
                elif intent == "back":
                    current_pos[1] -= step
                    print("⬇️ Moving back...")
                    commander.go_to(*current_pos, 0.0, 2.0)
                elif intent == "left":
                    current_pos[0] -= step
                    print("⬅️ Moving left...")
                    commander.go_to(*current_pos, 0.0, 2.0)
                elif intent == "right":
                    current_pos[0] += step
                    print("️ Moving right...")
                    commander.go_to(*current_pos, 0.0, 2.0)
                elif intent == "up":
                    current_pos[2] += step
                    print(" Moving up...")
                    commander.go_to(*current_pos, 0.0, 2.0)
                elif intent == "down":
                    current_pos[2] = max(0.1, current_pos[2] - step)
                    print(" Moving down...")
                    commander.go_to(*current_pos, 0.0, 2.0)
                elif intent == "excited":
                    print(" Excited jump!")
                    commander.go_to(current_pos[0], current_pos[1], current_pos[2] + 0.4, 0.0, 1.0)
                    time.sleep(1)
                    commander.go_to(*current_pos, 0.0, 1.0)
                    time.sleep(1)
                elif intent == "happy":
                    print(" Happy wiggle!")
                    commander.go_to(current_pos[0] - 0.1, current_pos[1], current_pos[2] + 0.2, 0.0, 1.0)
                    time.sleep(1)
                    commander.go_to(current_pos[0] + 0.2, current_pos[1], current_pos[2], 0.0, 1.0)
                    time.sleep(1)
                    commander.go_to(*current_pos, 0.0, 1.0)
                elif intent == "sad":
                    print(" Feeling sad...")
                    commander.go_to(current_pos[0], current_pos[1], max(0.2, current_pos[2] - 0.3), 0.0, 2.0)
                    time.sleep(2)
                    commander.go_to(*current_pos, 0.0, 2.0)
                elif intent == "spin":
                    print(" Spinning...")
                    commander.go_to(*current_pos, 180.0, 2.0)
                    time.sleep(3)
                    commander.go_to(*current_pos, -180.0, 2.0)
                    time.sleep(3)
                    commander.go_to(*current_pos, 0.0, 2.0)
                elif intent == "shake":
                    print(" Shaking head...")
                    commander.go_to(*current_pos, -30.0, 0.5)
                    time.sleep(0.5)
                    commander.go_to(*current_pos, 30.0, 0.5)
                    time.sleep(0.5)
                    commander.go_to(*current_pos, 0.0, 0.5)
                elif intent == "come here":
                    print(" Coming to you...")
                    current_pos[1] += 0.5
                    commander.go_to(*current_pos, 0.0, 2.0)
                elif intent == "stop":
                    if has_taken_off:
                        print(" Stopping and landing.")
                        commander.land(0.0, 2.0)
                        time.sleep(3)
                        break
                    else:
                        print(" Already landed.")
                time.sleep(0.5)
            if time.time() > next_idle_check:
                next_idle_check = time.time() + idle_check_interval
                idle_time = time.time() - last_interaction_time
                if idle_time > 20 and mood == "happy":
                    mood = "bored"
                    print(" Feeling a little bored...")
                elif idle_time > 40 and mood == "bored":
                    mood = "sad"
                    print(" Feeling ignored...")
                if has_taken_off:
                    if mood == "bored":
                        print(" Idle spin...")
                        commander.go_to(*current_pos, 180.0, 2.0)
                        time.sleep(3)
                        commander.go_to(*current_pos, 0.0, 1.0)
                        time.sleep(2)
                    elif mood == "sad":
                        print(" Drooping slightly...")
                        commander.go_to(current_pos[0], current_pos[1], max(0.1, current_pos[2] - 0.2), 0.0, 1.0)
                        time.sleep(1)
                        commander.go_to(*current_pos, 0.0, 1.0)
                        time.sleep(2)
print(" Connecting to Crazyflie...")
with SyncCrazyflie(URI, cf=Crazyflie(rw_cache=None)) as scf:
    commander = scf.cf.high_level_commander
    print(" Connected!")
    time.sleep(2)
    voice_control_loop(commander)
