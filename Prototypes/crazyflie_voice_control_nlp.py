import time
import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.high_level_commander import HighLevelCommander

# ========== SETUP ==========
URI = 'radio://0/80/2M'  # Adjust to your Crazyflie URI
cflib.crtp.init_drivers()

model = Model("model")  # Path to Vosk model directory
recognizer = KaldiRecognizer(model, 16000)
audio_q = queue.Queue()

# ========== AUDIO CALLBACK ==========
def audio_callback(indata, frames, time, status):
    if status:
        print("Audio error:", status)
    audio_q.put(bytes(indata))

# ========== INTENT RECOGNITION ==========
def interpret_intent(text):
    text = text.lower().strip()

    intent_map = {
        "takeoff": ["take off", "lift off", "start flying", "launch"],
        "land": ["land", "come down", "stop flying", "touch down"],
        "forward": ["go forward", "move forward", "fly forward"],
        "back": ["go back", "move back", "fly back"],
        "left": ["go left", "move left", "fly left"],
        "right": ["go right", "move right", "fly right"],
        "up": ["go up", "ascend", "fly higher"],
        "down": ["go down", "descend", "lower down"],
        "excited": ["get excited", "do a jump", "show excitement"],
        "happy": ["be happy", "wiggle", "dance"],
        "sad": ["be sad", "look sad"],
        "spin": ["spin", "twirl"],
        "shake": ["shake", "wiggle head", "shake head"],
        "come here": ["come here", "come closer", "fly to me"],
        "stop": ["stop", "halt", "freeze", "land now"]
    }

    for intent, phrases in intent_map.items():
        if any(phrase in text for phrase in phrases):
            return intent
    return None

# ========== VOICE CONTROL LOOP ==========
def voice_control_loop(commander: HighLevelCommander):
    print("🎙️ Ready — speak naturally. Try: 'Can you take off?', 'Come here', 'Do a jump', etc.")

    current_pos = [0.0, 0.0, 0.5]
    step = 0.3

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=audio_callback):
        while True:
            data = audio_q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower()
                if not text:
                    continue

                print(f"🗣️ Heard: {text}")
                intent = interpret_intent(text)

                if intent == "takeoff":
                    print("✈️ Taking off...")
                    commander.takeoff(current_pos[2], 2.0)
                    time.sleep(3)

                elif intent == "land":
                    print("🛬 Landing...")
                    commander.land(0.0, 2.0)
                    time.sleep(3)

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
                    print("➡️ Moving right...")
                    commander.go_to(*current_pos, 0.0, 2.0)

                elif intent == "up":
                    current_pos[2] += step
                    print("🔼 Moving up...")
                    commander.go_to(*current_pos, 0.0, 2.0)

                elif intent == "down":
                    current_pos[2] = max(0.1, current_pos[2] - step)
                    print("🔽 Moving down...")
                    commander.go_to(*current_pos, 0.0, 2.0)

                elif intent == "excited":
                    print("🤸 Excited jump!")
                    commander.go_to(current_pos[0], current_pos[1], current_pos[2] + 0.4, 0.0, 1.0)
                    time.sleep(1)
                    commander.go_to(current_pos[0], current_pos[1], current_pos[2], 0.0, 1.0)
                    time.sleep(1)

                elif intent == "happy":
                    print("😊 Happy wiggle!")
                    commander.go_to(current_pos[0] - 0.1, current_pos[1], current_pos[2] + 0.2, 0.0, 1.0)
                    time.sleep(1)
                    commander.go_to(current_pos[0] + 0.2, current_pos[1], current_pos[2], 0.0, 1.0)
                    time.sleep(1)
                    commander.go_to(*current_pos, 0.0, 1.0)

                elif intent == "sad":
                    print("😢 Feeling sad...")
                    commander.go_to(current_pos[0], current_pos[1], max(0.2, current_pos[2] - 0.3), 0.0, 2.0)
                    time.sleep(2)
                    commander.go_to(*current_pos, 0.0, 2.0)

                elif intent == "spin":
                    print("🌀 Spinning...")
                    commander.go_to(*current_pos, 180.0, 1.0)
                    time.sleep(1)
                    commander.go_to(*current_pos, -180.0, 1.0)
                    time.sleep(1)
                    commander.go_to(*current_pos, 0.0, 1.0)

                elif intent == "shake":
                    print("🐾 Shaking head...")
                    commander.go_to(*current_pos, -30.0, 0.5)
                    time.sleep(0.5)
                    commander.go_to(*current_pos, 30.0, 0.5)
                    time.sleep(0.5)
                    commander.go_to(*current_pos, 0.0, 0.5)

                elif intent == "come here":
                    print("🚶 Coming to you...")
                    current_pos[1] += 0.5
                    commander.go_to(*current_pos, 0.0, 2.0)

                elif intent == "stop":
                    print("🛑 Stopping and landing.")
                    commander.land(0.0, 2.0)
                    time.sleep(3)
                    break

# ========== MAIN ==========
print("🔗 Connecting to Crazyflie...")
with SyncCrazyflie(URI, cf=Crazyflie(rw_cache=None)) as scf:
    commander = scf.cf.high_level_commander
    print("✅ Connected!")
    time.sleep(2)  # Let estimator settle
    voice_control_loop(commander)
