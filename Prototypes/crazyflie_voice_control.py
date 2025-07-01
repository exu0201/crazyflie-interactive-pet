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

# ========== VOICE CONTROL LOOP ==========
def voice_control_loop(commander: HighLevelCommander):
    print("🎙️ Ready — say commands like 'take off', 'land', 'excited', 'spin', 'happy', 'sad', 'shake', 'come here', or 'stop'.")

    current_pos = [0.0, 0.0, 0.5]  # x, y, z
    step = 0.3  # movement increment in meters

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=audio_callback):
        while True:
            data = audio_q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower()

                if text:
                    print(f"🗣️ Heard: {text}")

                    if "take off" in text:
                        print("✈️ Taking off...")
                        commander.takeoff(current_pos[2], 2.0)
                        time.sleep(3)

                    elif "land" in text:
                        print("🛬 Landing...")
                        commander.land(0.0, 2.0)
                        time.sleep(3)

                    elif "forward" in text:
                        current_pos[1] += step
                        print("⬆️ Moving forward...")
                        commander.go_to(*current_pos, 0.0, 2.0)

                    elif "back" in text:
                        current_pos[1] -= step
                        print("⬇️ Moving back...")
                        commander.go_to(*current_pos, 0.0, 2.0)

                    elif "left" in text:
                        current_pos[0] -= step
                        print("⬅️ Moving left...")
                        commander.go_to(*current_pos, 0.0, 2.0)

                    elif "right" in text:
                        current_pos[0] += step
                        print("➡️ Moving right...")
                        commander.go_to(*current_pos, 0.0, 2.0)

                    elif "up" in text:
                        current_pos[2] += step
                        print("🔼 Moving up...")
                        commander.go_to(*current_pos, 0.0, 2.0)

                    elif "down" in text:
                        current_pos[2] = max(0.1, current_pos[2] - step)
                        print("🔽 Moving down...")
                        commander.go_to(*current_pos, 0.0, 2.0)

                    elif "excited" in text:
                        print("🤸 Excited jump!")
                        commander.go_to(current_pos[0], current_pos[1], current_pos[2] + 0.4, 0.0, 1.0)
                        time.sleep(1)
                        commander.go_to(current_pos[0], current_pos[1], current_pos[2], 0.0, 1.0)
                        time.sleep(1)

                    elif "spin" in text:
                        print("🌀 Spinning in place...")
                        commander.go_to(*current_pos, 180.0, 1.0)
                        time.sleep(1)
                        commander.go_to(*current_pos, -180.0, 1.0)
                        time.sleep(1)
                        commander.go_to(*current_pos, 0.0, 1.0)

                    elif "happy" in text:
                        print("😊 Happy wiggle!")
                        commander.go_to(current_pos[0] - 0.1, current_pos[1], current_pos[2] + 0.2, 0.0, 1.0)
                        time.sleep(1)
                        commander.go_to(current_pos[0] + 0.2, current_pos[1], current_pos[2], 0.0, 1.0)
                        time.sleep(1)
                        commander.go_to(*current_pos, 0.0, 1.0)

                    elif "sad" in text:
                        print("😢 Feeling sad...")
                        commander.go_to(current_pos[0], current_pos[1], max(0.2, current_pos[2] - 0.3), 0.0, 2.0)
                        time.sleep(2)
                        commander.go_to(*current_pos, 0.0, 2.0)

                    elif "shake" in text:
                        print("🐾 Shaking head...")
                        commander.go_to(*current_pos, -30.0, 0.5)
                        time.sleep(0.5)
                        commander.go_to(*current_pos, 30.0, 0.5)
                        time.sleep(0.5)
                        commander.go_to(*current_pos, 0.0, 0.5)

                    elif "stop" in text:
                        print("🛑 Stopping and landing.")
                        commander.land(0.0, 2.0)
                        time.sleep(3)
                        break

# ========== MAIN ==========
print("🔗 Connecting to Crazyflie...")
with SyncCrazyflie(URI, cf=Crazyflie(rw_cache=None)) as scf:
    commander = scf.cf.high_level_commander
    print("✅ Connected!")
    time.sleep(2)  # Allow estimator to settle
    voice_control_loop(commander)
