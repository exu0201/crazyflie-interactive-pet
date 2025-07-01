import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer

# Load model
model = Model("model")
rec = KaldiRecognizer(model, 16000)
q = queue.Queue()

# Audio callback to capture microphone input
def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

# Start microphone stream
print("🎙️ Listening for voice commands...")

with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=callback):
    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "")
            if text:
                print(f"🗣️ You said: {text}")

                # Command triggers
                if "take off" in text:
                    print("✈️ Takeoff triggered!")
                    # Call Crazyflie takeoff logic here

                elif "land" in text:
                    print("🛬 Landing triggered!")
                    # Call Crazyflie land logic here

                elif "stop" in text:
                    print("🛑 Stopping voice control.")
                    break
