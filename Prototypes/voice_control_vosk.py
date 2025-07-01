import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer
model = Model("model")
rec = KaldiRecognizer(model, 16000)
q = queue.Queue()
def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))
print("️ Listening for voice commands...")
with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=callback):
    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "")
            if text:
                print(f"️ You said: {text}")
                if "take off" in text:
                    print("️ Takeoff triggered!")
                elif "land" in text:
                    print(" Landing triggered!")
                elif "stop" in text:
                    print(" Stopping voice control.")
                    break
