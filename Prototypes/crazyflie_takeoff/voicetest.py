import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print("🎤 Speak now...")
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)

try:
    command = r.recognize_google(audio)
    print("You said:", command)
except sr.UnknownValueError:
    print("❌ Could not understand audio")
except sr.RequestError as e:
    print(f"🔌 API error: {e}")
