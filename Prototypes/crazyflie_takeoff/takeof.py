import time
import speech_recognition as sr
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.motion_commander import MotionCommander

URI = 'radio://0/80/2M'
cflib.crtp.init_drivers()

recognizer = sr.Recognizer()
mic = sr.Microphone()

def listen_command():
    with mic as source:
        print(" Say a command...")
        recognizer.adjust_for_ambient_noise(source, duration=1.0)
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio).lower()
        print(f"️ Heard: {command}")
        return command
    except:
        print("Could not understand")
        return ""

def reset_estimator(scf):
    print(" Resetting estimator...")
    scf.cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    scf.cf.param.set_value('kalman.resetEstimation', '0')
    print(" Waiting for estimator to settle...")
    time.sleep(2)

# Start flight
with SyncCrazyflie(URI, cf=Crazyflie(rw_cache=None)) as scf:
    print(" Connected to Crazyflie!")
    reset_estimator(scf)

    with MotionCommander(scf) as mc:
        print(" Taking off...")
        mc.take_off(0.5, 2.0)
        time.sleep(2)

        heading = 0  # initial facing direction
        flying = True

        while flying:
            command = listen_command()

            if "land" in command or "stop" in command:
                print(" Landing...")
                mc.land()
                flying = False
                break

            elif "forward" in command:
                print(" Moving forward...")
                mc.forward(0.3)
                time.sleep(1)

            elif "back" in command or "backward" in command:
                print(" Turning around for backward movement")
                mc.turn_left(180)
                time.sleep(1)
                mc.forward(0.3)
                time.sleep(1)
                mc.turn_right(180)  # restore heading
                time.sleep(1)

            elif "left" in command:
                print(" Turning left for leftward movement")
                mc.turn_left(90)
                time.sleep(1)
                mc.forward(0.3)
                time.sleep(1)
                mc.turn_right(90)
                time.sleep(1)

            elif "right" in command:
                print(" Turning right for rightward movement")
                mc.turn_right(90)
                time.sleep(1)
                mc.forward(0.3)
                time.sleep(1)
                mc.turn_left(90)
                time.sleep(1)

            else:
                print(" Unknown command.")
