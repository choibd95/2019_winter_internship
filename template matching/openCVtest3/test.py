import threading
import time

a = 0
b = 0

def alarm():
    global a
    global b

    while True:
        time.sleep(5)

t = threading.Thread(target=alarm)
t.daemon = True

t.start()

while True:
    print("receive")

