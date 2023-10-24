import Jetson.GPIO as GPIO
import time




#install GPIO library:
# sudo pip install Jetson.GPIO


# Set the GPIO pin numbers
button_pin = 31  # Replace with the actual pin number
led_green_pin = 33  # Replace with the actual pin number
led_red_pin = 35  # Replace with the actual pin number
status = 0

# Initial state and LED mapping
led_state = 0  # 0 for led1, 1 for led2
led_pins = [led_green_pin, led_red_pin]


def toggle_led_state():
    global led_state
    if led_state == 0:
        print("accendo verde")
        GPIO.output(led_green_pin, GPIO.HIGH)
        GPIO.output(led_red_pin, GPIO.LOW)
        led_state = 1
    else:
        print("accendo rosso")
        GPIO.output(led_green_pin, GPIO.LOW)
        GPIO.output(led_red_pin, GPIO.HIGH)
        led_state = 0



def process_1_GPIO(status):
    print("start")



    # Configure the GPIO pins
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    for pin in led_pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

    try:
        i  = 0
        while True:
            i = i+1

            button_state = GPIO.input(button_pin)
            print(i," button state:",button_state)

            if button_state == GPIO.HIGH:
                print("button premuto!!!")
                toggle_led_state()

            time.sleep(0.1)

    except KeyboardInterrupt:
        pass

    GPIO.cleanup()


process_1_GPIO(status)

































