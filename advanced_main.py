import Jetson.GPIO as GPIO
import time




#install GPIO library:
# sudo pip install Jetson.GPIO
# sudo groupadd -f -r gpio
#
# sudo usermod -a -G gpio your_user_name


# Set the GPIO pin numbers
button_pin = 31  # Replace with the actual pin number
led_green_pin = 33  # Replace with the actual pin number
led_red_pin = 35  # Replace with the actual pin number
status = 0

# Initial state and LED mapping
led_state = 0  # 0 for led1, 1 for led2
led_pins = [led_green_pin, led_red_pin]


def process_1_GPIO(st):
    print("start")
    status = 0



    # Configure the GPIO pins
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    for pin in led_pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

    try:
        i = 0
        while True:
            i = i+1


            print(i)
            if status == 0:
                GPIO.output(led_red_pin, GPIO.HIGH)
                GPIO.output(led_green_pin, GPIO.LOW)
            if status ==1:
                GPIO.output(led_red_pin, GPIO.LOW)
                GPIO.output(led_green_pin, GPIO.HIGH)

            button_state = GPIO.input(button_pin)
            if button_state == GPIO.HIGH:
                print("button premuto!!!")
                # Toggle the value
                if status == 0:
                    print("TO GREEN")
                    status = 1
                else:
                    print("TO RED")
                    status = 0



            time.sleep(0.2)

    except KeyboardInterrupt:
        pass

    GPIO.cleanup()


process_1_GPIO(status)

































