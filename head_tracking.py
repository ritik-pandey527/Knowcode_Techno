import cv2
import mediapipe as mp
import RPi.GPIO as GPIO
import time

# Servo Motor GPIO Pins
X_SERVO_PIN = 17  # GPIO pin for X-axis servo
Y_SERVO_PIN = 27  # GPIO pin for Y-axis servo

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(X_SERVO_PIN, GPIO.OUT)
GPIO.setup(Y_SERVO_PIN, GPIO.OUT)

# Set PWM frequency for servos
x_servo_pwm = GPIO.PWM(X_SERVO_PIN, 50)  # 50 Hz for servo
y_servo_pwm = GPIO.PWM(Y_SERVO_PIN, 50)

# Start PWM with the initial position
x_servo_pwm.start(7.5)  # Center position (7.5% duty cycle)
y_servo_pwm.start(7.5)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def map_range(value, in_min, in_max, out_min, out_max):
    """Map a value from one range to another."""
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def set_servo_angle(pwm, angle):
    """Set servo angle."""
    duty_cycle = map_range(angle, 0, 180, 2.5, 12.5)  # Map angle to duty cycle
    pwm.ChangeDutyCycle(duty_cycle)

cap = cv2.VideoCapture(0)  # Use the default camera

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for face landmarks
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get the nose tip as a reference point
                nose_tip = face_landmarks.landmark[1]
                x, y = int(nose_tip.x * frame.shape[1]), int(nose_tip.y * frame.shape[0])

                # Map x, y to servo range (0 to 180 degrees)
                x_mapped = map_range(x, 0, frame.shape[1], 0, 180)
                y_mapped = map_range(y, 0, frame.shape[0], 0, 180)

                # Move servos to the mapped positions
                set_servo_angle(x_servo_pwm, x_mapped)
                set_servo_angle(y_servo_pwm, y_mapped)

                # Draw the tracking point
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow('Head Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    x_servo_pwm.stop()
    y_servo_pwm.stop()
    GPIO.cleanup()
