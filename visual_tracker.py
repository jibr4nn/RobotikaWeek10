import cv2
import numpy as np
from controller import Robot

# P value for P controller
P_COEFFICIENT = 0.1

# Initialize the robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Initialize camera
camera = robot.getDevice('camera')
camera.enable(timestep)

# Initialize motors
motor_left = robot.getDevice('left wheel motor')
motor_right = robot.getDevice('right wheel motor')
motor_left.setPosition(float('inf'))
motor_right.setPosition(float('inf'))
motor_left.setVelocity(0)
motor_right.setVelocity(0)

# Main control loop
while robot.step(timestep) != -1:
    # Capture the image from the camera
    img = np.frombuffer(camera.getImage(), dtype=np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))

    # Convert the image to HSV color space
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Create a mask to detect the red ball (you can adjust the color ranges)
    mask = cv2.inRange(img, np.array([50, 150, 0]), np.array([200, 230, 255]))

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If there are any contours, find the largest one
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate the center of the largest contour (red ball)
        moments = cv2.moments(largest_contour)
        if moments['m00'] != 0:
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
            
            # Draw a circle at the center of the ball
            cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), -1)

            # Calculate the error (difference between the ball's center and image center)
            error = camera.getWidth() / 2 - center_x

            # Control the robot's velocity using a proportional controller
            motor_left.setVelocity(- error * P_COEFFICIENT)
            motor_right.setVelocity(error * P_COEFFICIENT)

    # Display the image with the ball and the proportional control in OpenCV
    # Convert the image back to BGR for OpenCV display
    img_bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow('Ball Tracking', img_bgr)

    # Wait for 1 ms to update the window and exit if user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release OpenCV windows
cv2.destroyAllWindows()
