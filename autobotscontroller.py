from controller import Robot, Camera, Motor
import numpy as np
import cv2

# Constants for the robots movement
SIMULATION_STEP = 64
FULL_SPEED = 6.28

class VisionNav:
    def __init__(self):
        self.robot_instance = Robot()
        self.vision_sensor = self.robot_instance.getDevice("camera")
        self.vision_sensor.enable(SIMULATION_STEP)
        self.motor_left = self.robot_instance.getDevice("left motor")
        self.motor_right = self.robot_instance.getDevice("right motor")
        self.motor_left.setPosition(float('inf'))
        self.motor_right.setPosition(float('inf'))
        self.motor_left.setVelocity(0.0)
        self.motor_right.setVelocity(0.0)


    def capture_and_process_image(self):
        #use the vision based data for analysis
        image_data = self.vision_sensor.getImageArray()
        grayscale_image = cv2.cvtColor(np.array(image_data), cv2.COLOR_BGR2GRAY)
        ret, binary_image = cv2.threshold(grayscale_image, 120, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self.extract_obstacle_data(contours)

    def extract_obstacle_data(self, data):
        obstacle_list = []
        for d in data:
            area = cv2.contourArea(d)
            if area > 0:
                M = cv2.moments(d)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                distance_estimate = 1000 / area  
                obstacle_list.append((distance_estimate, cx, cy))
        
        #sort the obstacles by distance from the robot to help guide robot to accomplish task
        obstacle_list.sort()  
        return obstacle_list
    
    #move the robot to where it needs to go using the data we got from sensors and the analysis

    def steer_toward_obstacle(self, obstacles):
        if not obstacles:
            self.stop_movement()
        closest_obstacle = obstacles[0]
        distance, central_x, central_y = closest_obstacle

        steering_error = 160 - central_x  
        adjustment = float(steering_error / 80)

        self.motor_left.setVelocity(FULL_SPEED - adjustment)
        self.motor_right.setVelocity(FULL_SPEED + adjustment)

            

    def stop_movement(self):
        self.motor_left.setVelocity(0)
        self.motor_right.setVelocity(0)

    #run robot

    def run_navigation_loop(self):
        while self.robot_instance.step(SIMULATION_STEP) != -1:
            detected_obstacles = self.capture_and_process_image()
            self.steer_toward_obstacle(detected_obstacles)

if __name__ == "__main__":
    autonomous_navigator = VisionNav()
    autonomous_navigator.run_navigation_loop()

