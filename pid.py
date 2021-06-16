import matplotlib.pyplot as plt
import numpy as np


class PID:
    def __init__(self, kp, ki, kd, sample_time=0.01):  # default sample time : 10ms
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.sample_time = sample_time
        self.first_flag = True
        self.last_error = 0
        self.feedback = 0
        self.integral = 0
        self.output = 0

    def update(self, set_point):
        """pid update method"""

        error = set_point - self.feedback
        if self.first_flag:
            '''first time have no integral item and derivative item'''
            derivative = 0
            '''first time complete'''
            self.first_flag = False
        else:
            self.integral += error
            derivative = (error - self.last_error)

        self.output = self.kp * error + self.ki * self.integral + self.kd * derivative

        '''update attribute'''
        self.last_error = error
        self.feedback = self.output

        return self.output

