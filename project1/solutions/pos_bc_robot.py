from base import RobotPolicy
import numpy as np
from sklearn.linear_model import LogisticRegression

class POSBCRobot(RobotPolicy):
    
    """ Implement solution for Part 2 below """
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, data):
        print("Using dummy solution for POSBCRobot")
        x_train = data['obs']
        y_train = data['actions']
        self.model.fit(x_train,y_train)
        
        pass

    def get_actions(self, observations):
        
        return self.model.predict(observations)
