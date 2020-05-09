from base import Regressor
import numpy as np
from sklearn.linear_model import LinearRegression

class PositionRegressor(Regressor):

    """ Implement solution for Part 1 below  """
    def __init__(self):
        self.model = LinearRegression()
    

    def train(self, data):
        print("Using dummy solution for PositionRegressor")
        n,height,wideth,channel = data['obs'].shape
        x_train = (data['obs'].reshape((n,height*wideth*channel)))/255.0
        
        self.model.fit(x_train,np.asarray([info['agent_pos'] for info in data['info']]))
        pass


    def predict(self, Xs):
        n,height,wideth,channel=Xs.shape
        Xs=Xs.reshape((n,height*wideth*channel))
        Xs=Xs/255.0
        predictions = self.model.predict(Xs)
        return predictions
