from base import RobotPolicy
import numpy as np

from sklearn.svm import LinearSVC

from sklearn.decomposition import PCA

#from sklearn import preprocessing


class RGBBCRobot(RobotPolicy):

    """ Implement solution for Part3 below """
    def __init__(self):
        self.model = LinearSVC()
        self.p = PCA(n_components=10)

    def train(self, data):
        print("Using dummy solution for RGBBCRobot")

        n,height,wideth,channel = data['obs'].shape

        x_train = data['obs'].reshape((n,height*wideth*channel))
        x_train = x_train/255.0

        x_trainpca=self.p.fit_transform(x_train)

        y_train = data['actions']

        self.model.fit(x_trainpca,y_train)

        #x_train_pca=self.pca.fit_transform(x_train)
        pass

    def get_actions(self, observations):
        observations = observations/255.0
        observationspca = self.p.transform(observations)
        return self.model.predict(observationspca)
       
