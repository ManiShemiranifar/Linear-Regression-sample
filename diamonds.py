import pandas as pd 
import matplotlib.pyplot as plt 
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

class Diamonds():

    def __init__(self):
        # Establishing data frame
        self.df = pd.read_csv("diamonds.csv")
        # preprocessing
        self.df.color = self.df.color.apply(list("JIHGFED").index)
        self.df.cut = self.df.cut.apply(list(["Fair", "Good", "Very Good", "Premium", "Ideal"]).index)
        self.df.clarity = self.df.clarity.apply(list(["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]).index)
        # x and y
        self.x = self.df.loc[:,(self.df.columns != "price")]
        self.y = self.df.price
        # Split data
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.x, self.y, test_size=0.3, random_state = 42)
        # Create and fit model
        self.model = LinearRegression()
        self.model.fit(self.xtrain, self.ytrain)
        # Predict
        self.ypred = self.model.predict(self.xtest)
        
    def RMSE(self):
        return sqrt(mean_squared_error(self.ytest, self.ypred))
    
    def MAE(self):
        return mean_absolute_error(self.ytest, self.ypred)

    def plot(self):
        plt.scatter(self.ytest, self.ypred)
        plt.xlabel("prices")
        plt.ylabel("Predicted prices")
        plt.show()

d1 = Diamonds()
d1.plot()
