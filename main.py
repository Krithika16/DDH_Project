
 #from the file 'estimator', import the class Estimator

#Import this classifier if you want to use 'outcome' as the diagnostic parameter
#Import this classifier if you want to use the alpha angle as the diagnostic parameter
from classifier2 import Classify2
from regression_autocrop import AutoCrop    #Import this classifier if you want to use the alpha angle as the diagnostic parameter
import tensorflow as tf

if __name__ == '__main__': #This code is run when the main.py file is run
   #Classify()
  # AutoCrop() #This instantiates the class Autocrop

   Classify2()    #in this case we will be using alpha
