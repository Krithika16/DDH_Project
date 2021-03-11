
'''
Basic Keras Code for a convolutional neural network
'''

#this is the classifier used if using alpha or calpha as the diagnostic parameter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanSquaredError
import matplotlib.pyplot as plt
from ..data_loader2 import DataGenerator2    #for this classifier we import only DataGenerator2 since we are using the alpha angle
from .model import conv0, conv1, conv2, conv3, resnet, resnet2, patientDetModel, resnet2Autocrop
import math
from tensorflow.keras.models import load_model, Model
import h5py
from ..data_loader2_autocrop import DataGeneratorCrop

#Function to diagnose DDH from the scans and patient details
def Classify2():    #again specifiy the function to be 2 and hence refere to the alpha angle

  # Training Parameters
  epochs = 1 #going through the dataset 50 times
  batch_size = 16 #Number of samples passed through CNN at one time for training data
  test_batch_size = 8 #batch size for testing data
  val_batch_size = 8 #batch size for validation data

  # Import Dataset, in this case the y label is the alpha angle and x is the scan
  data = DataGenerator2(width=256, height=256)  #in this case we have specified the width and height to be 256, larger than the standard in the dataloader file
  #Below, splits up the data so that it is within one of three categories: training, validation and testing datasets


#Extracting the relevant data for train, val and test sets and one-hot encoding the categorical data
  x_train = data.x_train
  gender_train = data.gender_train
  side_train = data.side_train
  indication_train = data.indication_train
  birthweight_train = data.birthweight_train
  #alpha_train = data.alphaAngle_train
  #beta_train = data.betaAngle_train
  y_train = data.y_train

  num_train_examples = np.shape(y_train)
  num_train_examples = num_train_examples[0]

  x_val = data.x_val
  gender_val = data.gender_val
  side_val = data.side_val
  indication_val = data.indication_val
  birthweight_val = data.birthweight_val
  #alpha_val = data.alphaAngle_val
  #beta_val = data.betaAngle_val
  y_val = data.y_val

  num_val_examples = np.shape(y_val)
  num_val_examples = num_val_examples[0]

  x_test = data.x_test
  gender_test = data.gender_test
  side_test = data.side_test
  indication_test = data.indication_test
  birthweight_test = data.birthweight_test
  #alpha_test = data.alphaAngle_test
  #beta_test = data.betaAngle_test
  y_test = data.y_test

  num_test_examples = np.shape(y_test)
  num_test_examples = num_test_examples[0]

  print("Training DataSet: " + str(x_train.shape) + " " + str(y_train.shape))
  print("Validation DataSet: " + str(x_val.shape) + " " + str(y_val.shape))
  print("Test DataSet: " + str(x_test.shape) + " " + str(y_test.shape))

  #Load Marta's model
  MartaModel = load_model('C:\Year_4_Courses\Masters_Project\Deep_learning_DDH\deep-learning-hip-dysplasia\\results\\tf2\\classifier__2020-06-04__08-55.h5')

  #Cut the last output layer to get only the features from the second last dense layer
  modelCut = Model(inputs=MartaModel.input, outputs=MartaModel.layers[-2].output)

  #Inputting the training, validation and test datasets into the pre-trained model to get the features that will then be input into the neural network that is to be trained
  outcomePred_Train = modelCut.predict(x_train)
  outcomePred_Val = modelCut.predict(x_val)
  outcomePred_Test = modelCut.predict(x_test)

  #shuffling the dataset batches to help training converge faster, reduce bias, and preventing model from learning the order of the training

  train_dataset = tf.data.Dataset.from_tensor_slices(({"outcome_pred" : outcomePred_Train, "gender" :gender_train, \
   "side" : side_train, "indication" : indication_train, "birthweight" : birthweight_train}, y_train)).batch(batch_size).shuffle(1000)
  train_dataset = train_dataset.repeat()

  val_dataset = tf.data.Dataset.from_tensor_slices(({"outcome_pred" : outcomePred_Val, "gender" :gender_val, \
   "side" : side_val, "indication" : indication_val, "birthweight" : birthweight_val}, y_val)).batch(val_batch_size).shuffle(1000)
  val_dataset = val_dataset.repeat()


  test_dataset = tf.data.Dataset.from_tensor_slices(({"outcome_pred" : outcomePred_Test, "gender" :gender_test, \
   "side" : side_test, "indication" : indication_test, "birthweight" : birthweight_test}, y_test)).batch(test_batch_size).shuffle(1000)
  test_dataset = test_dataset.repeat()



  """
  def lr_schedule(epoch):   #this is currently not being used


      #Learning Rate Schedule. Learning rate is scheduled to be reduced
      #after 80, 120, 160, 180 epochs. Called automatically every epoch as
      #part of callbacks during training.

      lr = 1e-2
      if epoch < 10:
          lr *= 1e-2
      elif epoch < 20:
          lr *= 1e-3
      elif epoch < 30:
          lr *= 1e-4
      elif epoch >= 40:
          lr *= 0.5e-4
      print('Learning rate: ', lr)
      return lr
      """

  # Network Parameters
  #WIDTH = data.WIDTH
  #HEIGHT = data.HEIGHT
  #CHANNELS = 1
  NUM_FEATURES = 1000 #This is the number of nodes in the last layer of Marta's network
  NUM_OUTPUTS = 1
  NUM_SIDE = data.SIDE
  NUM_GENDER = data.GENDER
  NUM_INDICATION = data.INDICATION


  #This section of code below chooses the model and compiles it with the necessary hyperparameters
  model = patientDetModel(NUM_FEATURES, NUM_OUTPUTS, NUM_SIDE, NUM_GENDER, NUM_INDICATION);  #model chosen is patientDetModel

  #Note: compile configures the model for training BUT DOESN'T TRAIN IT
  #Note: recall is sensitivity while precision is positive predictive value
  model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['binary_accuracy', tf.keras.metrics.Recall()])  #very important line about model characteristics
  model.summary() #prints information about the model that was trained

  # Prepare callbacks
  #lr_scheduler = LearningRateScheduler(lr_schedule)
  #lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=1, patience=5, min_lr=0.5e-6)
  #callbacks = [lr_reducer, lr_scheduler]

  start = time.time();

  #FIT TRAINS THE MODEL FOR A FIXED NUMBER OF EPOCHS (ITERATIONS ON A DATASET)
  history = model.fit(train_dataset,
          epochs=epochs,
          steps_per_epoch= int((num_train_examples/batch_size)*2),
          validation_data=val_dataset,
          validation_steps = math.ceil((num_val_examples/val_batch_size)*2))


  #This tests the model that was trained in the previous step - https://www.machinecurve.com/index.php/2020/11/03/how-to-evaluate-a-keras-model-with-model-evaluate/
  #Note: verbose shows the progress bar if 1 and doesn't show anything if it is 0
  #steps tells you the total number of batches fed forward before evaluating is considered to be complete
  evaluation = model.evaluate(test_dataset, verbose=1, steps = math.ceil((num_test_examples/test_batch_size)*2))    #after training and validation end (50 epochs), we finish with testing
  end = time.time()


  #Below are just details printed onto the screen for the user to inform them about the model's accuracy, etc.
  print('Classify Summary: Test Accuracy: %.2f Time Elapsed: %.2f seconds' % (evaluation[1], (end - start)) )
  print('Classify Summary: Test Loss: %.2f Time Elapsed: %.2f seconds' % (evaluation[0], (end - start)) )
  print('Classify Summary: Sensitivity: %.2f Time Elapsed: %.2f seconds' % (evaluation[2], (end - start)) )   #note sensitivity=recall

  # Plot Accuracy
  plt.plot(history.history["binary_accuracy"])
  plt.plot(history.history["val_binary_accuracy"])
  plt.ylabel("Accuracy")
  plt.xlabel("Epochs")
  plt.title('Classify Summary: Test Accuracy: %.2f Time Elapsed: %.2f seconds' % (evaluation[1], (end - start)))
  plt.legend(["Train Accuracy", "Validation Accuracy"], loc="upper left")

  import datetime
  file_time = datetime.datetime.today().strftime('_%Y-%m-%d__%I-%M')  #get date and time for today for filename
  plt.savefig('results/tf2/classifier_' + file_time + '.png')   #save graph
  model.save('results/tf2/classifier_' + file_time + '.h5')     #save model weights in h5 file
  plt.close()

  print(model.metrics_names)

  # Plot Loss
  plt.plot(history.history["loss"])
  plt.plot(history.history["val_loss"])
  plt.ylabel("Loss")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test Loss: %.2f Time Elapsed: %.2f seconds" % (evaluation[0], (end - start)))
  plt.legend(["Train Loss", "Validation Loss"], loc="upper left")
  plt.savefig('results/tf2/loss_' + file_time + '.png') #save loss graph
  plt.close()

  # Plot Sensitivity
  plt.plot(history.history["recall"])
  plt.plot(history.history["val_recall"])
  plt.ylabel("Sensitivity")
  plt.xlabel("Epochs")
  plt.title("Classify Summary: Test Sensitivity: %.2f Time Elapsed: %.2f seconds" % (evaluation[2], (end - start)))
  plt.legend(["Train Sensitivity", "Validation Sensitivity"], loc="upper left")
  plt.savefig('results/tf2/sensitivity_' + file_time + '.png')  #save sensitivity graph
