from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import math
import csv

import os
from skimage import io, transform
from tqdm import trange, tqdm
# importing one hot encoder from sklearn
# There are changes in OneHotEncoder class
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Constants
COLOR_CHANNELS = 3

class Stats:
    def __init__(self, min, max, mean, std):
        self.min = min
        self.max = max
        self.mean = mean
        self.std = std

class DataGenerator2:       #data generator2 is to use alpha or calpha as the diagnostic parameter

    def __init__(self,
        imagedir = 'C:\Year_4_Courses\Masters_Project\Deep_learning_DDH\deep-learning-hip-dysplasia\hip_images_marta\\', #insert here the directory where you store the hip images
        anglecsv =  'C:\Year_4_Courses\Masters_Project\Deep_learning_DDH\\Final_data_sample.csv', #insert here the file location of the csv with the patient data
        width = 128,    #insert here the image width
        height = 128,   #insert here the image height
        ratio1 = 0.8,   #this is the percentage for training, in this case 80%
        ratio2 = 0.1,   #this is the percentage for validation, 10% and hence the remaining 10% for testing
        #useBinaryClassify = True,   #we will be using a binary classification, 1 or 0
        #binaryThreshold = 60.0,     #above 60 is healthy, below 60 is diseased
        useNormalization = False,
        useWhitening = True,
        useRandomOrder = True):

        #also need to have self.GENDER, self.SIDE, self.INDICATION, self.BIRTHWEIGHT, self.ALPHA, self.BETA
        self.imagedir = imagedir
        self.anglecsv = anglecsv
        self.WIDTH = width
        self.HEIGHT = height
        self.CHANNELS = 1
        self.ratio1 = ratio1
        self.ratio2 = ratio2

        #self.useBinaryClassify = useBinaryClassify
        #self.binaryThreshold = binaryThreshold
        self.useCropping = False

        self.useNormalization = useNormalization
        self.useWhitening = useWhitening
        self.useRandomOrder = useRandomOrder

        print("Loading and formating image data ....")
        self.generate2()     #first function that leads onto the rest, go down to find it
        print("Loading and formating image data: Complete")
        print("Data size: Input Data", self.x_train.shape, " Truth Data:", self.y_train.shape);

    """
    def loadOldCSV(self):       #this was the loading process for the old csv file Michael had, not being used now
        angle_dict = {}
        with open(self.anglecsv) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # key = row['Current Standardized Name']

                # Get key and format key
                key = row['linked_images']
                key = int(key)
                if key < 10000:
                    key ='0' + str(key) + '.png'
                else:
                    key = str(key) + '.png'

                if row['Alpha'] != '' and row['Alpha'] != 'cm' and row['Alpha'] != 'Ia':
                    alpha = float(row['Alpha'])
                    beta = float(row['Beta'])
                    angle_dict[key] = [alpha, beta]
        return angle_dict

    def loadMartaCSV(self):     #this is the loading process I used, and the one currently used
        angle_dict = {}
        with open(self.anglecsv) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:      #as seen below you can use calpha or alpha, in this case the standard one used is calpha

                # Get key and format key
                key = row['Match 1']    #Match 1 contains the image name of the nonannotated images
                #if row['Alpha'] != '' and row['Alpha'] != 'cm' and row['Alpha'] != 'X' and row['Alpha'] != 'No ' and row['Beta'] != 'cm' and row['Alpha'] != 'no':
                #This line below is just extracting the values that are numbers
                if row['C Alpha'] != '' and row['C Alpha'] != 'cm' and row['C Alpha'] != 'X' and row['C Alpha'] != 'No ' and row['Beta'] != 'cm' and row['C Alpha'] != 'no':
                    alpha = float(row['C Alpha'])
                    #alpha = float(row['Alpha'])
                    beta = float(row['Beta'])       #the beta value was initially also saved though not used
                    angle_dict[key] = [alpha, beta]
        return angle_dict
    """

    #   KRITHIKA - This function loads both the binary outcome values and the patient details. This will be the function used
    def loadKrithikaCSV(self):
        outcome_dict = {}
        patient_details = {}
        coordinates_dict = {}
        with open(self.anglecsv) as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                key = row['Match 1']    #Match 1 contains the image name of the nonannotated images
                #if row['Alpha'] != '' and row['Alpha'] != 'cm' and row['Alpha'] != 'X' and row['Alpha'] != 'No ' and row['Beta'] != 'cm' and row['Alpha'] != 'no':
                #This line below is just extracting the values that are numbers
                #if row['C Alpha'] != '' and row['C Alpha'] != 'cm' and row['C Alpha'] != 'X' and row['C Alpha'] != 'No ' and row['C Alpha'] != 'no' \
                #and row['C Beta'] != 'No' and row['C Beta'] != ' ' and row['C Beta'] != 'X' \
                xCoordinate = row['X-pos for BB']
                yCoordinate = row['Y-pos for BB']
                coordinates_dict[key] = [xCoordinate, yCoordinate]
                if row['Indication'] != ' '\
                and row['Outcome'] != ' ' \
                and row['Birthweight in kg'] != '0' and row['Birthweight in kg'] != 'Unreccorded' and row['Birthweight in kg'] != 'Unknown' and row['Birthweight in kg'] != 'Unknown ':
                    #alpha = float(row['C Alpha'])
                    #beta = float(row['C Beta'])       #the beta value was initially also saved though not used
                    side = row['Side']
                    gender = row['Patient Sex']
                    indication = row['Indication']
                    birthweight = float(row['Birthweight in kg']) #This is the cleaned version of the birthweight data in kg
                    outcome = float(row['Binary Outcome'])

                    patient_details[key] = [side, gender, indication, birthweight]
                    outcome_dict[key] = [outcome]
        return outcome_dict, patient_details

    """
    #This was the old function used, where the output was the alpha angle. This isn't used anymore
    def formatAngleData(self):
        files = os.listdir(self.imagedir)

        files_with_angle = []
        angle_data = []
        for f in files:
            if f in self.angle_dict:        #this is to ensure all file names in Match 1 are also present in the image directory chosen
                angle_data.append(self.angle_dict[f])       #if this is the case the angle data will be saved for that file
                files_with_angle.append(f)      ##and the file name will also be stored

        return np.array(angle_data), files_with_angle   #note in this case the angle data and patient data lists are converted into numpy arrays
    """

    #This is the new function used, where the output is the binary outcome value. This includes other discrete pieces of patient data as well
    def formatOutcomeData(self):
        files = os.listdir(self.imagedir)

        files_with_angle = []
        outcome_data = []
        patient_data = []
        for f in files:
            if f in self.outcome_dict:        #this is to ensure all file names in Match 1 are also present in the image directory chosen
                outcome_data.append(self.outcome_dict[f])       #if this is the case the outcome data will be saved for that file
                files_with_angle.append(f)      ##and the file name will also be stored
                patient_data.append(self.patient_dict[f]) #And the relevant patient details will also be stored

        return np.array(outcome_data), files_with_angle, np.array(patient_data)   #note in this case the angle data and patient data lists are converted into numpy arrays

    """
    def cropimages(self, img):

        #Have done it previously so not necessary anymore

        return img
    """

    def loadImageData(self):
        files = self.files

        for f in tqdm(files):
            img = io.imread(self.imagedir + f)
            if(self.useCropping):
                img = cropimages(img)
            img = transform.resize(img, (self.HEIGHT, self.WIDTH, COLOR_CHANNELS), mode='constant')
            img = img[:,:,1]

            if f == files[0]:
                imgs = img[None,:]
            else:
                imgs = np.concatenate( (imgs, img[None,:]), axis=0)

        imgs = np.reshape( imgs, (-1, self.HEIGHT, self.WIDTH, self.CHANNELS))
        return imgs

    """
    #This is the old generate function, where the input was only the scan and the output was the alpha angle
    def generate(self):     #this is the initial function used to lead onto the others
        self.angle_dict = self.loadMartaCSV()       #run the function to get the angle values dictionary and the patient details dictionary
        self.angle_data, self.files = self.formatAngleData()    #run the function to get the files and angle/patient values which match and are present

        # Load image data
        self.image_data = self.loadImageData()

        # Grab angle data:
        self.angle_data = np.reshape(self.angle_data[:,0], (-1, 1)) #This only uses the alpha angle for all keys

        # Randomize data order
        if self.useRandomOrder:
            indices = [_ for _ in range(len(self.angle_data))]
            self.image_data = self.image_data[indices]
            self.angle_data = self.angle_data[indices]

        # Data preprocessing
        if self.useNormalization:
            self.image_data, self.img_min, self.img_max = self.normalize(self.image_data)

        if self.useWhitening:
            self.image_data, self.img_mean, self.img_std = self.whiten(self.image_data)

        if self.useBinaryClassify:
            self.angle_data = self.threshold(self.angle_data)   #refer to the threshold function to extract the angle data as 1 or 0s
        else:
            if self.useNormalization:
                self.angle_data, self.ang_min, self.ang_max = self.normalize(self.angle_data)
            if self.useWhitening:
                self.angle_data, self.ang_mean, self.ang_std = self.whiten(self.angle_data)

        # Split data into test/training/validation sets
        index1 = int(self.ratio1 * len(self.image_data)) # Split index
        self.x_train = self.image_data[0:index1, :]     #this is 80% training
        index2 = int((self.ratio1+self.ratio2) * len(self.image_data))
        self.x_val = self.image_data[index1:index2, :]  #then 10% validation
        self.x_test = self.image_data[index2:, :]   #and 10% testing

        if self.useBinaryClassify:      #Split the y labels now, the value of the angle
            self.y_train = self.angle_data[0:index1]
            self.y_val = self.angle_data[index1:index2]
            self.y_test = self.angle_data[index2:]

        else:
            self.y_train = self.angle_data[0:index1]
            self.y_test = self.angle_data[index1:index2]
            self.y_test = self.angle_data[index2:]
    """
    #This is the new generate function used, where the output is the binary outcome value and the inputs include both scans and other patient information
    def generate2(self):     #this is the initial function used to lead onto the others
        self.outcome_dict, self.patient_dict = self.loadKrithikaCSV()       #run the function to get the angle values dictionary and the patient details dictionary
        self.outcome_data, self.files, self.patient_data = self.formatOutcomeData()    #run the function to get the files and angle/patient values which match and are present

        # Load image data
        self.image_data = self.loadImageData()

#        print("Hi")
#        print(np.shape(self.patient_data))

        # Extract the necessary data from the dictionaries and reshape into column vectors
        self.outcome_data = np.reshape(self.outcome_data[:,0], (-1, 1)) #The (-1, 1) turns the matrix into a column vector
        #self.alphaAngle_data = np.reshape(self.patient_data[:,0], (-1, 1)) #This grabs the alpha angle for all the keys and turns it into a column vector
        #self.betaAngle_data = np.reshape(self.patient_data[:,1], (-1, 1)) #This grabs the beta angle for all the keys and turns it into a column vector
        self.side_data = np.reshape(self.patient_data[:,0], (-1, 1))
        self.gender_data = np.reshape(self.patient_data[:,1], (-1, 1))
        self.indication_data = np.reshape(self.patient_data[:,2], (-1, 1))
        self.birthweight_data = np.reshape(self.patient_data[:,3], (-1, 1))

        # Randomize data order
        if self.useRandomOrder:
            indices = [_ for _ in range(len(self.outcome_data))]
            self.image_data = self.image_data[indices]
            self.outcome_data = self.outcome_data[indices]
            #self.alphaAngle_data = self.alphaAngle_data[indices]
            #self.betaAngle_data = self.betaAngle_data[indices]
            self.side_data = self.side_data[indices]
            self.gender_data = self.gender_data[indices]
            self.indication_data = self.indication_data[indices]
            self.birthweight_data = self.birthweight_data[indices]

        #Change data type of the alpha, beta and birthweight arrays to be floats
        #self.alphaAngle_data = self.alphaAngle_data.astype(np.float64)
        #self.betaAngle_data = self.betaAngle_data.astype(np.float64)
        self.birthweight_data = self.birthweight_data.astype(np.float64)
        self.outcome_data = self.outcome_data.astype(np.float64)

        # Data preprocessing
        if self.useNormalization:
            self.image_data, self.img_min, self.img_max = self.normalize(self.image_data)

        if self.useWhitening:
            self.image_data, self.img_mean, self.img_std = self.whiten(self.image_data)

        #THE BELOW CODE IS TO DO WITH THE ALPHA ANGLE AND TURNING IT BINARY--> SEEING AS I AM NOT KEEPING ALPHA AS OUTPUT, DON'T NEED TO TURN IT BINARY
        #if self.useBinaryClassify:
        #    self.angle_data = self.threshold(self.angle_data)   #refer to the threshold function to extract the angle data as 1 or 0s
        #else:
        if self.useNormalization:
            #self.alphaAngle_data, self.alphaAngle_min, self.alphaAngle_max = self.normalize(self.alphaAngle_data)
            #self.betaAngle_data, self.betaAngle_min, self.betaAngle_max = self.normalize(self.betaAngle_data)
            self.birthweight_data, self.birthweight_min, self.birthweight_max = self.normalize(self.birthweight_data)
        if self.useWhitening:
            #self.alphaAngle_data, self.alphaAngle_mean, self.alphaAngle_std = self.whiten(self.alphaAngle_data)
            #self.betaAngle_data, self.betaAngle_mean, self.betaAngle_std = self.whiten(self.betaAngle_data)
            self.birthweight_data, self.birthweight_mean, self.birthweight_std = self.whiten(self.birthweight_data)

        def one_hot(array):
            unique, inverse = np.unique(array, return_inverse=True)
            onehot = np.eye(unique.shape[0])[inverse]
            return onehot
        #one-hot encoding the categorical data
        self.gender_data = one_hot(self.gender_data)
        self.side_data = one_hot(self.side_data)
        self.indication_data = one_hot(self.indication_data)

        #Number of different values in gender, side and indication
        self.GENDER = np.shape(self.gender_data)
        self.GENDER = self.GENDER[1]

        self.SIDE = np.shape(self.side_data)
        self.SIDE = self.SIDE[1]

        self.INDICATION = np.shape(self.indication_data)
        self.INDICATION = self.INDICATION[1]

        # Split the image data into test/training/validation sets
        index1 = int(self.ratio1 * len(self.image_data)) # Split index
        self.x_train = self.image_data[0:index1, :]     #this is 80% training
        index2 = int((self.ratio1+self.ratio2) * len(self.image_data))
        self.x_val = self.image_data[index1:index2, :]  #then 10% validation
        self.x_test = self.image_data[index2:, :]   #and 10% testing

        #Split the patient details data into test/training/validation sets
        #GENDER
        self.gender_train = self.gender_data[0:index1, :]
        self.gender_val = self.gender_data[index1:index2, :]
        self.gender_test = self.gender_data[index2:, :]

        #SIDE
        self.side_train = self.side_data[0:index1, :]
        self.side_val = self.side_data[index1:index2, :]
        self.side_test = self.side_data[index2:, :]

        #INDICATION
        self.indication_train = self.indication_data[0:index1, :]
        self.indication_val = self.indication_data[index1:index2, :]
        self.indication_test = self.indication_data[index2:, :]

        #BIRTHWEIGHT
        self.birthweight_train = self.birthweight_data[0:index1]
        self.birthweight_val = self.birthweight_data[index1:index2]
        self.birthweight_test = self.birthweight_data[index2:]

        """
        #ALPHA
        self.alphaAngle_train = self.alphaAngle_data[0:index1]
        self.alphaAngle_val = self.alphaAngle_data[index1:index2]
        self.alphaAngle_test = self.alphaAngle_data[index2:]

        #BETA
        self.betaAngle_train = self.betaAngle_data[0:index1]
        self.betaAngle_val = self.betaAngle_data[index1:index2]
        self.betaAngle_test = self.betaAngle_data[index2:]
        """
        #Don't think we need this
        #if self.useBinaryClassify:      #Split the y labels now, the value of the angle
        #    self.y_train = self.outcome_data[0:index1]
        #    self.y_val = self.outcome_data[index1:index2]
        #    self.y_test = self.outcome_data[index2:]

        #else:
        self.y_train = self.outcome_data[0:index1]
        self.y_val = self.outcome_data[index1:index2]
        self.y_test = self.outcome_data[index2:]


    #This function isn't being used anymore - don't need to turn the alpha angles into binary values of diseased or healthy
    #This function is used to classify if a person is diseased or not by setting a threshold on the predicted alpha angle
    def threshold(self, data):
        threshold = (data < self.binaryThreshold) * 1   #if the value of the alpha angle is below 60 the patient has a 1/diseased, if above or equal it has a 0/healthy
        onehot = np.concatenate( (1 - threshold, threshold), axis = 1)  # OneHot Encoding, mainly used for multiple classes
        return onehot[:, 0]     #notice how we only take the first column which corresponds to a 1 for below 60 and a 0 for above

    def next_batch(self, batch_size):
        length = self.x_train.shape[0]
        indices = np.random.randint(0, length, batch_size) # Grab batch_size values randomly
        return [self.x_train[indices], self.y_train[indices]]

    def normalize(self, data):
        max = np.max(data)
        min = np.min(data)
        return (data - min) / (max - min), min, max

    def whiten(self, data):
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std, mean, std

    def denormalize(self, data, min, max):
        return data * (max - min) + min

    def undo_whitening(self, data, mean, std):
        return data * std + mean

    def undo_whitening_and_denormalize(self, data, stats):
        return (data * stats.std + stats.mean) * (stats.max - stats.min) + stats.min

    def plot(self, index):
        plt.imshow(self.image_data[index,:,:,0])
        plt.show()

if __name__ == '__main__':
    data = DataGenerator2()     #important to use DataGenerator2 for the alpha angle
    xs, ys = data.next_batch(128)

    #plt.figure(1)
    #data.plot(1)

    plt.figure()
    plt.hist(ys)
    plt.show()

    plt.figure()
    plt.hist(data.y_train)
    plt.show()
