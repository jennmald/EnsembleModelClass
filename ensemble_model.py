#imports
# Tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import PolynomialFeatures
# Data & Plotting
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import statistics
import os

#Ensemble
# creates the ensemble
# num_models = the number of models that will be a part of the ensemble
# num_input = input size of each model and ensemble model
# num_output = output size of each model and ensemble model
# num_hidden = how many hidden layers for individual model
# node = number of nodes in each hidden layer
# xtrain = training x values
# ytrain = training y values
# xtest = test x values
# ytest = test y values
class Ensemble:
    def __init__(self,num_models, num_input, num_output, num_hidden, node, epoch, period, xtrain, ytrain, xtest, ytest):
        self.num_models = num_models
        self.num_inputs = num_input
        self.num_outputs = num_output
        self.hidden = num_hidden
        self.nodes = node
        self.epochs = epoch
        self.period = period
        self.x_train = xtrain 
        self.y_train = ytrain
        self.x_test = xtest
        self.y_test = ytest
        self.total_iter = int(self.epochs/self.period)
        #for normalizing data
        self.max_trainy = max(y_train) 
        #for storing each models information
        self.all_model_outputs = {}
        self.all_model_residues = {}
        self.all_model_loss = {}
        self.all_model_accuracy = {}
        
# Builds a functional model for numerical data
# Returns the completed model
    def buildModel(self):
        input_layer = Input(self.num_inputs) #inputs are always the same
        temp_hidden_layer = Dense(self.nodes, activation='relu')(input_layer) #creates first layer
        for l in range(self.hidden-1): #creates all other layers
            new_temp_hidden = Dense(self.nodes, activation='relu')(temp_hidden_layer)
            temp_hidden_layer = new_temp_hidden
        concat_layer = tf.keras.layers.Concatenate()([input_layer, temp_hidden_layer])
        output_layer = Dense(self.num_outputs)(concat_layer)
        model = Model(inputs=input_layer, outputs=output_layer)
        return model

# Trains each individual model by the period specified 
# Benchmarks progress and stores it in a dictionary
# Parameters 
# model = the model you wish to train
# model_number = the number of the model, to use for dictionary organization
    def train_by_period(self, model, model_number):
        #compile model
        model.compile(optimizer="Adam",loss='mse', metrics=['accuracy'])
        #fit model (data is already normal (0,1))
        #uses callbacks
        history = model.fit(self.x_train, self.y_train, epochs=self.period, callbacks=[call_back], verbose = 0)
        #stores the outputs for visualization later
        outputs = model.predict(self.x_test)
        #getting the loss and storing it per period in loss dictionary
        if model_number not in self.all_model_loss:
            self.all_model_loss[model_number]={}
            this_dict =  self.all_model_loss[model_number]
            this_dict[0] = history.history['loss']
        #getting the accuracy and storing it per period in loss dictionary
        if model_number not in self.all_model_accuracy:
            self.all_model_accuracy[model_number]={}
            accu_dict =  self.all_model_accuracy[model_number]
            accu_dict[0] = history.history['accuracy']
        #normalizing the outputs and storing them per period in the outputs dictionary
        normal_outputs = self.normalize_data(outputs)
        if model_number not in self.all_model_outputs:
            self.all_model_outputs[model_number]={}
            this_dict =  self.all_model_outputs[model_number]
            this_dict[0] = normal_outputs
        #computing the residues using the normalized data and storing them per period in the residues dictionary
        residue_list = self.residues(self.y_test, normal_outputs)
        residue_list = self.residues(self.y_test, outputs)
        if model_number not in self.all_model_residues:
            self.all_model_residues[model_number]={}
            this_dictr =  self.all_model_residues[model_number]
            this_dictr[0] = residue_list
        #trains model for entire epochs but checkpoints and reloads every period
        for i in range(1, self.total_iter):
            #build new model
            latest = tf.train.latest_checkpoint(checkpoint_dir)
            new_model = self.buildModel()
            new_model.compile(optimizer="Adam",loss='mse', metrics=['accuracy'])
            #load weights
            new_model.load_weights(latest)
            history = new_model.fit(self.x_train, self.y_train, epochs=self.period, callbacks=[call_back],verbose = 0)
            new_outputs = new_model.predict(self.x_test)
            #getting the accuracy and storing it per period in loss dictionary
            accu_dict =  self.all_model_accuracy[model_number]
            accu_dict[i] = history.history['accuracy']
            #getting the loss and storing it per period in loss dictionary
            loss_dict =  self.all_model_loss[model_number]
            loss_dict[i] = history.history['loss']
            #normalizing the outputs and storing them per period in the outputs dictionary
            new_normal_outputs = self.normalize_data(new_outputs)
            out_dict =  self.all_model_outputs[model_number]
            out_dict[i] = new_normal_outputs
            #computing the residues using the normalized data and storing them per period in the residues dictionary
            new_residue_list = self.residues(self.y_test, new_normal_outputs)
            res_dict =  self.all_model_residues[model_number]
            res_dict[i] = new_residue_list

# If needed this method will return data to a
# normalized state
# Often needed to assist in training more accurately
# Parameters
# to_norm = list of predictions between (0,1)
# max_trainy = max of the original train data (unnormalized)
    def normalize_data(self, to_norm):
        norm_data = to_norm*self.max_trainy
        return norm_data

# Creates the ensemble and averages the outputs 
# Returns the finished models and the dictionaries with the model's progress
    def ensemble(self):
        input_layer = Input(self.num_inputs) #create input layer
        ensemble_outputs = [] #store the outputs of the ensemble model
        for m in range(0, self.num_models):
            print('NEW MODEL')
            model = self.buildModel()
            ensemble_outputs.append(model(input_layer))
            #saving individual outputs for models -- to use for std dev
            self.train_by_period(model, m+1) 
        #averaging all output layers for the ensemble model
        average_outputs = tf.keras.layers.Average()(ensemble_outputs) 
        model = Model(input_layer, average_outputs)
        return ensemble_outputs,model,self.all_model_outputs,self.all_model_residues,self.all_model_loss,self.all_model_accuracy

# Finds the distance between predicted and actual points
# Parameters 
# true_values: the values you are expecting (usually y-test)
# predicted_values: the values the model has predicted
# returns a list of residues
    def residues(self, true_values, predicted_values):
        residues = []
        for i in range(0,len(true_values)):
            #indexing because of np array issues
            difference = true_values[i] - (predicted_values[i])[0]
            residues.append(difference) #regular list for plotting
        return residues

# Calculates the standard deviation between each point
# Parameters  
# populations = list of lists of predictions
# means = the average outputs the ensembles predicted
    def standard_dev(self, populations, means):
        std_list = []
        number_lists = len(populations)
        # how many points there are to loop through
        for index in range(0, len(means)):
            sum_squared = 0
            for prediction_list in range(0,len(populations)):
                temporary_list = populations[prediction_list]
                sum_squared += (temporary_list[index]-means[index])**2
            fraction = sum_squared/(number_lists)
            root = math.sqrt(fraction)
            std_list.append(root)
        return std_list

# ORGANIZE DATA
#Train Data
x_train = np.linspace(-20, 20, 100)
y_train = x_train**3 + np.random.normal(0,20,100)
#normalize y data ONLY
#( y - min ) / (max - min)
minTY = min(y_train)
maxTY = max(y_train)
norm_train_y = (y_train-minTY) / (maxTY - minTY)
norm_train_y = np.array(norm_train_y)

#POLYNOMIAL PREPROCESSING - created expanded data set
poly = PolynomialFeatures(3) #cubic
train_x_expanded = np.expand_dims(x_train, axis=1)
train_x_expanded = poly.fit_transform(train_x_expanded)

#Test Data
test_x = np.linspace(-40, 40, 200)
test_y = test_x**3
#normalize y data ONLY
#( y - min ) / (max - min)
minTestY = min(test_y)
maxTestY = max(test_y)
norm_test_y = (test_y-minTestY) / (maxTestY - minTestY)
norm_test_y = np.array(norm_test_y)
#expand the test data for polynomial fitting
test_x_expanded = np.expand_dims(test_x, axis=1)
test_x_expanded = poly.fit_transform(test_x_expanded)


#HOW TO USE

# period decided on for frequency of checkpoints
p = 1000
# checkpoint name and directory
checkpoint_path = "ensemble/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
call_back = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,verbose=1, period = p)

# test ensemble class
m = 20 #number of models
i = 4 #inputs
o = 1 #outputs
h = 4 #hidden layers
n = 512 #nodes
e = 10000 #epochs

test_ens = Ensemble(m, i, o, h, n, e, p, train_x_expanded, norm_train_y, test_x_expanded, test_y)

ens_outputs, ensemble_model, all_outputs, all_residues,all_loss,all_accuracy= test_ens.ensemble()
ensemble_model.summary()
ensemble_model.compile(optimizer="Adam",loss='mse', metrics=['accuracy'])
ensemble_history = ensemble_model.fit(train_x_expanded, norm_train_y, epochs = 10000, verbose = 0)
ensemble_outputs = ensemble_model.predict(test_x_expanded)

#plot of the loss
plt.plot(ensemble_history.history['loss'], color='c')
plt.title('Amount of Loss After Each Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train'], loc='upper right')
plt.show()

#plot of original data
plt.scatter(x_train,y_train, c= 'm')
plt.title('original train data')
plt.ylabel('y')
plt.xlabel('x')
plt.show()
#plot of original data
plt.scatter(test_x,test_y, c= 'm')
plt.title('original test data')
plt.ylabel('y')
plt.xlabel('x')
plt.show()

#normalize data
normal_ensemble = (ensemble_outputs+minTY)*(maxTY-minTY)
plt.scatter(x_train,y_train, c= 'm')
plt.scatter(test_x,normal_ensemble, c= 'c')
plt.title('predicted vs original train data')
plt.ylabel('y')
plt.xlabel('x')
plt.show()
#plot of original data
plt.scatter(test_x,test_y, c= 'm')
plt.scatter(test_x,normal_ensemble, c= 'c')
plt.title('predicted data vs actual test')
plt.ylabel('y')
plt.xlabel('x')
plt.show()

iterations = int(e/p)

# ALL RESIDUE PLOTS
figr,axr =  plt.subplots(m,iterations,figsize=(50,50))
set_xr = 0
for model_res in all_residues:
    resis = all_residues[model_res]
    set_yr =0
    for res_period in range(0,iterations):
        axr[set_xr][set_yr].plot(test_x, resis[res_period])
        axr[set_xr][set_yr].set_title('Residue: Model '+ str(set_xr+1)+' Period '+str(res_period))
        set_yr = set_yr+1
    set_xr = set_xr + 1   
plt.show()

modr = 1
for model_res in all_residues:
    r = all_residues[model_res]
    all_model_res = []
    for mr in range(0,iterations):
        for i in r[mr]:
            all_model_res.append(i)
    plt.plot(all_model_res)
    plt.title('Model '+ str(modr) + ' Residues')   
    plt.show()
    residual_sum_of_squares = 0
    for rsqu in all_model_res:
        residual_sum_of_squares+= (rsqu)**2
    print('Model '+ str(modr) + ' has residual sum squared of '+ str(residual_sum_of_squares))
    modr+=1
        

#ALL LOSS PLOTS
fig,ax =  plt.subplots(m,iterations,figsize=(30,30))
set_x = 0
set_y = 0

for model_loss in all_loss:
    losses = all_loss[model_loss]
    set_y = 0
    for loss_period in range(0,iterations):
        ax[set_x][set_y].plot(losses[loss_period])
        ax[set_x][set_y].set_title('Loss: Model '+ str(set_x+1)+' Period '+str(loss_period))
        set_y = set_y+1
    set_x = set_x + 1   
plt.show()

mod = 1
for model_loss in all_loss:
    l = all_loss[model_loss]
    all_model_loss = []
    for lp in range(0,iterations):
        for i in l[lp]:
            all_model_loss.append(i) 
    plt.plot(all_model_loss)
    plt.title('Model '+ str(mod) + ' Loss')   
    plt.show()
    mod+=1

#ALL ACCURACY PLOTS
figa,axa =  plt.subplots(m,iterations,figsize=(30,30))
set_xa = 0
for model_acc in all_accuracy:
    accus = all_accuracy[model_acc]
    set_ya = 0
    for accu_period in range(0,iterations):
        axa[set_xa][set_ya].plot(accus[accu_period])
        axa[set_xa][set_ya].set_title('Accuracy: Model '+ str(set_xa+1)+' Period '+str(accu_period))
        set_ya = set_ya+1
    set_xa = set_xa + 1   
plt.show()

moda = 1
for model_acc in all_accuracy:
    accus = all_accuracy[model_acc]
    all_model_acc = []
    for ma in range(0,iterations):
        for i in accus[ma]:
            all_model_acc.append(i) 
    plt.plot(all_model_acc)
    plt.title('Model '+ str(moda) + ' Accuracy')   
    plt.show()
    moda+=1

#ALL PREDICTION PLOTS
figo,axo =  plt.subplots(m,iterations,figsize=(50,70))
set_xo = 0
for model_out in all_outputs:
    temp_out = all_outputs[model_out]
    set_yo = 0
    for out_period in range(0,iterations):
        axo[set_xo][set_yo].plot(test_x,temp_out[out_period])
        axo[set_xo][set_yo].set_title('Predictions: Model '+ str(set_xo+1)+' Period '+str(out_period))
        set_yo = set_yo+1
    set_xo = set_xo + 1   
plt.show()

modo = 1
for model_out in all_outputs:
    temp_out = all_outputs[model_out]
    for mo in range(0,iterations):
        plt.plot(test_x, temp_out[mo], label = 'period'+str(mo))
    plt.title('Model '+ str(modo) + ' Prediction')
    plt.plot(test_x,test_y, c='r', label = 'Actual')
    plt.plot(test_x,normal_ensemble, c='k',label = 'Ensemble')
    #plt.legend()
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.show()
    modo+=1
