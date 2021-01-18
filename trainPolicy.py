import os
import sys
import numpy as np
import keras
from keras.utils import np_utils
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing import *
from keras.layers.pooling import *
from sklearn.model_selection import train_test_split
# 
import argparse

batch_size = 256
nb_epoch = 10000
WEIGHT_GAIN = 10.0

def gaussian_nll_weight(ytrue, ypreds):
    # Keras implmementation of multivariate Gaussian negative loglikelihood loss function. 
    # This implementation implies diagonal covariance matrix.
    from keras import backend as K
    
    n_dims = int(int(ypreds.shape[1])/2)
    mu = ypreds[:, 0:n_dims]
    logsigma = ypreds[:, n_dims:]
    weight = 1.0 + WEIGHT_GAIN*ytrue[:, n_dims:]  
    ytrue = ytrue[:, 0:n_dims]  
    
    diff = ytrue - mu
    mse = -0.5*K.sum(weight*K.square(diff/K.exp(logsigma)), axis=1)        
    sigma_trace = -K.sum(logsigma, axis=1)
    # log2pi = -0.5*n_dims*np.log(2*np.pi)
    
    log_likelihood = mse + sigma_trace # + log2pi   

    return K.mean(-log_likelihood)    

def CNNmodelJW(w,h,c, nb_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',activation='relu',input_shape=(w,h,c)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes*2, activation='linear'))

    model.compile(optimizer = adam(lr = 1e-5), loss=gaussian_nll_weight, metrics=['mse'])

    return model

def learningLoop(ITER, x_Data, y_Data, batch_size, nb_epoch):
    print('############################')
    print('### Traning Start ~~~!!! ###')
    print('Batch size:', batch_size, 'epoch size:', nb_epoch)
    
    checkpointer = ModelCheckpoint(filepath = './RUNS/' + str(ITER) + '/trained_policy.hdf5', verbose=1, save_best_only = True, period=100)
    
    tb_hist = keras.callbacks.TensorBoard(log_dir = DIRECTORY_NAME , histogram_freq = 0, write_graph = True, write_images = True)

    x_train, x_test, y_train, y_test = train_test_split(x_Data.astype('float32'), y_Data, test_size = 0.2)
    x_Data = y_Data = None
    
    model = CNNmodelJW(x_train.shape[1], x_train.shape[2], x_train.shape[3], y_train.shape[1]-2)
    print(model.summary())

    model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = batch_size, epochs = nb_epoch, callbacks = [checkpointer, tb_hist], verbose = 1)

    model.save_weights('./RUNS/' + str(ITER) + '/trained_policy_weight.hdf5')
    print('### Traning Finish ~~~!!! ###')
    print('#############################')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Weight-DAgger')
    argparser.add_argument('--policy', default='---', help='select the data to train the policy')
    args = argparser.parse_args()
    ITER = args.policy

    ### data load
    if ITER == 'BC' or str(ITER[:-1]) == 'WeightDAgger':
        x_data = np.load('./DATA/'+str(ITER)+'/seg.npy')     
        y_data = np.load('./DATA/'+str(ITER)+'/label.npy')   
    else:
        if int(ITER[-1]) == 1:
            x_data1 = np.load('./DATA/BC/seg.npy')
            y_data1 = np.load('./DATA/BC/label.npy')
        else:
            x_data1 = np.load('./DATA/'+str(ITER[:-1])+str(int(ITER[-1])-1)+'/aggregated_seg.npy')
            y_data1 = np.load('./DATA/'+str(ITER[:-1])+str(int(ITER[-1])-1)+'/aggregated_label.npy')

        x_data2 = np.load('./DATA/'+str(ITER)+'/seg.npy')
        y_data2 = np.load('./DATA/'+str(ITER)+'/label.npy')

        x_data = np.concatenate((x_data1, x_data2), axis=0)
        y_data = np.concatenate((y_data1, y_data2), axis=0)
        
        np.save("./DATA/"+str(ITER)+"/aggregated_seg", x_data)
        np.save("./DATA/"+str(ITER)+"/aggregated_label", y_data)
    
    x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], x_data.shape[2], 1)
    
    print("x_Data.shape : ", x_data.shape)
    print("y_Data.shape : ", y_data.shape)
##### Look-ahead Point
    _y_data = np.zeros((y_data.shape[0], 4))
    _y_data[:,:2] = y_data[:,:2]
    ### error
    _y_data[:,2] = y_data[:,2]
    _y_data[:,3] = y_data[:,2]
    ###
    y_data = _y_data
    print("y_Data.shape : ", y_data.shape)

####
    ### make directory
    DIRECTORY_NAME = str('./RUNS/' + str(ITER))
    print('@@@@@ check point dir: ', DIRECTORY_NAME, '@@@@@')
    if not(os.path.isdir(DIRECTORY_NAME)):
        os.makedirs(os.path.join(DIRECTORY_NAME))

    DIRECTORY_NAME = str('./RUNS/graph/_' + str(ITER))
    print('@@@@@ graph dir:', DIRECTORY_NAME, '@@@@@')
    if not(os.path.isdir(DIRECTORY_NAME)):
        os.makedirs(os.path.join(DIRECTORY_NAME))

    ### start training !
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))
    
    learningLoop(ITER, x_data, y_data, batch_size, nb_epoch)