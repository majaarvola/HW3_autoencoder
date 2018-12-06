# import modules
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# load the preprocessed data
COG1 = np.load("preprocessed_data/COG1_prob.npy")
COG160 =  np.load("preprocessed_data/COG160_prob.npy")
COG161 =  np.load("preprocessed_data/COG161_prob.npy")
all_families = np.concatenate((COG1,COG160,COG161))

# divide data into train (75%) and test (25%)
x_train, x_test = train_test_split(all_families, test_size=0.25)

# state the dimensions to use in autoencoder
input_dim = 400 # our sequences has dimension 400x1
encoding_dim = 25  # We want to compress the dimension with a factor of 16, from 400 to 25

# placeholder for input and encoded input
input = Input(shape=(input_dim,)) 
encoded_input = Input(shape=(encoding_dim,))

# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(400, activation='sigmoid')(encoded)

# Create the autoencoder model and for the encoder part
autoencoder = Model(input, decoded)
encoder = Model(input, encoded)

# Get the last layer of decoder part and create the decoder model
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

# compile the autoencoder with some optimiser and loss function
autoencoder.compile(optimizer='adam', loss='mean_squared_logarithmic_error')

# train the model 
result = autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=10,
                shuffle=True,
                validation_data=(x_test, x_test))

# extract encoded vector for sequences
encoded_vector = encoder.predict(all_families)  

# print encoded vector and save to txt file
print("The resulting encoded vector is: ")      
print(encoded_vector)
np.savetxt('resultat.txt', encoded_vector, delimiter=',') 

# visualisation of change in loss during training to make sure in converges
plt.figure
plt.plot(result.history['loss'], color ='green')
plt.plot(result.history['val_loss'], color ='orchid')
plt.xlim(0,50-1)
plt.title('Model Loss')
plt.ylabel('Loss Value')
plt.xlabel('Number of Epochs')
plt.legend(['Loss for training data', 'Loss for test data']) #
plt.show()