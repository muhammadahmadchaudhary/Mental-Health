import pandas as pd
import numpy as np 
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.model_selection import KFold
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model

model = load_model('mental_health_model.h5')

# Min and max values for scaling
min_max_values = {
    'GEN_005': (0, 5),
    'GEN_015': (0, 5),
    'GEN_020': (0, 5),
    'GEN_025': (0, 5),
    'GENDVHDI': (0, 5),
    'GEN_010': (0, 10),
    'GEN_030': (0, 5),
    'ALC_015': (0, 10),
    'ALC_020': (0, 10),
    'ALW_005': (0, 10),
    'ALWDVWKY': (0, 1000),
    'ALWDVDLY': (0, 1000),
    'ALWDVLTR': (0, 10),
    'ALWDVSTR': (0, 10),
    'DRGDVYA': (0, 10),
    'DHH_SEX': (1, 2),
    'DHHGAGE': (1, 5),
    'DHHGMS': (1, 5),
    'DHHDGHSZ': (1, 10),
    'DOALW': (0, 10)
}

def normalize_input(input_data, min_max_values):
    normalized_data = {}
    for feature, value in input_data.items():
        min_val, max_val = min_max_values[feature]
        normalized_data[feature] = (value - min_val) / (max_val - min_val)
    return normalized_data

# Function to map the predicted value to the corresponding category
def map_prediction_to_category(prediction):
    categories = {
        0: 'Poor',
        1: 'Fair',
        2: 'Good',
        3: 'Very good',
        4: 'Excellent',
        9: 'Not stated'
    }
    # Find the index of the maximum value in the prediction
    max_index = prediction.argmax()
    return categories.get(max_index, 'Unknown')

# Function to take input data and make predictions
def predict_mental_health(input_data):
    # Normalize the input data
    normalized_data = normalize_input(input_data, min_max_values)
    
    # Ensure the input data is in the correct format
    input_df = pd.DataFrame(normalized_data, index=[0])
    
    # Use the loaded model to make predictions
    prediction = model.predict(input_df)
    
    return prediction

# Function to get user input for all features
def get_user_input():
    user_input = {}
    user_input['GEN_005'] = float(input('Enter GEN_005: '))
    user_input['GEN_015'] = float(input('Enter GEN_015: '))
    user_input['GEN_020'] = float(input('Enter GEN_020: '))
    user_input['GEN_025'] = float(input('Enter GEN_025: '))
    user_input['GENDVHDI'] = float(input('Enter GENDVHDI: '))
    user_input['GEN_010'] = float(input('Enter GEN_010: '))
    user_input['GEN_030'] = float(input('Enter GEN_030: '))
    user_input['ALC_015'] = float(input('Enter ALC_015: '))
    user_input['ALC_020'] = float(input('Enter ALC_020: '))
    user_input['ALW_005'] = float(input('Enter ALW_005: '))
    user_input['ALWDVWKY'] = float(input('Enter ALWDVWKY: '))
    user_input['ALWDVDLY'] = float(input('Enter ALWDVDLY: '))
    user_input['ALWDVLTR'] = float(input('Enter ALWDVLTR: '))
    user_input['ALWDVSTR'] = float(input('Enter ALWDVSTR: '))
    user_input['DRGDVYA'] = float(input('Enter DRGDVYA: '))
    user_input['DHH_SEX'] = float(input('Enter DHH_SEX: '))
    user_input['DHHGAGE'] = float(input('Enter DHHGAGE: '))
    user_input['DHHGMS'] = float(input('Enter DHHGMS: '))
    user_input['DHHDGHSZ'] = float(input('Enter DHHDGHSZ: '))
    user_input['DOALW'] = float(input('Enter DOALW: '))  
    return user_input

# Get user input
user_input = get_user_input()

# Make a prediction
prediction = predict_mental_health(user_input)

# Assuming GENDVMHI is the first output, extract the first element from the prediction
gendvmhi_category = map_prediction_to_category(prediction[0])
print(f'Predicted GENDVMHI: {gendvmhi_category}')