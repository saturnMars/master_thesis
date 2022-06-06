from keras import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers
import keras
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def generate_data_sequences(df, window_size = 4, enconding_one_obs = False, verbose = False):
    feature_names = df.columns.tolist()

    # 1.1) Drop NaN
    idk_to_drop = df[df.isnull().any(axis = 1)].index
    if len(idk_to_drop) > 0:
        df = df.drop(index = idk_to_drop)
        
        if verbose:
            print(f"\nDropped {len(idk_to_drop)} NaN observations.\n")

    # 1.2) Turn the Pandas Dataframe into a numpy array
    data = df.to_numpy()

    # A) Create the windows
    Xs, Ys = [], []
    for idk in range(window_size, len(df)):

        # [X data] Retrieve the data sequence
        x_data = data[idk - window_size : idk]

        # [Y DATA] STATEGY for the target value 
        # a) Enconding one obs --> select only the single observations
        if enconding_one_obs and window_size != 1:
            y_data = x_data[-1, :].reshape(1, -1)

            if verbose: 
                print(f"STATEGY: Output with only 1 obs. INPUT WINDOW: {x_data.shape[0]} || OUTPUT WINDOW: {y_data.shape[0]}")

        # b) Select all the input sequence (as the x data --> pure autoencoding)
        else: 
            y_data = x_data
        
        # Save the data as Xs and Ys
        Xs.append(x_data)
        Ys.append(y_data)

        if verbose: 
            print(f"\nIDK: {idk} --> SEQUENCE: {idk - window_size}: {idk}")
            print(f"X {x_data.shape}:", x_data)
            print(f"Y {y_data.shape}:", y_data)

    # Save the list as numpy arrays
    x_values = np.array(Xs)
    y_values = np.array(Ys)
    return feature_names, x_values, y_values

def prepare_splitted_training_data(train_df, df_name, window_size = 4, enconding_one_obs = False, verbose = False):

    # A) Generate the sequence
    feature_names, x_values, y_values = generate_data_sequences(train_df, window_size, enconding_one_obs, verbose)

    # B) Split the data into train/validation sets
    valid_size = 0.2
    x_train, x_valid, y_train, y_valid = [np.array(dataset) for dataset in train_test_split(x_values, y_values, test_size = valid_size,
                                                                                            random_state = 101)]

    # C) Validity check
    assert x_train.shape[2] == x_valid.shape[2] == len(train_df.columns)
    
    # D) Visualize dimensions
    print(f"TRAIN SET ({int((1 - valid_size) * 100)} %) --> INPUT (X): {x_train.shape} || OUTPUT (Y): {y_train.shape}")
    print(f"VALID SET ({int(valid_size * 100)} %) --> INPUT (X): {x_valid.shape} || OUTPUT (Y): {y_valid.shape}")
    print(f"\n{len(feature_names)} FEATURES:\n--> " + '\n--> '.join(feature_names) + "\n")
  
    return feature_names, x_train, y_train, x_valid, y_valid

def prepare_training_data(df, df_name, window_size = 4, enconding_one_obs = False, verbose = False):

    # A) Generate the sequences
    feature_names, x_values, y_values = generate_data_sequences(df, window_size, enconding_one_obs, verbose)

    # B) Visualize the shape
    print(f"{df_name.upper()} INPUT: {x_values.shape} || OUTPUT: {y_values.shape}")
    if verbose: 
        print(f"\n{len(feature_names)} FEATURES:\n--> " + '\n--> '.join(feature_names) + "\n")
    
    return feature_names, x_values, y_values
            
def initialize_autoencoder(window_length, num_features, num_neurons = 16, num_layers = 2, enconding_one_obs = False, 
                           loss_function = "mse"):
 
    seed = 101
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Create the model (i.e. sequential layers)
    model = keras.Sequential(name = "LSTM_auto_encoder")
    
    # Set the weight initializer (with a fixed seed)
    uniform_initializer = initializers.HeUniform(seed = seed)

    # Compute the number of inner neurons
    inner_num_neurons = num_neurons//2 if num_layers == 2 else num_neurons//4

    # A) ENCODER LAYERS
    # A.1) ENCODER LAYER - INPUT 
    model.add(keras.layers.LSTM(num_neurons, kernel_initializer = uniform_initializer, 
                                batch_input_shape = (None, window_length, num_features), 
                                return_sequences = True, name = 'encoder_in')) 
    # A.2) ENCODER LAYER - MIDDLE 
    if num_layers == 3:
        model.add(keras.layers.LSTM(num_neurons//2, kernel_initializer = uniform_initializer, return_sequences = True,
                                    name = 'encoder_middle'))
    # A.3) ENCODER LAYER - OUTPUT
    model.add(keras.layers.LSTM(inner_num_neurons, kernel_initializer = uniform_initializer, return_sequences = False, 
                                name='encoder_out'))

    # B) MIDDLE LAYER (i.e., Bridge) ----------------------------------------------------
    if enconding_one_obs:
        dim_vector = 1
    else: 
        dim_vector = window_length

    # Repeat vector layer
    model.add(keras.layers.RepeatVector(dim_vector, name='encoder_decoder_bridge'))
    # -----------------------------------------------------------------------------------
    
    # C) DECODER LAYERS
    # C.1) DECODER LAYER - INPUT
    model.add(keras.layers.LSTM(inner_num_neurons, kernel_initializer = uniform_initializer, return_sequences = True, name = 'decoder_in'))
    
    # C.2) DECODER LAYER - MIDDLE
    if num_layers == 3:
        model.add(keras.layers.LSTM(num_neurons//2, kernel_initializer = uniform_initializer, return_sequences = True, 
                                    name = 'decoder_middle')) 
    # C.3) DECODER LAYER - OUTPUT
    model.add(keras.layers.LSTM(num_neurons, kernel_initializer = uniform_initializer, return_sequences = True, name = 'decoder_out'))

    # Repeat vector layer
    #model.add(keras.layers.RepeatVector(1, name='encoder_decoder_bridge_end'))
    # -----------------------------------------------------------------------------------
    
    # D) OUTPUT LAYER -------------------------------------------------------------------------------------------------
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(num_features, kernel_initializer = uniform_initializer),  
                                           name='output'))
    # -----------------------------------------------------------------------------------------------------------------
    
    # Model parameters (Loss function & optimizer)
    model.compile(loss = loss_function, optimizer = 'adam') # ['SGD', 'RMSprop', 'adam']
    
    # Build the model
    model.build()
    
    # Visualize the summary model
    model.summary()
    return model

def train_model(model, train_data, valid_data, num_epochs, batch_size, shuffle_flag = True, visualize_graph = True):
    
    # Unpack data
    x_train, y_train = train_data
    x_valid, y_valid = valid_data

    # Define the early stopping criterion
    early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 1e-3, patience = 5, verbose = 2, mode = 'auto', 
                               baseline = None, restore_best_weights = True) #1e-2
    
    # Train the model
    
    seqModel = model.fit(x = x_train, y = y_train, validation_data = (x_valid, y_valid),
                         epochs = num_epochs, batch_size = batch_size, 
                         shuffle = shuffle_flag, callbacks = [early_stop])
    if visualize_graph:
        train_loss = seqModel.history['loss']
        val_loss = seqModel.history['val_loss']
        x_values = range(1, len(train_loss) + 1)
        
        print(f"\nBest epoch N°{np.argmin(val_loss) + 1} --> Minimum validation Loss: {np.round(np.min(val_loss), 4)}\n")

        plt.figure()
        plt.title("Training phase", fontsize = 30, pad = 10)
        plt.plot(x_values, train_loss, label = "Train Loss")
        plt.plot(x_values, val_loss, label = "Validation Loss")
        plt.xlabel("Epoch",fontsize = 20)
        plt.ylabel("MSE Loss", fontsize = 20)
        plt.xlim(left = 1)
        if len(train_loss) <= 21:
            plt.xticks(np.arange(1, len(train_loss) + 1, step = 2)) 
        plt.legend() 
        plt.show()

    return seqModel

def visualize_norm_dist(norm_df):
    plt.figure()
   
    # 1) Input values for the graphs
    x_values = norm_df.index
    y_values = norm_df.values
    
    # 2) Visualize some statistics
    min_norm = np.min(y_values)
    max_norm = np.max(y_values)
    avg_norm = np.mean(y_values)
    median_norm = np.median(y_values)
    
    # 3) Boxplot
    plt.boxplot(y_values)
    #plt.xticks([]) 
    plt.ylabel("Norm", fontsize = 20)
    plt.show()
    
    print("\t\t\tMIN:", np.round(min_norm, 4))
    print("\t\t\tMAX:",  np.round(max_norm, 4))
    print("\t\t\tAVG:",  np.round(avg_norm, 4))
    print("\t\t     MEDIAN:",  np.round(median_norm, 4))
    
    # 4) Scatterplot
    alpha_value = 0.1 if len(y_values) > 400 else 0.8
    plt.scatter(x_values, y_values, alpha = alpha_value, s = 4, c = "orange")
    plt.hlines(y = avg_norm, xmin = x_values[0], xmax = x_values[-1], label = 'Mean', 
               linestyles = 'dotted', colors = "black", alpha = .8)
    plt.xticks(rotation = 20) 
    plt.xlabel("Date",fontsize = 20)
    plt.ylabel("Norm", fontsize = 20)
    plt.legend()
    plt.show()
    
    return min_norm, max_norm, avg_norm, median_norm

def visualize_distribution(inv_name, timestamps, norm_values, fault_ts, unique_faults, thresholds, prediction_window = 7, verbose = False):
    
    # 0) Assign the classes 
    # 0.A) CLASS 0: Nominal observations (i.e., default)
    norm_values.loc[:, "Class"] = "Normal"
    
    # 0.B) CLASS 1: Observations within a prediction window (i.e., 7 days before a failure events)
    # retrieve the period of the failure events
    period_failure_events = [{'start': pd.to_datetime(fault_event[3][0]), 'end':  pd.to_datetime(fault_event[3][1])} 
                             for fault_event in unique_faults]
    
    # Assign the class for each failure event
    pred_wind_class_name = "Within the prediction window"
    start_dates = set()
    for failure_event in period_failure_events:
        start_failure_event = failure_event['start']

        start_prediction_window = start_failure_event - pd.Timedelta(prediction_window, unit = "days")
        start_dates.add(start_failure_event.strftime('%Y-%m-%d'))

        if start_prediction_window != start_failure_event:
            norm_values.loc[start_prediction_window: start_failure_event, "Class"] = pred_wind_class_name
        
        #print("\nSTART PREDICTION WINDOW:", start_prediction_window)
        #print("FAILURE EVENT START:", start_failure_event)
        
    # 0.C) CLASS 2: Observations concerning failure events
    norm_values.loc[fault_ts, "Class"] = "Failure event"
    
    # 0.D) Save the list of classes
    classes = norm_values['Class']
    values = norm_values['KPI score']
        
    # 1) GRAPH PANEL - BOXPLOTS
    plt.figure(figsize=(13, 2))
    sns.boxplot(x = values, y = classes)
   
    # 1.2) Vertical lines - thresholds 
    plt.axvline(x = thresholds[0], linestyle = '--', color = 'orange', alpha = 0.8, label = "Threshold 1")
    plt.axvline(x = thresholds[1], linestyle = '--', color = 'red', alpha = 0.8, label = "Threshold 2")

    plt.xlabel("Values", fontsize = 20)
    plt.yticks(np.arange(len(classes.unique())), labels = [class_name + f" ({len(classes[classes == class_name])} obs.)" 
                                      for class_name in classes.unique()], fontsize= 20)
    plt.ylabel("")
    plt.legend()
    plt.show()
    
    # 2) DESCRIPTIVE STATISTICS
    if verbose: 
        classes_to_visualize = np.append([f'All data'], classes.unique())
        for idk, class_name in enumerate(classes_to_visualize):
            print("\n" + "-" * 30 + f"\nCLASS {idk}: {class_name}\n" + "-" * 30)

            if class_name != 'All data':
                cond = norm_values['Class'] == class_name
                class_values = norm_values.loc[cond, 'KPI score']
            else:
                class_values = norm_values.loc[:, 'KPI score']

            min_value = np.round(class_values.min(), 4)
            max_value = np.round(class_values.max(), 4)
            avg_value = np.round(class_values.mean(), 4)
            std_value = np.round(class_values.std(), 4)

            if class_name == "Normal":
                print("--> MIN:", min_value)
                print("--> MAX:", max_value)
                print("    "+ "-" * 12)
                print("--> AVG:", avg_value)
                print("--> STD:", std_value)
            else: 
                nominal_norm_values = norm_values.loc[norm_values['Class'] == 'Normal', 'KPI score']
                print(f"--> MIN: {min_value} \t[DELTA from class 'Normal'] --> MIN:", 
                      np.round(min_value - nominal_norm_values.min(), 2))
                print(f"--> MAX: {max_value} \t[DELTA from class 'Normal'] --> MAX:", 
                      np.round(max_value - nominal_norm_values.max(), 2))
                print("    " + "-" * 60)
                print(f"--> AVG: {avg_value} \t[DELTA from class 'Normal'] --> AVG:", 
                      np.round(avg_value - nominal_norm_values.mean(), 2))
                print(f"--> STD: {std_value} \t[DELTA from class 'Normal'] --> STD:", 
                      np.round(std_value - nominal_norm_values.std(), 2))
    
    # 3) GRAPH PANEL - SCATTERPLOT
    # PANEL 1: Figure 
    plt.figure(figsize=(25, 4))
    cutting_dim = len(timestamps) // 2
    if prediction_window > 0:
        class_sizes = {"Normal": 15, pred_wind_class_name: 30, "Failure event": 120}
    else:
        class_sizes = {"Normal": 30, "Failure event": 120}

    # 1) Scatterplot
    sns.scatterplot(x = timestamps[:cutting_dim], y = values[:cutting_dim], hue = classes[:cutting_dim], 
                    size = classes[:cutting_dim], sizes = class_sizes, alpha = .3)

    # 2) Horizontal lines - failure events
    for date in start_dates:
        window = [ts for ts in timestamps[:cutting_dim] if date == ts.strftime('%Y-%m-%d')]
        if len(window) > 0:
            plt.axvline(x = window[0], linestyle = '--', color = 'darkgrey')

    # 3) Vertical lines - thresholds 
    plt.hlines(y = thresholds[0], color = 'orange', xmin = timestamps[:cutting_dim][0], xmax =  timestamps[:cutting_dim][-1], alpha = 0.5, 
    label = "Threshold 1")
    plt.hlines(y = thresholds[1], color = 'red', xmin = timestamps[:cutting_dim][0], xmax =  timestamps[:cutting_dim][-1], alpha = 0.5,
    label = "Threshold 2")
    
    # 4) Graphical parameters
    plt.title(f"[{inv_name}] PERIOD 1: from '{timestamps[0].strftime('%Y-%m')}' to '{timestamps[cutting_dim].strftime('%Y-%m')}'",
             fontsize = 30, pad = 10, color = 'grey')    
    plt.ylabel("Values", fontsize = 30)
    plt.xticks(fontsize = 20) #rotation = 20
    plt.yticks(fontsize = 15) #rotation = 20
    plt.legend(fontsize = 15, markerscale = 2, loc = 'upper right')
    plt.show()

    # PANEL 2: Figure 
    plt.figure(figsize=(25, 4))

    # 1) Scatterplot
    sns.scatterplot(x = timestamps[cutting_dim:], y = values[cutting_dim:], hue = classes[cutting_dim:], 
                    size = classes[cutting_dim:], sizes = class_sizes, alpha = .3)

    # 2) Horizontal lines - failure events
    for date in start_dates:
        window = [ts for ts in timestamps[cutting_dim:] if date == ts.strftime('%Y-%m-%d')]
        if len(window) > 0:
            plt.axvline(x = window[0], linestyle = '--', color = 'darkgrey')

    # 3) Vertical lines - thresholds 
    plt.hlines(y = thresholds[0], color = 'orange', xmin = timestamps[cutting_dim:][0], xmax =  timestamps[cutting_dim:][-1], alpha = 0.5,
    label = "Threshold 1")
    plt.hlines(y = thresholds[1], color = 'red', xmin = timestamps[cutting_dim:][0], xmax =  timestamps[cutting_dim:][-1], alpha = 0.5,
    label = "Threshold 2")

    # 4) Graphical parameters
    plt.title(f"[{inv_name}] PERIOD 2: from '{timestamps[cutting_dim + 1].strftime('%Y-%m')}' to '{timestamps[-1].strftime('%Y-%m')}'",
             fontsize = 30, pad = 10, color = 'grey')
    plt.ylabel("Values", fontsize = 30)
    plt.xlabel("Date", fontsize = 30)
    plt.xticks(fontsize = 20) #rotation = 20
    plt.yticks(fontsize = 15) #rotation = 20
    plt.legend().remove()
    plt.show()
    
    print("-" * 17 + f"\n{len(start_dates)} FAILURE EVENTS\n" + "-" * 17 + "\n--> " + '\n--> '.join(sorted(start_dates)))