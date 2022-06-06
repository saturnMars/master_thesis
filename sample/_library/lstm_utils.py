from datetime import datetime
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping
from _library.fault_utils import load_priorities
from sklearn.model_selection import train_test_split
from os import path
import sklearn.metrics as metrics
from tensorflow import keras
#import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tensorflow_addons import metrics as tfa_metrics
from string import ascii_uppercase
from math import ceil

def load_failure_logs(folder_path, file_name, system_name, verbose = True):
    
    # File path
    file_path = path.join(folder_path, file_name)

    # Read the csv
    fault_df = pd.read_csv(file_path, parse_dates = [7, 8],  dtype = {'Inverter': 'Int64'}) 
    fault_df['Durata'] = fault_df['Durata'].apply(lambda value: pd.to_timedelta(value)) 

    # Retrive the inverters
    inverters = sorted([inv_number for inv_number in fault_df['Inverter'].unique() if not pd.isna(inv_number)])
    generalPlantBoxes = sorted([generalPlant for generalPlant in fault_df['Quadro Generale'].unique() if not pd.isna(generalPlant)])
    stringNames = sorted([string_name for string_name in fault_df['Stringa'].unique() if not pd.isna(string_name)], 
                         key = lambda name: int(name[1:]))
    unique_events = sorted([string_name for string_name in fault_df['Messaggio'].unique() if not pd.isna(string_name)])

    # label the events
    alarm_properties = load_priorities(system_name) 
    unique_events = {event : priority.lower() for priority, alarm_names in alarm_properties.items() 
                     for event in unique_events if event in alarm_names}

    # Visualize the information
    if verbose:
        print("-" * 110 + "\n" + "-" * 50, system_name.upper(), "-" * 50 + "\n" + "-" * 110)
        print(f"Logs concerning failure events have been loaded.\n")
        print("-" * 40, 'DATA AVAILABLE', "-" * 40)
        print(f"--> Inverter available ({len(inverters)}): ", ', '.join([str(num) for num in inverters]))
        print(f"--> Unique events ({len(unique_events)})") 
        print('\t--> ' + '\n\t--> '.join([f'{idk + 1}) ({priority.upper()}) {event}' for idk, (event, priority) in enumerate(unique_events.items())]))
        print(f"\n--> Unique string names available ({len(stringNames)}):", ', '.join(stringNames))
        print(f"\n--> General Plant box available ({len(generalPlantBoxes)}):\n\t-->", '\n\t--> '.join(generalPlantBoxes))
    return fault_df, unique_events

def fill_empty_ts(df, default_value = 0):
    
    # Get a example of row
    empty_row = df.iloc[0].apply(lambda cell: default_value)
    
    # Compute the missing timestamps
    theoretical_ts = pd.date_range(start = df.index[0], end = df.index[-1], freq = '1H')
    missing_ts = set(theoretical_ts) - set(df.index)
    print(f"Missing timestamps {len(missing_ts)} out of {len(df.index)}")
    
    # Fill the gaps
    empty_rows = []
    for ts in missing_ts:
        empty_row = empty_row.copy()
        empty_row.name = ts
        empty_rows.append(empty_row)
    
    print(f"--> SO: The dataframe has been filled with {len(missing_ts)} ({(round((len(missing_ts)/len(df))* 100, 2))} %) "\
          f"missing timestamps.\n")
        
    # Merge the dataframe
    df = pd.concat([df, pd.DataFrame(empty_rows)])
    df.sort_index(inplace = True)
    
    return df

def fill_stringBoxes_data(alarm, matrix_df, input_classes, output_classes, output_col_name, prefix_gb, system_name, verbose = False):

    # A) Retrive the column name
    # A.1) Retrieve the general box (e.g., CSP1.4 --> CSP(invNum).(generalBox))
    general_box = alarm['Quadro Generale'].split('.')
    if system_name == 'Soleto 1':
        general_box = general_box[-1] # FORMAT: CSP1.X [INV1]
    elif system_name in ['Soleto 2', 'Galatina']:
        general_box = general_box[0][-1]  # FORMAT: QCX.I1 [INV1]
    
    # A.2) Retrieve the string (e.g., s1)
    string_box = alarm['Stringa']

    # A.3) Build the complete name
    col_name = prefix_gb + str(general_box) 
    if isinstance(string_box, str): # is not NaN
         col_name += "_" + string_box
    
    # B) Retrieve the class 
    # B.1) Retrieve the class name
    alarm_name = alarm['Messaggio']
    
    # B.2) Retrive whether it is an alarm to consider as an input or as a label and its index
    if alarm_name in input_classes:
        idk_class = input_classes.index(alarm_name)
    elif alarm_name in output_classes:
        idk_class = output_classes.index(alarm_name)
        
        # Assign it to the output column
        col_name = output_col_name

    if verbose:
        print(f"\nCOLUMN: {col_name}\n" + "-" * 40)
        print(f"CLASS (idk: {idk_class}): {alarm_name}")
    
    # Retrieve the temporal duration  
    starting_ts = alarm['Inizio']
    ending_ts = alarm['Fine']
    hourly_reference = [pd.to_datetime(ts.strftime('%Y-%m-%d %H')) for ts in pd.date_range(
        start = starting_ts.strftime('%Y-%m-%d %H'), 
        end = pd.to_datetime(ending_ts.strftime('%Y-%m-%d %H')), 
        freq = "1H")]
    
    if verbose:
        print(f"START: {starting_ts} || END: {ending_ts}")
    
    # Compute the temporal duration for each hourly reference
    temporal_durations = dict()
    for idk, ref_hour in enumerate(hourly_reference):
        next_hour = ref_hour + pd.Timedelta(1, unit = 'h')
        
        # CASE 1: The first reference hour
        if idk == 0:
            duration = next_hour - starting_ts
            
        # CASE 2: The last reference hour
        elif idk + 1 == len(hourly_reference):
            duration = ending_ts - ref_hour 
            
        # GENERAL CASE: The entire hour
        else:
            duration = next_hour - hourly_reference[idk]
            
        # Save the pair as (reference hour, elapsed minutes)
        to_min = lambda seconds: int(seconds / 60)
        temporal_durations[ref_hour] = to_min(duration.total_seconds())
    
    # Assign the values to the matrix 
    for reference_hour, elapsed_minutes in temporal_durations.items():
        matrix_df.loc[reference_hour, col_name][idk_class] += elapsed_minutes
        
        if verbose:
            print(f"REFERENCE HOUR: {pd.to_datetime(reference_hour).strftime('%Y-%m-%d (%H:%M)')} "\
                  f"[{elapsed_minutes} minutes] --> "\
                  f"{', '.join([str(counter) for counter in matrix_df.loc[reference_hour, col_name]])}")

def fill_generalized_stringBoxes_data(alarm, matrix_df, input_classes, output_classes, output_col_name, prefix_gb, 
                                      system_name, verbose = False):

    # A) Retrive the column name
    # A.1) Retrieve the general box (e.g., CSP1.4 --> CSP(invNum).(generalBox))
    general_box = alarm['Quadro Generale'].split('.')
    if system_name == 'Soleto 1':
        general_box = general_box[-1] # FORMAT: CSP1.X [INV1]
    elif system_name in ['Soleto 2', 'Galatina']:
        general_box = general_box[0][-1]  # FORMAT: QCX.I1 [INV1]

    # A.2) Retrieve the string (e.g., s1)
    string_name = alarm['Stringa']

    # A.3) Build the complete name
    col_name = prefix_gb + str(general_box) 
    if isinstance(string_name, str): # is not NaN
         col_name += '_strings_time'
    
    # B) Retrieve the class 
    # B.1) Retrieve the class name
    alarm_name = alarm['Messaggio']
    
    # B.2) Retrive whether it is an alarm to consider as an input or as a label and its index
    if alarm_name in input_classes:
        idk_class = input_classes.index(alarm_name)
    elif alarm_name in output_classes:
        idk_class = output_classes.index(alarm_name)
        
        # Assign it to the output column
        col_name = output_col_name

    if verbose:
        print(f"\nCOLUMN: {col_name}\n" + "-" * 40)
        print(f"CLASS (idk: {idk_class}): {alarm_name}")
    
    # Retrieve the temporal duration  
    starting_ts = alarm['Inizio']
    ending_ts = alarm['Fine']
    hourly_reference = [pd.to_datetime(ts.strftime('%Y-%m-%d %H')) for ts in pd.date_range(
        start = starting_ts.strftime('%Y-%m-%d %H'), 
        end = pd.to_datetime(ending_ts.strftime('%Y-%m-%d %H')), 
        freq = "1H")]
    
    if verbose:
        print(f"START: {starting_ts} || END: {ending_ts}")
    
    # Compute the temporal duration for each hourly reference
    temporal_durations = dict()
    for idk, ref_hour in enumerate(hourly_reference):
        next_hour = ref_hour + pd.Timedelta(1, unit = 'h')
        
        # CASE 1: The first reference hour
        if idk == 0:
            duration = next_hour - starting_ts
            
        # CASE 2: The last reference hour
        elif idk + 1 == len(hourly_reference):
            duration = ending_ts - ref_hour 
            
        # GENERAL CASE: The entire hour
        else:
            duration = next_hour - hourly_reference[idk]
            
        # Save the pair as (reference hour, elapsed minutes)
        to_min = lambda seconds: int(seconds / 60)
        temporal_durations[ref_hour] = to_min(duration.total_seconds())
    
    # Assign the values to the matrix 
    for reference_hour, elapsed_minutes in temporal_durations.items():

        if col_name in matrix_df.columns:

            # a) Increase the minutes 
            matrix_df.loc[reference_hour, col_name][idk_class] += elapsed_minutes

            if verbose:
                print(f"REFERENCE HOUR: {pd.to_datetime(reference_hour).strftime('%Y-%m-%d (%H:%M)')} "\
                      f"[{elapsed_minutes} minutes] --> "\
                      f"{', '.join([str(counter) for counter in matrix_df.loc[reference_hour, col_name]])}")

            # Save the string name for this reference hour
            if 'strings_time' in col_name:

                # b) Build the metadata
                stringList_col_name = prefix_gb + str(general_box) + '_faulty_strings'
                string_number = int(string_name[-1])

                if matrix_df.loc[reference_hour, stringList_col_name][idk_class][string_number] == 0:
                    matrix_df.loc[reference_hour, stringList_col_name][idk_class][string_number] += 1

                if verbose:
                    display(matrix_df.loc[reference_hour, stringList_col_name])

def normalized_faulty_strings_counter(df_row, col_name, stringBoxes_config, verbose = False):

    # Retrieve the general box (i.e., QCX_faulty_strings)
    general_box_name = col_name.split('_')[0]

    # Retrive the number of strings for this general box
    total_strings = stringBoxes_config[general_box_name]['num_strings']

    # faulty_strings = len(df_row[col_name])
    counted_items = np.count_nonzero(df_row[col_name], axis = 1, keepdims = True)

    # Assign the new value
    normalized_num_strings = np.round(counted_items / total_strings, 4) 
    
    if verbose:
        print('\nName:', col_name)
        print("general_box_name", general_box_name)
        print(f"Original {df_row[col_name].shape}:", df_row[col_name])
        print(f"Items {counted_items.shape}", counted_items)
        print(f"Normalized {normalized_num_strings.shape}", normalized_num_strings)
            
    return normalized_num_strings


# ------------- DATA PREPARATION ------------------------
def reset_seeds(seed_num):
    np.random.seed(seed_num) 
    random.seed(seed_num)
    tf.random.set_seed(seed_num)

def OLD_generate_temporal_sequences_num_obs(df, window_size, verbose = False):
    Xs =  []
    for idk in range(window_size, len(df)):

        # Temporal window
        x_data = df[idk - window_size : idk]

        # Save the data as Xs and Ys
        Xs.append(x_data)
    timestamps = df.index[window_size:]
    
    if verbose:
        print("Discared timestamps:", len(df.index[:window_size]))
        
    return np.array(Xs), timestamps


def OLD_generate_data_sequences_num_obs(df, output_cols, window_size = 4, enconding_one_obs = True, verbose = False):
    x_feature_names = df.columns.tolist()
    
    # Separate the output columns from the others
    for col in output_cols:
        x_feature_names.remove(col)

    # 1.2) Turn the Pandas Dataframe into a numpy array
    input_df = df[x_feature_names].to_numpy()
    output_df = df[output_cols].to_numpy()

    # A) Create the windows
    Xs, Ys = [], []
    for idk in range(window_size, len(df)):

        # [X data] Retrieve the data sequence
        x_data = input_df[idk - window_size : idk]

        # [Y DATA] STATEGY for the target value 
        # a) Enconding one obs --> select only the single observations
        if enconding_one_obs and window_size != 1:
            y_data = output_df[idk]

            if verbose: 
                print(f"STATEGY: Output with only 1 obs. INPUT WINDOW: {x_data.shape[0]} || OUTPUT WINDOW: {y_data.shape[0]}")

        # b) Select all the input sequence (as the x data --> pure autoencoding)
        else: 
            if window_size == 1:
                y_data = output_df[idk - window_size : idk].reshape(-1)
        
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
    return x_feature_names, x_values, y_values

def generate_temporal_window(df_row, df, window_size, x_feature_names, output_cols):
    starting_ts = df_row.name - pd.Timedelta(window_size - 1, unit = 'hours')
    if starting_ts in df.index:
        data = {
            'x': df.loc[starting_ts : df_row.name, x_feature_names].to_numpy(),
            'y': df.loc[df_row.name, output_cols].to_numpy(dtype = int)
        }
    else:
        data = None

    return data

def generate_data_sequences(df, output_cols, window_size = 12, enconding_one_obs = True, num_inverters = 1, verbose = False):
    x_feature_names = df.columns.tolist()
    
    # Separate the output columns from the others
    for col in output_cols:
        x_feature_names.remove(col)
        
    # Computing the window
    windowed_data = df.apply(func = lambda df_row: 
                             generate_temporal_window(df_row, df, window_size, x_feature_names, output_cols), 
                             axis = 1)
    windowed_data.dropna(inplace = True)
    
    # Retrieve the x data, y data and thier timestamps
    timestamps = windowed_data.index 

    x_values = np.vstack(windowed_data.apply(lambda item: np.ravel(item['x']))).reshape(len(windowed_data), window_size * num_inverters, 
                                                                                        -1)
    y_values = np.vstack(windowed_data.apply(lambda item: np.ravel(item['y']))).reshape(len(windowed_data), -1)
    
    return x_values, y_values, x_feature_names, timestamps

def generate_temporal_window_prod_version(df_row, df, window_size):

    # Compute the starting timestamp
    starting_ts = df_row.name - pd.Timedelta(window_size - 1, unit = 'hours')
    
    # Retrieve the data for this temporal window
    if starting_ts in df.index:
        data = df.loc[starting_ts:df_row.name, :].to_numpy()
    else:
        data = None
    return data

def generate_data_sequences_prod_version(df, window_size = 12, verbose = False):
    
    # Computing the window
    windowed_data = df.apply(func = lambda df_row: generate_temporal_window_prod_version(df_row, df, window_size), axis = 1)
    windowed_data.dropna(inplace = True)
    
    # Retrieve the x data, y data and thier timestamps
    timestamps = windowed_data.index    
    x_values = np.vstack(windowed_data.apply(lambda item: np.ravel(item))).reshape(len(windowed_data), window_size, -1)
    
    return x_values, timestamps

def prepare_splitted_training_data(train_df, output_cols, window_size = 4, enconding_one_obs = False, valid_size = 0.2,
                                   verbose = False):

    # A) Generate the sequence
    x_values, y_values, feature_names, timestamps = generate_data_sequences(train_df, output_cols, window_size, enconding_one_obs, verbose)

    # B0) Isolate potential problematic class (turnaround when there is only one instance to carry out the statified split)
    nonZeros = np.sum(y_values, axis = 0) 

    class_idk_to_select = [j for j in range(y_values.shape[1]) if j not in np.argwhere(nonZeros == 1)]
    if y_values.ndim == 3:
        df_to_statify = y_values[:, :, class_idk_to_select]
    else:
        df_to_statify = y_values[:, class_idk_to_select]
    
    # B) Split the data into train/validation sets
    try: 
        x_train, x_valid, y_train, y_valid = train_test_split(x_values, y_values, test_size = valid_size, 
                                                              stratify = df_to_statify, random_state = 101, shuffle = True)
    except ValueError:
        x_train, x_valid, y_train, y_valid = train_test_split(x_values, y_values, test_size = valid_size, random_state = 101, 
                                                              shuffle = True)
        print("\nISSUE: Error when tried to stratify the train/validation split.")
    
    print("x_train", x_train.shape)
    print("x_valid", x_valid.shape)
    print(f"TOTAL COLS: {len(train_df.columns)}, OUTPUT COLS: {len(output_cols)}, --> INPUTS: {len(train_df.columns) - len(output_cols)}")

    # C) Validity check
    assert x_train.shape[2] == x_valid.shape[2] == (len(train_df.columns) - len(output_cols))
    
    # D) Visualize dimensions
    print(f"TRAIN SET ({int((1 - valid_size) * 100)} %) --> INPUT (X): {x_train.shape} || OUTPUT (Y): {y_train.shape} "\
          f"\n--> {list(zip(['CLASS' + str(j + 1) for j in range(y_train.shape[1])], np.sum(y_train, axis = 0)))}")
    
    print(f"\nVALID SET ({int(valid_size * 100)} %) --> INPUT (X):  {x_valid.shape} || OUTPUT (Y):  {y_valid.shape}"\
          f"\n--> {list(zip(['CLASS' + str(j + 1) for j in range(y_valid.shape[1])], np.sum(y_valid, axis = 0)))}\n")
    
    if verbose:
        print(f"\n{len(feature_names)} FEATURES:\n--> " + '\n--> '.join(feature_names) + "\n")
    return (x_train, y_train), (x_valid, y_valid), feature_names, timestamps


# ------------------- LSTM ------------------------------------------
def initialize_lstm(num_neurons, window_length, num_input_features, num_output_classes, enconding_one_obs = True, num_hid_layers = 3):

    # Set the seed
    seed = 101
    reset_seeds(seed_num = seed)
    
    # Create the model (i.e. sequential layers)
    model = keras.Sequential(name = f"lstm") #_{datetime.now().strftime('%Yy_%mm_%dd_%Hh%Mm')}
    
    # Set the weight initializer (with a fixed seed)
    initializer = initializers.GlorotNormal(seed) 
    
    input_shape = (None, window_length, num_input_features)
    print("input_shape", input_shape)

    # ------------------- LSTM ----------------------------------
    sequence_flag = False if enconding_one_obs else True
    # -------- LAYER 1: LSTM -------------- 
    if num_hid_layers == 1:
        model.add(keras.layers.LSTM(num_neurons, kernel_initializer = initializer, batch_input_shape = input_shape, return_sequences = sequence_flag, dropout = 0.1, name = 'lstm_in'))
    else:
        model.add(keras.layers.LSTM(num_neurons, kernel_initializer = initializer, batch_input_shape = input_shape, return_sequences = True,dropout = 0.3, name = 'lstm_in'))

    if num_hid_layers == 2:
        # -------- LAYER 2: LSTM --------------
        model.add(keras.layers.LSTM(num_neurons//2, kernel_initializer = initializer, return_sequences = sequence_flag, 
                                    dropout = 0.15, name='lstm_inner1'))
    else:
        model.add(keras.layers.LSTM(num_neurons//2, kernel_initializer = initializer, return_sequences = True, 
                                    dropout = 0.2, name='lstm_inner1'))
    
    # -------- LAYER 2bis: LSTM --------------
    #model.add(keras.layers.LSTM(num_neurons//4, kernel_initializer = initializer, return_sequences = True,  seed = seed, 
                                #dropout = 0.2, name='LSTM_inner2'))

    # -------- LAYER 2ter: LSTM --------------
    #model.add(keras.layers.LSTM(num_neurons//6, kernel_initializer = initializer, return_sequences = True,  seed = seed, 
                               # dropout = 0.2, name='LSTM_inner3'))
    if num_hid_layers == 3:
        # -------- LAYER 3: LSTM -------------- 
        sequence_flag = False if enconding_one_obs else True
        model.add(keras.layers.LSTM(num_neurons//4, kernel_initializer = initializer, return_sequences = sequence_flag,  
                                    dropout = 0.1, name='lstm_out'))
    
    # -------- LAYER 4: Sigmoid -------------- 
    model.add(keras.layers.Dense(num_output_classes, kernel_initializer = initializer, activation = 'sigmoid', 
                                 name ='output_layer'))
     
    # Define the metrics to evaluate the network
    metrics_list = []
    
    # F1 score
    metrics_list.append(tfa_metrics.F1Score(num_classes = num_output_classes, average = 'macro', threshold = 0.5))
    
    # B) Precision (global and for each class)
    metrics_list.append(tf.keras.metrics.Precision(name = 'Precision'))
    metrics_list.append(tf.keras.metrics.Recall(name = 'Recall', thresholds = 0.5))
    for idk_class in range(num_output_classes):
        metrics_list.append(tf.keras.metrics.Precision(
            class_id = idk_class, 
            name = f'Precision_{ascii_uppercase[idk_class]}'
        ))
        metrics_list.append(tf.keras.metrics.Recall(
            class_id = idk_class, 
            name = f'Recall_{ascii_uppercase[idk_class]}'
        ))
    
    # A) (Binary) accuracy
    metrics_list.append(tf.keras.metrics.BinaryAccuracy())

    model.compile(
        run_eagerly = False,
        loss = tf.keras.losses.BinaryCrossentropy(from_logits = False),
        optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9),
        metrics = metrics_list
    )
    
    # Build the model
    model.build()

    # Visualize the summary model
    model.summary()
    return model

def train_model(model, train_data, valid_data, num_epochs, batch_size, shuffle_flag = True, visualize_graph = True, 
                saving_graph_path = None):
    
    # Unpack data
    x_train, y_train = train_data
    x_valid, y_valid = valid_data

    # Define the early stopping criterion
    early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 1e-4, patience = 5, mode = 'auto', 
                               baseline = None, restore_best_weights = False) # verbose = 2,
    
    # Train the model
    seqModel = model.fit(x = x_train, y = y_train, validation_data = (x_valid, y_valid), epochs = num_epochs, 
                         batch_size = batch_size, shuffle = shuffle_flag, callbacks = [early_stop])
    
    # use_multiprocessing = True
    idk_best_epoch = np.argmax(seqModel.history['val_f1_score'])
    best_perfomances = {metrics_name.replace('_', ' ').capitalize(): np.round(score_list[idk_best_epoch], 4)
                        for metrics_name, score_list in seqModel.history.items()}
    best_perfomances['best_epoch'] = idk_best_epoch + 1

    if visualize_graph:
        num_metrics = len(seqModel.history.keys()) // 2
        num_cols = 4
        fig, axes = plt.subplots(nrows = ceil(num_metrics / num_cols), ncols = num_cols, figsize = (40, 20), sharex = True)
        fig.suptitle('Perfomance', fontsize = 50, y = 1.1)
    
        print("\n" + "-" * 40, "PERFOMANCE", "-" * 40, "\n")
        idk_row, idk_column = 0, 0
        for idk, metrics_name in enumerate(list(seqModel.history.keys())[:num_metrics]):
            train_perfomance = np.array(seqModel.history[metrics_name])
            valid_perfomance = np.array(seqModel.history['val_' + metrics_name])
            x_values = range(1, len(train_perfomance) + 1)
            
            metrics_name = metrics_name.replace('_', ' ').capitalize()
            
            if metrics_name != 'Loss':
                train_perfomance = train_perfomance * 100
                valid_perfomance = valid_perfomance * 100
           
            axes[idk_row, idk_column].set_title(metrics_name, fontsize = 30, pad = 10)
            axes[idk_row, idk_column].plot(x_values, train_perfomance, label = f"Train {metrics_name}")
            axes[idk_row, idk_column].plot(x_values, valid_perfomance, label = f"Validation {metrics_name}")
            axes[idk_row, idk_column].set_xlabel("Epoch", fontsize = 30)
            axes[idk_row, idk_column].set_ylabel("[%]" if metrics_name != 'Loss' else metrics_name, fontsize = 23, labelpad = 5)
            axes[idk_row, idk_column].tick_params(labelsize = 14)
            axes[idk_row, idk_column].tick_params(axis = 'x', grid_alpha = 0.2)
            axes[idk_row, idk_column].tick_params(axis = 'y', grid_alpha = 0.5, grid_linestyle = "--", grid_linewidth = 2)
            axes[idk_row, idk_column].set_xlim(left = 1)
            axes[idk_row, idk_column].grid()
            
            if len(train_perfomance) <= 21:
                axes[idk_row, idk_column].set_xticks(np.arange(1, len(train_perfomance) + 1, step = 2)) 
            
            if idk == 0:
                 axes[idk_row, idk_column].legend()
                
            # Change row/column counters
            if idk_column + 1 < axes.shape[1]:
                idk_column += 1
            else:
                idk_column = 0
                idk_row += 1
        
        fig.tight_layout(pad = 3)

        if saving_graph_path:
            fig.savefig(saving_graph_path, bbox_inches='tight', pad_inches = 1)
        plt.show()

    print(f"BEST EPOCH (early stopping): {best_perfomances['best_epoch']}\n" + "-" * 20)
    print('--> ' + '\n--> '.join([f'{metrics_name}: {np.round(score , 4)}' 
                                for metrics_name, score in list(best_perfomances.items())[:-1]]))

    return seqModel, best_perfomances

def evaluate_predictions(predictions, targets, class_names, verbose = True):
    perfomance = dict()
    for idk_class in range(len(class_names)):
        
        # Retrieve the class name
        class_name = class_names[idk_class] 
        
        # Retrieve the data concerning this class 
        class_predictions = predictions[:, idk_class]
        class_target_values = targets[:, idk_class]
        
        # Retrieve the number of hourlu observations  
        num_hourly_events = np.nonzero(class_target_values)[0].shape[0]

        if verbose: 
            print("\n\t\t\t" + "-" * 80 + "\n\t\t\t\t" + class_name, f"(HOURLY EVENTS: {num_hourly_events})\n\t\t\t" +  "-" * 80)
            
            conf_matrix = metrics.confusion_matrix(class_target_values, class_predictions)
            label_names = ['(0) No high alarm', '(1) High alarm'] if num_hourly_events > 0 else None
            metrics.ConfusionMatrixDisplay(conf_matrix, display_labels = label_names).plot(cmap = 'Greens') 
            plt.title('Confusion Matrix', fontsize = 40, y = 1.1)
            plt.ylabel('TARGET CLASSES', fontsize = 18)
            plt.xlabel('PREDICTED CLASSES', fontsize = 18, labelpad = 10)
            plt.show()

            if len(conf_matrix.ravel()) == 4:
                tn, fp, fn, tp = conf_matrix.ravel()
                print(f"\t\t     TN:  {tn} || FP: {fp} \n\t\t     FN: {fn} || TP:  {tp}\n")
            
        # Decide how to deal when there are no events
        if num_hourly_events == 0:
            zero_divison_stategy = 0
        else:
            zero_divison_stategy = 'warn'
            
        perfomance[class_name] = { 
            'F1 score': metrics.f1_score(class_target_values, class_predictions, zero_division = zero_divison_stategy), 
            'Recall': metrics.recall_score(class_target_values, class_predictions, zero_division = zero_divison_stategy), 
            'Precision': metrics.precision_score(class_target_values, class_predictions, zero_division = zero_divison_stategy), 
            'Accuracy': metrics.accuracy_score(class_target_values, class_predictions),
            'Misclassification ratio':metrics.zero_one_loss(class_target_values, class_predictions),
            'Hourly events': num_hourly_events
        }
        
        if num_hourly_events == 0:
            if perfomance[class_name]['F1 score'] == 0:
                perfomance[class_name]['F1 score'] = np.nan 
            if perfomance[class_name]['Recall'] == 0:
                perfomance[class_name]['Recall'] = np.nan
            if perfomance[class_name]['Precision'] == 0:
                perfomance[class_name]['Precision'] = np.nan

        if verbose: 
            print("\n" + "-" * 10, "CLASS METRICS", "-" * 10)
            print("ACCURACY: ", np.round(perfomance[class_name]['Accuracy'] * 100, 2), "%")
            print("PRECISION:", np.round(perfomance[class_name]['Precision'] * 100, 2), "%")
            print("RECALL:   ", np.round(perfomance[class_name]['Recall'] * 100, 2), "%")
            print("F1-SCORE: ", np.round(perfomance[class_name]['F1 score'] * 100, 2), "%")
            print("MISCLASSIFICATION RATIO:", np.round(perfomance[class_name]['Misclassification ratio'] * 100, 2), "%")
    return perfomance