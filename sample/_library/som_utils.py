import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from _library.fault_utils import find_fault_observation
from os import path, makedirs
from minisom import MiniSom
from matplotlib.patches import RegularPolygon
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from sklearn.model_selection import train_test_split
from datetime import timedelta
from collections import Counter, defaultdict
from math import floor

# -----------------------------------------------------------------------------------
# -------------- FUCTION: Create subfolders for a SOM config ------------------------
# -----------------------------------------------------------------------------------
def create_somVersion_folders(saving_folder_path, subfolders, dataset_name, som_config, merge_inv_data, pre_processing_steps = None):
    
    if not isinstance(som_config, str):
        # --------- SOM CONFIGURATION --------------------

        # 0) Simplify the number of epoches (added K/M)
        if som_config["epoch"] >= 10**6:
            simplified_epoch = str(som_config["epoch"]//10**6) + "M"
        elif som_config["epoch"] >= 1000: 
            simplified_epoch = str(som_config["epoch"]//1000) + "K"
        else: 
            simplified_epoch = str(som_config["epoch"])
        
        # 1) Concatenate all the parameters in a string
        config = f"{som_config['dim_grid']}grid_{simplified_epoch}epoch_{som_config['learning_rate']}lr_" \
              f"{som_config['sigma']}sigma_{som_config['neighborhood_function']}Func"
        if merge_inv_data:
            config += "_allInvData"
  
        # ---------------- PRE-PROCESSING STEPS ---------------------
        str_steps = dataset_name.replace("-", "").replace(" ", "_")[:-9]
        if pre_processing_steps["Linear regression for AC power outliers"]:
            if pre_processing_steps["Linear regression for AC power outliers (Test set)"]:
                str_steps += "_fullReg"
            else:
                str_steps += "_linReg" 
        if pre_processing_steps["Data detrending"]:
            str_steps += "_detrended"
        if pre_processing_steps["Three-phase average"]:
            str_steps += "_avgThreePhases"
        
        if 'extra_param' in pre_processing_steps.keys():
            str_steps += "_" + str(pre_processing_steps["extra_param"])

        # ----------------- MERGE THEM TOGETHER --------------------
        som_version  = config + "_" + str_steps
        print("\n SOM VERSION: ", som_version, "\n")
    else:
        som_version = som_config
        
    # --------------- CREATE THE SUB-FOLDERS ------------------------------
    for subfolder in subfolders:
        version_path = path.join(saving_folder_path, subfolder, som_version)
        if not path.exists(version_path):
            makedirs(version_path)
            print(f" The folder '{subfolder}/{som_version}' has been created!")
    return som_version
            
# -----------------------------------------------------------------------------------
# -------- FUCTION: Create a numerical matrix from a dataframe ----------------------
# ----- STAT (for NaN) = [None, mean_values, zero_values, last_valid_obs] -----------
# -----------------------------------------------------------------------------------
def to_num_matrix(df, stat_nan_values = "delate"):   
   
    # Save the useful information before discarding them (i.e., keeping only numerical values)
    columns = df.columns.tolist()

    # Fill nan in the scarse ambiental parameters --> Strategy: Use the mean value
    columns_with_nan = df.columns[df.isnull().any()].tolist()
    empty_cols = []
    
    if stat_nan_values:
        function_method = None
        nan_values = []

        for col in columns_with_nan:
            nan_values_counter = len(df[df[col].isna()])

            if stat_nan_values == "mean_values":
                filling_value = round(df[col].mean(), 2)
                if np.isnan(filling_value):
                    filling_value = 0
                    empty_cols.append(col)
                df[col] = df[col].fillna(value = filling_value)
            elif stat_nan_values == "zero_values":
                filling_value  = 0
                df[col] = df[col].fillna(value = filling_value)
            elif stat_nan_values == "interpolate":
                df[col] = df[col].astype(np.float64).interpolate(method = 'linear')
            elif stat_nan_values == "delate":
                idk_to_drop = df[df[col].isnull()].index
                df.drop(index = idk_to_drop, inplace = True)
                #print(f"Dropping {len(idk_to_drop)} observations.")  
                    
            # Filling the NaN values
            nan_values.append((col, nan_values_counter, len(df)))

        if columns_with_nan:
            print(40*"-" + f"\nPRE-PROCESSING (Filling NaN values)\n" + 40*"-")
            print(f"Filled up the NaN values ({len(nan_values)}) in "\
                  f"{len(columns_with_nan)} column(s) with the stategy: '{stat_nan_values}'")
            print([col_name.split('(')[0].rstrip() + ': '+str(nan_values) + f' obs. ({round(nan_values/tot*100, 2)}%)' 
                   for col_name, nan_values, tot in nan_values])
            
            if empty_cols:
                print(f"\n\tISSUE(S): All NaN values (using STRAT: zero_values)\n\t\t", empty_cols)

    else:
        if columns_with_nan:
            print(f"ISSUE: Some NaN values have been detected in the columns: {columns_with_nan}."\
                  "\nThese values should be fixed.")  
        
    # 2D matrix (ROW: Observations, COLUMNS: Numerical values)
    data = np.array(df.values)
    timestamps = df.index.tolist()
    print(f"\n--> INPUT MATRIX: {data.shape}")
    print(f"\n--> FEATURES ({len(columns)})")
    print("\t" + ('\n\t').join(columns))
    
    return data, columns, timestamps

def find_data_periods(data, verbose = False):

    if isinstance(data, pd.DataFrame):
        timestamps = np.array(data.index)
    else:
        timestamps = data
    
    # 1) Generate the periods
    detected_idk_periods = [idk_timestamp for idk_timestamp, diff in enumerate(np.diff(timestamps)) 
                            if pd.Timedelta(diff).days > 0]
    num_periods = len(detected_idk_periods) + 1

    if verbose:
        print("-" * 30, f"PERIODS: {num_periods} ({data.shape[0]} obs.)", "-" * 30)

    all_periods = []
    start_idk = 0
    for loop_counter in range(len(detected_idk_periods) + 1):

        # Retrieve the starting
        starting_date = pd.to_datetime(timestamps[start_idk])
        
        # Retrieve the ending dates
        if loop_counter == len(detected_idk_periods):
            idk_cutting_ts = len(timestamps) - 1
        else:
            idk_cutting_ts = detected_idk_periods[loop_counter]
        ending_date = pd.to_datetime(timestamps[idk_cutting_ts])  
            
        if verbose:
            print(f"PERIOD ({loop_counter + 1}) --> FROM '{pd.to_datetime(starting_date).strftime('%Y-%m-%d (%H:%M)')}' "\
                  f"(idk:{start_idk}) TO '{pd.to_datetime(ending_date).strftime('%Y-%m-%d (%H:%M)')}' (idk:{idk_cutting_ts})")
        
        # Select the period
        all_periods.append({'start': starting_date, 'end': ending_date})
        
        # Update the starting index
        start_idk = idk_cutting_ts + 1
    return all_periods

def split_test_validation_sets(test_data, valid_dim = 0.5, verbose = False):

    # Find Periods 
    all_periods = find_data_periods(test_data, verbose)
        
    # 2) Selecting randomly the periods for the validation & test sets
    num_validation = int(floor(len(all_periods) * valid_dim))
    idk_validation_set = np.random.default_rng(seed = 99).choice(len(all_periods), size = num_validation, replace = False)

    idk_test_set = np.setdiff1d(range(len(all_periods)), idk_validation_set)
    if verbose:
        print("\n", "-" * 10, f"VALIDATION DIMENSION ({valid_dim * 100} %): {num_validation}/{len(all_periods)}", "-" * 10)
        print(f"VALIDATION SET ({len(idk_validation_set)} PERIODS): indexes -->", idk_validation_set)
        print(f"TEST SET ({len(idk_test_set)} PERIODS): indexes -->", idk_test_set)
    
    # 3) Generate the test/validation sets
    valid_set = pd.concat([test_data[all_periods[idk]['start']: all_periods[idk]['end']] 
                for idk in idk_validation_set])
    test_set = pd.concat([test_data[all_periods[idk]['start']: all_periods[idk]['end']] 
                for idk in idk_test_set])  
    if verbose:
        print(f"\nVALID SET ({valid_dim * 100} %): {len(valid_set)} obs. ({(round((len(valid_set)/len(test_data))*100, 2))} %)")
        print(f" TEST SET ({(1 - valid_dim) * 100} %): {len(test_set)} obs. ({(round((len(test_set)/len(test_data))*100, 2))} %)")

    return valid_set, test_set, num_periods

def k_fold_split(data_matrix, timestamps, k = 3, verbose = False):
    
    # 1) Find the periods 
    all_periods = find_data_periods(timestamps, verbose)
    num_periods = len(all_periods)
    
    print(f"Periods: {num_periods} ({len(timestamps)} obs.)\n")

    # 2) Retrive data for each period
    periods_idks = np.arange(num_periods)
    periods_data = []
    for idk in periods_idks:

        # 2.1) Retrieve the starting/ending timestamps
        starting_ts = all_periods[idk]['start']
        ending_ts =  all_periods[idk]['end']
       
        # 2.2) Retrieve the indexes for the starting/ending timestamps
        idk_start = np.argwhere(timestamps == starting_ts)[0][0]
        idk_end = np.argwhere(timestamps == ending_ts)[0][0]
        period_ts = timestamps[idk_start : idk_end + 1]

        # 2.3) Retrive and save the numerical data
        period_data = data_matrix[idk_start : idk_end + 1]
 
        # 2.4) Append the data to this fold
        periods_data.append({'timestamps': period_ts, 'data': period_data})

        if verbose:
            print(f"PERIOD {idk + 1}/{num_periods}:", period_data.shape)
            print("--> START:", starting_ts)
            print("-->   END:", ending_ts)
    
    # 3) Sort periods according to the dimensions
    periods_data = sorted(periods_data, key = lambda item: len(item['data']), reverse = True)

    # 4) Retrive the number of periods per fold
    num_periods_per_fold = num_periods // k
    num_extra_periods = num_periods % k

    if verbose: 
        print("\nNum periods per fold", num_periods_per_fold)
        print("\nNum extra Periods", num_extra_periods)

    # 5) Generate the K-folds
    list_idks = list(range(num_periods_per_fold * k))
    rng = np.random.default_rng(seed = 101)
    kfolds_data = []
    kfolds_timestamps = []
    for idk_fold in range(k):

        # 5.1) Pick up some indexes randomly
        rand_indexes = rng.choice(list_idks, size = num_periods_per_fold, replace = False)
        
        # 5.2) Retrieve the data for the selected periods
        fold_data = np.concatenate([periods_data[idk]['data'] for idk in rand_indexes])
        fold_ts = np.concatenate([periods_data[idk]['timestamps'] for idk in rand_indexes])

        print(f"FOLD {idk_fold + 1} --> SHAPE: {fold_data.shape} --> PERIODS ({rand_indexes.shape[0]}): {rand_indexes}")
        
        # Save the fold
        kfolds_data.append(fold_data)
        kfolds_timestamps.append(fold_ts)

        # Delate the selected indexes
        for idk in rand_indexes:
            list_idks.remove(idk)

    # Add the extra periods
    if num_extra_periods > 0: 

        # Sort the periods according to their dimensions
        sorted_dim_folds = sorted([{'idk':idk, 'dim': kfolds_data[idk].shape[0]} for idk in range(k)], 
                                 key = lambda item: item['dim'], reverse = True)
        if verbose: 
            print("\nThere is/are some extra period(s)...")
            print("PREVIOUS FOLDS:", sorted_dim_folds, "\n")
        
        # Attach each extra periods in the folds with fewer observations
        for idk_pos in range(1, num_extra_periods + 1):

            # Retrieve the data for this (extra) period 
            extra_period = periods_data[- idk_pos]['data']
            extra_period_ts = periods_data[- idk_pos]['timestamps']

            # Find the fold (i.e., idk) to be used
            fold = sorted_dim_folds[- idk_pos]
            pos = fold['idk']
            
            # Concatenate the existing data with the extra period
            kfolds_data[pos] = np.concatenate([kfolds_data[pos], extra_period])
            kfolds_timestamps[pos] = np.concatenate([kfolds_timestamps[pos], extra_period_ts])

            print(f"EXTRA PERIOD {idk_pos} ({extra_period.shape[0]} obs.)--> FOLD {fold}")

        # Check the final dimension of the fold 
        sorted_dim_folds = sorted([{'idk':idk, 'dim': kfolds_data[idk].shape[0]} for idk in range(k)], 
                                 key = lambda item: item['dim'], reverse = True)
        print("\nFOLDS:", sorted_dim_folds)

    return kfolds_data, kfolds_timestamps

# ----------------------------------------------------------------------
# -------- FUCTION: Train the Self-Organizing Map ----------------------
# ----------------------------------------------------------------------
def train_som(train_matrix, dim_grid, epoch, learning_rate, sigma, map_topology, neighborhood_function, activation_distance, 
              shuffling_flag, verbose):

    # Visualize the parameters
    print(50*"-" + f"\n\tPARAMETERS: Self-Organizing Map (SOM)\n" + 50*"-")
    print(f"(SOM) GRID SIZE: {dim_grid}x{dim_grid}")
    print(f"(SOM) EPOCH: {epoch} {'(with shuffling)' if shuffling_flag else ''}")
    print(f"(SOM) LR: {learning_rate}")
    print(f"(SOM) SIGMA: {sigma}")
    print(f"(SOM) MAP TOPOLOGY: '{map_topology}'")
    print(f"(SOM) NEIGHBOURHOOD FUNCTION: '{neighborhood_function}'")
    print(f"(SOM) ACTIVATION DISTANCE: '{activation_distance}'\n")

    # SOM: Self-Organizing Map
    som = MiniSom(x = dim_grid, y = dim_grid, 
                  input_len = train_matrix.shape[1], 
                  sigma = sigma, 
                  learning_rate = learning_rate, 
                  neighborhood_function = neighborhood_function, 
                  topology = map_topology,
                  activation_distance = activation_distance,
                  random_seed = 99)
    
    # Weight initialization 
    if train_matrix.shape[1] > 1: # MINIMUM FEATURES: 2 (for PCA initialization)
        som.pca_weights_init(train_matrix)
    
    # Train phase
    print(10 * "-", "Training the SOM", 10 * "-", "\n")
    som.train(train_matrix, epoch, random_order = shuffling_flag, verbose = True)
    
    # Quantization error (SOURCE: GitHub repo)
    # Returns the quantization error computed as the average distance 
    # between each input sample and its best matching unit. 
    quantization_error = som.quantization_error(train_matrix)
    weights = som.get_weights()

    # Quantization error (SOURCE: GitHub answer)
    # "it simply tells you how much information you lose in case that you quantize your data with the SOM. 
    # If the quantization error is 0 the weights of your network are exactly as the original data. 
    # To know if the SOM is reliable, you have to test it for your specific application"
    # ---------------------
    # "You need to tune the SOM to have the quantization error that you desire. 
    # More clusters means lower quantization error. 
    # The best solution only depends in how many clusters there's in your data"
    
    # Visualize the distances 
    if verbose:
        distance_map = som.distance_map()
        columns = ["N" + str(j) for j in range(som.distance_map().shape[0])]
        
        print(f"\n DISTANCE MAP {distance_map.shape}:")
        display(pd.DataFrame(distance_map, columns = columns, index = columns))

        # "They will specify clusters or centers of vectors."
        # print(f"\nWEIGHTS {weights.shape}: \nN(0, 0):", np.round(weights[0, 0, :] ,2), "\nN(0, 1):", np.round(weights[0, 1, :] ,2))
    
    return som, quantization_error, weights


# -----------------------------------------------------------------------------
# ------ ----- Compute all the variables needed for computing the metrics ----
# -----------------------------------------------------------------------------
def compute_metrics_for_test_set(inv_name, kpi_scores, thresholds, test_timestamps, test_data, test_obs_to_ignore, fault_df, 
                                 prediction_window = 7, use_starting_failure_ts = False, verbose = False):
    
    # B) Create the WARNINGS
    inv_warnings = create_warning(kpi_scores, thresholds, test_obs_to_ignore)
    
    # C) Retrieve the INVERTER'S FAUL EVENTS
    # C.1) Isolate only the (General) faults & Log - High
    events_to_consider = ["General Fault", "Log - High"]
    filtered_fault_df = fault_df[fault_df["Tipo"].isin(events_to_consider)]
    
    # C.2) Find out the fault events for the inverter
    fault_events, unique_faults = find_fault_observation(filtered_fault_df, test_data, inv_name,
                                                         include_faults_notRelatedToInverters = False, verbose = False)
    failure_events = dict([(ts, fault) for ts, faults in fault_events.items() for fault in faults])           
    if use_starting_failure_ts: 
        failure_events = dict([(fault[3][0], fault[2]) for ts, fault in failure_events.items()])
    
    # Compute the raw metrics (i.e., TP, FP, TN, FN)                 
    metrics = compute_correct_wrong_predictions(test_timestamps, failure_events, inv_warnings.index, prediction_window, verbose = False)
    true_positive, true_negative, false_positive, false_negative = metrics
    
    # Compute the F1-score
    f1_score, recall, precision = compute_f1_score(true_positive, false_positive, false_negative, true_negative, verbose=True)
        
    return f1_score, recall, precision

def compute_train_grid_search(som, train_matrix, train_timestamps, sliding_window, std_multipliers):

    # 1) Compute the mapped space (train data)
    train_mapped_space, train_obs_counter = compute_mapping_space_minimal(som, train_matrix, verbose = False)
    num_train_obs = train_matrix.shape[0]

    # 2) Retrieve the mapping space
    dim_mapping_space = (len(som._xx), len(som._yy))

    # 3) [TRAIN DATA] Find thresholds (by computing the KPI on the train data)
    thresholds, train_probs = compute_train_kpi_scores(train_timestamps, train_obs_counter, train_mapped_space, num_train_obs, 
                                                        dim_mapping_space, sliding_window, std_multipliers, compute_graphs = False, 
                                                        diagnostic_mode = False, verbose = False)
    return thresholds, train_probs, train_obs_counter

def compute_test_grid_search(som, train_obs_counter, num_train_obs, test_matrix, test_timestamps, sliding_window, verbose = False):

    # 1) Compute the mapped space (test data)
    test_mapped_space, test_obs_counter = compute_mapping_space_minimal(som, test_matrix)
    
    # 2) Retrieve the mapping space
    dim_mapping_space = (len(som._xx), len(som._yy))
    
    # 3) Compute the kpi scores (test data)
    kpi_scores_df = compute_test_kpi_scores(test_timestamps, train_obs_counter, test_mapped_space, dim_mapping_space, num_train_obs, 
                                            sliding_window, verbose)
    return kpi_scores_df

def compute_metrics_kfolds_validation(som, inv_name, train_matrix, train_timestamps, test_matrix, test_timestamps, test_obs_to_ignore,
                                      num_folds, kpi_sliding_window, fault_df, prediction_window = 7):

    # TRAIN DATA: Compute the threshold and train artefacts (e.g., train probs)
    std_multipliers = [3, 5]
    thresholds, train_probs, train_obs_counter = compute_train_grid_search(som, train_matrix, train_timestamps, kpi_sliding_window, 
                                                                           std_multipliers)
    # 1) Divide the test data into K folds
    test_timestamps = np.array(test_timestamps)
    kfolds_data, kfolds_timestamps = k_fold_split(test_matrix, test_timestamps, k = num_folds)

    kfolds_start_time = time.time()
    kfolds_metrics = []

    # 2)  Compute the perfomance per each fold
    for idk_fold, fold_test_data in enumerate(kfolds_data):

        # 2.1) Retrieve the timestamps related to this fold data
        fold_test_timestamps = kfolds_timestamps[idk_fold]

        print("\n" + "-" * 40, f"[{inv_name}] FOLD {idk_fold + 1}", "-" * 40)
        print(f"\n[{inv_name}] FOLD {idk_fold + 1} --> DATA: {fold_test_data.shape} --> {fold_test_timestamps.shape}")

        # 2.2) Retrieve the observations to ignore for this fold 
        if test_obs_to_ignore != None:
            fold_obs_to_ignore = list(filter(lambda ts: ts in test_obs_to_ignore, fold_test_timestamps))
                                
            if len(fold_obs_to_ignore) == 0:
                fold_obs_to_ignore = None
            else:
                print(f"\t--> FOLD OBS TO IGNORE: {len(fold_obs_to_ignore)}/{len(test_obs_to_ignore)}")
        else:
            fold_obs_to_ignore = None

        # 2.2) Compute the kpi scores for this fold
        num_train_obs = train_matrix.shape[0]
        kpi_scores_df = compute_test_grid_search(som, train_obs_counter, num_train_obs, test_matrix, test_timestamps, 
                                                 kpi_sliding_window, verbose = False)
        
        # 2.3) Compute the metrics for this fold
        print("-"* 110 + f"\n\t\tGRID SEARCH B2: [FOLD {idk_fold + 1}] Computing all the variables for "\
              "assessing the F1-score\n\t\t\t (i.e., KPI Thresholds, KPI scores, warnings, raw metrics)\n" + "-"* 110)
        test_data = pd.DataFrame(fold_test_data, index = fold_test_timestamps)
        fold_f1_score, fold_recall, fold_precision = compute_metrics_for_test_set(inv_name, kpi_scores_df, thresholds, test_timestamps, 
                                                                                  test_data, test_obs_to_ignore, fault_df, 
                                                                                  prediction_window, use_starting_failure_ts = True,
                                                                                  verbose = False)
        kfolds_metrics.append((fold_f1_score, fold_recall, fold_precision))

        # 2.3B) Visualize the perfomance for this fold
        print("-"* 110 + f"\n\t\t\t\tGRID SEARCH B3: [FOLD {idk_fold + 1}] Metrics \n" + "-"* 110)
        print(f"\t\t\t\t\t[F1 SCORE: {np.round(fold_f1_score, 4)}]\n\t\t\t\t\t   RECALL: {np.round(fold_recall, 4)}"\
          f"\n\t\t\t\t\tPRECISION: {np.round(fold_precision, 4)}\n")

    # 2.4) End time elapsed
    kfolds_time_elapsed = str(timedelta(seconds = (time.time() - kfolds_start_time)))

    # 3) Compute the average metrics for all the folds
    kfolds_metrics = np.concatenate(kfolds_metrics).reshape(3, -1)
    avg_folds_metrics = np.nansum(kfolds_metrics, axis = 0)/num_folds
    avg_folds_metrics = {'f1': avg_folds_metrics[0], 'recall': avg_folds_metrics[1],  'precision': avg_folds_metrics[2]}
    return avg_folds_metrics, kfolds_metrics, kfolds_time_elapsed                  
    
# -----------------------------------------------------------------------------
# --------------------------- Grid Search ------------------------------
# -----------------------------------------------------------------------------
def grid_search(inv_name, train_matrix, train_timestamps, test_matrix, test_timestamps, test_obs_to_ignore, params, map_topology,
                activation_distance, som_folder_path, fault_df, config_type, shuffling_flag, verbose = True):
    
    # Unpack the parameters
    epoch_values, dim_grid_values, learning_rate_values, sigma_values, neighborhood_functions = params
    
    # Compute the total of confgurations
    total_config = len(epoch_values) * len(dim_grid_values) * len(learning_rate_values) * len(sigma_values) * len(neighborhood_functions)
    config_num = 0
    
    # Inizilize the variables used to find the best (and second best) perfomance 
    best_perfomance = {'f1': -1, 'recall': -1, 'precision': -1}
    second_best_perfomance = {'f1': -2, 'recall': -2, 'precision': -2}
    
    # Compute each configuration
    for epoch in epoch_values:
        for dim_grid in dim_grid_values:
            for learning_rate in learning_rate_values:
                for sigma in sigma_values:
                    for neighborhood_function in neighborhood_functions:
                        
                         # 0.A) Create the config name
                        if epoch >= 10**6:
                            simplified_epoch = str(epoch//10**6) + "M" 
                        elif epoch >= 1000: 
                            simplified_epoch = str(epoch//1000) + "K"
                        else: 
                            simplified_epoch = str(epoch)
                        config_num += 1
                        config = f"{dim_grid}grid_{simplified_epoch}epoch_{learning_rate}lr_"\
                        f"{sigma}sigma_{neighborhood_function}Func" 
                        
                        # 0.B) Skip this combination if it is invalid
                        if (neighborhood_function == "bubble") and (sigma < 1):
                            print(f"\n--> ({config_num}) This config ({config}) has been skipped ")
                            continue 
                        if (dim_grid <= 20) and (learning_rate == 0.01):
                            print(f"\n--> ({config_num}) This config ({config}) has been skipped.")
                            continue 
                        if (dim_grid > 20) and (learning_rate == 0.001):
                            print(f"\n--> ({config_num}) This config ({config}) has been skipped.")
                            continue 
                        
                        # 1) Train the Self-Organizing Map (SOM)
                        som_start_time = time.time()
                        som, quantization_error, weights = train_som(train_matrix, dim_grid, epoch, learning_rate,
                                                                     sigma, map_topology, neighborhood_function,
                                                                     activation_distance, shuffling_flag, verbose=False)
                        som_time_elapsed = str(timedelta(seconds = (time.time() - som_start_time)))
                        print("-"* 110 + f"\n\t\tGRID SEARCH A: The SOM has been trained (config: {config_num}/{total_config})"\
                              "\n\t\t --> "  + config + "\n" + "-"* 110)
                       
                        # 2) Retrieve the k-fold test datasets # test_matrix, test_timestamps
                        print("-"* 110 + "\n\t\tGRID SEARCH B: K-Fold validation (i.e., Compute the average metrics "\
                              "across a folded test data)\n" + "-"* 110)
                        num_folds = 3
                        kpi_sliding_window = 24
                        avg_folds_metrics, kfolds_metrics, kfolds_time_elapsed = compute_metrics_kfolds_validation(som, inv_name, 
                                                                                                                   train_matrix, 
                                                                                                                   train_timestamps, 
                                                                                                                   test_matrix,
                                                                                                                   test_timestamps, 
                                                                                                                   test_obs_to_ignore,
                                                                                                                   num_folds,
                                                                                                                   kpi_sliding_window, 
                                                                                                                   fault_df)
                        #Visualize the perfomance
                        print("-"* 110 + f"\n\t\t\t\tGRID SEARCH B4: [{num_folds}-FOLDS METRICS] Metrics \n" + "-"* 110)
                        print(f"\t\t\tCONFIG {config_num}/{total_config}: {config}\n")
                        print(f"\t\t\t\t\t[F1 SCORE: {np.round(avg_folds_metrics['f1'], 4)}]\n\t\t\t\t\t   RECALL: "\
                              f"{np.round(avg_folds_metrics['recall'], 4)} \n\t\t\t\t\t"\
                              f"PRECISION: {np.round(avg_folds_metrics['precision'], 4)}\n")

                        print("-" * 30, "FOLDS PERFORMANCE", "-" * 30)
                        print('\n'.join([f'FOLD {idk + 1} --> F1: {np.round(kfolds_metrics[idk][0], 4)} '\
                                         f'|| RECALL: {np.round(kfolds_metrics[idk][1], 4)} '\
                                         f'|| PRECISION: {np.round(kfolds_metrics[idk][2], 4)}' 
                                         for idk in range(kfolds_metrics.shape[0])]))
            
                        # 2.2) Save the first and second best trained SOM using the validation performance
                        if avg_folds_metrics['f1'] > best_perfomance['f1']:
                            best_config = config
                            best_trained_som = som
                            print(f"\n{config}: has been selected as the best, so far")

                            # Update the best metrics
                            best_perfomance['f1'] = avg_folds_metrics['f1'], 
                            best_perfomance['recall'] = avg_folds_metrics['recall']
                            best_perfomance['precision'] = avg_folds_metrics['precision']
         
                        elif avg_folds_metrics['f1'] > second_best_perfomance['f1']:
                            second_best_config = config
                            second_best_trained_som = som

                            # Update the second best metrics
                            second_best_perfomance['f1'] = avg_folds_metrics['f1'], 
                            second_best_perfomance['recall'] = avg_folds_metrics['recall']
                            second_best_perfomance['precision'] = avg_folds_metrics['precision']
                     
                        # 3.3) Save the perfomance (of the test set) into a txt file
                        # Create/open the file
                        file_name = f"{inv_name}_som_folds_performance_{config_type}_ALT.txt"
                        perfomance_file = open(path.join(som_folder_path, file_name),  mode = "a")
                        
                        # Build the summary for the perfomance
                        perfomance_text = f"QUANTIZATION ERROR (Train): {round(quantization_error, 6)}\n"
                        perfomance_text += f"F1-SCORE: {round(avg_folds_metrics['f1'], 4)} "\
                                           f"({', '.join([str(np.round(value, 4)) for value in kfolds_metrics[:, 0]])})\n"
                        perfomance_text += f"RECALL: {round(avg_folds_metrics['recall'], 4)}\n"
                        perfomance_text += f"PRECISION: {round(avg_folds_metrics['precision'], 4)}"
                        computational_time = f"SOM TIME: {som_time_elapsed}\nK-FOLDS TIME: {kfolds_time_elapsed}\n"

                        # Write the perfomance on a TXT file
                        perfomance_file.write(f"CONFIG ({config_num}/{total_config}): " + config + "\n" + perfomance_text + "\n" + 
                                              computational_time + "-" * 50 + "\n")
                        # Close the file
                        perfomance_file.close()
                        
                        # ------------------------------------------------------------
                        # -------------- OLD CODE (Validation/Test split) ------------
                        # ------------------------------------------------------------
                        # 2) VALID SET: Compute the metrics to carry out the hyper-parameter optimization
                        #print("-"* 110 + "\n\t\tGRID SEARCH B: [VALIDATION SET] Computing all the variables for "\
                              #"assessing the F1-score\n\t\t\t (i.e., KPI Thresholds, KPI scores, warnings, raw metrics)\n" + "-"* 110)
                        #kpi_sliding_window = 24 
                        #valid_kpi_start_time = time.time()
                        #valid_f1_score, valid_recall, valid_precision = compute_metrics_for_test_set(som, inv_name, train_matrix,
                                                                                                     #train_timestamps, valid_matrix,
                                                                                                     
                        # valid_timestamps,valid_obs_to_ignore, 
                                                                                                    # kpi_sliding_window, fault_df, 
                                                                                                     #verbose = True)
                        #valid_kpi_time_elapsed = str(timedelta(seconds = (time.time() - valid_kpi_start_time)))
                        #valid_perfomance = f"QUANTIZATION ERROR(Train): {round(quantization_error, 6)}\n"\
                                           #f"F1-SCORE: {round(valid_f1_score, 4)}\nRECALL: {round(valid_recall, 4)}"\
                                           #f"\nPRECISION: {round(valid_precision, 4)}"
                        #print("-"* 110 + "\n\t\tGRID SEARCH B: All the metrics have been computed\n")
                        #print(f"\t\t--> GRID SEARCH: PERFORMANCE [VALIDATION SET] --> \n{valid_perfomance}" + "\n")
                        #print("-"* 110)
                        
                        # 2.B) Save the perfomance (of the test set) into a txt file
                        #file_name = f"{inv_name}_som_valid_performance_{config_type}.txt"
                        #perfomance_valid_file = open(path.join(som_folder_path, file_name),  mode = "a")
                        #perfomance_valid_file.write(f"CONFIG ({config_num}/{total_config}): " + config + "\n" + valid_perfomance + "\n" + 
                                              #f"SOM TIME: {som_time_elapsed}\n" + f"KPI TIME: {valid_kpi_time_elapsed}\n"
                                              #+ "-"* 50 + "\n")
                        #perfomance_valid_file.close()
  

                        # 3) TEST SET: Compute the metrics to assess the final perfomance 
                        #print("-"* 110 + "\n\t\tGRID SEARCH C: [TEST] Computing all the variables for assessing the F1-score\n\t\t\t"\
                              #"(i.e., KPI Thresholds, KPI scores, warnings, raw metrics)\n" + "-"* 110)
                        #kpi_start_time = time.time()
                        #kpi_sliding_window = 24 
                        #f1_score, recall, precision = compute_metrics_for_test_set(som, inv_name, train_matrix, train_timestamps, 
                                                                                   #test_matrix, test_timestamps, test_obs_to_ignore, 
                                                                                   #kpi_sliding_window, fault_df, verbose=True)
                        # kpi_time_elapsed = str(timedelta(seconds = (time.time() - kpi_start_time)))
                        #perfomance = f"QUANTIZATION ERROR(Train): {round(quantization_error, 6)}\n"\
                                     #f"F1-SCORE: {round(f1_score, 4)}\nRECALL: {round(recall, 4)}\nPRECISION: {round(precision, 4)}"
                        #print("-"* 110 + "\n\t\tGRID SEARCH D: All the metrics have been computed\n")
                        #print(f"\t\t--> GRID SEARCH: PERFORMANCE [TEST SET] --> \n{perfomance}" + "\n")
                        #print("-"* 110)

                        
                
    # Creating the subfolders
    # a) Best version
    best_version_path = path.join(som_folder_path, best_config)
    if not path.exists(best_version_path):
        makedirs(best_version_path)
    
    # a) Best version 
    best_perfomance_file = open(path.join(best_version_path, f"som_best_performance.txt"),  mode="w+")
    best_perfomance_file.write(best_config + f"\nPERFOMANCE: F1-SCORE = {best_perfomance['f1']} " + 
                               f"(RECALL: {best_perfomance['recall']}, PRECISION: {best_perfomance['precision']})")
    best_perfomance_file.close()
    
    #try: 
        # b) Second best version
        #second_version_path = path.join(som_folder_path, second_best_config)
        #if not path.exists(second_version_path):
           #makedirs(second_version_path)
        # b) Second best version 
        #second_best_perfomance_file = open(path.join(second_version_path, f"som_secondBest_performance.txt"),  mode="w+")
        #second_best_perfomance_file.write(second_best_config + f"\nPERFOMANCE: F1-SCORE = {second_best_perfomance[0]} " + 
                                   #f"(RECALL: {second_best_perfomance[1]}, PRECISION: {second_best_perfomance[2]}")
        #second_best_perfomance_file.close()
        
    #except UnboundLocalError:
        #print("No second best available")
        #return best_trained_som, best_config, None, None
                
    return best_trained_som, best_config 

def compute_mapping_space_minimal(som, matrix, verbose = False):
    
    # Mapping space
    mapped_space = som.win_map(matrix, return_indices = True) 
    
    obs_counter = dict()
    for neuron in sorted(mapped_space.keys()):

        # Indexes of the observations mapped in this neuron
        idk_obs_mapped_into_neuron = mapped_space[neuron]
        
        # Save the number of observations mapped in this neuron
        num_obs = len(idk_obs_mapped_into_neuron)
        obs_counter[neuron] = num_obs
    return mapped_space, obs_counter

# -----------------------------------------------------------------------------
# -------- FUCTION: Compute e visualize the mapped space ----------------------
# -----------------------------------------------------------------------------
def compute_mappig_space(som, train_matrix, test_matrix, verbose=False):
    
    # Mapping space
    train_mapped_space = som.win_map(train_matrix, return_indices=True) 
    test_mapped_space = som.win_map(test_matrix, return_indices=True) 
    
    # VISUALIZE A: Mapped space of TRAIN DATA
    print(80*"-" + f"\n\t\t\t\tMapping outcomes (TRAIN DATA)\n" + 80*"-")
    print(f"OBSERVATIONS: {len(train_matrix)} ({int(round(len(train_matrix)/1000, 0))} K)")
    print(f"MAP NEURONS ({len(train_mapped_space.keys())}/{len(som._xx) * len(som._yy)}): ")
    
    if verbose:
        print(f"{sorted(list(train_mapped_space.keys()))}")
    
    train_obs_counter = dict()
    for neuron in sorted(train_mapped_space.keys()):
        idk_obs_mapped_into_neuron = train_mapped_space[neuron]
        
        num_obs = len(idk_obs_mapped_into_neuron)
        train_obs_counter[neuron] = num_obs
        
        if verbose:
            print(f"\nNEURON {neuron}: {num_obs} observations ({round(num_obs/len(train_matrix) * 100, 2)} %)"\
                  f" \n{sorted(idk_obs_mapped_into_neuron[:15])} ...")

    # VISUALIZE B: Mapped space of TEST DATA
    print(80*"-" + f"\n\t\t\t\tMapping outcomes (TEST DATA)\n" + 80*"-")
    print(f"OBSERVATIONS: {len(test_matrix)} ({int(round(len(test_matrix)/1000, 0))} K)")
    print(f"MAP NEURONS ({len(test_mapped_space.keys())}/{len(som._xx) * len(som._yy)}): ")
    
    if verbose:
        print(f"{sorted(list(test_mapped_space.keys()))}")
          
    test_obs_counter = dict()
    for neuron in sorted(test_mapped_space.keys()):
        # Indexes of the observations mapped in this neuron
        idk_obs_mapped_into_neuron = test_mapped_space[neuron]
        
        # Save the number of observations mapped in this neuron
        num_obs = len(idk_obs_mapped_into_neuron)
        test_obs_counter[neuron] = num_obs
        
        if verbose:
            print(f"\nNEURON {neuron}: {num_obs} observations ({round(num_obs/len(test_matrix) * 100, 2)} %)"\
                  f" \n{sorted(idk_obs_mapped_into_neuron[:15])} ...")
        
    return train_mapped_space, test_mapped_space, train_obs_counter, test_obs_counter
# -----------------------------------------------------------------------------
# ------------ FUCTION: Visualize the neuron probabilities --------------------
# SOURCE: https://github.com/JustGlowing/minisom/blob/master/examples/HexagonalTopology.ipynb
# -----------------------------------------------------------------------------
def plot_neur_freq(som, mapped_space, obs_counter, saving_folder, dataset_name, visual_output = True):
    xx, yy = som.get_euclidean_coordinates()
    umatrix = som.distance_map()
    weights = som.get_weights()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle(f"SOM mapped space ({dataset_name.replace('_', ':')} data)", fontsize=35, y = 0.95)
    sns.set_theme(style="whitegrid")
    
    # Add iteractively the hexagons
    freq_values = []
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            wy = yy[(i, j)] * np.sqrt(3) / 2
            
            try:
                freq = obs_counter[(i, j)]
                freq_values.append(freq)
            except KeyError:
                freq = 0
                freq_values.append(freq)
            
            hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, radius=.95 / np.sqrt(3),
                                 facecolor = cm.Blues(umatrix[i, j]), alpha = .6, edgecolor='gray')
            ax.add_patch(hex)
            
    # Highlight the activated neuron
    marker = "o"
    color = "orange"
    for neuron in sorted(mapped_space.keys()):
        
        # place a marker on the winning position for the sample xx
        wx, wy = som.convert_map_to_euclidean(neuron) 
        wy = wy * np.sqrt(3) / 2
        plt.plot(wx, wy, marker, markerfacecolor='None', markeredgecolor=color, 
                 alpha = 0.8, markersize=8, markeredgewidth=2)
    
    # X-Ticks
    if weights.shape[0] >= 28:
        xrange = np.arange(1, weights.shape[0] + 1, step = 2)
    else:
        xrange = np.arange(1, weights.shape[0] + 1)
    plt.xticks(xrange - 1, xrange)

    # Y-Ticks
    if weights.shape[1] >= 28:
        yrange = np.arange(1, weights.shape[1] + 1, step = 2)
    else:
        yrange = np.arange(1, weights.shape[1] + 1)
    plt.yticks((yrange * np.sqrt(3) / 2) - 0.87, yrange)
    
    # Legend
    legend_elements = [Line2D([0], [0], marker=marker, color=color, label='Neuron activated',
                              markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(0.4, 1.02), loc='upper left', 
               borderaxespad=0., ncol=1, fontsize=14)
    
    # Color bar
    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=0.3)
    #norm = Normalize(vmin = min(freq_values), vmax = max(freq_values))
    
    cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Blues, alpha = 0.6)#=, orientation='vertical',  alpha=.4, norm = norm
                     
    cb1.ax.get_yaxis().labelpad = 30
    cb1.set_label("Distance from neurons in the neighbourhood", rotation=270, fontsize=20)
    plt.gcf().add_axes(ax_cb)
    
    # Save the figure
    file_name = f"mapped_space_{dataset_name}.png"
    graph_path = path.join(saving_folder, file_name)
    plt.savefig(graph_path, bbox_inches='tight', pad_inches=1)
    
    if visual_output:
        plt.show()
    plt.close()
# -----------------------------------------------------------------------------
# ---- FUCTION: Compute the neuron probabilities for all the train data -------
# ----------------- Prob (CELL i, OBS: all train) -->  P(i,TRAIN) -------------
# -----------------------------------------------------------------------------
def compute_train_probabilities(train_obs_counter, dim_mapping_space, num_train_obs, verbose):        
    print(70*"-" + f"\nA) Computing the cell probabilities [P(i,TRAIN)] for all the train data.\n" + 70*"-")
    
    # Compute the probabilities for each available neurons --> Prob (CELL i, OBS: all train) -->  P(i,TRAIN)
    train_probs = np.zeros(shape = dim_mapping_space)
    for neuron_pos in train_obs_counter.keys():
        
        # Compute the probability 
        num_obs_neuron = train_obs_counter[neuron_pos]
        prob = num_obs_neuron / num_train_obs
        train_probs[neuron_pos] = prob
        
        if verbose:
            print("\nNEURON:", neuron_pos, "--> p = ", np.round(train_probs[neuron_pos], 4))
        
    # Check the validity by summing up all the probabilities
    probs_sum = train_probs.sum()
    if probs_sum != 1:
        print(f"WARNING: There's an error in the computed probabilities! SUM PROB: {probs_sum}")
        
    print("\n\t\t\t\tDONE")
    return train_probs

# -------------------------------------------------------------------------------
# ---- FUCTION: Compute the neuron probabilities for all the observations -------
# --------------------- Prob (CELL i, obs d) -->  P(i,obs) ----------------------
# -----------------------------------------------------------------------------
def compute_obs_probabilities(timestamps, test_mapped_space, dim_mapping_space, sliding_window, verbose=False):
    print("\n" + 70*"-"+ f"\nB) Computing the cell probabilities for the observations [P(i,d)].\n"\
          f"SLIDING WINDOW: {sliding_window} hours\n" + 70*"-")
    
    if not verbose:
        print(f"\nComputing the cell probabilities for {len(timestamps)} observations...")
    start_time = time.time()
    
    # 1) For each observation --> retrive the previous (-24 hours) observations (with their indexes)
    obs_prob = dict()
    for idk, timestamp in enumerate(timestamps):
        start_time_find_period = time.time()
        if verbose: 
            print(40*"-" + f"\n{idk + 1}) OBSERVATION: {timestamp.strftime('%Y-%m-%d, %H:%M')}\n" + 40*"-")
        
        # Compute the period 
        start_ts = timestamp - pd.Timedelta(sliding_window - 1, unit="hours") # 24 hours 
        period = pd.date_range(start_ts, timestamp, freq= "H")

        # Retrieve the actual observations included in the 24-hour period
        obs_period = [(idk, timestamp) for idk, timestamp in enumerate(timestamps) if timestamp in period]
       
        # Retrieve thier indexes --> IN CASE there is the observation Timestamp - 24 h (e.g., T: 10:00 --> T-1: 9:00) 
        is_start_ts_included = [True for idk, timestamp in obs_period if (timestamp == period[0])] 
        if is_start_ts_included:
            idk_tw_obs = [idk for idk, ts in obs_period]
            if verbose:
                print(f"{len(obs_period)} observations (time window: {sliding_window} hours): "\
                      f"{[ts.strftime('%H:%M') for idk, ts in obs_period]}\n")
        else:
            if verbose:
                print(f"No enough observations ({len(obs_period)}) in the time window of {sliding_window} hours.\n"\
                      f"Data available: {(obs_period[-1][1] - obs_period[0][1]).components[1]} hours.\n")
            continue

        # 2) Inizialize the probability matrix (i.e., zeros)
        tw_probs = np.zeros(shape = dim_mapping_space)
        total_tw_obs = len(idk_tw_obs)
        
        # 3) Compute the number of daily observations mapped in the cell i (over the total daily observation)
        for idk, neuron_pos in enumerate(sorted(test_mapped_space.keys())):
         
            # Compute the number of daily (test) observation mapped in the cell 
            daily_obs_per_neuron = [idk_obs for idk_obs in test_mapped_space[neuron_pos] if idk_obs in idk_tw_obs]

            # Compute the probability
            if len(daily_obs_per_neuron) > 0:
                cell_prob = len(daily_obs_per_neuron) / total_tw_obs
                tw_probs[neuron_pos] = cell_prob
                if verbose:
                    print(f"\t({idk + 1}) NEURON {neuron_pos}: observations mapped {len(daily_obs_per_neuron)}/{total_tw_obs} "
                          f"--> p = {np.round(tw_probs[neuron_pos], 4)}")
        # 4) Save the probability 
        obs_prob[timestamp.strftime('%Y-%m-%d, %H:%M')] = tw_probs
        
        start_time_find_obs_elapsed = str(timedelta(seconds = (time.time() - start_time_find_period)))
        print(f"END TIMESTAMP {timestamp.strftime('%Y-%m-%d, %H:%M')} --> ", start_time_find_obs_elapsed)
    
    start_time_elapsed = str(timedelta(seconds = (time.time() - start_time)))
    print(f"END TIMESTAMP {timestamp.strftime('%Y-%m-%d, %H:%M')} --> ", start_time_elapsed)    
    print("\n\t\t\t\tDONE")
    return obs_prob

# --------------------------------------------------------------------------------------
# -- EXPERIMENTAL FUNCTION: Compute the neuron probabilities for all the observations --
# --------------------- Prob (CELL i, obs d) -->  P(i,obs) -----------------------------
# --------------------------- NOTE: it's working ---------------------------------------
# --------------------------------------------------------------------------------------
def retriveActualDailyObs(df, row):
    try:
        obs = df.loc[row["Start Timestamp"]:row.name, "Index"].tolist()
        return obs
    except KeyError:
        #print(f"ERROR: No actual observations found in this sliding window --> ROW '{row.name}'")
        return []
    
def compute_obs_probabilities_EXPERIMENTAL(timestamps, mapped_space, dim_mapping_space, sliding_window, verbose=False,
                                          diagnostic_mode = False):
    print("\n" + 70*"-"+ f"\nB) [EXPERIMENTAL] Computing the cell probabilities for the observations [P(i,d)].\n"\
          f"SLIDING WINDOW: {sliding_window} hours\n" + 70*"-")
    
    # Minimum number of observations within a sliding window
    if diagnostic_mode:
        hard_lower_limit = 1
        print("MODE: Diagnostic (sliding window = 1 obs.)")
    else:
        hard_lower_limit = 14 
    
    start_time = time.time()
    print(f"\nComputing the cell probabilities for {len(timestamps)} observations...")
    
    # Create a dataframe
    timestamp_df = pd.DataFrame(data = list(range(len(timestamps))), index = timestamps, columns = ["Index"])
    
    # A) Compute the starting timestamp (T - 24)
    timestamp_df["Start Timestamp"] = timestamp_df.index - pd.Timedelta(sliding_window - 1, unit="hours")
    
    # B) Compute the theoretical period (T - 24h, T)
    compute_theoretical_sw_period = lambda row: pd.date_range(start = row["Start Timestamp"], end = row.name, freq = "H")
    timestamp_df["Theoretical Period"] = timestamp_df.apply(compute_theoretical_sw_period, axis=1)
    
    # C) Discover the actual timestamps found in the dataset within the theoretical period 
    timestamp_df["Sliding window"] = timestamp_df.apply(lambda row: retriveActualDailyObs(timestamp_df, row), axis = 1)

    # D) Compute the number of daily observation
    timestamp_df["Total daily obs"] = timestamp_df["Sliding window"].apply(lambda obs_period: len(obs_period))

    # E) Keep only the timestamps having a full sliding window (there are enough observations within the sliding window)
    # E.1) Count the daily observations
    freq_daily_obs = Counter(timestamp_df["Total daily obs"].tolist()).most_common()
    
    # E.2) Find the minimum daily obs
    # E.2.A ) Find the daily observations occurring at least in the the 5% of the dataset
    minimum_percentage_occurreces = 0.05
    freq_threshold = int(len(timestamp_df) * minimum_percentage_occurreces)
    freq_daily_obs = [(daily_obs, freq) for daily_obs, freq in freq_daily_obs if freq >= freq_threshold]
    freq_daily_obs = sorted(freq_daily_obs, key = lambda item: item[0])
    
    # E.2.B) Find minimum daily obs among the daily observations occurring in more than the 5% of the dataset
    minimum_daily_obs = freq_daily_obs[0][0]
    minimum_daily_obs = hard_lower_limit if minimum_daily_obs < hard_lower_limit else minimum_daily_obs
    print(f"Minimum daily observation: {minimum_daily_obs} ", 
        ('(i.e., that occurrs at least in the 5% of the dataset)' if freq_daily_obs[0][0] >= hard_lower_limit else ''))
    
    # E.3) Discard the sliding winding having less than the minimum daily obs
    valid_sliding_windows = timestamp_df[timestamp_df["Total daily obs"] >= minimum_daily_obs]
    
    if len(valid_sliding_windows) == 0:
        print("\nISSUE: Not enough observations in the sliding windows "\
              f"\nMinimum observations in the 24-hour sliding window:{hard_lower_limit} || "\
              f"Maxiumum found: {valid_sliding_windows['Total daily obs'].max()}\n")
        display(timestamp_df)
        return None
    timestamp_df = valid_sliding_windows
    
    # F) Find the neurons that have mapped these daily observations
    find_neurons = lambda sliding_window_period: Counter([neuron_pos for neuron_pos, obs_mapped in mapped_space.items()
                                                          for obs in obs_mapped
                                                          if obs in sliding_window_period])
    timestamp_df["Neurons"] = timestamp_df["Sliding window"].apply(find_neurons)
    
    # G) Compute the frequencies (i.e., probabilitis) for the activated neurons
    compute_prob = lambda row: [(neuron, freq / row["Total daily obs"]) for neuron, freq in row["Neurons"].items()]
    timestamp_df["Activated neuron probabilities"]= timestamp_df.apply(compute_prob, axis = 1)
    
    # H) Create an empty matrix the overall probabilities
    time_window_matrices = [np.zeros(shape = dim_mapping_space) for j in range(len(timestamp_df))]
    timestamp_df["Neuron probabilities"] = time_window_matrices
    
    # I) Assign the probabilities of the activated neurons in the matrix
    neuron_probabilities = list(timestamp_df["Neuron probabilities"])
    for idk, neurons in enumerate(list(timestamp_df["Activated neuron probabilities"])):
        
        if verbose:
            print("\nTS", idk)
            
        for neuron, prob in neurons:
            neuron_probabilities[idk][neuron] = prob
            
            if verbose: 
                print(neuron, "--> p = ", prob)
                
        if verbose:
            print("SUM:", np.sum(neuron_probabilities[idk]))
    
    # J) Create a dictionary for each timestamp (KEY: timestamp, VALUE: matrix of probabilities)
    obs_prob = dict(zip(timestamp_df.index, neuron_probabilities))
    print(f"\nKPI COMPUTED: {len(obs_prob.keys())}/{len(timestamps)} ({round((len(obs_prob.keys())/len(timestamps))*100, 2)} %)")

    print(f"\nEND Computing probs: --> ", str(timedelta(seconds = (time.time() - start_time))))
    return obs_prob

# -----------------------------------------------------------------------------
# ------------ FUCTION: Compute the KPI for an observation --------------------
# -----------------------------------------------------------------------------
def compute_single_kpi(timestamp, train_probs, obs_probs, verbose=True): 
    if verbose:
        print(80*"-" + f"\n\t\t\t\tComputing the KPI for '{timestamp}'\n" + 80*"-")

    if not obs_probs:
        return -1
        
    # Inizialize the matrix
    kpi_matrix = np.zeros(shape = (train_probs.shape[0], train_probs.shape[1]))
    
    # Retrieve the daily probabilities 
    try:
        timestamp_probs = obs_probs[timestamp]
    except KeyError:
        if verbose:
            print(f"ISSUE ({timestamp}): Probabilities are unavailable.")
        return np.nan
    
    # Retrieve the activated neurons (those with the probability > 0)
    if verbose:
        activated_neurons = np.argwhere(timestamp_probs != 0)
        print("Activated NEURONS:")
        print(activated_neurons)
    
    # Compute the matrix KPI
    diff = np.abs(train_probs - timestamp_probs) # Cell i: P(Train) - P(obs day)
    ratio_diff = (1 - diff) / (1 + diff)
    kpi_matrix = timestamp_probs * ratio_diff
    
    # Compute the final KPI
    kpi = kpi_matrix.sum()
    if verbose:
        print(f"KPI: {round(kpi, 4)}")
        
    return kpi
# -----------------------------------------------------------------------------
# ------------ FUCTION: Compute the KPI for all the timestamps --------------------
# -----------------------------------------------------------------------------
def compute_kpi(timestamps, train_obs_counter, test_mapped_space, dim_mapping_space, num_train_obs, sliding_window, verbose=False, 
                diagnostic_mode = False):
    # PART A: P(i, TRAIN)
    train_probs = compute_train_probabilities(train_obs_counter, dim_mapping_space, num_train_obs, verbose)

    if verbose:
        mapped_obs = defaultdict(list)
        for neuron, list_idk_obs in test_mapped_space.items():
            if isinstance(list_idk_obs, list):
                for idk_obs in list_idk_obs:
                    mapped_obs[neuron].append(timestamps[idk_obs].strftime('%Y-%m-%d (%H:%M)'))
            else:
                mapped_obs[neuron].append(timestamps[list_idk_obs].strftime('%Y-%m-%d (%H:%M)'))
        mapped_obs = sorted(mapped_obs.items(), key = lambda value: value[0])
        print("-" * 20, "Mapped neurons", "-" * 20)
        for neuron, list_obs in mapped_obs:
            print("\nN:", neuron, "-->", ', '.join(list_obs))

    verbose = False

    # PART B: P(i, obs d)
    if sliding_window is not None:
        obs_probs = compute_obs_probabilities_EXPERIMENTAL(timestamps, test_mapped_space, dim_mapping_space, sliding_window, 
                                                           verbose, diagnostic_mode)
    else:
        test_days = sorted(set(timestamp.date() for timestamp in test_timestamps))
        obs_probs = compute_obs_probabilities(test_days, test_mapped_space, dim_mapping_space, sliding_window, verbose = False)
        
    # clear_output(wait=True)
    
    # Compute the KPI  for each timestamp
    print(80*"-" + f"\nC) Computing the KPI (SLIDE WINDOW: {sliding_window} hours)\n" + 80*"-")
    kpi_scores = []
    for timestamp in timestamps:
        kpi = compute_single_kpi(timestamp, train_probs, obs_probs, verbose = verbose)
        kpi_scores.append(kpi)
        
    # KPI scores: drop Nan values (i.e., timestamps without probabilities, as the slide window was not covered)
    kpi_scores = np.array(kpi_scores)
    valid_kpi_scores = kpi_scores[~ np.isnan(kpi_scores)]
    print(f"\nKPI scores are available for {len(valid_kpi_scores)} timestamps "\
          f"({round(len(valid_kpi_scores)/len(timestamps) * 100, 2)} %) out of {len(timestamps)}.")

    # Visualize some statistics    
    # Compute some descriptive statistic 
    valid_kpi_scores = kpi_scores[~ np.isnan(kpi_scores)]
    print("\n"+ 60*"-"+ "\n\tAVERAGE KPI (with sliding window)\n" + 60*"-")
    print(f"MIN: {round(valid_kpi_scores.min(), 4)} || MAX: {round(valid_kpi_scores.max(), 4)} || "\
          f"AVG: {round(np.mean(valid_kpi_scores), 4)} || STD: {round(np.std(valid_kpi_scores), 4)}")
    
    return kpi_scores, train_probs, obs_probs
# -----------------------------------------------------------------------------
# ------------ FUCTION: Compute the threshold for the KPI- --------------------
# -----------------------------------------------------------------------------
def compute_thresholds(kpi_scores, std_multipliers = [3, 5]):
    # Drop NaN values
    kpi_scores = kpi_scores[~ np.isnan(kpi_scores)]
    
    # Compute mean (μ)
    mean = np.mean(kpi_scores)
    print("MEAN (μ): ", round(mean, 4))
    
    # Compute the standar deviation (σ)
    std = np.std(kpi_scores)
    print("STAND. DEV. (σ): ", round(std, 4))
    
    # Threshold 1
    threshold_1 = mean - (std_multipliers[0] * std)
    print(f"\nTHRESHOLD 1 (μ - {std_multipliers[0]}σ): ", round(threshold_1, 4))
    
    # Threshold 2
    threshold_2 = mean - (std_multipliers[1] * std)
    print(f"THRESHOLD 2 (μ - {std_multipliers[1]}σ): ", round(threshold_2, 4))

    return threshold_1, threshold_2

# -----------------------------------------------------------------------------
# ----------- FUCTION: Compute the warnings using the KPI scores --------------
# -----------------------------------------------------------------------------

def compute_kpi_scores_minimal(som, inv_data, train_neuron_prob, sliding_window, verbose = False):   
    
    # 0) Generete the numerical matrix 
    input_matrix, columns, timestamps = to_num_matrix(inv_data)
    
    # 1) Compute the mapped space suing the trained som
    mapped_space, obs_counter = compute_mapping_space_minimal(som, input_matrix, verbose = False)
    
    if verbose:
        mapped_obs = defaultdict(list)
        for neuron, list_idk_obs in mapped_space.items():

            if isinstance(list_idk_obs, list):
                for idk_obs in list_idk_obs:
                    mapped_obs[neuron].append(timestamps[idk_obs].strftime('%Y-%m-%d (%H:%M)'))
            else:
                mapped_obs[neuron].append(timestamps[list_idk_obs].strftime('%Y-%m-%d (%H:%M)'))
        mapped_obs = sorted(mapped_obs.items(), key = lambda value: value[0])
        print("-" * 20, "Mapped neurons", "-" * 20)
        for neuron, list_obs in mapped_obs:
            print("N:", neuron, "-->", ', '.join(list_obs))
    
    # 2) Compute the KPI
    # 2.1) Compute neuron probabilities for the observations
    dim_mapping_space = (len(som._xx), len(som._yy))
    obs_probs = compute_obs_probabilities_EXPERIMENTAL(timestamps, mapped_space, dim_mapping_space, sliding_window,
                                                       verbose = False)
    display(obs_probs)

    # Compute the KPI  for each timestamp
    print(80*"-" + f"\nC) Computing the KPI (SLIDE WINDOW: {sliding_window} hours)\n" + 80*"-")
    kpi_scores = []
    for timestamp in timestamps:
        kpi = compute_single_kpi(timestamp, train_neuron_prob, obs_probs, verbose = verbose)
        kpi_scores.append(kpi)
        
    # KPI scores: drop Nan values (i.e., timestamps without probabilities, as the slide window was not covered)
    kpi_scores = np.array(kpi_scores)
    valid_kpi_scores = kpi_scores[~ np.isnan(kpi_scores)]
    print(f"\nKPI scores are available for {len(valid_kpi_scores)} timestamps "\
          f"({round(len(valid_kpi_scores)/len(timestamps) * 100, 2)} %) out of {len(timestamps)}.")
    
    kpi_scores_df = pd.DataFrame(kpi_scores, index = timestamps, columns = ["KPI score"])
    return kpi_scores_df

def compute_test_kpi_scores(test_timestamps, train_obs_counter, test_mapped_space, dim_mapping_space, num_train_obs, sliding_window, 
                            verbose, diagnostic_mode = False):

    print("\n" + 80 * "-" + "\n\t\t\t\tComputing the Test KPI\n" + 80 * "-")
    kpi_scores, *_ = compute_kpi(test_timestamps, train_obs_counter, test_mapped_space, dim_mapping_space, 
                                 num_train_obs, sliding_window, verbose, diagnostic_mode)

    kpi_scores_df = pd.DataFrame(kpi_scores, index = test_timestamps, columns = ["KPI score"])
    return kpi_scores_df

def compute_train_kpi_scores(train_timestamps, train_obs_counter, train_mapped_space, num_train_obs, dim_mapping_space, sliding_window, 
                            std_multipliers, compute_graphs = False, diagnostic_mode = False, verbose = False):

    # 1.1) Compute the TRAIN KPIs to compute the thresholds
    print("\n" + 80 * "-" + "\n\t\t\tComputing the train KPI for creating the thresholds. \n" + 80 * "-")
    train_kpi_scores, train_probs, obs_probs = compute_kpi(train_timestamps, train_obs_counter, train_mapped_space, dim_mapping_space, 
                                                           num_train_obs, sliding_window, verbose, diagnostic_mode)
    if compute_graphs:
        y = train_kpi_scores
        x = train_timestamps
        plt.ylabel("KPI Score")
        plt.boxplot(y)
        plt.show()
        plt.scatter(x, y, alpha = .1)
        plt.ylabel("KPI Score")
        plt.show()

    # 1.2) Compute the threshold --> using the KPI scores of train observations
    print("\n" + 80 * "-" + "\n\t\t\tComputing KPI thresholds. \n" + 80 * "-")        
    thresholds = compute_thresholds(train_kpi_scores, std_multipliers)
    
    return thresholds, train_probs

def compute_kpi_scores(som, inv_name, train_matrix, train_timestamps, test_matrix, test_timestamps, std_multipliers, sliding_window, 
                       graphs_folder = None, visualize_graphs = False, compute_graphs = True, diagnostic_mode = False):
    
    # 1a) Compute the mapped space (train datta)
    train_mapped_space, train_obs_counter = compute_mapping_space_minimal(som, train_matrix, verbose = False)
    num_train_obs = train_matrix.shape[0]

    # 1b) Compute the mapped space (test datta)
    test_mapped_space, test_obs_counter = compute_mapping_space_minimal(som, test_matrix, verbose = False)

    # 1c) Graphical representations of train & test data
    if compute_graphs:
        plot_neur_freq(som, train_mapped_space, train_obs_counter, graphs_folder, dataset_name = f"{inv_name}_Train",
                       visual_output = visualize_graphs)
        plot_neur_freq(som, test_mapped_space, test_obs_counter, graphs_folder, dataset_name = f"{inv_name}_Test",
                       visual_output = visualize_graphs)

    # 2) Retrieve the mapping space
    dim_mapping_space = (len(som._xx), len(som._yy))

    # 3) [TRAIN DATA] Find thresholds (by computing the KPI on the train data)
    thresholds, train_probs = compute_train_kpi_scores(train_timestamps, train_obs_counter, train_mapped_space, num_train_obs, 
                                                        dim_mapping_space, sliding_window, std_multipliers, compute_graphs, 
                                                        diagnostic_mode, verbose = False)

    # 4) [TEST DATA] Compute the KPI SCORES 
    verbose = False
    kpi_scores_df = compute_test_kpi_scores(test_timestamps, train_obs_counter, test_mapped_space, dim_mapping_space, num_train_obs, 
                                            sliding_window, verbose, diagnostic_mode)

    return kpi_scores_df, thresholds, train_probs

# -----------------------------------------------------------------------------
# ----------- FUCTION: Compute the warnings using the KPI scores --------------
# -----------------------------------------------------------------------------
def create_warning(kpi_scores, thresholds, obs_to_ignore = None, consecutive_num_obs = [2], keep_zero_values = False, 
                   lower_is_an_anomaly = True, diagnostic_mode = False):
    print("\n" + 80 * "-" + f"\n\t\t\t\t Warning levels \n" + 80 * "-")
    
    # Remove potential unwanted observavations (i.e., discarted using the regression)
    if obs_to_ignore is not None:
        print(f"OBSERVATIONS TO IGNORE: {len(obs_to_ignore)} observations")
        kpi_scores = kpi_scores.drop(index = obs_to_ignore)
    
    # CONDITION 1: numerical KPI thresholds
    if lower_is_an_anomaly:
        under_t1 = kpi_scores[kpi_scores["KPI score"] < thresholds[0]]
        under_t2 = kpi_scores[kpi_scores["KPI score"] < thresholds[1]]
    else: 
        under_t1 = kpi_scores[kpi_scores["KPI score"] > thresholds[0]]
        under_t2 = kpi_scores[kpi_scores["KPI score"] > thresholds[1]]
        
    # Drop empty KPI values
    kpi_scores.dropna(how="all", inplace = True)
    
    # Create the basic warning levels (< t1 --> 1, < t2 --> 3)
    kpi_scores.loc[:, "Warning level"] = 0
    kpi_scores.loc[under_t1.index,"Warning level"] = 1
    kpi_scores.loc[under_t2.index,"Warning level"] = 3
    
    # Select only valid warnings (warning level > 0)
    positive_warnings_idk = kpi_scores[kpi_scores["Warning level"] > 0].index
  
    # Compute the derivative --> (x1 - x0)
    if lower_is_an_anomaly:
        check_derivative = lambda diff: "Degradation" if diff < 0 else "Improvement"
    else: 
        check_derivative = lambda diff: "Degradation" if diff > 0 else "Improvement"
    kpi_scores.loc[:, "Behaviour"] = kpi_scores["KPI score"].diff().apply(lambda diff: check_derivative(diff)
                                                                          if not np.isnan(diff) else diff)
    
    # CONDITION 2: Set a warning level to 0 to in case the KPI score is improving (i.e, non-negative derivative)
    if not diagnostic_mode:
        improvement_cond = kpi_scores["Behaviour"] == "Improvement"
        kpi_scores.loc[improvement_cond, "Warning level"] = 0
   
    # CONDITION 3: Increase the warning level in case a persistence  (i.e., two or more consequtive warnings)
    positive_warnings_idk = kpi_scores[kpi_scores["Warning level"] > 0].index
    
    for num_obs in consecutive_num_obs:  
        for timestamp in positive_warnings_idk: 
            time_delta = pd.Timedelta(num_obs - 1, unit="hour") # --> FOR: Soletto 1&2, Galatina
            prev_obs = (pd.to_datetime(timestamp) - time_delta).strftime('%Y-%m-%d %H:%M:%S')
            
            if prev_obs in positive_warnings_idk:
                num_warning_increase = 1
                kpi_scores.loc[timestamp, "Warning level"] += num_warning_increase
    
    # Create the warnings --> Isolate only the warnings with a positive warning level
    kpi_scores.drop(columns = "Behaviour", inplace = True)
    
    if keep_zero_values:
        warnings = kpi_scores
    else:
        warnings = kpi_scores[kpi_scores["Warning level"] > 0]
    
    # Display visually the overall outcomes
    print(f"STATEGY: Numerical thresholds (T1: {round(thresholds[0], 4)}, T2: {round(thresholds[1],4)}) with:"\
          "\n\t(A) TIME PERSISTENCE PENALIZATION (>= 2 obs.) "\
          "\n\t(B) KPI scores with a DEGRADATION TREND (i.e., negative derivative)\n")
    print(f"WARNINGS: {len(warnings)}/{len(kpi_scores)}")
    dates = sorted(set(pd.to_datetime(timestamp).date().strftime('%Y-%m-%d') for timestamp in warnings.index))
    print(f"WARNING DATES ({len(dates)})")
    print("\n".join(dates))

    # Visualize the type of warnings
    grouped_warnings = warnings.groupby(by = "Warning level").count()["KPI score"].to_frame()
    grouped_warnings.sort_values(by = "Warning level", ascending = False, inplace=True)
    grouped_warnings.rename(columns = {"KPI score": "Timestamps"})
    print("-" * 30, "Grouped warning levels", "-" * 30)
    display(grouped_warnings)
 
    return warnings

# -----------------------------------------------------------------------------
# --------------- FUCTION: Warnings, analyse the faul side  -------------------
# -----------------------------------------------------------------------------
def analyse_fault_side(fault_events, inv_warnings, prediction_window):
    print(40*"-" + f"\nFAULT SIDE: ({len(fault_events)}) fault event timestamps. \n" + 40*"-")
    
    fault_event_timestamps = fault_events.keys()
    fault_event_dates = sorted(set(ts.strftime('%Y-%m-%d') for ts in fault_event_timestamps))
    print(f"FAULT DATES ({len(fault_event_dates)}):", (", ".join(fault_event_dates)))
    #print(f"WARNING DATES: ", (", ").join(sorted(set(pd.to_datetime(ts).strftime('%Y-%m-%d') for ts in inv_warnings.index))))
    
    # dict: KEY --> timestamp, values --> list of faults
    list_true_warnings = []
    list_missed_fault_event = []  
    for idk, fault_timestamp in enumerate(fault_events.keys()):
        start_prediction_ts = fault_timestamp - pd.Timedelta(prediction_window, unit="days")
        faults = fault_events[fault_timestamp]
        
        print("-" * 60 + f"\nFAULT TIMESTAMP ({idk + 1}/{len(fault_events.keys())}):", fault_timestamp.strftime('%Y-%m-%d (%H:%M)'))
        print(f"PREDICTION WINDOW (-{prediction_window} days): {start_prediction_ts}\n" + "-" * 60)
        
  
        for idk_fault, fault in enumerate(faults):
            print(f"   FAULT EVENT {idk_fault + 1}/{len(faults)}: \n   {fault[1]} ({fault[0]}): {fault[2]}"\
                  f"\n   [{fault[3][0].strftime('%Y-%m-%d (%H:%M)')} - {fault[3][1].strftime('%Y-%m-%d (%H:%M)')}]")

            # Detect the true predictions --> The warning is within the prediction window
            true_warnings = [(warning_ts, inv_warnings.loc[warning_ts, "Warning level"])
                             for warning_ts in inv_warnings.index.tolist()
                             if start_prediction_ts <= pd.to_datetime(warning_ts) <= fault_timestamp]

            if len(true_warnings) > 0:
                print(f"\n       --> FOUND SOME VALID WARNINGS ({len(true_warnings)} timestamps)!")
                
                # Save each true warning 
                for warning_ts, warning_level in true_warnings:
                    warning_timestamp = pd.to_datetime(warning_ts)
                    list_true_warnings.append({
                            "Fault timestamp": fault_timestamp.strftime('%Y-%m-%d, %H:%M'),
                            "Fault": fault,
                            "Prediction": warning_timestamp.strftime('%Y-%m-%d, %H:%M'),
                            "Advance day": (warning_timestamp - fault_timestamp).to_pytimedelta(),
                            "Warning level": warning_level
                        })
                    
                # Visualize the true warnings
                dates = sorted(set(pd.to_datetime(timestamp).strftime('%Y-%m-%d') for timestamp, wl in true_warnings))
                print(f"       --> WARNINGS DATES ({len(dates)}):", ", ".join(dates))
                print(f"       --> ADVANCE DAYS: {(fault_timestamp - pd.to_datetime(dates[0])).components[0]} day(s)\n")
            else:
                print("\n       --> [No warnings triggered.That's bad!]\n")
                list_missed_fault_event.append(fault)
    return list_true_warnings, list_missed_fault_event

# -----------------------------------------------------------------------------
# ------------- FUCTION: Warnings, analyse the warning side  ------------------
# -----------------------------------------------------------------------------
def analyse_warning_side(fault_events, inv_warnings, prediction_window):
    print(40*"-" + f"\nWARINING SIDE: Warning timestamps ({len(inv_warnings)}) \n" + 40*"-")
    
    if len(inv_warnings) == 0:
        print("No warnings")
        return [], []
        
    # Fault event timestamps
    fault_event_timestamps = fault_events.keys()
    
    correct_warnings = []
    wrong_warnings  = []
    for idk, warning_timestamp in enumerate(inv_warnings.index):
        warning_timestamp = pd.to_datetime(warning_timestamp)
        
        # Start timestamp of the prediction window
        end_prediction_ts = warning_timestamp + pd.Timedelta(prediction_window, unit="days")
        print(f"\n{idk + 1}/{len(inv_warnings.index)}) WARNING TIMESTAMP: ", warning_timestamp.strftime('%Y-%m-%d, %H:%M'))
        print(f"      PREDICTION WINDOW:  {end_prediction_ts.strftime('%Y-%m-%d, %H:%M')} (+{prediction_window} days)")
        
        matched_timestamps = [fault_timestamp for fault_timestamp in fault_event_timestamps
                              if warning_timestamp <=  fault_timestamp <= end_prediction_ts]
                    
        if len(matched_timestamps) == 0:
            wrong_warnings.append(warning_timestamp)
            print(f"      No fault events in this period! This is a false warning")
        else:
            correct_warnings.append(warning_timestamp)
            print(f"      Found fault timestamps in period: {len(matched_timestamps)} "\
                  "(dates:", (", ").join(sorted(set(ts.strftime('%Y-%m-%d') for ts in matched_timestamps))), ")")
            
    print(f"\nCorrect warnings ({round(len(correct_warnings)/len(inv_warnings.index) * 100, 2)} %): "\
          f"{len(correct_warnings)}/{len(inv_warnings.index)}")
    print(f"Wrong warnings ({round(len(wrong_warnings)/len(inv_warnings.index) * 100, 2)} %):  "\
          f"{len(wrong_warnings)}/{len(inv_warnings.index)}\n")  
    
    return correct_warnings, wrong_warnings

# -----------------------------------------------------------------------------
# ----------------- FUCTION: Compute the true negative  ------------------------
# -----------------------------------------------------------------------------
def compute_correct_wrong_predictions(timestamps, fault_events, inv_warnings, prediction_window, verbose = False):
    if verbose:
        print(f"\nTIMESTAMP ({len(timestamps)})", timestamps[:10], "...")
        print(f"\nFAULT EVENTS ({len(fault_events)})", [timestamp.strftime('%Y-%m-%d %H:%M:%S') 
                                                        for timestamp in list(fault_events.keys())[:10]], "...")
        print(f"\nWARNINGS ({len(inv_warnings)})", inv_warnings.tolist()[:10], "...\n")

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    
    print("\n" + 50 * "-" + f"\nComputing the correct and wrong predictions \n" + 50 * "-")
    for fault_ts, fault_desc in fault_events.items():
        fault_ts = pd.to_datetime(fault_ts)

        # Period
        start_prediction_ts = fault_ts - pd.Timedelta(prediction_window, unit="days")
        period_warnings = [warning_ts for warning_ts in inv_warnings
                          if start_prediction_ts <=  pd.to_datetime(warning_ts) <= fault_ts]
        noWarnings = True if len(period_warnings) == 0 else False

        if verbose: 
            print(f"[FAULT SIDE] TS: {fault_ts} --> START WINDOW ({prediction_window}): {start_prediction_ts}")

        if noWarnings:
            false_negative += 1
        else:
            true_positive += 1
    
    for warning_ts in inv_warnings:
        timestamp = pd.to_datetime(warning_ts)
        
        end_prediction_ts = timestamp + pd.Timedelta(prediction_window, unit="days")
        period_faults = [fault_ts for fault_ts, fault_desc in fault_events.items() 
                         if timestamp <=  fault_ts <=  end_prediction_ts]
        noFaults = True if len(period_faults) == 0 else False
        
        if verbose: 
            print(f"[WARNING SIDE] TS: {timestamp} --> END WINDOW ({prediction_window}): {end_prediction_ts}")
            
        if noFaults:
            false_positive += 1
            
    for idk, timestamp in enumerate(timestamps):
        timestamp = pd.to_datetime(timestamp) 
        
        is_fault =  True if len([fault_ts for fault_ts, fault_desc in fault_events.items() if fault_ts == timestamp]) > 0 else False
        is_warning = True if len([warning_ts for warning_ts in inv_warnings if pd.to_datetime(warning_ts) == timestamp]) > 0 else False
        
        if not is_warning:
            end_prediction_ts = timestamp + pd.Timedelta(prediction_window, unit="days")
            period_faults = [fault_ts for fault_ts, fault_desc in fault_events.items() 
                         if timestamp <=  fault_ts <=  end_prediction_ts]
            period_noFaults = True if len(period_faults) == 0 else False

            if verbose: 
                print(f"[ALL] TS: {timestamp} --> END WINDOW ({prediction_window}): {end_prediction_ts}")

            if period_noFaults:
                true_negative += 1

    print("TP: ", true_positive, "TN: ", true_negative, "\nFP: ", false_positive, "FN: ", false_negative)
    return true_positive, true_negative, false_positive, false_negative


# -----------------------------------------------------------------------------
# ----- FUCTION: Compute the function to optimize for training the SOM  ------
# -----------------------------------------------------------------------------
def compute_f1_score(true_positive, false_positive, false_negative, true_negative, verbose=True):
    print("\n" + 50 * "-" + f"\nComputing the F1 score \n" + 50 * "-")
    
    if verbose:
        print("TP: ", true_positive, " TN: ", true_negative)
        print("FP: ", false_positive, " FN: ", false_negative, "\n")
    
    # METRICS A: Recall (a.k.a., TPR: True Positive Rate = TP/TP+FN)
    try:
        recall = true_positive / (true_positive + false_negative)
    except ZeroDivisionError: # CASE: THERE ARE NO FAULTS! (TP nor FN)
        recall = np.nan
        print("NO faults (TP or FN)")
    
    # METRIC B: Precision (correct positive prevision / retrieved positive cases)
    try:
        precision = true_positive / (true_positive + false_positive)
    except ZeroDivisionError: # --> THERE ARE NO WARNINGS
        precision = np.nan
        print("ISSUE: NO warnings (TP or FP)")
    
    # F1 score
    try:
        f1_score = 2 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError: # CASE: Recall&Precision = 0 --> 0 TP (i.e., zero true warnings)
        f1_score = 0
    
    if verbose:
        print("RECALL (TP/TP+FN): ", recall)
        print("PRECISION (TP/TP+FP)", precision)
        
    print("F1 score: ", round(f1_score, 4))
    return f1_score, recall, precision

# -----------------------------------------------------------------------------
# ----- FUCTION: Compute the function to optimize for training the SOM  ------
# -----------------------------------------------------------------------------
def fallout_recall_ratio(true_positive, false_positive, false_negative, true_negative, verbose=True):
    print("\n" + 50 * "-" + f"\nComputing the FALL-OUT/RECALL ratio \n" + 50 * "-")
    
    if verbose:
        print("TP: ", true_positive, " TN: ", true_negative)
        print("FP: ", false_positive, " FN: ", false_negative, "\n")
    
    # A) Recall (a.k.a., TPR: True Positive Rate = TP/TP+FN)
    try:
        recall = true_positive / (true_positive + false_negative)
    except ZeroDivisionError: # --> THERE ARE NO FAULTS!
        recall = np.nan
        print("NO faults (i.e., TP = 0, FN = 0)")
        
    # B) Fall-out (a.k.a., FPR: False Positive Rate = FP/TN + FP)
    fall_out = false_positive / (true_negative + false_positive)
    
    # C) Compute the ratio --> lower is better
    # BEST CASE  --> (= 0) --> 0/1 --> fall-out(0) / recall (1)
    # WORST CASE --> (= inf) --> 1/0 --> fall-out(1) / recall (0)
    # TYPICAL CASE A --> (0.16) --> 0.1/0.6 --> fall-out(0.1) / recall (0.6)
    # TYPICAL CASE A --> (0.125) --> 0.1/0.8 --> fall-out(0.1) / recall (0.8)
    
    # Add a regularization term to avoid dividing by zero or losing information when a variable is zero
    regularization_term = 0.00001 
    fall_out_recall_ratio = (fall_out + regularization_term) / (recall + regularization_term)

    print("\nFALL-OUT/RECALL RATIO: ", round(fall_out_recall_ratio, 4))
    print(f"[Fall-out ({round(fall_out, 4)}) / Recall ({round(recall, 4)})]")
        
    return fall_out_recall_ratio

# -----------------------------------------------------------------------------
# ---------------------- FUCTION: Compute the metrics  ------------------------
# -----------------------------------------------------------------------------
def compute_metrics(true_positive, false_positive, false_negative, true_negative, verbose):
    print("\n" + 22*"-" + f" METRICS " +  22*"-")
    
    if verbose:
        print("TP: ", true_positive, " TN: ", true_negative)
        print("FP: ", false_positive, " FN: ", false_negative, "\n")
        
    # METRICS 1: Recall (a.k.a., TPR: True Positive Rate = TP/TP+FN)
    try:
        recall = true_positive / (true_positive + false_negative)
    except ZeroDivisionError: # --> THERE ARE NO FAULTS!
        recall = np.nan
        print("ISSUE: NO faults (TP or FN)")
    
    # METRICS 2: Miss Rate (a.k.a., FNR: False Negative Rate = FN/TP+FN)
    try:
        miss_rate = false_negative / (true_positive + false_negative)
    except ZeroDivisionError: # --> THERE ARE NO FAULTS
        miss_rate = np.nan
        print("ISSUE: NO faults (TP or FN)")
    
    # METRICS 3: Fall-out (a.k.a., FPR: False Positive Rate = FP/TN + FP)
    try:
        fall_out = false_positive / (true_negative + false_positive)
    except ZeroDivisionError:
        fall_out = np.nan
        print("ISSUE: No FP nor TN --> That's super nice!")
    
    # METRICS 6: False discovery rate 
    try: 
        false_discovery_rate = false_positive / (true_positive + false_positive)
    except ZeroDivisionError:
        false_discovery_rate = np.nan
        print("ISSUE: No TP nor FP --> No positive instances have been generated!")
    
    # METRICS 4: Precision (correct positive prevision / retrieved positive cases)
    try:
        precision = true_positive / (true_positive + false_positive)
    except ZeroDivisionError: # --> THERE ARE NO WARNINGS
        precision = np.nan
        print("ISSUE: NO warnings (TP or FP)")
    
    # METRICS 5: F1 score
    try: 
        f1_score = 2 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError: 
        f1_score = np.nan
    
    print(52 * "-")
    print("A) F1 SCORE                           :", round(f1_score * 100, 2), "%")
    print("B1) RECALL/HIT RATE          (TP/TP+FN):", round(recall * 100, 2), "%")
    print("B2) MISS RATE (1 - Hit Rate) (FN/FN+TP):", round(miss_rate * 100, 2), "%")
    print("C) PRECISION                 (TP/TP+FP):", round(precision * 100, 2), "%")
    print("D) FALL-OUT (FP Rate)        (FP/FP+TN):", round(fall_out * 100, 2), "%")
    print("E) FALSE DISCOVERY RATE      (FP/TP+FP)", round(false_discovery_rate * 100, 2), "%")
    print(52 * "-")
    
    if verbose:
        paper_recall = np.mean([93, 98, 92])
        paper_miss_rate = np.mean([7, 2, 8])
        paper_fall_out = np.mean([13, 18, 1])
        
        print("Delta (from the paper)")
        print(f"RECALL ({round(paper_recall, 0)} %): ", round((recall * 100) - paper_recall, 2), "%")
        print(f"MISS RATE ({round(paper_miss_rate, 0)} %): ", round((miss_rate *100) - paper_miss_rate, 2), "%")
        print(f"FALL-OUT ({round(paper_fall_out, 0)} %): ", round((fall_out *100) - paper_fall_out, 2), "%\n")
        
    return recall, miss_rate, fall_out, precision, f1_score