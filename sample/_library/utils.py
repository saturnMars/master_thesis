import pandas as pd
import numpy as np
from os import listdir, path
from random import randrange
from scipy.stats import zscore

                   # --- 0 ---------- 1 --------- 2 ------ 3 ------ 4 --------- 5 -------- 6 -------- 7 ---
SYSTEM_NAMES_FULL = ["Binetto 1","Binetto 2", "Cantore", "Emi", "Soleto 1", "Soleto 2","Galatina", "Verone"]
SYSTEM_NAMES = ["Binetto 1","Binetto 2", "Soleto 1", "Soleto 2","Galatina"]
SYSTEM_NAMES_UNIGE = ["FV","Q05_FV"]
SUBFOLDERS = ["Cleaned", "1-hour sampling", "1-hour averaged sampling", "Residuals", "Residuals_analytical", 'Failure events',  None]

def load_datasets(system_name, subfolder = None, verbose = True):
    print("-"*80,"\n\t\t\t\tPV SYSTEM -->",system_name.upper(), f"\n{'-'*80}")

    # Build the path of the loading folder(s)
    folder_path = path.join("data", system_name.upper())
    if system_name != SYSTEM_NAMES[0] and system_name != SYSTEM_NAMES[1]:
        folder_path = path.join(folder_path, system_name.upper())
        
    if subfolder:
        folder_path = path.join(folder_path, "Imported data", subfolder.lower().capitalize())
    else:
        folder_path = path.join(folder_path, "Imported data")

    # Potential folder of string inverter data (only for binotto 1&2)
    string_inv_folder_path = path.join(folder_path, "String Inverters")

    # Retrieve file names 
    inv_files = [file for file in listdir(folder_path) if file.endswith('.csv') and ("INV" in file.upper() or "SYSTEM" in file.upper())]
    raw_irr_file = [file for file in listdir(folder_path) if file.endswith('.csv') and ("irr" in file or "amb" in file)]

    if path.exists(string_inv_folder_path):
        string_inv_files = [file for file in listdir(string_inv_folder_path) if file.endswith('.csv') and "INV" in file.upper()]
    else:
        string_inv_files = []

    # Load the data as Pandas dataframes
    inv_data = dict()
    inv_names = []
    inv_files.sort() # Sort inverter files alphabetically 
    print("\nLoading inverter data...")
    for file_name in inv_files:
        inv_name = file_name.split("_")[1]
        inv_names.append(inv_name)
        inv_data[inv_name] = pd.read_csv(path.join(folder_path, file_name), parse_dates=[0], dtype={"Allarme":"string"})
    print(f"{system_name.upper()}: OK, component data loaded ({len(inv_data)}) --> {', '.join([name.upper() for name in inv_names])}")

    # IF AVAILABLE: Load string inverter data (in case of Binetto 1 & 2)
    if string_inv_files:
        print("\nLoading string inverter data...")
        string_inv_files.sort()
        string_inv_names = []
        string_inv_data = dict()
        for file_name in string_inv_files:
            string_inv_name = file_name.split("_")[1] + file_name.split("_")[2]
            string_inv_names.append(string_inv_name)
            string_inv_data[string_inv_name] = pd.read_csv(path.join(folder_path, "String Inverters", file_name), parse_dates=[0])
        print("{0}: OK, string inverter data loaded ({1})".format(system_name.upper(), len(string_inv_data)))
    else:
        string_inv_names = None
        string_inv_data = None
        
    # Retrieve raw data of irradiance values
    if raw_irr_file:
        print("\nLoading irradiance values...")
        raw_irr_data = [pd.read_csv(path.join(folder_path, file_name), parse_dates=[0], dtype={"Irradiance (W/mq)":"Int64"}) 
                        for file_name in raw_irr_file][0]
        print(f"{system_name.upper()}: OK, raw irradiance data ({len(raw_irr_data)} observations) have been loaded\n")
    else:
        raw_irr_data = None
        
    print("-"*80, f"\nFINISHED!: All datasets have been loaded. "\
          f"(SYS: {len(inv_data)}{' - STR_INV: ' + str(len(string_inv_data)) if string_inv_data else ''} - IRR FILE: {len(raw_irr_file)})")
    print("-"*80)
    
     # TASK: Show example of available columns 
    if verbose:
        to_load = inv_names[0]
        datetime_column_name = inv_data[to_load].columns[0]
        period = (inv_data[to_load].iloc[0, 0], 
                  inv_data[to_load].iloc[-1, 0])
        period_length =  period[1] - period[0]

        print("-"*80, f"\nEXAMPLE --> {system_name}: {to_load.upper()} "\
              f"(FROM '{period[0].strftime('%Y-%m-%d')}' TO '{period[1].strftime('%Y-%m-%d')}': {period_length.components[0]} days).\n{'-'*80}")
        inv_data[to_load].info()

        if string_inv_files:
            strInv_to_load = string_inv_names[randrange(len(string_inv_names) - 1)]
            special_strInv_to_load = string_inv_names[52 if system_name == SYSTEM_NAMES[0] else 0]
            
            print("-"*80, "\nEXAMPLE --> {0}: String {1}\n{2}".format(system_name, strInv_to_load, "-"*80))
            string_inv_data[strInv_to_load].info()
            
            print("-"*80, f"\nSPECIAL STRING INVERTER --> {system_name}: String {special_strInv_to_load}\n", '-'*80)
            string_inv_data[special_strInv_to_load].info()
            
    return folder_path, inv_data, inv_names, raw_irr_data, string_inv_data, string_inv_names

def load_amb_cond(system_name = "Galatina"):
    # Folders
    system_path = path.join("data", system_name.upper(), system_name.upper())
    folder_path = path.join(system_path, "Ambiental conditions")
    
    # List files
    file_names =  [file for file in listdir(folder_path) if file.endswith('.csv')]
    
    # Read each file concerning the year
    yearly_data = [pd.read_csv(path.join(folder_path, file_name), sep = ";", parse_dates=[0]) for file_name in file_names]
    
    # Combine all the yearly data
    amb_cond = pd.concat(yearly_data)
    
    # Drop useless columns
    flag_columns = ["flag_t", "flag_umr", "flag_prec", "flag_vv", "flag_rad", "flag_pres", "flag_dv"]
    empty_columns = ["radsolare", "Unnamed: 15"]
    amb_cond.drop(columns = flag_columns + empty_columns , inplace=True)

    # Rename columns
    amb_cond.rename(inplace=True, columns = {
            "data": "Date/Time", 
            "temperatura": "Amb. Temp (°C)", 
            "umr": "Humidity (%)",
            "precipitazione":"Rainfall (mm)",
            "vvento" : "Wind speed (m/s)",
            "dvento" : "Wind direction (°)",
            "pressione" : "Atmospheric Pressure (hPa)"
        })

    # Drop rows which do not have the temperature values (the pivotal variable)
    amb_cond.dropna(subset=["Amb. Temp (°C)"], inplace=True)

    # Sort according to the timestamp
    amb_cond.sort_values(by="Date/Time", inplace=True)
    
    # Reset the indexes
    amb_cond.reset_index(inplace=True, drop=True)

    # Reorder columns 
    new_col_order =  amb_cond.columns.tolist()
    new_col_order.remove(amb_cond.columns[6])
    new_col_order.insert(3, amb_cond.columns[6])
    amb_cond = amb_cond.reindex(columns=new_col_order)

    return amb_cond

# -----------------------------------------------------
# FUNCTION: Find the outliers by adopting the z-score
# -----------------------------------------------------
def find_outliers(df, threshold = 3, verbose = False): 
    numerical_df = df.select_dtypes(include = np.number).dropna(how="all")
    
    # THE MEASURAMENT: Z-score
    # It describes a value's relationship to the mean of a group of values 
    # It's the number of standard deviations by which the observed value is above/below the mean value of what is being measured
    z_score = np.abs(zscore(numerical_df, nan_policy = "omit")) 
    z_score.columns = ["[z-score] "+ col for col in z_score.columns]
    
    # Filter the observations that are above the threshold (from literature: 3 is a typical cut-off point to detect the outliers)
    if len(z_score.columns) == 4:         
        # FIll possible nan values 
        z_score.iloc[:, 3].fillna(z_score.iloc[:, 0], inplace=True)

        cond_outlier = (z_score.iloc[:, 0] > threshold) & (z_score.iloc[:, 1] > threshold) & (z_score.iloc[:, 2] > threshold) & (z_score.iloc[:, 3] > threshold)
    elif len(z_score.columns) == 3:
        cond_outlier = (z_score.iloc[:, 0] > threshold) & (z_score.iloc[:, 1] > threshold) & (z_score.iloc[:, 2] > threshold)
    elif len(z_score.columns) == 2:
         cond_outlier = (z_score.iloc[:, 0] > threshold) & (z_score.iloc[:, 1] > threshold)
    else:
        cond_outlier = z_score.iloc[:, 0] > threshold 

    # Find the outliers according to the threshold of the z-scores
    outliers_idk = z_score[cond_outlier].index
    outliers = df.loc[outliers_idk, :]
    
    if verbose & len(outliers) != 0:
        print("-"*40)
        print(f"Z-score values (threshold: {threshold})")
        print("MIN (z-score):", round(np.min(min(z_score[cond_outlier].values.tolist())), 2), \
              "\nMAX (z-score):",round(np.max(max(z_score[cond_outlier].values.tolist())), 2))
        print("-"*20, f"\nObservations flagged as outliers: {len(outliers)}", "-"*20)
        print("MIN (value):", round(np.min(min(outliers.select_dtypes(include = np.number).values.tolist())), 2), \
              "\nMAX (value):", round(np.max(max(outliers.select_dtypes(include = np.number).values.tolist())), 2))
        display(pd.concat([outliers, z_score[cond_outlier]], axis=1))
        
    return outliers

# ------------------------------------------------------------------------------------------
# FUNCTION: Compute a weighted average value according to the observation's neighbours (KNN)
# ------------------------------------------------------------------------------------------
def weighted_knn(df, idk_position, list_outliers, column,  K = 6, verbose=True, force_integer=False, Fill_nan = True):
    # Define the interval before and after the position
    k_before = K//2 
    k_after = K//2
    
    # Define the range of neighbours 
    idk_range = np.arange(idk_position - k_before, idk_position + k_after + 1)
    
    # Find the indexes of other outliers (for excluding them in the weighted average)
    if idk_position in list_outliers:
        list_outliers.remove(idk_position)
    
    # Detect potential outliers in its neighbours
    problematic_idk_neighbours = [idk_outlier for idk_outlier in list_outliers if idk_outlier in idk_range]
        
    if len(problematic_idk_neighbours) != 0:
        if verbose:
            print(f"\nOPS: Problematic IDK neighbours discovered {len(problematic_idk_neighbours)} "\
                  f"({round(((len(problematic_idk_neighbours)/len(idk_range))*100), 2)} %) --> {problematic_idk_neighbours}")
        
        # Try to increase the number of neighbours 
        MAX_K = 30
        if len(problematic_idk_neighbours) >= K//2:
            new_k = K*2 
            if new_k <= MAX_K:
                if verbose: 
                    print(f"Trying with more neighbours ({new_k})...")
                weighted_knn(df, idk_position, list_outliers, column,  K = new_k, verbose=False)
            else:
                if verbose:
                    print("Reach maximum of K neighbours")
        
        # Remove problematic neighbours
        idk_range = idk_range[~ np.isin(idk_range, problematic_idk_neighbours)]
        
        if verbose:
            print("\nEnd process of increasing number of neighbours (i.e., number of problematic neigbours is acceptable (fewer than K/2)")
            print(f"The problematic neighbours ({len(problematic_idk_neighbours)}) have been removed from the neighbour list")
    
    # Find the neigbourhood 
    idk_outlier_pos = np.argwhere(idk_range == idk_position)[0][0]
    neighbours_idk = np.delete(idk_range, idk_outlier_pos)
    
    # Define the neighbourhood
    neighbours = df.loc[neighbours_idk, :][column] # iloc
    outlier_value = df.loc[idk_position, :][column]
    
    # Tackle nan values
    if Fill_nan:
        neighbours = neighbours.fillna(value = 0)
    else:
        neighbours = neighbours.dropna()
    
    if len(neighbours) == 0:
        if Fill_nan:
            original_value = df.loc[idk_position, column]
            print(f"\nISSUE ({column}): No neighbours found! using the original value ({original_value}). [STRAT: Fill_nan]")
            return original_value
        else:
            print(f"\nISSUE ({column}): No neighbours found! Returning a NaN value [STRAT: Drop_nan]")
            return np.nan
    
    if verbose:
        print(f"\nVARIABLE: {column}")
        print("STRAT for NaN values: ", "Filling with zero values" if Fill_nan else "Dropping them")
        print(f"NEIGHBOURES ({len(neighbours)} out of {K}):"\
              f"{neighbours.loc[(neighbours.index < idk_position)].tolist()} \--/ "\
              f"{neighbours.loc[(neighbours.index > idk_position)].tolist()}")
    
    # Define the weights
    dist_idk = [np.abs(idk - idk_position) for idk in neighbours.index] #
    penalize = lambda dist: min(dist_idk)/dist if dist != min(dist_idk) else 1 #min(dist_idk)/(1 + (dist/10))
    weights = [round(penalize(dist), 2) for dist in dist_idk]
    
    if verbose:
        print("WEIGHTS:", weights)
       
    # Compute the weighted average 
    neighbours = np.array(neighbours)
    weighted_average_value = np.average(neighbours, weights = weights)
    
    # Round and cast the value to an integer value
    if force_integer:
        candidate_value = int(round(weighted_average_value, 0))
    else:
        candidate_value = round(weighted_average_value, 3)
    
    if verbose:
        print(f"COMPUTED VALUE: {candidate_value}")
        print("-" * 80)
    return candidate_value   