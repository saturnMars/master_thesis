import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from os import path, listdir

# -----------------------------------------------------------------------------------
# ------------------------ FUCTION: Linear regression -------------------------------
# -----------------------------------------------------------------------------------
def make_linear_regression(X,y, weighted=False, constant = False):
    if constant:
        X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
        
    if weighted:
        weights = 1/X["Irradiance (W/mq)^2"]
        print("X[0]:, ", X["Irradiance (W/mq)^2"])
        print("W: ", weights)
        model = sm.WLS(y, X, weights)
    else: 
        model = sm.OLS(y, X)
    
    fitted_model = model.fit()
    return fitted_model

def check_residual(cols, reg_model):    
    for idk, col in enumerate(cols):
        print("\t\t" + 40 * "-" + f" VAR {idk + 1}: {col} " + 40 * "-")
        fig = sm.graphics.plot_regress_exog(reg_model, col)
        fig.set_size_inches(20, 12)
        fig.tight_layout(pad=1.0)
        plt.show()
        plt.close()

def linear_regression(df, y_col, features, irr_threshold = True, verbose=True):
    
    # Select only daylight observations
    cond_latenight = ((df[y_col].index.hour >= 20) & (df[y_col].index.hour <= 23)) 
    cond_earlymorning =  ((df[y_col].index.hour >= 0) & (df[y_col].index.hour <= 7))
    cond = cond_latenight | cond_earlymorning
    night_indexes = df[cond].index
    day_indexes = list(set(df.index) - set(night_indexes))
    df_day = df.loc[day_indexes,:].sort_index()
    
    # Fill obs with a NaN value
    df_day = df_day.fillna(method="ffill") #OR: dropna(inplace=True)
    
    # Select a minimum irradiance value --> tackle outliers
    if irr_threshold:
        minimum_irradiance_value = 50
        df_day = df_day[df_day["Irradiance (W/mq)"] >= minimum_irradiance_value]
    
    # Select the featues (Y: Pac, X: Irr, Amb Temp, Humidity)
    Y_acPower = df_day[y_col]
    
    # Select the features that will be used as predictors
    X_values = df_day[features]
    
    # See graphically a simplified representation
    if verbose:
        fig, axes = plt.subplots(nrows = len(features), figsize=(20, 6 * len(features)))
        sns.set_theme(style="whitegrid")
        if len(features) == 1:
            axes = np.array([axes])
            
        fig.suptitle("Data pre-processing",fontsize=30, color='dimgray', y=1.01) 
        colors = ["red", "orange", "blue"]
        for idk, col in enumerate(features):
            sns.scatterplot(data = df_day, x = col, y="Pac R (kW)", color= colors[idk], s = 10, alpha=0.8,ax = axes[idk])
            axes[idk].set_title(f"{y_col} = f({col.split(' ')[0]})", fontsize="xx-large")
            axes[idk].tick_params(axis='y', which='major', grid_linestyle = "-.", grid_alpha=0.8)
        fig.tight_layout(pad=1.0)
        plt.show()
    
    # Linear regression: Pac = f(solarIrr, ambTemp, humidity)
    model = make_linear_regression(y = Y_acPower, X = X_values)
    return model

def compute_outliers_above_threshold(linear_model, y_var, new_df_data, x_features, perc_error_threhsold, verbose = True):
    
    # TACKLE the annoying warning 'SettingWithCopyWarning'
    new_df_data._is_copy = None 

    if len(new_df_data) == 0:
        return []
 
    # Compute the predictions
    new_df_data["Predicted Pac"] = linear_model.predict(new_df_data[x_features])

    # a) Transfrom the negative predicted values into zero values
    neg_values_cond = new_df_data["Predicted Pac"] < 0
    new_df_data.loc[neg_values_cond, "Predicted Pac"] = 0
    
    # b) Transform the float prediction into an integer Power values (as the original one)
    new_df_data["Predicted Pac"] = new_df_data["Predicted Pac"].round(decimals = 0)
    new_df_data["Predicted Pac"] = new_df_data["Predicted Pac"].astype("int")
    
    # Compute the absolute and percentage errors
    new_df_data["obs error"] = np.abs(new_df_data[y_var] - new_df_data["Predicted Pac"])
    new_df_data["Percentage error"] = new_df_data["obs error"]/new_df_data["Predicted Pac"]
    
    # Visualize 
    cond = new_df_data["Percentage error"] > perc_error_threhsold
    potential_outliers = new_df_data[cond]

    print("-" * 70 + f"\n{len(potential_outliers)} observations ({round(len(potential_outliers)/len(new_df_data)*100, 2)} %) "\
          f"above the percentage error threshold (= {perc_error_threhsold})\n", "-" * 70)
    
    if verbose:  
        # 0: Assign a lable for plotting them
        new_df_data["AsOutlier"] = "No"
        new_df_data.loc[potential_outliers.index, "AsOutlier"] = "Yes"
        
        # 1: Plot the graphs 
        n_rows = len(x_features)
        fig, axes = plt.subplots(nrows = n_rows, figsize = (15, 4 * n_rows))
        sns.set_theme(style="whitegrid")
        if len(x_features) == 1:
            axes = np.array([axes])

        fig.suptitle("Detecting outliers using a simple linear regression", fontsize=30, color='dimgray', y=1.01) 
        colors = ["green","red", "orange", "yellow", "blue"]
        
        for idk, col in enumerate(x_features):
            # Actual points 
            sns.scatterplot(data = new_df_data, x = col, y=y_var, color = colors[idk],hue = "AsOutlier", markers = "AsOutlier",
                            s = 8, alpha=0.9, ax = axes[idk])
            
            # Predicted points
            sns.scatterplot(x = new_df_data[col], y = new_df_data["Predicted Pac"], 
                            color="grey",ax = axes[idk], label=f"Predicted {col}", s = 6, alpha=0.8)
            
            # Some graphical parameters
            axes[idk].set_title(f"Partial regression plot: Pac = f({col})", fontsize="xx-large")
            axes[idk].tick_params(axis='y', which='major', grid_linestyle = "-.", grid_alpha=0.8)
        
        # Visualize the graph
        fig.tight_layout(pad=1.0)
        plt.show()
        
        # Remove the artefact (created for the graph)
        new_df_data.drop(columns = ["AsOutlier"])
        
    # Remove the artefacts
    new_df_data = new_df_data.drop(columns = ["Predicted Pac", "obs error", "Percentage error"])
    
    return potential_outliers.index.tolist()


# --------------------------------------------------------------------------------------------
# --(Pre-processing: Step 1) Find outliers using a simple linear regression on the AC power --
# --------------------------------------------------------------------------------------------
def find_pac_outliers_lin_reg(df, num_features = 3, perc_error_threhsold = 2, verbose=True, y_power = True):
    
    # ---------- Y ------------- X1 ----------------- X2 ---------------X3------
    if y_power:
        cols = ["Pac R (kW)", "Irradiance (W/mq)",  "Amb. Temp (°C)", "Humidity (%)"]
        print("Y: AC Power")
    else:
        cols = ["Cc 1 (A)", "Irradiance (W/mq)",  "Amb. Temp (°C)", "Humidity (%)"]
        print("Y: Current (A)")
    y_col = cols[0]
    x_cols = cols[1:num_features + 1]
    
    df["Irradiance (W/mq)^3"] = df["Irradiance (W/mq)"] ** 3
    x_cols.append("Irradiance (W/mq)^3")

    df["Irradiance (W/mq)^2"] = df["Irradiance (W/mq)"] ** 2
    x_cols.append("Irradiance (W/mq)^2")

    #df["interaction_irr_temp"] = df["Irradiance (W/mq)"] * df["Amb. Temp (°C)"]
    #x_cols.append("interaction_irr_temp")

    x_cols = sorted(x_cols, reverse=True)
    
    # Perform the linear regression
    lin_model = linear_regression(df, y_col, features = x_cols, irr_threshold=True, verbose=False)
    print("FORMULA: Y = X1^3 + X1^2 + X1 + X2 \n  --> Pac (kW) = Irr^3 + Irr^2 + Irr + Amb. Temp") #+ (Irr * Amb.Temp)
    display(lin_model.summary())
    
    # Check residual 
    check_residual(x_cols, lin_model)
    
    # Predict the AC power with the selected features 
    new_data = df[[y_col] + x_cols]
    potential_outliers = compute_outliers_above_threshold(lin_model, y_col, new_data, x_cols, perc_error_threhsold, verbose = True)
    
    # Drop artefacts
    df.drop(columns = ["Irradiance (W/mq)^3", "Irradiance (W/mq)^2"], inplace = True) #, "interaction_irr_temp"
  
    return potential_outliers, lin_model

# --------------------------------------------------------------------------------------------
# ----------------------- (Pre-processing: Step 2) Detrending --------------------------------
# --------------------------------------------------------------------------------------------
def detrending(df, feature, daily_obs = 16, num_weeks = 2, verbose = True):

    # Visualize the possible trend
    #if verbose:
        #days = 7 * (num_weeks//2)
        #analysis = seasonal_decompose(df[feature],  model='additive', extrapolate_trend='freq', period = days * daily_obs, 
                                       #two_sided = False)
        #plt.rcParams.update({'figure.figsize': (10,10)})
        #analysis.plot(observed = False, seasonal=True, resid=False).suptitle(f'Additive Decomposition [{feature}]', fontsize=18)
        #plt.show()
    
    # 1) Compute the moving average 
    moving_average_col = feature + f"_{num_weeks}w"
    df[moving_average_col] = df[feature].rolling(window = daily_obs * 7 * num_weeks, min_periods = None).mean()
        
    # 2) Compute the detrended values
    detrend_col = feature + f"detrend_{num_weeks}w"
    detrended_values = df[feature] - df[moving_average_col]
    df[detrend_col] = detrended_values
    
    # 3) Compute the moving average over the detranded feature
    detrend_mean_col = feature + f"detrend_{num_weeks}w_mean"
    df[detrend_mean_col] = df[detrend_col].rolling(window = daily_obs * 7 * num_weeks, min_periods = 1).mean()
    
    # Visualize the moving average
    cols_to_visualize = [feature, moving_average_col, detrend_mean_col]
    if verbose: 
        colors = ["grey", "blue", "red"]
        df[cols_to_visualize].plot(color = colors, linewidth=1, figsize=(6,3))

        # Some graphical parameters
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.title(feature, fontsize=20)
        plt.xlabel('Time', fontsize=16)
        plt.ylabel(feature, fontsize=16)
        plt.legend(labels = [feature, f'{feature}: {num_weeks}-week mean',f'{feature}: {num_weeks}-week detrend mean'], fontsize=11)
        plt.show()
    
    # Drop artefacts
    df.drop(columns = cols_to_visualize + [detrend_col], inplace=True)
    
    return detrended_values

def save_parameters(saving_parameters_path, inv_names, inv_linear_model, perc_error_threshold, data_std_scalers, daily_obs_detrending,
                    num_weeks_detrending, file_name = "Parameters.txt"):

    file_path = path.join(saving_parameters_path, file_name)

    with open(file_path, 'w') as txt_file:
        txt_file.write(f"[Detrending] daily_obs_detrending: {daily_obs_detrending}\n\n")
        print(f"The number of 'daily observations' have been saved.")

        txt_file.write(f"[Detrending] num_weeks_detrending: {num_weeks_detrending}\n\n")
        print(f"The'num_weeks_detrending' have been saved.")

        txt_file.write(f"[Regression model] Percentage error threshold: {perc_error_threshold}\n")
        print(f"The 'percentage error threshold' have been saved.")
            
        for inv_name in inv_names:

            # Regression Regression
            if inv_name in inv_linear_model.keys():
                regression_model = inv_linear_model[inv_name]
                regression_model.save(path.join(saving_parameters_path, inv_name + "_lin_reg_model.p" ))
                print(f"\n[{inv_name}] The regression model has been saved.")

                # regression_params = regression_model.params.to_dict()
                # txt_file.write(f"\n[Regression model] [{inv_name}]\n")
                # for var_name, coeff in regression_params.items():
                # txt_file.write(f"{var_name}: {coeff}\n")
            else:
                print("\n[{inv_name}] WARNING: The regression model has not been found!")

            # Standard scaler
            scaler = data_std_scalers[inv_name]
            txt_file.write(f"\n[Standard Scaler] [{inv_name}]\n")
            txt_file.write("FEATURES:\n" + ','.join(scaler.feature_names_in_)+ "\n")
            txt_file.write("MEAN:\n" + ','.join([str(value) for value in scaler.mean_]) + "\n")
            txt_file.write("VARIANCE:\n" + ','.join([str(value) for value in scaler.var_]) + "\n")
            print(f"[{inv_name}] The paramaters (i.e., mean and variance) of the standard scaler have been saved. ")  

def read_preProcessing_parameters(params_folder, file_name = "Parameters.txt"):
    params_file_path = path.join(params_folder, file_name)
    
    inv_scaler_parameters = dict()
    with open(params_file_path, 'r') as params_file:
        lines = params_file.readlines()

        # Regression percent param
        regression_line = [line.strip().split(':') for line in lines if 'regression' in line.lower()]
        percent_threshold = int(regression_line[0][1].strip())
        
        # Detrending parameters
        detrending_lines = [line.strip().split(':') for line in lines if 'detrending' in line.lower()]
        detrending_params = [{'parameter': param_name, 'value': int(param_value.strip())} 
                             for param_name, param_value in detrending_lines]
        daily_obs_detrending = detrending_params[0]['value']
        num_weeks_detrending = detrending_params[1]['value']
        
        for idk, line in enumerate(lines): 
            if 'standard scaler' in line.lower():
                inv_name = line.strip().split("]")[1].strip()[1:]
                feature_names = lines[idk + 2].strip().split(",")
                mean_values = [float(value) for value in lines[idk + 4].strip().split(",")]
                variance_values = [float(value) for value in lines[idk + 6].strip().split(",")]

                inv_scaler_parameters[inv_name] = {'features': feature_names, 'mean_values': mean_values, 
                                                   'variance_values': variance_values}

    return percent_threshold, daily_obs_detrending, num_weeks_detrending, inv_scaler_parameters

def read_regression_models(params_folder):
    inv_regression_models = dict()
    models_files = sorted([file for file in listdir(params_folder) if file.endswith('.p')])
    
    for picke_file in models_files:
        file_path = path.join(params_folder, picke_file)
        inv_name = picke_file.split("_")[0]
        loaded_model = RegressionResults.load(file_path)
        
        inv_regression_models[inv_name] = loaded_model
    return inv_regression_models
