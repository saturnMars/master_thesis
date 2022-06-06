from math import ceil
from matplotlib.colors import Normalize
from string import ascii_uppercase
from os import path, makedirs
from collections import Counter, defaultdict
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

PRE_STEPS_CONFIGS = [
    "1hour",
    "1hour_avgThreePhases",
    "1hour_detrended",
    "1hour_detrended_avgThreePhases",
    "1hour_Reg",
    "1hour_Reg_detrended",
    "1hour_Reg_avgThreePhases",
    "1hour_Reg_detrended_avgThreePhases",
    "1hour_fullReg",
    "1hour_fullReg_detrended",
    "1hour_fullReg_avgThreePhases",
    "1hour_fullReg_detrended_avgThreePhases",
    # -------------------------
    "1hour_averaged",
    "1hour_averaged_avgThreePhases",
    "1hour_averaged_detrended",
    "1hour_averaged_detrended_avgThreePhases",
    "1hour_averaged_Reg",
    "1hour_averaged_Reg_detrended",
    "1hour_averaged_Reg_avgThreePhases",
    "1hour_averaged_Reg_detrended_avgThreePhases",
    "1hour_averaged_fullReg",
    "1hour_averaged_fullReg_detrended",
    "1hour_averaged_fullReg_avgThreePhases",
    "1hour_averaged_fullReg_detrended_avgThreePhases"
]

# --------------------------------------------------------------------------------
# ---------- FUNCTION: read the SOM perfomance log file --------------------------
# --------------------------------------------------------------------------------
def read_som_perfomance_file(file_path, variable_to_sort_out = "f1"):
    
    # Read the file
    configs_perfomance = []
    som_performance = dict()
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
      
        for numRow in range(0, len(lines), 8):
            
            # ROW 0: Configiguration name
            config = lines[numRow].rstrip()
                                    
            # ROW 1: Quantization error (AVAILABLE ONLY FOR THE FULL RAW VERSION)                     
            quantization_error = float(lines[numRow + 1].split(":")[1].strip())
            
            # ROW 2: F1 score                
            f1_score = float(lines[numRow  + 2].split(":")[1].split('(')[0].strip())
            fold_f1_scores = [float(item.strip(')\n ')) for item in lines[numRow  + 2].split(":")[1].split('(')[1].split(',')]
            
            # ROW 3: Recall    
            recall = float(lines[numRow  + 3].split(":")[1].strip())
                                    
            # ROW 3: Precision                
            precision = float(lines[numRow  + 4].split(":")[1].strip())
           
            # ROW 5&6: Computational times 
            som_time = lines[numRow + 5].strip()
            kpi_time = lines[numRow + 6].strip()
            
            # Save the information
            som_performance[config] = (quantization_error, f1_score, recall, precision, fold_f1_scores, (som_time, kpi_time))

            # Save the config as a Pandas Series
            parts = config.split(":")[1].split("_")
            config_parts = (int(parts[0].replace("grid", "")),
                            parts[1].replace("epoch", ""),
                            float(parts[2].replace("lr", "")),
                            float(parts[3].replace("sigma", "")), 
                            parts[4].replace("Func", ""))
            configs_perfomance.append(pd.Series(
                    data = [*config_parts, quantization_error, f1_score, fold_f1_scores, recall, precision], 
                    index = ["Dim grid", "Num epoch", "Learning rate", "Sigma", "Function",  "Quantization Error", 
                             "F1 Score", "Fold F1 scores", "Recall", "Precision"]))
        file.close()               
    som_performance_df = pd.DataFrame(configs_perfomance)

    # Sort out the configurations
    if variable_to_sort_out.lower() == "f1":
        idk_var = 1
    elif variable_to_sort_out.lower() == "recall":
        idk_var = 3
    elif variable_to_sort_out.lower() == "precision":
        idk_var = 4
    best_configurations = sorted(som_performance.items(), 
                                 key = lambda config: config[1][idk_var] if not np.isnan(config[1][idk_var]) else 0, 
                                 reverse = True)
    best_score_performance = best_configurations[0][1][:4]
    
    return som_performance_df, best_configurations, best_score_performance


# --------------------------------------------------------------------------------
# --------------------- FUNCTION: Generate single graph --------------------------
# --------------------------------------------------------------------------------
def plot_grid_findings(filtered_df, metrics_name, graph, varPlot, varValue, limits, visualize_yLabel):
    
     # 1) Plot the lines (for each sigma value)
    sigma_value = filtered_df["Sigma"]#.astype("int")
    hue_normalization = Normalize(vmin = sigma_value.min(), vmax = sigma_value.max())
    color_map = sns.color_palette("viridis", as_cmap = True).reversed()
    
    sns.lineplot(x = filtered_df["Num epoch"], y = filtered_df[metrics_name], hue = sigma_value,
                 palette = color_map, hue_norm = hue_normalization, ax = graph, zorder = 2) # crest, YlOrBr
    
    # A confidence interval for a mean gives us a range of plausible values for the population mean.  
    sns.lineplot(x = filtered_df["Num epoch"], y = filtered_df[metrics_name], estimator = np.mean, ci = 95,
                 color = "darkgray", linestyle = "--", linewidth = 4, ax = graph, alpha = 0.2, zorder = 1) 
        
    # 1.2) Set major visual parameters (e.g., title, legend and limits)
    # 1.2.1) Title
    plot_title = []
    for idk, var in enumerate(varPlot):
        if var == "Dim grid":
            plot_title.append("GRID DIMENSION: " + r"$\bf{" + f"{varValue[idk]}x{varValue[idk]}" + "}$")
        else:
            plot_title.append(var + r": $\bf{" + str(varValue[idk]) + "}$")
    
    graph.set_title(' || '.join(plot_title), fontsize = 55, pad=20)
    
    # 1.2.2) Legend
    legend_labels = [int(value) if value.is_integer() else value for value in sigma_value.unique().tolist()]
    legend = graph.legend(legend_labels, fontsize = 25, title = "Sigma", title_fontsize = 35, shadow = True, borderpad = 0.5)
    for line in legend.get_lines():
        line.set_linewidth(10)
    
    # 1.2.3) Limits
    graph.set_xlim(limits[0])
    graph.set_ylim(limits[1])
    
    # 1.3) Set minor graphical parameters
    graph.set_xlabel("Epochs", fontsize = 45, color= "dimgrey")
    if visualize_yLabel:
        graph.set_ylabel(metrics_name, fontsize = 50, color= "dimgrey", labelpad = 20)
    else:
        graph.set(ylabel=None)
   
    graph.tick_params(axis='both', which='major', labelsize = 30)
    graph.tick_params(axis = "y", grid_linestyle = "-.", grid_linewidth = 1, pad = 10)
    graph.tick_params(axis = "x", grid_linewidth = 2, grid_alpha = 0.6, labelsize = 35, pad = 20)
    graph.grid()

# --------------------------------------------------------------------------------
# --------------------- FUNCTION: Generate the graphical panel ------------------
# --------------------------------------------------------------------------------
def plot_analysis(som_perfomance, system_name, inv_name, saving_folder_path, dataset_type, var_panels, var_graphColums, var_graphRows, verbose = True):
    metrics_names = ["Quantization Error", "F1 Score", "Recall", "Precision"]
    
    # A graphical panel for each grid dimension
    panels_values = som_perfomance[var_panels].unique()
    remaining_col = som_perfomance.drop(
        columns = [var_panels, var_graphColums, var_graphRows, *metrics_names] + ["Sigma", "Num epoch"]).columns.tolist()
    
    if len(remaining_col) > 0:
        remaining_value = som_perfomance[remaining_col].squeeze().unique()
    
    for panel_var_value in panels_values:
        df = som_perfomance[som_perfomance[var_panels] == panel_var_value].drop(columns = [var_panels])
        
        # A graphical panel for each metric according to the grid dimension
        for idk_metrics, metrics_name in enumerate(metrics_names):
            
            if var_graphColums == var_graphRows:
                colRow_values = som_perfomance[var_graphColums].unique()
                n_cols = 2 if len(colRow_values) <= 10 else 3
                n_rows = ceil(len(colRow_values)/n_cols)
                extra_subplots = True if ((n_rows * n_cols) - len(colRow_values)) > 0 else False
            else:
                col_values = som_perfomance[var_graphColums].unique()
                row_values = som_perfomance[var_graphRows].unique()
                n_cols = len(col_values)
                n_rows = len(row_values)
                extra_subplots = False
            
            # Create the figure (i.e., visual panel)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(25 * n_cols, 15 * n_rows), 
                                     tight_layout= {"pad": 6}, facecolor = "white")

            # Set the title for the columns/rows
            #if n_rows > 1:
                #fig.supxlabel(var_graphColums, fontsize = 70, y = 0.975)
            #if n_cols > 1:
                #fig.supylabel(var_graphRows, fontsize = 60)
                
            if axes.ndim == 1:
                axes = np.array([axes])
                
            # Make invisible the last graph in extra subplots
            if extra_subplots:
                for row in range(1, extra_subplots + 1):
                    axes[-row, -1].set_visible(False)
                    
            # Set the figure title 
            panel_title = r"$\bf{" + system_name.replace(' ','\\ ') + f" [{inv_name}]: SOM PERFOMANCE".replace(' ','\\ ') + "}$: " \
            + f" ({ascii_uppercase[idk_metrics]}) {metrics_name}\n"
            if var_panels == "Dim grid":
                panel_title += f"(GRID: {panel_var_value}x{panel_var_value}, "
            else:
                panel_title += f"({var_panels.upper()}: {panel_var_value.capitalize()})"
                
            fig.suptitle(panel_title, fontsize=130, color='dimgray', y = 1.01)
            
            idk_row = 0
            idk_column = 0
            
            # CASE 1: 
            if var_graphColums == var_graphRows:
                for col_value in colRow_values:

                    # Retrieve the data
                    cond = df[var_graphColums] == col_value
                    filtered_df = df[cond]

                    # Use a masked name for the i-th graph
                    graph = axes[idk_row, idk_column]

                    # Plot the graph
                    x_limit = (filtered_df["Num epoch"].iloc[0], filtered_df["Num epoch"].iloc[-1])
                    y_limit = (som_perfomance[metrics_name].min() * 0.95, som_perfomance[metrics_name].max() * 1.05)
                    
                    firstCol = True if idk_column == 0 else False
                    var_title = [var_graphColums]
                    value_title = [col_value]

                    if len(remaining_col) > 0:
                        if remaining_col[0].upper() == "LEARNING RATE":
                            var_title.append("LR")
                        else: 
                            var_title.append(remaining_col[0].upper())
                        
                        value_title.append(filtered_df[remaining_col[0]].unique()[0])
                    plot_grid_findings(filtered_df, metrics_name, graph, var_title, value_title, [x_limit, y_limit],
                                      visualize_yLabel = firstCol)

                    # Increment the row/column
                    if idk_column + 1 < axes.shape[1]:
                        idk_column += 1
                    else:
                        idk_column = 0
                        idk_row += 1
       
            else:
                for col_value in col_values:
                    for row_value in row_values:
                        
                        # Retrieve the data
                        col_cond = df[var_graphColums] == col_value
                        row_cond = df[var_graphRows] == row_value
                        filtered_df = df[col_cond & row_cond]
                        
                         # Use a masked name for the i-th graph
                        graph = axes[idk_row, idk_column]

                        # Plot the graph
                        x_limit = (filtered_df["Num epoch"].iloc[0], filtered_df["Num epoch"].iloc[-1])
                        y_limit = (som_perfomance[metrics_name].min() * 0.95, som_perfomance[metrics_name].max() * 1.05)

                        firstCol = True if idk_column == 0 else False
                        var_title = [var_graphColums, var_graphRows]
                        value_title = [col_value, row_value]

                        plot_grid_findings(filtered_df, metrics_name, graph, var_title, value_title, [x_limit, y_limit],
                                          visualize_yLabel = firstCol)

                        # Increment the row/column
                        if idk_column + 1 < axes.shape[1]:
                            idk_column += 1
                        else:
                            idk_column = 0
                            idk_row += 1
            
            # Create the subfolder for each metrics (if it does not exits)            
            saving_path = path.join(saving_folder_path, metrics_name)
            if not path.exists(saving_path):
                makedirs(saving_path)

            # Save the figure as a PNG image
            file_name = f"{inv_name}_{dataset_type}_SOM_behaviour_" + str(panel_var_value) + var_panels.upper()[:4] + "_ " \
            + metrics_name.replace(' ', '_') + ".png"
            fig.savefig(path.join(saving_path, file_name), bbox_inches='tight', pad_inches = 2)
            
            if verbose: 
                plt.show()
            else:
                print(f"{ascii_uppercase[idk_metrics]}) The graph for the '{metrics_name.upper()}' have been created and saved.")
            plt.close()
            
# --------------------------------------------------------------------------------
# ------------ FUNCTION: Visualize metrics for each som configuration  ------------
# --------------------------------------------------------------------------------
def visualize_metrics(metrics_configs, fault_profile, fault_profiles_available, prediction_window, save_to_file, verbose = False):
    
    inv_ranked_configs = []

    # A.0) VISUALIZE FAULTS
    if verbose:
        inv_df = metrics_configs[list(metrics_configs.keys())[0]]
        inv_df["Fault Profile"] = pd.array(inv_df["Fault Profile"])
        inv_faults = inv_df[inv_df["Fault Profile"] == str(fault_profile)]["Faults"].unique()
        inv_faults = inv_faults[0].split(",")

        # A.1) Visualize the faults
        if inv_faults[0] != "[]":

            print(40 * "-" + f"\nALL {len(inv_faults)} FAILURE EVENTS (types: '{', '.join(fault_profile)}')\n" + 40 * "-") 
            for idk, rawFaultString in enumerate(inv_faults):
                fault = rawFaultString.split("\\n")
                print(f"{idk + 1}) {fault[0].strip()} \n{fault[1]}")
        else:
            print("[No events/alarms found]\nSorry, here there are nothing for you. :/ \n\n"\
                  "The metrics will be presented anyway. \nOne should expect only false warnings (i.e., fall-out)\n")

        print(70 * "-" + "\n")
        print(85 * "-" + "\n\t\tCOMPARISONS of the combinations of pre-processing steps" + "\n" + 85 * "-")

    ranked_config = dict()
    inv_fault_anticipation = dict()
    best_perfomance = dict()

    for idk_warning, warning_level in enumerate([4, 3, 2, 1]):
        perfomance_list = []

        # Visualize the dataframe with all the perfomance
        views = ["A", "B", "C", "D"]
        print("\n" + "-" * 40 + f" {views[idk_warning]}) COMPARISION OF THE PRE-PROCESSING STEPS " + "-" * 32)
        print("-" * 43 + f" WARNINGS with (levels >= {warning_level}) " + "-" * 44)
        for idk, config in enumerate(metrics_configs.keys()):
            df = metrics_configs[config]
            df["Fault Profile"] = pd.array(df["Fault Profile"])

            # Filter the dataframe
            cond1 = df["Prediction Window (days)"] == prediction_window
            cond2 = df["Fault Profile"] == str(fault_profile)
            filtered_df = df[cond1 & cond2]

            # Select only one warning level
            cond3 = filtered_df["Warning levels (>=)"] == warning_level
            further_filtered_df = filtered_df[cond3]

            # Visualize warnings of the fault 
            raw_fault_warnings = further_filtered_df["Fault warnings"].tolist()[0]
            fault_warnings = raw_fault_warnings.split("('")[1:]
            
            if verbose:
                print("\n" + "-" * 40)
                print(f"CONFIG {idk +1}/{len(metrics_configs.keys())}:", config)
                print(f"FAILURE EVENTS DETECTED: {len(fault_warnings)}/{len(inv_faults)} "\
                      f"({round((len(fault_warnings)/len(inv_faults))*100, 1)} %)")
                print("-" * 40)

            for idk_warning, rawStringfault in enumerate(fault_warnings):
                rawFault, rawWarnings = rawStringfault.split(", [")

                # A) EXTRACT PARTS FROM THE "RAW FAULT STRING"
                fault_message, fault_start, fault_end = [item.strip("'") for item in rawFault.split("_")]
               
                # B) WARNINGS
                rawWarnings = rawWarnings.split(",")
                if verbose and idk == 0:
                    print("\t" + "-" * 40)
                    print(f"\n\tFAULT/ALARM ({idk_warning + 1}/{len(fault_warnings)}): {fault_message} "\
                          f"\n\t--> FROM {fault_start} TO {fault_end}")
                    print(f"\t--> WARNINGS ({len(rawWarnings)}):")

                ts_format = "%Y-%m-%d (%H:%M)"
                warnings = []
                for item in rawWarnings:
                    if not item.isspace():
                        timestamp, warningLevel = item.split("_")
                        timestamp = timestamp.strip("' ")
                        warningLevel = int(warningLevel.strip("'])]"))
                        
                        if verbose and idk == 0:
                            print(f"\t\t--> WARNING (L{warningLevel}):", timestamp)

                        warnings.append({"Timestamp": pd.to_datetime(timestamp, format= ts_format),
                                         "Warninge level": warningLevel})

                # Get event anticipation
                fault = {"Message":fault_message, 
                         "Start": pd.to_datetime(fault_start, format = ts_format), 
                         "End": pd.to_datetime(fault_end, format = ts_format)}
                first_anticipation, last_anticipation = get_event_anticipation(fault, warnings) 
                
                days, hours, minutes, *_ = first_anticipation.components
                if verbose and idk == 0:
                    print(f"\t--> ANTICIPATION (1° Warning): {days} day(s) {hours}, hour(s) and {minutes} minute(s).")
                if len(warnings) > 1:
                    last_days, last_hours, last_minutes, *_ = last_anticipation.components

                    if verbose and idk == 0:
                        print(f"\t--> LAST WARNING: {last_days} day(s), {last_hours} hour(s) and {last_minutes} minute(s). ")

                if fault_profile == ["General Fault", "Log - High"]:
                    if warning_level in [1, 2]:
                        if "best_for_inverter" in config:
                            try: 
                                inv_fault_anticipation[f"L{warning_level}"].append(first_anticipation)
                            except KeyError:
                                inv_fault_anticipation[f"L{warning_level}"] = [first_anticipation]

            # Turn them into percentages
            further_filtered_df = further_filtered_df.apply(lambda value:(value * 100)).round(decimals = 2)
            metrics_cols = ["F1 score", "Recall", "Miss rate", "Precision", "Fall out"]

            try: 
                further_filtered_df = further_filtered_df[metrics_cols]
            except KeyError:
                metrics_cols = metrics_cols[1:]
                further_filtered_df = further_filtered_df[metrics_cols]

            # Save only the metrics (i.e., Recall, Miss rate, ...) as a Pandas Series
            perfomance_list.append(pd.Series(data = further_filtered_df.squeeze(),
                                             name = str("(CONFIG. " + str(idk + 1) + ") " + config)))
            
        # Create a dataframe with all the perfomances
        overall_perfomance = pd.DataFrame(perfomance_list)
        try:
            overall_perfomance = overall_perfomance.sort_values(by = ["F1 score", "Recall", "Precision", "Fall out"],
                                                                ascending = [False, False, False, True])
        except KeyError:
            overall_perfomance = overall_perfomance.sort_values(by = ["Recall", "Precision", "Fall out"],
                                                                ascending = [False, False, True])
        overall_perfomance.columns = [col + " (%)" for col in overall_perfomance.columns]

        # Visualize the dataframe
        if save_to_file:
            print(tabulate(overall_perfomance, headers = 'keys', tablefmt = 'psql'))
        else:
            display(overall_perfomance)
        
        # Visualize comparison between the configs available
        if len(overall_perfomance) > 1:

            for metrics in ['F1 score (%)', 'Recall (%)', 'Precision (%)']:
                print("\n" + "-" * 50 + f" {metrics} " + "-" * 50)
                perfomance = overall_perfomance[metrics]
                all_configs_available = perfomance.index.tolist()

                for config in all_configs_available[:1]:
                    reference_f1_score = perfomance[config]
                    if np.isnan(reference_f1_score):
                        continue
                    print("-" * 110)
                    print(f"REFERENCE: {'||'.join(config.split('_'))} --> {metrics}: {reference_f1_score} %")
                    print("-" * 110)

                    remaining_configs = all_configs_available.copy()
                    remaining_configs.remove(config)
                    for compared_config in remaining_configs:
                        f1_score = perfomance[compared_config]
                        diff = f1_score - reference_f1_score
                        print(f"--> {'||'.join(compared_config.split('_'))} --> [{round(diff, 2)}%]")

        best_perfomance[f"L{warning_level}"] = overall_perfomance.iloc[0].to_dict()
        
        # Save the ranked configurations (to compute some statistics afterwards)
        ranks = range(1, len(metrics_configs.keys()) + 1)
        configs = overall_perfomance.index
        ranked_config[f"warning{warning_level}"] = list(zip(ranks, configs, 
                                                            overall_perfomance["F1 score (%)"],
                                                            overall_perfomance["Recall (%)"], 
                                                            overall_perfomance["Precision (%)"],
                                                            overall_perfomance["Fall out (%)"]))

    inv_ranked_configs.append(ranked_config)
    
    if inv_fault_anticipation:            
        print("\n" + "-" * 10 + " ANTICIPATIONS " + "-" * 20)
        for level, anticipation in inv_fault_anticipation.items():
            print(f"WARNING LEVEL >= {level}:\n" + 20 * "-")
            #print(('\n').join([f"EVENT {idk + 1} --> " + str(item) for idk, item in enumerate(anticipation)]))
            avg_anticipation = np.mean(np.array(anticipation))
            print( 22 * "-" + f"\n[AVG] {avg_anticipation}\n")

    # Print the metrics of the paper
    paper_recall = [93, 98, 92]
    paper_missRate = [7, 2, 8]
    paper_fallOut = [13, 18, 1]
    print("\n"+ "-" * 10 + " METRICS STATED IN THE PAPER " + "-" * 10)
    print(f"RECALL (paper): {round(np.mean(paper_recall), 2)}% --> avg{paper_recall}\n"\
          f"MISS RATE (avg paper): {round(np.mean(paper_missRate), 2)} % --> avg{paper_missRate}"\
          f"\nFALL-OUT (avg paper): {round(np.mean(paper_fallOut), 2)} % --> avg{paper_fallOut}") 
    
    return inv_ranked_configs, inv_fault_anticipation, best_perfomance


# --------------------------------------------------------------------------------
# ------------ FUNCTION:Weight the som perfomance for each inverter  ------------
# --------------------------------------------------------------------------------
def merge_and_weigh_som_perfomance(inv_som_performance, threshold, verbose = False):
    weighted_som_perfomance = defaultdict(list)
    
    if threshold != -1:
        print(f"\nConsider only the configuration included in the 'top {int(threshold * 100)}%\n" + "-" * 80)
        
    for inv_name, som_perfomance in inv_som_performance.items():
        
        # Retrieve the ordered list of the configs       
        inv_best_configs = som_perfomance["best_configs"]

        print(f"[{inv_name}] Reading and merging som perfomance "\
              f"{'with threshold ' + str(int(threshold*100)) + '%' if threshold != -1 else ''} "\
              f"(i.e., {len(inv_best_configs)} configs.)")
        
        f1_scores = [metrics[1] for config_name, metrics in inv_best_configs]
        max_f1_score = np.nanmax(f1_scores)

        if verbose:
            print("-" * 40 + f" {inv_name} " + "-" * 40)
            print("F1-SCORE [MAX]: ", max_f1_score)

        # Extract items
        for config_perfomance in inv_best_configs:
            config_name, (quantization_error, f1_score, recall, precision, fold_f1_scores, computational_time) = config_perfomance
            normalized_f1_score = f1_score / max_f1_score

            if verbose:
                print("\nCONFIG:", config_name)
                print("F1 SCORE:", f1_score)
                print("[NORMALIZED] F1 SCORE", normalized_f1_score)

            if threshold != -1:
                if normalized_f1_score >= (1 - threshold):
                    weighted_som_perfomance[config_name].append({"normalized": normalized_f1_score, "raw_score": f1_score})
                else:
                    weighted_som_perfomance[config_name].append({"normalized": 0, "raw_score": f1_score})

                    if verbose:
                        print(f"Discarding this configuration (Normalized F1 score: {round(normalized_f1_score, 4)} ||"\
                              f" F1 score: {round(f1_score, 4)})")
            else:
                weighted_som_perfomance[config_name].append({"normalized": normalized_f1_score, "raw_score": f1_score})
    return weighted_som_perfomance

# --------------------------------------------------------------------------------
# ----- FUNCTION: Compute the average value among all the f1 scores  ------------
# --------------------------------------------------------------------------------
def compute_average_scores(weighted_som_perfomance, threshold):
    for config_name in weighted_som_perfomance.keys():   

        # Normalized and ordinary f1 scores
        normalized_f1_scorse = [item["normalized"] for item in weighted_som_perfomance[config_name]]
        f1_scores = [item["raw_score"] for item in weighted_som_perfomance[config_name]]

        # Averaged values
        avg_normalized = np.nanmean(normalized_f1_scorse)
        avg_scores = np.nanmean(f1_scores)

        # Save the new value for each configuration
        weighted_som_perfomance[config_name] = ((avg_normalized, normalized_f1_scorse), (avg_scores, f1_scores))

    # Sort the list according to the averaged normalized f1 score (i.e., avg_normalized)
    sorted_weighted_som_perfomance = sorted(weighted_som_perfomance.items(), 
                                     key = lambda config: config[1][0][0] if not np.isnan(config[1][0][0]) else 0, 
                                     reverse = True)
    return sorted_weighted_som_perfomance

# --------------------------------------------------------------------------------
# -------- FUNCTION: Visualize or save the normalized som configurations  ---------
# --------------------------------------------------------------------------------
def visualize_weighted_som_configurations(weighted_som_perfomance, threshold, system_name, dataset_type, labels, file_version, log_folder_path,
                                          save_to_file):
   
    if save_to_file: 
        console_stdout = sys.stdout
        log_file_name = f"weighted_som_{dataset_type}_performance{'_top' + str(int(threshold * 100)) + '%' if threshold != -1 else ''}.txt"
        log_file_path = path.join(log_folder_path, log_file_name)
        log_file = open(log_file_path,  mode = "w+")

        print(f"[{system_name}] The findings will be saved in a txt file (i.e., {log_file_name})")
        sys.stdout = log_file
    
    print("-" * 35 + " WEIGHTED SOM PERFOMANCE FOR EACH CONFIGURATION "\
          f"{'[with SOM within TOP ' + str(int(threshold * 100)) + '%] ' if threshold != -1 else ''}" 
          + "-" * 22 + "\n\t" + "-" * 28 + f" ({file_version}) " + "-" * 20 + "\n")

    previous_inv_available = -1
    for idk, (config_name, metrics) in enumerate(weighted_som_perfomance):
        
        # Retrieve the metrics
        (avg_normalized, normalized_f1_scores), (avg_scores, f1_scores) = metrics

        # Count the availble inverter data 
        normalized_f1_scores = np.array(normalized_f1_scores)
        inverters_available = np.count_nonzero(normalized_f1_scores[~ np.isnan(normalized_f1_scores)])
     
        if previous_inv_available != inverters_available:
            print("-" * 25 + f" INVERTERS AVAILABLE: {inverters_available}/{len(normalized_f1_scores)}" + "-" * 25)
            previous_inv_available = inverters_available
        
        print(f"TOP {idk + 1}: " + config_name)
        print(f"--> [AVG] Normalized F1-SCORE: {np.round(avg_normalized, 4)} ({np.round(avg_scores * 100, 2)} %)\n"\
              f"--> NORMALIZED: {dict(zip(labels, np.round(normalized_f1_scores, 4)))}")
        inv_scores_labelled = dict(zip(labels, [str(np.round(score * 100, 1)) + ('%' if not np.isnan(score) else '') 
                                                for score in f1_scores]))
        print(f"--> ACTUAL:    {inv_scores_labelled}\n")
    
    if save_to_file:
        sys.stdout.close()
        sys.stdout = console_stdout
            
# --------------------------------------------------------------------------------
# -------- FUNCTION: Get the temporal anticipation of a faulty event  ------------
# --------------------------------------------------------------------------------
def get_event_anticipation(fault, warnings):

    anticipation_warnings = []
    for warning in warnings:
        warning_ts = warning["Timestamp"]

        anticipation = fault["Start"] - warning_ts
        anticipation_warnings.append(anticipation)

    anticipation_warnings = sorted(anticipation_warnings, reverse = True)
    
    first_anticipation = anticipation_warnings[0]
    last_anticipation =  anticipation_warnings[-1]
    return first_anticipation, last_anticipation