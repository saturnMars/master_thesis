import numpy as np
import pandas as pd
from dateutil.parser import ParserError
from os import path

def load_priorities(system_name):
    freq_file_name = system_name.capitalize() + " - Fault_frequencies.xlsx"
    path_logs = path.join("data", system_name.upper(), system_name.upper(), freq_file_name)

    try:
        fault_freq = pd.read_excel(path_logs, sheet_name = ["Faults"])["Faults"]
    except FileNotFoundError:
        return None
    
    if 'priority' not in fault_freq.columns:
        return None

    fault_freq = fault_freq[["priority", "Messaggio"]]
    fault_freq.dropna(how="all", inplace=True)
    
    # Modify the values
    fault_freq.loc[:, "priority"] = fault_freq.loc[:, "priority"].map({"ALTA": "High", "MEDIA": "Medium", "BASSA": "Low"})
    fault_freq["Messaggio"] = fault_freq["Messaggio"].apply(lambda text: text.split("]")[-1])
    
    # Save the list of messagges for each priority
    raw_dict = fault_freq.to_dict(orient = "index")
    priorities = dict()
    for idk_row, item in raw_dict.items():
        priority = item["priority"]
        message = item["Messaggio"].strip()
        
        try:
            priorities[priority].append(message)
        except KeyError:
            priorities[priority] = [message]
            
    return priorities
    
def load_faults(system_name, include_faults_notRelatedToInverters = False, load_fault_logs = True, 
                log_priority = ["High", "Medium", "Low"], load_stringBox_data = True, load_faults = True, verbose=True):
    print(100 * "-" + f"\n\t\t\t\t\tFAULTS: {system_name}")
    print(f"\t\t\t\tPRIORITIES: {', '.join(log_priority)} {'& faults' if load_faults else ''}\n" + 100 * "-")

    if load_faults:
        # IMPORT: General faults
        fault_file_name = "Storico Guasti.xlsx"
        general_faults = pd.read_excel(path.join("data", fault_file_name), header = [0], skiprows = [1])
        general_faults = general_faults.drop(columns = general_faults.columns[0]).rename(columns = {"Unnamed: 1": "Colonne"})

        general_columns = general_faults["Colonne"].iloc[0:6].tolist()

        # Select the fault of the selected system
        general_faults = general_faults[system_name.upper()]

        indexes_to_split = range(0, len(general_faults), len(general_columns))
        list_fault_descriptions = np.split(general_faults, indexes_to_split[1:])
        for fault_desc in list_fault_descriptions:
            fault_desc.index = general_columns 

        # Create a dataframe 
        fault_df = pd.DataFrame(data = list_fault_descriptions, columns = general_columns)

        # Drop empty rows 
        fault_df.dropna(subset=["Componente Guasto"], inplace=True)
        fault_df["Causa Guasto"].fillna("Unknown", inplace = True)

        # Dealing with a missing starting timestamp 
        fault_df["Data Inizio"] = fault_df["Data Inizio"].fillna(value = fault_df["Data Fine"])
        fault_df["Orario Inizio"] = fault_df["Orario Inizio"].fillna(value = fault_df["Orario Fine"])

        # Merge the date and time 
        fault_df["Inizio"] = pd.to_datetime(
            fault_df["Data Inizio"].apply(lambda date: date.strftime("%Y-%m-%d")) + " " + 
            fault_df["Orario Inizio"].apply(lambda time: time.strftime("%H:%M"))
        )
        fault_df["Fine"] = pd.to_datetime(
            fault_df["Data Fine"].apply(lambda date: date.strftime("%Y-%m-%d")) + " " +
            fault_df["Orario Fine"].apply(lambda time: time.strftime("%H:%M"))
        )

        # Highlight the inverter
        fault_df["Tipo"] = "General Fault"
        select_inv_name = lambda message: message[:10].split(" ")[1] if "INV" in message[:10].upper() else "-"
        fault_df["Inverter"] = fault_df["Componente Guasto"].apply(lambda message: int(select_inv_name(message)) 
                                                                   if select_inv_name(message).isdigit() else pd.NA)
        # Drop the merged columns
        fault_df.drop(columns = ["Data Inizio","Data Fine", "Orario Inizio", "Orario Fine"], inplace=True)
        new_order_columns = [fault_df.columns.tolist()[-1]] + fault_df.columns.tolist()[:-1]
        fault_df = fault_df.reindex(columns = new_order_columns)
        fault_df.sort_values(by = ["Inizio"], inplace = True)
        fault_df.reset_index(inplace = True, drop=True)

        # --------- [SOLETO 1] Add retrieved information manually -----------------
        # A) INV 4: "GENERAL FAULT (-) inverter - sostituzione di 1 ventilayore e 1 scheda adapter slot"
        # --> a) LOG - MEDIUM (Inverter con produzione a 0) INV4 U090226 250kWp: Inverter con produzione a 0
        #       [2020-06-18 (11:36) -  2020-06-20 (09:39)]
        # --> b) GENERAL FAULT (-) inverter - sostituzione di 1 ventilayore e 1 scheda adapter slot
        #       [2020-06-20 (08:30) -  2020-06-20 (10:30)]
        fault_s1 = "inverter - sostituzione di 1 ventilayore e 1 scheda adapter slot"
        cond_fault_s1 = fault_df["Componente Guasto"] == fault_s1
        fault_df.loc[cond_fault_s1 , "Inverter"] = 4

        # --------- [SOLETO 2] Add retrieved information manually -----------------
        # A) INV 2: "Inveter -Scheda driver per IGBT fase T"
        # SINCE: 
        #--> LOG - HIGH (Allarme inverter) Inverter 2 250kWp U090223: [0x20000] Desaturazione IGBT inverter (0x200AD400)
        #                           [2021-01-19 (07:58) -  2021-01-21 (14:58)]
        #--> LOG - MEDIUM (Inverter con produzione a 0) Inverter 2 250kWp U090223: Inverter con produzione a 0
        #                           [2021-01-19 (11:05) -  2021-01-21 (14:58)]
        # AND THEN:
        # --> GENERAL FAULT (desaturazione IGBT ) Inveter -Scheda driver per IGBT fase T
        #                           [2021-01-21 (15:00) -  2021-01-21 (16:00)]
        fault_s2_a = "Inveter -Scheda driver per IGBT fase T"
        cond_fault_s2_a = fault_df["Componente Guasto"] == fault_s2_a
        fault_df.loc[cond_fault_s2_a , "Inverter"] = 2

        # B) INV 2: inverter - scheda alimentazione
        # SINCE:
        # --> LOG - MEDIUM (Ritardo comunicazione dispositivo) Inverter 2 250kWp U090223: Ritardo comunicazione dispositivo
        #                   [2020-10-12 (09:11) -  2020-10-13 (13:15)]
        # THEN:
        # --> GENERAL FAULT (sbalzo di corrente) inverter - scheda alimentazione
        #                   [2020-10-13 (14:30) -  2020-10-13 (15:30)]
        fault_s2_b = "inverter - scheda alimentazione"
        cond_fault_s2_b = fault_df["Componente Guasto"] == fault_s2_b
        fault_df.loc[cond_fault_s2_b , "Inverter"] = 2

        # Drop the fault of "scheda di comunicazione slot RS485" since it's not related to inverter operation 
        cond_fault_s0 = fault_df["Componente Guasto"].str.contains("scheda di comunicazione")
        idk_to_drop = fault_df[cond_fault_s0].index.tolist()
        fault_df.drop(index = idk_to_drop, inplace = True)
        if len(idk_to_drop) > 0:
            print(f"0) {len(idk_to_drop)} fault(s) called 'scheda di comunicazione' have/has been discarted. "\
                  "As it's not related to inverter operation.")

        if not include_faults_notRelatedToInverters:
            faults_to_drop = fault_df[fault_df["Inverter"].isnull()]
            print(f"\n0) DROPPING ({len(faults_to_drop)}/{len(fault_df)}) anonimous faults:")
            print("-", "\n- ".join(faults_to_drop["Componente Guasto"].tolist()), "\n")
            fault_df.drop(index = faults_to_drop.index, inplace = True)

        if load_stringBox_data:
            stringBox_related_faults = [
                'datexel - scheda PV isolation di un quadro di campo ed 1 concentratore',
                'fusibili stringhe',
                'moduli FV'
            ]
            fault_df = fault_df[fault_df['Componente Guasto'].isin(stringBox_related_faults)]
            print("\n0) SELECTING only the string box-related faults\n")

        if verbose:
            print(100 * "-" + f"\n\t\t\t\t\tGeneral faults\n" + 100 * "-")
        else:
            # Count the faults for each inverter
            grouped_faults = fault_df.groupby(by = ['Inverter']).count()['Tipo']
            if len(grouped_faults) > 0:
                count_faults = [f"INV{inv_number}: {counter}" for inv_number, counter in grouped_faults.items()]
            else:
                count_faults = [str(len(fault_df))]
            
            print(f"--> A) General faults loaded ({', '.join(count_faults)})")
    else:
        print("--> [A) General faults skipped]")
        fault_df = pd.DataFrame()
    
    # -----------------------LOAD FAUL LOGS ----------------------------------------
    # Dictionary of the logs concerning the inverter data

    if load_fault_logs or load_stringBox_data:
        fault_logs = dict()

        log_file_name = system_name.capitalize() + " - Storico Allarme.xlsx"
        path_logs = path.join("data", system_name.upper(), system_name.upper(), log_file_name)
        
        # Load the Excel file
        fault_logs_sheets = pd.read_excel(path_logs, skiprows = [0], sheet_name = None)
        sheet_names = fault_logs_sheets.keys()
        inv_names =  [key for key in fault_logs_sheets.keys() if "INV" in key]

         # Read the priorities
        priorities = load_priorities(system_name)

        if not priorities:
            print(f"\nWARNING:[{system_name}] Priority column not found.\nLoading priorities of the default PV System (i.e., Soleto 1)\n")
            priorities = load_priorities(system_name = 'Soleto 1')

        if load_fault_logs:
            fault_logs_inv_sheets = {sheet_name : sheet for sheet_name, sheet in fault_logs_sheets.items() if sheet_name in inv_names}

            # A) Read the inverter data
            for inv_name in inv_names:
                inv_df = fault_logs_inv_sheets[inv_name]

                # Rename the columns
                inv_df = inv_df.rename(columns = {
                        "Tipologia Evento" : "Causa Guasto",
                        "Messaggio":"Componente Guasto", 
                        "Ricevuto il": "Inizio",
                        "Rientrato il": "Fine"
                    })

                # Extract the inv number and assign it to the column "Inverter"
                inv_number = inv_name.split(".")[1].strip()
                inv_df["Inverter"] = int(inv_number)

                # Detect the property of the log
                detect_priority = lambda message: "Log - " + str([priority for priority in priorities.keys() 
                                                                  for fault in priorities[priority] 
                                                                  if fault in message]).replace("['", "").replace("']", "")
                inv_df["Tipo"] =  inv_df["Componente Guasto"].apply(detect_priority)

                # Reoder columns
                if len(fault_df) > 0:
                    inv_df = inv_df.reindex(columns = fault_df.columns)

                # Save the dataframe
                inv_dict_name = inv_name.replace(".", "") 
                fault_logs[inv_dict_name] = inv_df

            if not verbose:
                print(f"--> B) Inverter logs loaded ({', '.join([f'{inv_name}: {len(logs)}' for inv_name, logs in fault_logs.items()])})")
        else:
            if not verbose:
                print("--> [B) Inverter logs have been skipped (0)]")
          
        # B) Attach to the inverter data their "String Box"
        if load_stringBox_data:
            stringBox_names = sorted(set(sheet_names) - set(inv_names))

            for stringBox_name in stringBox_names:
                stringBox_df = fault_logs_sheets[stringBox_name]

                # Rename the columns
                stringBox_df = stringBox_df.rename(columns = {
                            "Tipologia Evento" : "Causa Guasto",
                            "Messaggio":"Componente Guasto", 
                            "Ricevuto il": "Inizio",
                            "Rientrato il": "Fine"
                        })

                # Extract the inv number and assign it to the column "Inverter"
                name_parts = stringBox_name.split('.')

                # CASE 1 --> SCHEMA NAME: CSPX.1
                if system_name == "Soleto 1": 
                    inv_reference = int(name_parts[0][-1].strip())

                # CASE 2 --> SCHEMA NAME: QC1.IX
                elif system_name in ["Soleto 2", "Galatina"]:
                    inv_reference = int(name_parts[1][-1].strip())
                stringBox_df["Inverter"] = inv_reference

                # Detect the property of the log
                detect_priority = lambda message: "Log_stringBox - " + str([priority for priority in priorities.keys() 
                                                                      for fault in priorities[priority] 
                                                                      if fault in message]).replace("['", "").replace("']", "")
                stringBox_df["Tipo"] =  stringBox_df["Componente Guasto"].apply(detect_priority)

                # Merge StringBox data with its inverter reference
                inv_ref_name = "INV" + str(inv_reference)
                if inv_ref_name in fault_logs.keys():
                    fault_logs[inv_ref_name] = pd.concat([fault_logs[inv_ref_name], stringBox_df])
                else:
                    fault_logs[inv_ref_name] = stringBox_df

                if verbose:
                     print(f"Merging '{stringBox_name}' with '{inv_ref_name}' data")

            if not verbose:
                print(f"--> C) String-box logs loaded ({', '.join([f'{inv_name}: {len(logs)}' for inv_name, logs in fault_logs.items()])})")
 
        # Concat all the logs loaded
        dataframes = [fault_df] + [fault_logs[inv_name] for inv_name in fault_logs.keys()]
        fault_df = pd.concat(dataframes, ignore_index = True)

        # Filter the logs according to the priority selected
        # a) Log properities
        priorites = ["Log - " + item.capitalize() for item in log_priority]
        # b) General fault
        priorites.append("General Fault")
        # c) String box fault
        if load_stringBox_data:
            priorites.extend(["Log_stringBox - " + item.capitalize() for item in log_priority])
        # Filter the dataframe
        fault_df = fault_df[fault_df["Tipo"].isin(priorites)]
        
        # Sort values
        fault_df.sort_values(by = ["Inizio"], inplace = True)
        fault_df.reset_index(inplace = True, drop=True)
        
        if verbose:
            print(120 * "-" + f"\n\t\t\t\tGeneral faults & Inverter logs {'& StringBox logs'if load_stringBox_data else ''} "\
                  f"(with priority: {log_priority})\n" + 120 * "-")
            grouped_fault_causes = fault_df.groupby("Causa Guasto").count().rename(columns = {"Tipo": "Logs"})
            grouped_fault_causes.sort_values(by = "Logs", ascending=False, inplace=True)
            grouped_fault_causes = grouped_fault_causes["Logs"]
            grouped_fault_causes.rename(axis = "FAULT CAUSES ({len(grouped_fault_causes.index)})", inplace=True)
            
            display(grouped_fault_causes)
            print("\t\t" + 90 * "-")

            display(fault_df)
        else:
            print("\nLoading completed!")
            fault_causes = [cause for cause in fault_df["Causa Guasto"].unique() if cause != "-"]
            print(f"\nFAUL CAUSES ({len(fault_causes)}):\n" + "-" * 20)
            print("\n".join([f"{idk + 1}) {cause}" for idk, cause in enumerate(sorted(fault_causes))]))
            
    return fault_df

# ---------------------------------------------------------------
# --------------------- FIND FAULT TIMESTAMPS ------------------
#---------------------------------------------------------------
def compare_fault_df_ts(inv_faults, timestamps, verbose = True):
    fault_periods = inv_faults["range_period"].tolist()
    fault_indexes = inv_faults.index

    fault_events = dict()
    for idk, range_period in enumerate(fault_periods):
        fault_idk = fault_indexes[idk]
        common_period_timestamps = np.intersect1d(range_period.tolist(), timestamps)
        
        if verbose:
            display(inv_faults.loc[fault_idk, :])
            print(f"COVERED RANGE: {inv_faults.loc[fault_idk, 'range_period'][0]} --> {inv_faults.loc[fault_idk, 'range_period'][-1]}")

        if len(common_period_timestamps) > 0:
            for timestamp in common_period_timestamps:
                fault_desc = inv_faults.loc[fault_idk, ["Tipo", "Causa Guasto", "Componente Guasto", "Period"]].tolist()
                try:
                    fault_events[timestamp].append(fault_desc)
                except KeyError: 
                    fault_events[timestamp] = [fault_desc]

                if verbose:
                    print(f"COMMON TS: {timestamp}")
    
    if verbose:
        for common_timestamp, faults in fault_events.items():
            faults = sorted(faults, key = lambda fault_desc: fault_desc[0])

            print("\n" + common_timestamp.strftime("%Y-%m-%d (%H:%M)") +  f"--> {len(faults)} faults in this TS\n" + "-" * 40)
            print("\n".join(fault[0] +" - " + fault[1] +" - " +  fault[2] +"\n" + str(fault[3][0]) + " - " + str(fault[3][1]) 
                            for fault in faults))
    return fault_events

def extract_fault_period(merged_col):
    period = []
    parts = merged_col.split("|")
    
    # STARTING TIMESTAMP
    try:
        period.append(pd.to_datetime(parts[0]))
    except ParserError:
        inferred_start = pd.to_datetime(parts[1]) - pd.Timedelta(1, unit= "hour")
        period.append(inferred_start)
        
    # ENDING TEMESTAMP 
    try:
        period.append(pd.to_datetime(parts[1]))
    except ParserError:
        inferred_end = pd.to_datetime(parts[0]) + pd.Timedelta(1, unit= "hour")
        period.append(inferred_end)
    return period

def find_fault_observation(fault_df, inv_df, inv_name, include_faults_notRelatedToInverters = True, tolerance = [False, True], 
                           verbose = True):  

    # Extract the number of the inverter from the inverter name
    inv_number = int(inv_name[-1:])
    
    # Build the filtering condition according to the flag 'include_faults_notRelatedToInverters'
    if include_faults_notRelatedToInverters:
        cond = (fault_df["Inverter"] == inv_number) | pd.isna(fault_df["Inverter"])
    else:
        cond = fault_df["Inverter"] == inv_number
        
    # Filter the faults/allarm logs
    inv_faults = fault_df[cond]

    if verbose:
        print(f"FILTERED FAULT DF (INV{inv_number}) --> {len(inv_faults)} failure events.")

    # TACKLE the annoying warning 'SettingWithCopyWarning'
    inv_faults._is_copy = None 
    
     # Extraxt the Starting and ending timestamps (save them as a Python list)
    inv_faults["Period"] = inv_faults[["Inizio", "Fine"]].astype(str).agg('|'.join, axis=1)
    inv_faults["Period"] = inv_faults["Period"].apply(extract_fault_period)
    inv_faults = inv_faults.drop(columns = ["Inizio", "Fine"])
    
    if verbose:
        print("\n" + 60*"-", f"\n\t\t\t{inv_name}: {len(inv_faults)} fault events\n" + 60*"-")
    
    # Create a list of fault periods
    time_tolerance = pd.Timedelta(59, unit="min")
    consider_start_tolerance = lambda start_fault: (start_fault - time_tolerance) if tolerance[0] else start_fault
    consider_end_tolerance = lambda end_fault: (end_fault + time_tolerance) if tolerance[1] else end_fault
    inv_faults["range_period"] = inv_faults["Period"].apply(lambda col: pd.date_range(start = consider_start_tolerance(col[0]),
                                                                                      end = consider_end_tolerance(col[1]),
                                                                                      freq = "1min", inclusive = "both"))
    # Compare the timestamps with the fault timestamps --> Find common 
    fault_events = compare_fault_df_ts(inv_faults, inv_df.index.tolist(), verbose)
    
    # Extraxt the unique fault events
    unique_faults = set([(fault[0], fault[1], fault[2], (str(fault[3][0]), str(fault[3][1]))) 
                         for ts, faults in fault_events.items() for fault in faults])
    unique_faults = sorted(unique_faults, key = lambda fault_desc: fault_desc[0])
    return fault_events, unique_faults

def find_first_fault_warning(warnings, faults, prediction_window):
    temporal_delta = pd.Timedelta(prediction_window, unit = "days")
    
    # Get and retrieve unique fault events
    unique_faults = set([(fault[0], fault[1], fault[2], fault[3][0], fault[3][1]) for fault in faults.values()])
    
    faults_warnings = []
    for fault_type, fault_cause, fault_mess, start_fault_ts, end_fault_ts in unique_faults:
        print(f"\nFAULT: ({fault_type}) {fault_cause} - {fault_mess} \n\t--> {str(start_fault_ts)} - {str(end_fault_ts)}")
        
        # Get all the fault warnings
        all_fault_warnings = [(warning_ts, warnings.loc[warning_ts, "Warning level"]) for warning_ts in warnings.index
                              if (start_fault_ts - temporal_delta) < warning_ts < start_fault_ts]
        all_fault_warnings = sorted(all_fault_warnings, key = lambda item: item[0])
        
        if len(all_fault_warnings) > 0:
            print(f"\t--> {len(all_fault_warnings)} warnings have been found for this fault")
            
            # Get the first warning
            first_warning = all_fault_warnings[0]
            anticipation = first_warning[0] - start_fault_ts
           
            # Save the findings
            faults_warnings.append({
                "fault": fault_mess + "_" + start_fault_ts.strftime('%Y-%m-%d (%H:%M)') + "_" + end_fault_ts.strftime('%Y-%m-%d (%H:%M)'),
                "warnings": [warning_ts.strftime('%Y-%m-%d (%H:%M)') + "_" + str(warning_level) 
                              for warning_ts, warning_level in all_fault_warnings], 
                "first_warning": f"(WL: {first_warning[1]}) " + first_warning[0].strftime('%Y-%m-%d (%H:%M)') + ": " 
                    + str(anticipation)
            })
        else:
            print(f"\t--> ISSUE: No warnings for the fault (prediction window: -{prediction_window} days)")

    return faults_warnings

def train_test_split(fault_df, df, inv_name, priorities_to_consider = ["High"], time_window = 15, temporal_tolerance = [False, True], 
                    verbose = True):

    faults_to_consider = ["General Fault"]
    faults_to_consider.extend(['Log - ' + item.capitalize() for item in priorities_to_consider])
    faults_to_consider.extend(['Log_stringBox - ' + item.capitalize() for item in priorities_to_consider])
    
    # Filter the fault dataset
    cond_important_fault = fault_df["Tipo"].isin(faults_to_consider)
    important_fault_df = fault_df[cond_important_fault]

    # Find the fault events for this inverter
    fault_events, unique_faults = find_fault_observation(important_fault_df, df, inv_name, 
                                                         include_faults_notRelatedToInverters = True,
                                                         tolerance = temporal_tolerance,
                                                         verbose = False)
    if verbose:
        print("\n" + "-" * 20 + " COMPUTING THE FAULT PERIODS " +  "-" * 20)
        
    test_period_timestamps = set()
    for idk, (fault_type, fault_cause, fault_component, period) in enumerate(unique_faults):
        if verbose:
            print(f"FAULT {idk + 1}/{len(unique_faults)}: ({fault_type}, {fault_cause}) {fault_component}\n" + "-" * 50)
            print(f"STARTING: {period[0]}\nENDING: {period[1]}")
        
        # Convert the period in timestamps
        period = [pd.to_datetime(ts) for ts in period]

        # Add the tolerance
        time_tolerance = pd.Timedelta(59, unit="min")
        if not temporal_tolerance[0]: # since '1-hour averaged' --> considers hour - 55 min 
            period[0] = period[0] + time_tolerance
        if temporal_tolerance[1]:
            period[1] = period[1] + time_tolerance
            
        # Consider only the hour component (exclude the minutes) to match them to the 1-hour sample dataset
        period = [date.strftime('%Y-%m-%d %H') for date in period]
        
        # Compute the starting and ending date to compute the test set periods
        tolerance_days = pd.Timedelta(time_window, unit="days")
        starting_date = pd.to_datetime(period[0]) - tolerance_days
        ending_date = pd.to_datetime(period[1]) + tolerance_days

        if verbose:
            print(f"\nTEST SET PERIOD (+- {time_window} days): FROM '{starting_date}' TO '{ending_date}'")
            
        # Compute the houly period
        fault_period_ts = pd.date_range(start = starting_date, end = ending_date, freq = "1H")
        test_period_timestamps.update(fault_period_ts)
        if verbose:
            print(f"HOURLY TIMESTAMP PERIOD (START FAULT - {time_window} days: END FAULT + {time_window} days):", len(fault_period_ts))
            print("TOTAL TEST TIMESTAMPS:", len(test_period_timestamps), "\n")
            
    # Fault periods generated for the inverter data
    test_period_timestamps = sorted(test_period_timestamps)
    if verbose:
        print("-" *90, "\n\t\tTEST TIMESTAMPS\n" + "-" * 90)
        print(f"TIMESTAMP PERIODS GENERATED FOR THE TEST SET: {len(test_period_timestamps)}/{len(df)} --> "\
              f" {round(len(test_period_timestamps)/len(df)*100, 2)} %")
    
    # A) Compute the TEST set (i.e., compare the timestamps generated with the original dataset)
    common_ts = np.intersect1d(test_period_timestamps, df.index.tolist())
    test_idk = df.loc[common_ts, :].index
    
    if verbose:
        print(f"COMMON TIMESTAMPS WITH THE ORIGINAL DATASET: {len(common_ts)} / {len(df.index)} "\
              f"({round((len(common_ts) / len(df.index)) *100, 2)} %)")
           
    # B) Compute the TRAIN SET
    train_idk = df[~ df.index.isin(test_idk)].index
    
    return train_idk, test_idk


def create_anomaly_df_pnn(fault_df, df, inv_name, priorities_to_consider = ["High"], time_window = 0, temporal_tolerance = [False, True], 
                    verbose = True):
    
    faults_to_consider = ["General Fault"]
    faults_to_consider.extend(['Log - ' + item.capitalize() for item in priorities_to_consider])
    faults_to_consider.extend(['Log_stringBox - ' + item.capitalize() for item in priorities_to_consider])
    
    # Filter the fault dataset
    cond_important_fault = fault_df["Tipo"].isin(faults_to_consider)
    important_fault_df = fault_df[cond_important_fault]

    # Find the fault events for this inverter
    fault_events, unique_faults = find_fault_observation(important_fault_df, df, inv_name, 
                                                         include_faults_notRelatedToInverters = True,
                                                         tolerance = temporal_tolerance,
                                                         verbose = False)
    if verbose:
        print("\n" + "-" * 20 + " COMPUTING THE FAULT PERIODS " +  "-" * 20)
        
    test_period_timestamps = set()
    for idk, (fault_type, fault_cause, fault_component, period) in enumerate(unique_faults):
        if verbose:
            print(f"FAULT {idk + 1}/{len(unique_faults)}: ({fault_type}, {fault_cause}) {fault_component}\n" + "-" * 50)
            print(f"STARTING: {period[0]}\nENDING: {period[1]}")
        
        # Convert the period in timestamps
        period = [pd.to_datetime(ts) for ts in period]

        # Add the tolerance
        time_tolerance = pd.Timedelta(59, unit="min")
        if not temporal_tolerance[0]: # since '1-hour averaged' --> considers hour - 55 min 
            period[0] = period[0] + time_tolerance
        if temporal_tolerance[1]:
            period[1] = period[1] + time_tolerance
            
        # Consider only the hour component (exclude the minutes) to match them to the 1-hour sample dataset
        period = [date.strftime('%Y-%m-%d %H') for date in period]
        
        # Compute the starting and ending date to compute the test set periods
        tolerance_days = pd.Timedelta(time_window, unit="days")
        starting_date = pd.to_datetime(period[0]) - tolerance_days
        ending_date = pd.to_datetime(period[1]) + tolerance_days

        if verbose:
            print(f"\nTEST SET PERIOD (+- {time_window} days): FROM '{starting_date}' TO '{ending_date}'")
        
        # Compute the houly period
        fault_period_ts = pd.date_range(start = starting_date, end = ending_date, freq = "1H")
        test_period_timestamps.update(fault_period_ts)
        if verbose:
            print(f"HOURLY TIMESTAMP PERIOD (START FAULT - {time_window} days: END FAULT + {time_window} days):", len(fault_period_ts))
            print("TOTAL TEST TIMESTAMPS:", len(test_period_timestamps), "\n")
            
    # Fault periods generated for the inverter data
    test_period_timestamps = sorted(test_period_timestamps)
    if verbose:
        print("-" *90, "\n\t\tTEST TIMESTAMPS\n" + "-" * 90)
        print(f"TIMESTAMP PERIODS GENERATED FOR THE TEST SET: {len(test_period_timestamps)}/{len(df)} --> "\
              f" {round(len(test_period_timestamps)/len(df)*100, 2)} %")
    
    ## add anomaly class
    df.loc[:, "Anomaly"] = 0
    
    # A) Compute the TEST set (i.e., compare the timestamps generated with the original dataset)
    common_ts = np.intersect1d(test_period_timestamps, df.index.tolist())
    test_idk = df.loc[common_ts, :].index
    df.loc[common_ts, "Anomaly"] = 1
    
    return df