DATE: 14/01/2022
AUTHOR: Bronzini Marco 

-----------------------------------------------------------------------------------------------------------------
--------------------------------------------- FOLDER STRUCTURE --------------------------------------------------
-----------------------------------------------------------------------------------------------------------------

--------------------------------------------- [GENERAL NOTE] ----------------------------------------------------
There are two types of photovoltaic systems (and thus, datasets) with different data structures and features available. 
Accordingly, some early steps (i.e., importing and cleaning processes) are carried out with different notebooks implementing coherent strategies.
- Type A: Binetto 1, Binetto 2
- Type B: Emi, Cantore, Soleto 1, Soleto 2, Galatina, Verone
-----------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------
[FOLDER 0] _library (i.e., various utils used in the notebooks)
----------------------------------------------------------------
    --> ./diagnostic_utils.py    (i.e., utils used to compute the DIAGNOSTIC KPI)
    --> ./fault_utils.py         (i.e., utils used to handle FAULTS and LOG ALARMS)
    --> ./som_outcome_utils.py   (i.e., utils used to deal with SOM PERFORMANCE FINDINGS)
    --> ./som_pre_utils.py       (i.e., utils used to carry out various PRE-PROCESSING STEPS for the SOM)
    --> ./som_utils.py           (i.e., utils used to manage all aspects of the SOM)
    --> ./utils.py               (i.e., utils used to handle general operations on the datasets)

------------------------------------------------
[FOLDER 1] ETL (i.e., Extract, Transform, Load) 
------------------------------------------------
    --------------------------------------------------------------------------------------------------------
    --> Import                  (i.e., Reading the raw monthly excel files and creating an atomic dataset)
    --------------------------------------------------------------------------------------------------------
        --> ./SAMPLE - ETL - Import data (Emi&co.).ipynb
        --> ./SAMPLE - ETL - Import data (Bin.1&2).ipynb

        ------------------------------------------------------------------------------------------
        [NOTE]: Steps carried out:
            a) Loading the monthly data (excel files)
            b) Handling appropriatly the multiple excel sheets
            c) Merging all the montly data into a single dataframe
            d.1) Saving the new atomic dataframe as a CSV file 
            d.2) Saving also the raw irradiance values (which may be used later)

        INPUTS: Montly data (i.e., one excel file for each month)
            [PARAMETER: "system_name" (select one among those included in the variable 'system_names')]
        OUTPUTS: Single CSV file with all the montly data available (../imported data)
        -------------------------------------------------------------------------------------------
    ---------------------------------------------------------------------------------------------------
    --> Cleaning             (i.e., carring out multiple steps for cleaning the datasets)
    ---------------------------------------------------------------------------------------------------
        --> ./SAMPLE - Data Cleaning - Emi & co.ipynb
        --> ./SAMPLE - Data Cleaning - Binetto.ipynb

        ------------------------------------------------------------------------------------------------
        [NOTE]: Steps carried out:
            a) A quick data exploration (i.e., descriptive statistics)
            [Only for Binetto 1&2] Merge the two sources of ambiental temperatue values 
            b) Check uniqueness of the datetimes and eventually try to fix duplicated observations
            c) Discard features (i.e., columns) that are confirmed empty or corrupted.
            d) Discover and fix (i.e., using a weighted K-NN) outliers in the features
                d.1) AC voltages
                d.2) Solar irradiance values
                d.3) [Only for Emi&co.] inverter temperature values 
                d.4) [Only for Binetto 1&2] Generated Energy (kWh)
            e) Save the cleaned dataset 

        INPUTS: the atomic CSV file of the PV system (i.e., output from import notebooks)
            [PARAMETER: "system_name" (select one among those included in the variable 'SYSTEM_NAMES_FULL')]
        OUTPUTS: A cleaned dataset saved as a CSV file (../cleaned)
        ------------------------------------------------------------------------------------------------

    ---------------------------------------------------------------------------------------------------
    --> ./SAMPLE - Sampling correction - 1 hour.ipynb
    ---------------------------------------------------------------------------------------------------
        a) Merge the irradiance values (i.e., 'raw_irr_data.csv') with those included in the inverter data

        b) Generate hourly dataset (using simultaneously two different stategies)
            1) Generete the 1-hour sampling dataset [STAT 1, hourly_inv_data] 
                - i.e., keeping only obervation of the hour
            2) Generete the 1-hour averaged dataset [STAT 2, averaged_hourly_inv_data] 
                - i.e., computing averaged values from 'T' to 'T - 55 minutes'

        c) Integrate the ambiental condition retrieved from another data source (ONLY IF AVAILABLE for that PV system)
            PARAMETER: 'use_amb_temp' [default: TRUE --> for Soleto 1, Soleto 2, Galatina || FALSE --> for the others] 

        d) Save both datasets inside two different folders (i.e., 1-hour sampling' & '1-hour averaged sampling')
    
        --------------------------------------------------------------------------------------------
        INPUTS: The cleaned CSV file related to a PV system (i.e., output from cleaning notebooks) 
            [This notebook has never been tested, but it MIGHT also work using the output files 
            generated using the import notebooks (i.e., skipping all the cleaning steps)]
        PARAMETERS: 
            a) "system_name" (select one among those included in the variable 'SYSTEM_NAMES_FULL')
        OUTPUTS: CSV files of the two hourly sampling datasets (generated using the two different strategies)
             placed in two sub-folders placed in the main folder called 'imported data' 
        ---------------------------------------------------------------------------------------------
        
        -------------------------------------------------------------------------------------------------
                  AFTER DEEP ANALYSIS ONLY THE dataset '1-hour averaged sampling' will be used
        -------------------------------------------------------------------------------------------------

------------------------
[FOLDER 2] Exploration
------------------------
    
    ---------------------------------------------------------------------------------------------------
    USED ONLY TO EXPLORE THE DATASETS IN THE EARLY STAGES, IT SHOULD NOT BE USED DURING THE DEVELOPMENT
    ---------------------------------------------------------------------------------------------------
    
    -------------------------------------------------------------------------------------------------------------------------------------
    --> ./SAMPLE - Data Exploration - Numerical & temporal distributions.ipynb  
        (i.e., Generate graphs for numerical and temporal distributions)
    -------------------------------------------------------------------------------------------------------------------------------------
        0) PARAMETERS: 
            a) "system_name" (select one among those included in the variable 'SYSTEM_NAMES_FULL')
            b) "subfolder" (select one among: "Cleaned", "1-hour sampling", "1-hour averaged sampling", None)
        a) Generate and save boxplots to display the NUMERICAL DISTRIBUTIONS.
        b) Generate and save histograms to display the TEMPORAL DISTRIBUTIONS.
        c) Generate and save heatmaps to display the CORRELATIONS COEFFICIENTS.

    ---------------------------------------------------------------------------------------------------------------------------------------
    --> ./SAMPLE - ETL - Simple fault analysis.ipynb     (i.e., Statistical analysis of the log alarms 
                                                                --> "[PV SYSTEM]-Storico Allarme.xlsx")
    ---------------------------------------------------------------------------------------------------------------------------------------
        PARAMETER: system_name (select one among those included in the variable 'SYSTEM_NAMES_FULL')

    ------------------------------------------------------------------------------------------------------------------------------------
    --> ./SAMPLE - Data Exploration - Various stuff.ipynb                (i.e., various utils used to check specific issues)
    ------------------------------------------------------------------------------------------------------------------------------------
        a) Numerical comparisons between columns
        b) Test out the theoretical formula for the power (Ampere * Voltage)
        c) Isolate problematic temporal or numerical range for specific variable 
        d) Analyse the three-phase behaviour on a random day 
        e) Analyse sampling of the dataset 
    ------------------------------------------------------------------------------------------------

---------------
[FOLDER 3] UC1
---------------
    -------------------------------------------------------------------------------------------------------------------------------------
    --> ./SAMPLE - UC1 - Self-organizing map            (i.e., main notebook in which the Self-Organizing Map (SOM) is implemented)
    -------------------------------------------------------------------------------------------------------------------------------------
        a) Loading the dataset regarding the Photovoltaic system selected

        b) Technical transformation of the datasets (e.g., timestamps as indexes)

        c) Load failure events (general faults & alarm logs) of the photovoltaic system
            PARAMETERS:
            - Load alarm logs (default: True)
            - Load alarm logs from string boxes (default: False)
            - Load anonimous faults (default: False)

        d) Split the dataset into TRAIN and TEST sets (there are two possible stategies)
            PARAMETERS: 
            - "simple_train_test_split": Decide which splitting stategy to use (default: False)
            - "test_fault_priorities": Decide which type of failure events including in the test set 
                                      (default: "General Fault" & "Log - High")
            - "days_to_include": Decide the temporal tolerance when considering a period including a failure event 
                                (X - start_event, end_event + X)

        e) Carring out some pre-processing processes (selected using the variable 'pre_processing_steps')
            - It's highly recommended to never skip the process of 'data standardization'.

        f) Train the Self-Organizing Map (SOM)
            PARAMETERS:
            - Grid_search (True./False) - values to test out are included in the notebook cell 'Training phase'
            - Pre-trained (True./False)

        g) Compute the KPI scores (for each hourly observation using a sliding window)
            PARAMETER: "sliding_window": number of hours to include in the sliding window of the KPI (default: previous 24 hours)

        h) Compute the warnings using the computed KPI scores

        i) Quick visualization of the warnings generated (analysing the fault perspective as well as the warning perspective)

        j) Compute the correct and wrong predictions (i.e., True./False Positive, True./False Negative)

        k) Compute the final metrics (F1 score, Recall, Miss Rate, Fall-out and Precision)

        ----------------------------------------------------------------------------------------------------------------------------------
        INPUTS: "1-hour sampled cleaned dataset" of the PV system (output of the notebook 'SAMPLE - Sampling correction - 1 hour.ipynb')
    
        PARAMETERS:
            a.1) "system_name" (select one among those included in the variable 'SYSTEM_NAMES')
            a.2) "subfolder" (select one among: "1-hour sampling", "1-hour averaged sampling")
            b) "grid_search_preProcessesing": whether testing out all possible combiantions of pre-processing steps 
    
        OUTPUTS: a folder called "SOMs" which includes several findings:
            - "Trained SOM": this folder could contain:
                - the trained SOM 
                - the performance of the grid search (as txt files for each inverter)
            - "Graphs": graphical representation of the SOM neurons after the training process (i.e., which neurons have been activated)
            - "KPI scores": KPI scores and their thresholds which were computed using a specific trained SOM
            - "Warnings": Warnings with their warning level and raw KPI score generated using a specific trained SOM
            - "Metrics": All the metrics (e.g., F1 score) generated using a specific trained SOM as well as: 
                - The warnings correctly generated 
                - Failure events correctly detected
        ---------------------------------------------------------------------------------------------------------------------------------

    -------------------------------------------------------------------------------------------------------------------------------------
    --> ./SAMPLE - UC1 - Visualize SOM findings            (i.e., Visualize the som findings)
    --------------------------------------------------------------------------------------------------------------------------------------
        0) PARAMETERS: 
            a) "system_name" (select one among those included in the variable 'SYSTEM_NAMES_FULL')
        a) Visualize findings of the grid search 
            a.1) Numerical perfomance 
                PARAMETER: "save_to_file" (i.e., decide whether visualize the output in the notebook or save them in a txt file)
                - [View 1] for each inverter
                - [View 2] Considering all inverters
                - [View 2.bis] considering all PV systems
            a.6) View perfomance viually
                - [View 3] Generate graph using the perfomance of the som (X: Number of Epoches || Y: Metrics)
                    --> For each metrics (Quantization errror, F1 Score, Recall, Precision)
                        --> For each inverter available
                    - PARAMETERS: 
                        - "var_panels": Variable to consider for generating each grahical panel 
                                        [default: (neighbourhood) 'Function']
                        - "var_graphColums": variable to consider for generating the graphs on the columns of the graphical panel 
                                        [default: 'Dim grid']
                        - "var_graphRows": variable to consider for generating the graphs on the rows of the graphical panel 
                                        [default: 'Dim grid']
            b) View the perfomance of trained som (using a specific pre-processing version) - [View 4]
                - The cells will load the metrics included in the folder "SOMs./Metrics"
                - Although the cells have been developed to load multiple som pre-processing versions 
                  (used to compare the perfomance of different pre-processing steps)
                    - They are now used to load only one pre-processing version (selected after a deep analysis)
                - PARAMETERS
                    - "save_metrics_to_file": whether saving the results on a txt file or visualizing them in the notebook
                    - "prediction_window": there are multiple metrics available for different prediction windows (1, 2, 3, 4, 5, 6, 7)
                                       [default: 7 days]
            b.2) View the averaged days in advance of the failiure events detected
 
    --------------------------------------------------------------------------------------------------------------------------------------
    --> ./SAMPLE - UC1 - Experiments                (i.e., various experiments concerning som parts using the input data for the som)
    --------------------------------------------------------------------------------------------------------------------------------------
        0) PARAMETERS: 
            a) "system_name" (select one among those included in the variable 'SYSTEM_NAMES_FULL')
            b) "subfolder" (select one among: "Cleaned", "1-hour sampling", "1-hour averaged sampling", None)
        a) Check visually (i.e., using a graph) the temporal distribution of some variable (e.g., Irradiance, E. Totale)
        b) Emperical rule for the grid dimension for the SOM
        c) Analysis of missing timestamps

    --------------------------------------------------------------------------------------------------------------------------------------
    --> ./SAMPLE - UC1 - Diagnostic KPI                (i.e., implementation of the Perfomance Ratio)
    --------------------------------------------------------------------------------------------------------------------------------------
        0) PARAMETERS: 
            a) "system_name" (select one among those included in the variable 'SYSTEM_NAMES_FULL')
            b) "subfolder" (select one among: "1-hour sampling", "1-hour averaged sampling")
        a) Carry out some transformations:
            - Computing the Generated Energy (using the cumulative variable called 'E.Totale')
            - Discarding problematic periods (e.g., due to corrupted irradiance values)
        b) Outlier correction: Find and try to fix the outliers (i.e., negative values) of the variable 'Generated Energy'
        c) Outlier correction: Tackle missing hourly observations
        d) Compute the daily performance ratio using a sliding window
            PARAMETERS: "sliding_window": number of previous months to consider for computing the ratio 
                        [suggested range: from 1 to 12 months, "-1" is used to compute the comulative]
        e) Find and try to fix potential problematic perfomance ratios (i.e., values exceeding 100%) 
        f) Generate and save the diagnostic KPI (i.e., daily perfomance ratio) using a temporal graph
        g) Correlation with the fault and alarm logs 
    
        -----------------------------------------------------------------------------------------------------------------------------------
        INPUTS: "1-hour sampled cleaned dataset" of the PV system (output of the notebook 'SAMPLE - Sampling correction - 1 hour.ipynb')
        OUTPUTS: a folder called "Diagnostic" which includes several findings:
            - graphs of the generated perfomance ratios
        ---------------------------------------------------------------------------------------------------------------------------------

-------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------- PIPELINE -------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------

------------ Data preparation ------------------
1) SAMPLE - ETL - Import data.ipynb
2) SAMPLE - Data Cleaning - Emi & co.ipynb
3) SAMPLE - Sampling correction - 1 hour.ipynb

----------------------------------- Use Case 1 --------------------------------
4a) SAMPLE - UC1 - Self-organizing map [ONLY FOR Soleto 1, Soleto 2 & Galatina]
4b) SAMPLE - UC1 - Diagnostic KPI

--------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------