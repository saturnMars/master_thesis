DATE: 05/04/2022
AUTHOR: Bronzini Marco, Gianmaria Tarantino

-----------------------------------------------------------------------------------------------------------------
--------------------------------------------- FOLDER STRUCTURE --------------------------------------------------
-----------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------
[FOLDER 0] _library (i.e., various utils used in the notebooks)
----------------------------------------------------------------
    --> ./fault_utils.py         (i.e., utils used to load the alarm logs)
    --> ./utils.py               (i.e., utils used to handle general operations on the inverter datasets)
    --> ./lstm_utils.py		     (i.e., utils used to manage the ML approach and its data preparation stages)

-------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------- THE NOTEBOOKS -------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------
------------ 0: Data preparation ------------------
--------- (common part with the UC1) --------------

------------------------------------------------
[FOLDER 1] ETL (i.e., Extract, Transform, Load) 
------------------------------------------------
    --------------------------------------------------------------------------------------------------------
    --> Import                  (i.e., Reading the raw monthly excel files and creating an atomic dataset)
    --------------------------------------------------------------------------------------------------------
        --> ./SAMPLE - ETL - Import data (Emi&co.).ipynb

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
        --> ./SAMPLE - [PRODUCTION] SAMPLE - Data Cleaning - Emi & co.ipynb

        ------------------------------------------------------------------------------------------------
        [NOTE]: Steps carried out:
            a) Load data and perform some checks (Exploration section can be skipped)
            b) Fix uniqueness of datetimes: Check uniqueness of the datetimes and eventually try to fix duplicated observations
            c) Carry out some transformations: Discard features (i.e., columns) that are confirmed empty or corrupted (Note that these columns are not the ones taken into account as input for the model)
            d) [Skipped for production]: Detect and correct outliers: Discover and fix (i.e., using a weighted K-NN) outliers in the features [NOte that this step has to be performed only with train data and not with test data]
                d.1) AC voltages
                d.2) Solar irradiance values
                d.3) inverter temperature values
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
            1) Generate the 1-hour sampling dataset [STAT 1, hourly_inv_data] 
                - i.e., keeping only obervation of the hour. Do not consider!
            2) Generate the 1-hour averaged dataset [STAT 2, averaged_hourly_inv_data] 
                - i.e., computing averaged values from 'T' to 'T - 55 minutes'

        c) Integrate the ambiental condition retrieved from another data source (ONLY IF AVAILABLE for that PV system)
            PARAMETER: 'use_amb_temp' [default: TRUE --> for Soleto 1, Soleto 2, Galatina || FALSE --> for the others] 

        d) Save both datasets inside two different folders (i.e., 1-hour sampling' & '1-hour averaged sampling')
    
        --------------------------------------------------------------------------------------------
        INPUTS: The cleaned CSV file related to a PV system (i.e., output from cleaning notebooks) 
        PARAMETERS: 
            a) "system_name" (select one among those included in the variable 'SYSTEM_NAMES_FULL')
        OUTPUTS: CSV files of the two hourly sampling datasets (generated using the two different strategies)
             placed in two sub-folders placed in the main folder called 'imported data' 
        ---------------------------------------------------------------------------------------------
        
        -------------------------------------------------------------------------------------------------
                  AFTER DEEP ANALYSIS ONLY THE dataset '1-hour averaged sampling' will be used
        -------------------------------------------------------------------------------------------------


--------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------- (UC2) THE NOTEBOOKS --------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------

-------------------------------------------------------------------------------------------------------------------------------------
------------------------------------ (1) SAMPLE - UC2 - Create the failure events dataframe ----------------------------------------- 
-------------------------------------------------------------------------------------------------------------------------------------

1) --> ./sample/UC2/SAMPLE - UC2 - Create the failure events dataframe.ipynb
AIM: Read all the alarm logs concerning the string boxes of a PV system included in the Excel file '[system_name] - Storico Allarme.xlsx'.

        a) Select Pv system

        b) Retrieve failure events
            1) alarm priorities selection (for the model training we need "High" and "Medium" but for model application we will use only the "Medium" alarms)
            2) select which data to load ('stringBox_alarms': True, all other params must be set to False)
			3) load the failure events
			4) carry out some trasnformation (compute end_datetime, compute alarm duration, retrieve plant box names, retrieve string names)
			5) visualize the outcome [this can be skipped]
			
        c) Save the dataframe

PARAMETERS:
- [A.1) Selecting the PV system] --> "system_name"
- [B.1) Select the priorities to load] --> "fault_priorities" (do not change whather is not strictly necessary)

 INPUT --> Excel File: './system_name/[system_name] - Storico Allarme.xlsx'
OUTPUT --> CSV file: './system_name/Imported data/Failure events/{fault_priorities}_failureEvent_logs.csv


-------------------------------------------------------------------------------------------------------------------------------------
------------------------------------- (2) SAMPLE - UC2 - Diagnostics - Detect failure events.ipynb ---------------------------------------- 
-------------------------------------------------------------------------------------------------------------------------------------
2) --> ./sample/UC2/SAMPLE - UC2 - Diagnostics - Detect failure events.ipynb
AIM: Predict failure events with high priority (i.e., events that imply a power stoppage) using the logs of failure events 
     with medium priority of the previous X hours (possible value range: 1-256 hours)
	a) Read the CSV file (i.e., ...failureEvent_logs.csv) created previously
	b) Prepare the data for the trained model (i.e. data preparation)
	c) create the new data space:
		c.1) Retrieve the number of strings for each string box: the config file is loaded
		c.2) Generate the column names
		c.3) Generate the inverter names
		c.4) Fill the new data space
		c.5) Integrate the inverter data with the failure event logs (i.e., DC current, DC voltage)and the ambiental condition data with the failure event logs (e.g., Solar irradiance, Amb. Temperature, Humidity)
		c.6) Remove redundant features: there are unnecessary pairs (artifacts) computed that must be removed
		c.7) Fill the empty timestamps
		c.8) Standardize the data: the standard scaler parameters are loaded to normalize data
	d) Load a trained model [for each inverter]: use the parameter merged_invs_config = True 
		d.1) Configurations: set the num_neurons and window_length params for each inverter that will be used to load the proper model
		d.2) Load the trained models for each inverter
	e) Predict high-priority alarms
		- last_k_hours is the parameter that can be set to specify the last hours to get the prediction
		- a minimum temporal period will be used (e.g., the trained model loaded required only the past 48 hours)
	f) Normalize the predictions:
		- for the last_k_hours timestamps the predicted high-priority alarms are computed
		- 0: the alarm is predicted to be not present
		- 1: the alarm is predicted to be present
	
PARAMETERS READ FROM FILEs:
- [Retrieve the number of strings for each string box] --> JSON file loaded: ./Params/stringBoxes_config.json
- [Standardize data: Read the parameters for standardizing the data] --> CSV file loaded: ./Params/INVx_stdScaler_generalizedApproch.txt
- [Load the trained model: Load the trained models for each inverter] --> load keras saved models --> default folder: ./Trained models - Generalized version/...

PARAMETERS:
- [A.1) Selecting the PV system] --> "system_name"
- [Data preparation: Select only the relevant period] --> 'minimum_days_required'
	- Default: 11 (~ 256 hours)
- [Integrate the inverter data: Load the inverter data] --> "dataset_name" (default path: ./Imported data/{dataset_name}) --> default: '1-hour averaged sampling'
- [Integrate the inverter data: Fill the empty timestamps] --> "fill_empty_ts" (i.e., tackle potential missing values from the inverter source) --> default: True 
- [Load the trained model: Configurations to load] --> "avg_config" (i.e., use or not use an average configuration across all the inverters) (default: ...)
- [Load the trained model: Configurations to load] --> "config_to_load" (i.e., which trained model load and use) --> DO NOT CHANGE THEM IF NOT STRICTLY NECESSARY





-------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------- PIPELINE -------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------
------------ 0: Data preparation ------------------
--------- (common part with the UC1) --------------
---------------------------------------------------
0.a) SAMPLE - ETL - Import data.ipynb
0.b) SAMPLE - Data Cleaning - Emi & co.ipynb
0.c) SAMPLE - Sampling correction - 1 hour.ipynb

NB: Look at the UC1 readme file for more information about these notebooks

---------------------------------------------------
------------ 1: UC2: Data import -------------
---------------------------------------------------
1) --> ./sample/UC2/SAMPLE - UC2 - Create the failure events dataframe.ipynb

---------------------------------------------------
---------------- 2: UC2: Use case 2 ---------------
---------------------------------------------------
2) --> ./sample/UC2/SAMPLE - UC2 - Diagnostics - Detect failure events.ipynb