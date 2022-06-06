DATE: 03/03/2022
AUTHOR: Bronzini Marco, Gianmaria Tarantino

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
    --> ./fault_utils.py         (i.e., utils used to handle FAULTS and LOG ALARMS. These utils are only used in the train phase and for creating the test set labelling alarm and faults as anomalies.)
    --> ./som_outcome_utils.py   (i.e., utils used to deal with SOM PERFORMANCE FINDINGS. These utils can be skipped)
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


---------------
[FOLDER 3] UC1
---------------
    -------------------------------------------------------------------------------------------------------------------------------------
    --> ./SAMPLE - UC1 - Self-organizing map            (i.e., main notebook in which the Self-Organizing Map (SOM) is implemented)
    -------------------------------------------------------------------------------------------------------------------------------------
        a) Loading the dataset regarding the Photovoltaic system selected

        b) Technical transformation of the datasets (e.g., timestamps as indexes)

        c) Load failure events (general faults & alarm logs) of the photovoltaic system
            PARAMETERS [only for training and testing purposes]:
            - Load alarm logs (default: True)
            - Load alarm logs from string boxes (default: False)
            - Load anonimous faults (default: False)

        d) Split the dataset into TRAIN and TEST sets (there are two possible stategies)
            PARAMETERS [only for training and testing purposes]: 
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
    --> ./SAMPLE - UC1 - [PRODUCTION] SAMPLE - UC1 - Self-organizing map           (i.e., main notebook in which the trained Self-Organizing Map (SOM) is used)
    -------------------------------------------------------------------------------------------------------------------------------------
        1) Load data
		
			1.A) Loading the dataset regarding the Photovoltaic system selected

			1.B) Technical transformation of the datasets (e.g. drop unnecessary columns, drop duplicated data)

        2) Data pre - processing

			2.A) Read parameters and models:
				- detrending parameters
				- std scaler params
				- regression model
				
			2.B) Apply the preprocessing steps:
				- compute Dc power
				- regression
				- detrending
				- scaling
			
		3) SOM
		
			3.A) read the SOM model names
			
			3.B) create the folders where saving the KPI values returned (KPI is the name used for the function which manipulates the output of the som) and warnings
			
			3.C) load the SOM model
			
			3.D) compute the KPI values:
				 - given the last 24 temporal hours a single real value in [0,1] will be returned.
				 Note that the notebook gets a test sample of multiple hours and computes all the KPI values, thus setting num_selected_kpi = 2,
				 the last two hours computed will be returned.
				
			3.E) loading the numerical thresholds for generating the warnings
			
			3.F) generate the warnings associated to each KPI computed
			
        ----------------------------------------------------------------------------------------------------------------------------------
        INPUTS:
		
		1.	nome impianto e inverter
		2.	modello di som trainato (.p) per quell’impianto e inverter e relativo file di probabilità di occupazione celle calcolato sul train set (.csv)
		3.	file csv di threshold calcolate per l’inverter in fase di addestramento modello
		4.	misurazioni a frequenza di campionamento oraria delle precedenti 24 ore temporali. E.g  alle 20:00 del 15/11 uso le misurazioni che partono dal primo 
			istante successivo alle 20:00 del 14/11 (cioè 20:05 del 14/11). N.B. gestiamo la casistica di misurazioni assenti nelle ore notturne:
			a.	voltaggio e corrente DC 
			b.	voltaggi e correnti trifase (quindi 3 voltaggi AC e 3 correnti AC)
			c.	potenza DC (calcolata moltiplicando voltaggio DC * corrente DC)
			d.	potenza AC
			e.	energia generata
			f.	temperatura inverter
			g.	irradianza solare registrata sul piano dei moduli (con un solarimetro)
			h.	temperatura ambientale
    
        OUTPUTS:
		
		Per ogni ora diurna della giornata viene calcolato un warning level sulla base delle misurazioni delle 24 ore temporali precedenti che assume valori interi in [0,2].
		L’output del modello prognostico è una 2-upla del tipo:
		timestamp(orario), warning level
		Quando viene sollevato un warning level >= 1 si predice il verificarsi di un evento grave entro 7 giorni (un allarme high o un fault).
        ---------------------------------------------------------------------------------------------------------------------------------

    --------------------------------------------------------------------------------------------------------------------------------------
    --> ./SAMPLE - UC1 - Diagnostic KPI                (i.e., implementation of the Perfomance Ratio)
    --------------------------------------------------------------------------------------------------------------------------------------
        A.4) Carry out some transformations:
            - Computing the Generated Energy (using the cumulative variable called 'E.Totale')
		A.5) Descriptive statistics (skip)
        B) Outlier correction: Find and try to fix the outliers (i.e., negative values) of the variable 'Generated Energy'
        C) Outlier correction: Tackle missing hourly observations
        D.0 - D.2) Compute the daily performance ratio using a sliding window
            PARAMETERS: "sliding_window": number of previous months to consider for computing the ratio 
                        [suggested range: 12 months, "-1" is used to compute the comulative]
        D.3) Find and try to fix potential problematic perfomance ratios (i.e., values exceeding 100%) 
        E) Generate and save the diagnostic KPI (i.e., daily perfomance ratio) using a temporal graph
        F) Correlation with the fault and alarm logs (skip)
    
        -----------------------------------------------------------------------------------------------------------------------------------
        INPUTS: "1-hour sampled cleaned dataset" of the PV system (output of the notebook 'SAMPLE - Sampling correction - 1 hour.ipynb')
        OUTPUTS: a folder called "Diagnostic" which includes several findings:
            - graphs of the generated perfomance ratios
        ---------------------------------------------------------------------------------------------------------------------------------

		
    --------------------------------------------------------------------------------------------------------------------------------------
    --> ./SAMPLE-UC1-unige-Diagnostic KPI from Maps connector data               (i.e., implementation of the Perfomance Ratio and Performance Ratio with temperature correction)
    --------------------------------------------------------------------------------------------------------------------------------------
        A.1) Select the PV system: only FV is allowed for Pr with temp correction since we don't have data for other systems
		
		A.2) Retrieve its parameters:
            - nominal power
			- beta (use it when computing PR with temp correction)
			
        A.3) Data wrangling and create the dataset from csv data
        A.3 bis) Visualize data (skip)
		
        A.4) Carry out some transformations to the datatset:
            - Computing the Generated Energy (using the active power since we don't have the generated energy)
			- fit missing values
			- quality check (skip)
			
		A.5) Choose which KPI to use
		
        B) Outlier correction: Find and try to fix the outliers (i.e., negative values) of the variable 'Generated Energy'
        C) Outlier correction: Tackle missing hourly observations
        D.0 - D.2) Use this points if at A.5) you have choosen PR
			Compute the daily performance ratio using a sliding window
            PARAMETERS: "sliding_window": number of previous months to consider for computing the ratio 
                        [suggested range: 12 months, "-1" is used to compute the comulative]
        D.3) Find and try to fix potential problematic perfomance ratios (i.e., values exceeding 100%) 
        E) Generate and save the diagnostic KPI (i.e., daily perfomance ratio) using a temporal graph
        
		F.0 - F.2) Use this points if at A.5) you have choosen PR with temp correction
			Compute the daily performance ratio using a sliding window
            PARAMETERS: "sliding_window": number of previous months to consider for computing the ratio 
                        [suggested range: 12 months, "-1" is used to compute the comulative]
        F.3) Find and try to fix potential problematic perfomance ratios (i.e., values exceeding 100%) 
        G) Generate and save the diagnostic KPI (i.e., daily perfomance ratio) using a temporal graph
        
    
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
2) [PRODUCTION] SAMPLE - Data Cleaning - Emi & co.ipynb
3) SAMPLE - Sampling correction - 1 hour.ipynb

----------------------------------- Use Case 1 --------------------------------
4a) [PRODUCTION] SAMPLE - UC1 - Self-organizing map [ONLY FOR Soleto 1, Soleto 2 & Galatina]
4b) SAMPLE - UC1 - Diagnostic KPI


------------ Unige -------------------------------
4c) SAMPLE-UC1-unige-Diagnostic KPI from Maps connector data

--------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------