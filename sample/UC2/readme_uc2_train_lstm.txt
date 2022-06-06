------------------------------- UC2: LSTM ---------------------------
--> ./SAMPLE - UC2 - LSTM with generalized failures logs.ipynb


--------------------------------------------------------------------------------------------- 
------------- SAMPLE - UC2 - LSTM with generalized failures logs ------------------------- 
---------------------------------------------------------------------------------------------

INPUT:  
	1) [Alarms] Dataset generated using 'SAMPLE - UC2 - Create the failure events dataframe'
	2) [Inverter] Dataset generated using 'SAMPLE - Sampling correction - 1 hour'


PARAMS/FLAGS
1) [ONLY FOR DEBUG] Turnaround to use the CPU (instead of the GPU)

2) [1.1) Selecting the PV system] --> 'system_name'

3) [Merge all the inverter data] --> 'merge_inverter_data'
	--> Flag to enable the modality of put all the dataset of each inverter into a one enormous dataset 

4) [Logistic LSTM] --> 'grid_search' & 'pre_trained'
--> IF both are set to FALSE, the notebook will run a simple train using the parameter included in the cell 'Hyperparameters'

5) [Logistic LSTM: Train the LSTM]: Hyperparameter used to carry out the training phase
--> There are three different scenarios:
	a) Use an average configuration for all the inverter (avg_config = True)
	b) Use a configuration for the grouped inverter data (merge_inverter_data = True)
	c) Use a different configuration for all the inverters 
		(avg_config = False & merge_inverter_data = False)

OUTPUT: Folder ./SAMPLE/data/{system_name}/{system_name}/UC2 - LSTM/...
--> 'Trained models - Generalized Version'
--> 'Test metrics - Generalized Version'
--> [Grid search - Generalized Version]