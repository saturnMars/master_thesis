import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import _library.fault_utils as fault_utils
from matplotlib.dates import DateFormatter, MonthLocator
from math import ceil
from os import path

# -------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- PERFORMANCE RATIO -------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------

PV_NOMINAL_POWERS = {
    "SOLETO 1": 993.60, # kWp
    "SOLETO 2": 446.40, # kWp
    "GALATINA": 993.60,  # kWp
    "CANTORE": 990, # kWp
    "EMI": 990, # kWp
    "VERONE": 999, # kWp
    "FV": 80, # kWp
    "Q05_FV": 21 # kWp
}

PV_params = {
    "FV_beta": -0.31, # %/C 
    "Q05_FV_beta": -0.31 # %/C
}

def load_PV_nominal_power(system_name):
    pv_nominal_power = PV_NOMINAL_POWERS[system_name.upper()]
    return pv_nominal_power

def load_PV_params(system_name):
    beta = PV_params[system_name.upper() + "_beta"]
    return beta

# --------------------------------------------------------------
# -- FUNCTION: Reference yield (i) of a PV system relative -----
# ------- to the time period i [UNIT: peak sun hours] ----------
# --------------------------------------------------------------
def reference_yield(measured_solar_irradiance, verbose = False):
    
    # Standard Test Condition (STC): 1000 W/mq
    stc_solar_irradiance = 1000 
    
    # Reference yield
    reference_yield = measured_solar_irradiance / stc_solar_irradiance
    
    if verbose:
        print("-" * 5 + " B) REFERENCE YIELD " + "-" * 5)
        print(f"MEASURED SOLAR IRRADIANCE: {measured_solar_irradiance} W/mq")
        print(f"STC SOLAR IRRDIANCE: {stc_solar_irradiance} W/mq")
        print(f"REFERENCE YIELD: {round(reference_yield, 4)} peak sun hours\n")
    return reference_yield

# --------------------------------------------------------------
# -- FUNCTION:  Specific yield (i) of a PV system relative -----
# ------ to the time period i (UNIT: kWh/kWp) ------------------
# - i.e., Number of hours needed for the PV system to generate,- 
# ------- operating at the nominal power (kWp), the energy (kWh)-
# ---------------- generated in the time period i---------------
# --------------------------------------------------------------
def specific_yield(generated_energy, nominal_PV_power, verbose = False):
    nominal_PV_power = nominal_PV_power * 1
    specific_yield = generated_energy / nominal_PV_power
    
    if verbose:
        print("-" * 5 + " A) SPECIFIC YIELD " + "-" * 5)
        print(f"GENERATED ENERGY: {' ' if generated_energy < 100 else ''}{round(generated_energy, 1)} kWh")
        print(f"INVERTER NOMINAL POWER: {nominal_PV_power} kWp")
        print(f"SPECIFIC YIELD: {round(specific_yield, 4)} kWh/kWp\n")
    return specific_yield

# --------------------------------------------------------------
# -- FUNCTION: Performance Ratio (i) is the ratio of the ------
# --- actual and theoretically possible energy outputs ---------
# ------- relative to the time period i (UNIT: [%]) -------------
# --------------------------------------------------------------
def perfomance_ratio(specific_yield, reference_yield):
    pr_ratio = specific_yield / reference_yield
    pr_ratio = pr_ratio * 100
    return pr_ratio

def perfomance_ratio_temp_correction(specific_yield, reference_yield, beta, module_temp):
    pr_ratio = specific_yield / (reference_yield *(1-beta/100*(module_temp - 25)))
    pr_ratio = pr_ratio * 100
    return pr_ratio

def compute_single_perfomance_ratio(nominal_PV_power, measured_solar_irradiance, generated_energy, verbose = False):
    
    # A) Compute the Specific Yield (SY)
    specific = specific_yield(generated_energy, nominal_PV_power, verbose)
    
    # B) Compute the Reference Yield (RY)
    reference =  reference_yield(measured_solar_irradiance, verbose)
    
    # C) Compute the Perfomance Ratio (PR)
    perfomance = perfomance_ratio(specific, reference)
    
    if verbose:
        print("-" * 5 + " C) PERFORMANCE RATIO " + "-" * 5)
        print(f"PERFOMANCE RATIO: {round(perfomance, 2)} %\n")
    
    return perfomance

def compute_single_perfomance_ratio_temp_correction(nominal_PV_power, measured_solar_irradiance, generated_energy, beta, module_temp,verbose = False):
    
    # A) Compute the Specific Yield (SY)
    specific = specific_yield(generated_energy, nominal_PV_power, verbose)
    
    # B) Compute the Reference Yield (RY)
    reference =  reference_yield(measured_solar_irradiance, verbose)
    
    # C) Compute the Perfomance Ratio (PR)
    perfomance = perfomance_ratio_temp_correction(specific, reference, beta, module_temp)
    
    if verbose:
        print("-" * 5 + " C) PERFORMANCE RATIO " + "-" * 5)
        print(f"PERFOMANCE RATIO: {round(perfomance, 2)} %\n")
    
    return perfomance




def compute_perfomance_ratios(df, all_df_dates, date, sliding_window, nominal_power, verbose = False):
    #sliding window keeps the number of days
    if sliding_window == -1:
        starting_date = all_df_dates[0]
    # if we have one day we have to compute the KPI on that day ( we compute it at 23.59 of that day)
    else:
        # Compute the starting date (n days - 1)
        starting_date = date - pd.Timedelta(sliding_window-1, unit="days")

    if verbose:
        print("\n"+ "-" * 48 + f"\nPERIOD: FROM '{starting_date}' TO '{date}' "\
              f"({(date - starting_date).days} days)\n"  + "-" * 48 + "\n")

    # Too few observation. The sliding window for computing the KPI is unavailable. :/"
    if starting_date not in all_df_dates:
        if verbose:
            print(f"[{starting_date}] This period is not available. "\
                  f"Only {(date - all_df_dates[0]).days} days available in this period. That's so sad!")
        return None

    # A) Compute the GENERATED ENERGY (kWh) within this period
    total_energy_start = df.loc[str(starting_date), "E. totale (kWh)"].iloc[0]
    total_energy_end = df.loc[str(date), "E. totale (kWh)"].iloc[-1]
    generated_energy = total_energy_end - total_energy_start

    if verbose:
        print("\tA) Computing the GENERATED ENERGY (kWh)...\n\t" + "-" * 50)
        print(f"\t{starting_date}:  {round(total_energy_start, 2)} kWh")
        print(f"\t{date}: {round(total_energy_end, 2)} kWh")
        print(f"\tGENERATED ENERGY: {round(generated_energy, 2)} kWh "\
              f"[{round(generated_energy/1000, 2)} MWh] [{round(generated_energy/1000000, 2)} GWh] ")
        print("\t\t\t[AVG] ", round((generated_energy/(30 *sliding_window)/1000), 2), "MWh/day")

    # B) Compute the solar irradiance measured in this period [kWh/mq]
    obs_sliding_window = df.loc[str(starting_date):str(date)]
    obs_sliding_window["Irradiance (Wh/mq)"] = obs_sliding_window["Irradiance (W/mq)"] * 1 # hour
    summed_irradiance = obs_sliding_window["Irradiance (Wh/mq)"].sum()

    if verbose:
        print("\n\tB) Computing the SOLAR IRRADIANCE (kWh/mq)...\n\t" + "-" * 70)
        print(f"\tSLIDING WINDOW ({len(set(obs_sliding_window.index.date))} days): "\
              f"FROM '{obs_sliding_window.index[0].strftime('%Y-%m-%d (%H:%M)')}' "\
              f"TO '{obs_sliding_window.index[-1].strftime('%Y-%m-%d (%H:%M)')}' ")
        print(f"\tSUMMED SOLAR IRRADIANCE: {summed_irradiance} Wh/mq [{summed_irradiance/1000} kWh/mq]")
    
    # C) Compute the perfomance Ratio
    perfomance_ratio = compute_single_perfomance_ratio(nominal_PV_power = nominal_power,
                                                       measured_solar_irradiance = summed_irradiance,
                                                       generated_energy = generated_energy,
                                                       verbose = False)
    if verbose:
        print("\n\tC) Computing the Perfomance Ratio...\n\t" + "-" * 70)
        print(f"\tPerfomance Ratio: {round(perfomance_ratio, 2)} %")
    
    # FINAL OUTCOME
    daily_ratio = {
        "Starting Date" : str(starting_date),
        "Generated energy (kWh)": generated_energy,
        "Summed Solar Irradiance (Wh/mq)": summed_irradiance,
        "Perfomance Ratio (%)" : perfomance_ratio
    }
    if sliding_window != -1:
        daily_ratio["Sliding window (months)"] = sliding_window
    
    return daily_ratio

def compute_module_temp(df, all_df_dates, date, sliding_window, verbose = False):
    
    if sliding_window == -1:
        starting_date = df_dates[0]
    else:
        # Compute the starting date (n days - 1)
        starting_date = date - pd.Timedelta(sliding_window-1, unit="days")

    if verbose:
        print("\n"+ "-" * 48 + f"\nPERIOD: FROM '{starting_date}' TO '{date}' "\
              f"({(date - starting_date).days} days)\n"  + "-" * 48 + "\n")

    # Too few observation. The sliding window for computing the KPI is unavailable. :/"
    if starting_date not in all_df_dates:
        if verbose:
            print(f"[{starting_date}] This period is not available. "\
                  f"Only {(date - df_dates[0]).days} days available in this period. That's so sad!")
        return None

    ## select the slice of df related to dates of interest
    obs_sliding_window = df.loc[str(starting_date):str(date)]
    module_temp = sum(obs_sliding_window.loc[:,'Generated Energy (kWh)']* obs_sliding_window.loc[:,'Panel Temp (°C)'])/sum(obs_sliding_window.loc[:,'Generated Energy (kWh)'])
    
    return module_temp


def compute_perfomance_ratios_temp_correction(df, all_df_dates, date, sliding_window, nominal_power, beta, module_temp, verbose = False):
    
    if sliding_window == -1:
        starting_date = df_dates[0]
    else:
        # Compute the starting date (n days - 1)
        starting_date = date - pd.Timedelta(sliding_window-1, unit="days")

    if verbose:
        print("\n"+ "-" * 48 + f"\nPERIOD: FROM '{starting_date}' TO '{date}' "\
              f"({(date - starting_date).days} days)\n"  + "-" * 48 + "\n")

    # Too few observation. The sliding window for computing the KPI is unavailable. :/"
    if starting_date not in all_df_dates:
        if verbose:
            print(f"[{starting_date}] This period is not available. "\
                  f"Only {(date - df_dates[0]).days} days available in this period. That's so sad!")
        return None

    # A) Compute the GENERATED ENERGY (kWh) within this period
    total_energy_start = df.loc[str(starting_date), "E. totale (kWh)"].iloc[0]
    total_energy_end = df.loc[str(date), "E. totale (kWh)"].iloc[-1]
    generated_energy = total_energy_end - total_energy_start

    if verbose:
        print("\tA) Computing the GENERATED ENERGY (kWh)...\n\t" + "-" * 50)
        print(f"\t{starting_date}:  {round(total_energy_start, 2)} kWh")
        print(f"\t{date}: {round(total_energy_end, 2)} kWh")
        print(f"\tGENERATED ENERGY: {round(generated_energy, 2)} kWh "\
              f"[{round(generated_energy/1000, 2)} MWh] [{round(generated_energy/1000000, 2)} GWh] ")
        print("\t\t\t[AVG] ", round((generated_energy/(30 *sliding_window)/1000), 2), "MWh/day")

    # B) Compute the solar irradiance measured in this period [kWh/mq]
    obs_sliding_window = df.loc[str(starting_date):str(date)]
    obs_sliding_window["Irradiance (Wh/mq)"] = obs_sliding_window["Irradiance (W/mq)"] * 1 # hour
    summed_irradiance = obs_sliding_window["Irradiance (Wh/mq)"].sum()

    if verbose:
        print("\n\tB) Computing the SOLAR IRRADIANCE (kWh/mq)...\n\t" + "-" * 70)
        print(f"\tSLIDING WINDOW ({len(set(obs_sliding_window.index.date))} days): "\
              f"FROM '{obs_sliding_window.index[0].strftime('%Y-%m-%d (%H:%M)')}' "\
              f"TO '{obs_sliding_window.index[-1].strftime('%Y-%m-%d (%H:%M)')}' ")
        print(f"\tSUMMED SOLAR IRRADIANCE: {summed_irradiance} Wh/mq [{summed_irradiance/1000} kWh/mq]")
    
    # C) Compute the perfomance Ratio
    perfomance_ratio = compute_single_perfomance_ratio_temp_correction(nominal_PV_power = nominal_power,
                                                       measured_solar_irradiance = summed_irradiance,
                                                       generated_energy = generated_energy,
                                                       beta= beta, module_temp = module_temp,
                                                       verbose = False)
    if verbose:
        print("\n\tC) Computing the Perfomance Ratio...\n\t" + "-" * 70)
        print(f"\tPerfomance Ratio: {round(perfomance_ratio, 2)} %")
    
    # FINAL OUTCOME
    daily_ratio = {
        "Starting Date" : str(starting_date),
        "Generated energy (kWh)": generated_energy,
        "Summed Solar Irradiance (Wh/mq)": summed_irradiance,
        "Perfomance Ratio (%)" : perfomance_ratio
    }
    if sliding_window != -1:
        daily_ratio["Sliding window (months)"] = sliding_window
    
    return daily_ratio

# -------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- DATA PRE-PROCESSING -------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------
def neg_outlier_correction(df, outliers):
    for idk, outlier in enumerate(outliers):
        print(f"\nOUTLIER ({idk + 1}/{len(outliers)}): {outlier}\n" + "-" * 35)

        # 0) Get the value of the outlier
        generated_energy = df.loc[outlier, "Generated Energy (kWh)"]
        outlier_cumulative_energy = df.loc[outlier, "E. totale (kWh)"]

        # 0) Get the value of the previous observation
        idk_outlier = np.argwhere(np.array(df.index) == outlier)[0]
        idk_prev_obs = df.index[idk_outlier -1].tolist()[0]
        prev_cumulative_energy = df.loc[idk_prev_obs, "E. totale (kWh)"]
        delta_time = df.loc[idk_prev_obs, :].name - df.loc[outlier,:].name 

        # Detect the order of magnitude of the delta 
        magnitude_order = np.log10(np.abs(generated_energy)).astype("int")

        print(f"\t[OUTLIER] GENERATED ENERGY: {round(generated_energy, 4)} kWh [Order of Magnitude: {magnitude_order}]")
        print(f"\t[OUTLIER]  CUMULATIVE GENERATED ENERGY: {round(outlier_cumulative_energy, 2)} kWh")
        print(f"\t[PREVIOUS] CUMULATIVE GENERATED ENERGY: {np.round(prev_cumulative_energy, 2)} kWh "\
              f"({df.loc[idk_prev_obs, :].name.strftime('%Y-%m-%d (%H:%M)')}, {delta_time}) \n")

        # Detect whethere the cumulative counter may or may not be resetted
        resetted_counter_threshold = 4
        print("\t" + "-" * 80)

        # STRAT 0: Recalibrate the counter
        if magnitude_order > resetted_counter_threshold:
            df.loc[:idk_prev_obs, "E. totale (kWh)"] -= np.abs(generated_energy)
            print("\t[STRAT 0] Recalibrated the previous values of the cumulative counter... \n\t"\
                  "The cumulative counter for the generated energy (i.e., 'E. totale (kWh)) may have been resetted. \n")

        # STAT 1: Set the generated enegery to zero
        df.loc[outlier, "Generated Energy (kWh)"] = 0
        print("\t[STRAT 1] Setting the value to zero. The cumulative counter could be corrupted.")

        print("\t" + "-" * 80)
        print(f"\n\t[NEW VALUE] GENERATED ENERGY: ", df.loc[outlier, "Generated Energy (kWh)"].round(2), "kWH\n")

# -------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- GRAPHICAL PANEL -------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------
def retrieve_failure_periods(perfomance_ratio_values, normality_threshold, verbose = False):
    
    # 0) Check for empty periods and try to generate an artifical period (JUST ONLY FOR THE GRAPHICAL VISUALIZATION)
    original_length = len(perfomance_ratio_values)
    for idk, diff_dates in enumerate(np.diff(perfomance_ratio_values.index.tolist())):
        diff_days = diff_dates.days
        kpi = perfomance_ratio_values.iloc[idk + 1]
        date = perfomance_ratio_values.index[idk + 1]
        
        if diff_days >= 2:
            # Previous observation that is available
            prev_kpi = perfomance_ratio_values.iloc[idk]
            prev_obs_date = perfomance_ratio_values.index[idk]
            
            # Generate the artificial period 
            empty_period = (prev_obs_date + pd.Timedelta(1, unit="day"), 
                            date - pd.Timedelta(1, unit="day"))
            generated_dates = pd.date_range(start = empty_period[0], end = empty_period[1], freq = "D")
            generated_kpi = np.mean([prev_kpi, kpi])
            
            new_artificial_data = [pd.Series(data = generated_kpi, index = [date], name = perfomance_ratio_values.name) 
                                   for date in generated_dates]
            # Append the new period
            perfomance_ratio_values = perfomance_ratio_values.append(new_artificial_data)   
            
            if verbose:
                print("\nIDK:", idk)
                print("Diff:", diff_days)
                print("DATE:", date)
                print(f"KPI: {round(kpi, 2)} %")

                print(f"KPI [prev]: {round(prev_kpi, 2)} %")
                print("GENERATED DATES:", generated_dates)
                print(f"GENERATED KPI: {round(generated_kpi, 2)} %")
    if verbose:            
        print("DATE ADDED ARTIFICIALLY: ", len(perfomance_ratio_values) - original_length)  
    
    # A) Highlight days below the threhsold
    cond = perfomance_ratio_values.round(decimals = 0) <= normality_threshold
    dates_below_thresholds = sorted(perfomance_ratio_values[cond].index.tolist())
   
    
    # B) Extract the periods from the list of dates
    period_cutOff = [idk_date for idk_date, diff_dates in enumerate(np.diff(dates_below_thresholds)) 
                     if diff_dates.days > 2]
    if verbose: 
        print(f"DATES BELOW THRESHOLD ({normality_threshold})")
        print('\n'.join([f"{idk}) {date}" for idk, date in enumerate(dates_below_thresholds)]))
        print(period_cutOff)
    
    if len(period_cutOff) > 0:
        period_below_thresholds = []        
        start = 0
        for idk_period in range(len(period_cutOff) + 1):

            if idk_period < len(period_cutOff):
                end = period_cutOff[idk_period]
            else:
                end = len(dates_below_thresholds) - 1
                
            period = (dates_below_thresholds[start].strftime("%Y-%m-%d"), 
                      dates_below_thresholds[end].strftime("%Y-%m-%d"),
                      dates_below_thresholds[end] - dates_below_thresholds[start] + pd.Timedelta(1, unit="day"))
            period_below_thresholds.append(period)
            start = end + 1
    else:
        if len(dates_below_thresholds) > 0:
            period_below_thresholds = [(dates_below_thresholds[0].strftime("%Y-%m-%d"), 
                                       dates_below_thresholds[-1].strftime("%Y-%m-%d"),
                                       dates_below_thresholds[-1] - dates_below_thresholds[0] + pd.Timedelta(1, unit="day"))]
        else: 
            period_below_thresholds = []
    return period_below_thresholds
    
# ------------------------------------------------------------------------------
# --------- FUNCTION: Generate general graphical panel ------
# ------------------------------------------------------------------------------
def generateGraphicalPanel(n_col, n_row, system_name, sub_title):
    
    # Create the graphical figure
    fig, axes = plt.subplots(n_row, n_col, figsize= (20 * n_col, 5 * n_row), facecolor = "white")

    if axes.ndim == 1:
        axes = np.array([axes])

    # Figure title
    fig.suptitle(f"[{system_name.upper()}] " + r"$\bf{" + "Perfomance Ratio".replace(" ", "\\ ") + "}$", 
                 fontsize = 50, color = "dimgray", y = 1.01)
    fig.supxlabel(sub_title,  fontsize = 30, color = "dimgray", y = 0.9 if n_row == 2 else 0.952) #y = 
    
    return fig, axes

# ------------------------------------------------------------------------------
# --------- FUNCTION: Generate the historical graph with the inverter data ------
# ------------------------------------------------------------------------------
def generate_historicalGraphs(df, graph, inv_name, normality_threshold, plot_moving_average_lines = True, 
                             visualize_under_threshold_period = True):
    df.index =  pd.to_datetime(df.index)
    num_days = (df.index.date[-1] - df.index.date[0]).days
    
    # DATA (for the graph)
    perfomance_ratio_values = df["Perfomance Ratio (%)"]
    
    # Retrieve the periods with failures events
    if visualize_under_threshold_period:
        period_below_thresholds = retrieve_failure_periods(perfomance_ratio_values, normality_threshold)
        print("\n"+ "-" * 25 + f" {inv_name} " + "-" * 25)
        print("\n".join([f"PERIOD {idk + 1}: FROM '{period[0]}' TO '{period[1]}' ({period[2].days} days)" 
                        for idk, period in enumerate(period_below_thresholds)]))
    
    # Moving average (for the PR values) - 1 week
    num_weeks = 1
    perfomance_ratio_moving_average_1w = df["Perfomance Ratio (%)"].rolling(window = 18 * 7 * num_weeks, min_periods=1).mean()
    
    # Moving average (for the PR values) - 4 week
    num_weeks = 4
    perfomance_ratio_moving_average_4w = df["Perfomance Ratio (%)"].rolling(window = 18 * 7 * num_weeks, min_periods=1).mean()
    
    # PLOT THE DATA
    # a) Raw values
    sns.lineplot(x = perfomance_ratio_values.index, y = perfomance_ratio_values, ax = graph,
                 lw = 2, color = "forestgreen", label = "Perfomance Ratio", zorder = 5)
    
     # d) Mean
    graph.axhline(perfomance_ratio_values.mean(), 
                  color ="seagreen", linestyle = "--", lw = 2, alpha = 0.4, zorder = 1,
                  label=f"Mean ({round(perfomance_ratio_values.mean(), 2)} %)")
    
    if plot_moving_average_lines:

        # b1) Moving average - 1 Week
        sns.lineplot(x = perfomance_ratio_moving_average_1w.index, y = perfomance_ratio_moving_average_1w, ax = graph, 
                     lw = 4, color = "mediumseagreen", alpha = 0.4, label = f"Moving Average (~ 1 weeks)", zorder = 3)
        
    # c) Warning threshold 
    graph.axhline(normality_threshold, 
                  color ="crimson", linestyle = "-", lw = 2, alpha = 0.4, zorder = 1,
                  label=f"Normality threshold ({normality_threshold} %)")
    
    if visualize_under_threshold_period:

        # D) Problematic period (text + highlighted period)
        for idk_period, (starting_date, ending_date, time_elapsed) in enumerate(period_below_thresholds):

            # X POSITION
            if time_elapsed.days >= 20:
                text_pos_x = (pd.to_datetime(starting_date) + pd.Timedelta(time_elapsed.days//2 - 7, unit="days")).date()
                pr_values = perfomance_ratio_values[starting_date:text_pos_x]
            else:
                text_pos_x = (pd.to_datetime(starting_date) - pd.Timedelta(13, unit="days")).date()
                pr_values = perfomance_ratio_values[text_pos_x:starting_date]

            if text_pos_x < df.index.date[0]:
                text_pos_x = df.index.date[0]
            if text_pos_x + pd.Timedelta(5, unit="days") > df.index.date[-1]:
                text_pos_x = df.index.date[-1]

            min_pr_value = np.nanmin(pr_values)

            if not np.isnan(min_pr_value):
                text_pos_y = min_pr_value - 20
            else:
                text_pos_y = perfomance_ratio_values.mean() + 10

            if idk_period != 0:
                prev_period = period_below_thresholds[idk_period - 1][1]
                temporal_distance = (text_pos_x - pd.to_datetime(prev_period).date()).days

                if temporal_distance <= 30:
                    text_pos_y = text_pos_y - (idk_period*10)

            # GRAPHICAL ELEMENT: Text
            elapsed_days = time_elapsed.days if time_elapsed.days != 0 else 1
            white_spaces = 2 if elapsed_days <= 10 else (1 if elapsed_days <= 100 else 0)
            text = starting_date + "\n" + (' ' * white_spaces) + f"({elapsed_days} days)"
            graph.text(x = text_pos_x, y = text_pos_y, s =  text,
                       fontsize = 10, color = "black", bbox=dict(facecolor='white', edgecolor='red', boxstyle='round, pad=0.5'),
                       zorder = 7)

            # GRAPHICAL ELEMENT: Vertical Lines for the starting and ending dates
            graph.axvline(pd.to_datetime(starting_date), color ="crimson", linestyle = "--", lw = 2, alpha = 0.4, zorder = 1)
            graph.axvline(pd.to_datetime(ending_date), color ="crimson", linestyle = "--", lw = 2, alpha = 0.4, zorder = 1)

            # GRAPHICAL ELEMENT: Area of the problematic period
            y_values = [102 for j in range(time_elapsed.days)]
            x_values = pd.date_range(pd.to_datetime(starting_date), pd.to_datetime(ending_date))
  
            if idk_period == 0:
                graph.fill_between(x_values, y_values, color="crimson", alpha=0.05, label = "Problematic periods")
            else:
                graph.fill_between(x_values, y_values, color="crimson", alpha=0.05)

    # Set some graphical parameters: Title
    graph.set_title(f"Inverter N°{inv_name[-1]}", fontsize = 40, pad = 10)
    
    # Set some graphical parameters: Y Limit
    graph.set_ylim(ymin = 0, ymax = 102)
    graph.set_xlim(xmin = df.index[0] - pd.Timedelta(2, unit="days"), xmax = df.index[-1] + pd.Timedelta(1, unit="days"))
    
    # Set some graphical parameters: axes labels
    graph.set_ylabel("Perfomance Ratio [%]", fontsize = 15, color= "dimgrey")
    
    # Set some graphical parameters: Tick parameters
    if num_days >= 3000:
        interval_months = 4
    if num_days >= 800:
        interval_months = 3
    elif num_days >= 500:
        interval_months = 2
    else:
        interval_months = 1
    graph.xaxis.set_major_locator(MonthLocator(interval = interval_months)) 
    graph.xaxis.set_minor_locator(MonthLocator(interval = 1))
    graph.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    
    # Set some graphical parameters: Tick visual parameters
    graph.tick_params(axis='both', which='major', labelsize = 15, direction = "in", length = 7)
    graph.tick_params(axis = "y", which="major", right = True)
    graph.tick_params(axis = "x", which='major', labelsize = 18, pad = 5, direction = "out", length = 8,
                      grid_linestyle = "-.", grid_alpha = 0.5, width = 1)
    graph.tick_params(axis = "x", which='minor', grid_linestyle = "-.", grid_alpha = 0.2, length = 0)
    
    # Set some graphical parameters: General graph
    graph.grid(which = "both")
    graph.legend(fontsize = 11, shadow = True, borderpad = 0.7, borderaxespad = 1, loc = "lower left")

    return period_below_thresholds

# -------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- FAILURE EVENT CORRELATION -----------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------
def create_failure_events_matrix(fault_df, df, inv_name, verbose = False):
    print("\n" + "-" * 50)
    print("A) Computing the FAILURE EVENT matrix")
    print("-" * 50)
    
    df_dates = np.array(sorted(set(df.index.date)))
    print(f"[{inv_name}] DATA AVAILABLE: {len(df_dates)} days")
    
    # Retrieve the timestamps concerning failure events on the inverter
    inv_fault_events, inv_unique_faults = fault_utils.find_fault_observation(fault_df, df, inv_name, 
                                                                             include_faults_notRelatedToInverters = False, 
                                                                             tolerance = [True, False], verbose = False)    
    if verbose:
        for idk, (fault_type, fault_cause, fault_message, fault_period) in enumerate(inv_unique_faults):
            print("-" * 100 + f"\nFAULT {idk +1}: ({fault_type}) - [{fault_cause}] {fault_message}\n" + "-" * 100 + 
                  f"\nFROM '{pd.to_datetime(fault_period[0]).strftime('%Y-%m-%d (%H:%M)')}' TO "
                  f"'{pd.to_datetime(fault_period[1]).strftime('%Y-%m-%d (%H:%M)')}' --> "\
                  f"(DURATION: {pd.to_datetime(fault_period[1]) - pd.to_datetime(fault_period[0])})\n")
    
    # Retrieve the failure dates
    failure_event_dates = np.array(sorted(set(timestamp.date() for timestamp in inv_fault_events.keys())))
    print(f"[{inv_name}] FAILURE EVENTS: {len(failure_event_dates)} days")
    
    # Create the matrix 
    matrix = np.zeros(shape = len(df_dates), dtype = np.int16)
    for idk, failure_event_date in enumerate(failure_event_dates):
        idk_date = np.argwhere(df_dates == failure_event_date)[0]
        matrix[idk_date] = 1
        print(f"\t--> FAILURE EVENT {' ' if idk < 9 else ''}{idk +1}/{len(failure_event_dates)}:", failure_event_date)
        
    check_non_zeros = np.nonzero(matrix)[0]
    if len(check_non_zeros) != len(failure_event_dates):
        print(f"\n[ISSUE] 'Non zero values': {len(check_non_zeros)} days "\
              f"!= 'Failure events': {len(failure_event_dates)} days")
 
    return df_dates, matrix

def create_perfomance_ratio_matrix(kpi_values, threshold, simple_strat= False, sliding_window = 7, verbose = False):
    print("\n" + "-" * 50)
    print("B) Computing the PERFOMACE RATIO matrix")
    print("-" * 50)
    
    # Create the empty matrix
    all_kpi_dates = np.array(list(set(kpi_values.index.date)))
    matrix = np.zeros(shape = all_kpi_dates.shape[0], dtype = np.int16)
         
    # Set the value '1' to the dates with a KPI under the threshold
    if simple_strat:
        
         # Selecting the timestamps under the numerical threshold
        dates_below_thresholds = kpi_values[kpi_values <= threshold].index.date.tolist()
        print(f"DATES UNDER THE THRESHOLD: {len(dates_below_thresholds)} days "\
          f"({round((len(dates_below_thresholds)/len(kpi_values))*100, 2)} %)")
    
        if len(under_threshold_dates) == 0:
            print("No KPI values under this threshold. That's nice.")
        else:
            print(f"KPI under this threshold: {len(under_threshold_dates)} days.")
        
        for idk, date in enumerate(under_threshold_dates):
            idk_date = np.argwhere(all_kpi_dates == date)[0]
            matrix[idk_date] = 1
            print(f"--> {' ' if idk < 9 else ''}({idk +1})", date)
        
        # Check its validity
        check_non_zeros = np.nonzero(matrix)[0]
        if len(check_non_zeros) != len(under_threshold_dates):
            print(f"\n[ISSUE] 'Non zero values': {len(check_non_zeros)} days "\
                  f"!= 'Failure events': {len(under_threshold_dates)} days")
    else:
        print(f"Computing the matrix for all the {len(all_kpi_dates)} days")
        for idk, date in enumerate(all_kpi_dates):
            
            # Create the period of the sliding window (day: day - sliding window)
            starting_date = date - pd.Timedelta(sliding_window, unit = 'days')
            
            # Retrieve the KPi scores of this period
            sliding_window_df = kpi_values[starting_date:date]
            
            # Find the observations having their KPi score under the theshold
            under_threshold_obs = sliding_window_df[sliding_window_df <= threshold]
            
            if len(under_threshold_obs) > 0:
                matrix[idk] = 1

                if verbose:
                    print(f"\nDATE {idk + 1}/{len(all_kpi_dates)}: '{date}' "\
                      f"(-{(date - starting_date).days} days, '{starting_date}')")
                    print(f"Observations having their KPI score under the threshold ({threshold} %): {len(under_threshold_obs)}")
                    print('--> ' + '\n--> '.join([timestamp.strftime('%Y-%m-%d') for timestamp in under_threshold_obs.index]))
                    print("\n" + '-' * 40 + f"\n[STAT] Setting the day '{date}' to 1\n" + '-' * 40)
            
    return matrix

def generate_heatmap(corr_scores, system_name, saving_path, sliding_window):
    sns.set_theme(style="whitegrid")

    inv_names = list(corr_scores[list(corr_scores.keys())[0]].keys())
    
    # Visual panel dimensions
    n_columns = 2 if len(inv_names) != 1 else 1
    n_rows = ceil(len(inv_names)/n_columns)
    
    # Create the figure
    fig, axes = plt.subplots(n_rows, n_columns, squeeze=False,
                             figsize=(14 * n_columns, 9 * n_rows),
                             tight_layout= {"w_pad": 4, "h_pad":3})
    dynamic_title = "Correlation Matrices" if len(inv_names) > 1 else "Correlation Matrix"
    fig.suptitle(f"[{system_name.upper()}] " + r"$\bf{" + dynamic_title.replace(" ", "\\ ") + "}$", 
                 fontsize = 70, color = "dimgray", y = 1)


    # Make invisible the last graph in extra subplots
    if n_rows > 1:
        extra_subplots = (axes.shape[0] * axes.shape[1]) - len(inv_names)
        if extra_subplots > 0:
            for row in range(1, extra_subplots + 1):
                axes[-row, -1].set_visible(False)

    idk_row = 0
    idk_column = 0
    
    # A sub-plot for each inverter
    for inv_name in inv_names:
        
        # Create the list of values to plot
        list_value =  [pd.Series(corr_scores[fault_profile][inv_name], name = fault_profile.replace('_', "\n")) 
                       for fault_profile in corr_scores.keys()]
        df = pd.DataFrame(list_value)
        #display(df)
        
        # Generate the heatmap
        sns.heatmap(df, ax = axes[idk_row, idk_column],
                    annot = True, fmt =".1g", 
                    annot_kws = {"fontsize":15},
                    cmap = "coolwarm", center = 0, vmin = -1, vmax=1,
                    cbar_kws = {"shrink": .5, "ticks": np.arange(-1, 1.5, step = .5), "pad":0.06}, 
                    linewidths = 0.1, linecolor='lightgrey')
        
        # Color bar
        cbar = axes[idk_row, idk_column].collections[0].colorbar
        cbar.ax.set_ylabel("Correlation coefficient", fontsize=20, color='dimgray', labelpad = 10) # labelpad = -80
        cbar.ax.tick_params(axis='y', width=2, length=5, labelsize =15, 
                            labelleft = True, labelright = False,
                            direction = "inout", color="white")

        # Set title
        axes[idk_row, idk_column].set_title(f"INVERTER N°{inv_name[-1]}", fontsize=55, color= "dimgrey", pad=20)
        
        # Set title axes
        axes[idk_row, idk_column].set_xlabel("Thresholds of the 'Perfomance Ratio'", 
                                             fontsize = 20, fontweight='bold', labelpad = 15)
        axes[idk_row, idk_column].set_ylabel("Profiles of failure events", 
                                             fontsize = 20, fontweight='bold', labelpad = 15)
        
        # Set ticks
        axes[idk_row, idk_column].tick_params(axis='both', which='major', labelsize=20)
        axes[idk_row, idk_column].tick_params(axis='y', which='major', labelrotation = 0)
        
        # Change row/column counters
        if idk_column + 1 < axes.shape[1]:
            idk_column += 1
        else:
            idk_column = 0
            idk_row += 1
        
    name_to_save = f"PerfomanceRatio_correlation_{sliding_window}daysSL.png"
    path_to_save = path.join(saving_path, name_to_save)
    fig.tight_layout(pad = 2)
    fig.savefig(path_to_save, bbox_inches='tight', pad_inches = 1)
    fig.show()