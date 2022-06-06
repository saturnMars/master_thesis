from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

class MultidimPolynomialModel:

    def __init__(self, degree: int):
        self._degree = degree
        self._model = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        if len(X.shape) != 2 or len(Y.shape) != 1 or X.shape[0] != Y.shape[0]:
            raise Exception("Bad shapes in input data: X: {}, Y: {}".format(X.shape, Y.shape))

        # build polynomial features of required degree from data
        feats = PolynomialFeatures(self._degree, interaction_only=False, include_bias=False).fit(X)

        # compute powers of data points by transforming the whole training set
        t_X = feats.transform(X)

        # fit a linear model against powers of input variables
        linmod = LinearRegression(fit_intercept=True).fit(t_X, Y)

        # save the model
        self._model = {"feats": feats, "linmod": linmod}

    def predict(self, X: np.ndarray, force_non_negative_values = False, force_zero_with_low_irr = False, force_integer_values = False) -> np.ndarray:
        if self._model is None:
            raise Exception("Model not initialised")

        # compute powers of data points
        t_X = self._model["feats"].transform(X)

        # use the linear model for predictions on powers of input data
        predictions = self._model["linmod"].predict(t_X)

        if force_non_negative_values:
            predictions[predictions < 0] = 0

        if force_zero_with_low_irr:
            minimum_irr_value = 50
            irr_values = X[:, 1] 
            obs_under_minimum = np.argwhere(irr_values <= minimum_irr_value)
            predictions[obs_under_minimum] = 0

        if force_integer_values:
            predictions = predictions.astype(int)
        
        return predictions

    def load(self, filename: str):
        self._model = joblib.load(filename)

    def save(self, filename: str):
        joblib.dump(self._model, filename)

    def get_model(self):
        return self._model['linmod']

    def get_features(self):
        return self._model['feats']


# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

def fit_best_polynomial_model(inputs, output, valid_x_data, valid_y_data, degree_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                              force_non_negative_values = False, force_zero_with_low_irr = False, force_integer_values = False, 
                              valid_metrics = 'rmse', verbose = True):
    # INPUT:
    # degree_list: array of degree for polynom to be fitted
    # OUTPUT:
    # model_best: best MultidimPolynomialModel fitted
    # degree_best: degre of best model fitted
    
    best_score = np.iinfo(np.int32).max
    print(f"LOSS FUNCTION: {valid_metrics.upper()} error")
    for degree in degree_list:
        print(f"\n--> Computing a polynomial model with a {degree}° degree...\n" + "-" * 60)
        
        # Fit the model 
        model = MultidimPolynomialModel(degree)
        model.fit(X = inputs, Y = output)
        
        # Predict the values
        valid_pred = model.predict(valid_x_data, force_non_negative_values, force_zero_with_low_irr)
        
        # Compute the metrics
        print('-' * 20, 'VALIDATION METRICS', '-' * 20)
        mae, rmse, wape, pos_wape, neg_wape = compute_metrics(valid_y_data, valid_pred)

        # Compare the perfomance with the best recored one
        if valid_metrics == 'rmse':
            reference_metrics = rmse
        elif valid_metrics == 'wape':
            reference_metrics = wape
        elif valid_metrics == 'mae':
            reference_metrics = mae
        elif valid_metrics == 'neg_wape':
            reference_metrics = neg_wape
        elif valid_metrics == 'pos_wape':
            reference_metrics = pos_wape

        if reference_metrics < best_score:
            best_score = reference_metrics
            model_best = model
            degree_best = degree
            print("\n\t\t\t [THE BEST, so far...]")

    if verbose:
        print("\n" + "-" * 50)
        print(f"BEST FITTED MODEL (degree: {degree_best}):")
        print(f"--> Terms ({len(model_best.get_features().get_feature_names_out())}):", 
               ', '.join(model_best.get_features().get_feature_names_out()))
        print(f"--> Coeff ({len(model_best.get_model().coef_)}):", ', '.join([str(np.round(item, 6)) 
                                                                           for item in model_best.get_model().coef_]))
        print("--> Intercept:", round(model_best.get_model().intercept_, 4))
        print("-" * 50)
    
    return model_best, degree_best

def split_data(x_data, y_data, test_dim):
    train_x_data, test_x_data, train_y_data, test_y_data = train_test_split(x_data, y_data, test_size = test_dim, 
                                                                            shuffle = True, random_state = 101)
    print("\n" + "-" * 50)
    print(f"DATA: {x_data.shape[0]} obs. --> TRAIN: {train_x_data.shape[0]} obs. || "\
          f"TEST ({int(test_dim*100)}%): {test_x_data.shape[0]} obs.")
    print("-" * 50)
    return train_x_data, train_y_data, test_x_data, test_y_data

# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
def compute_metrics(actual, predicted):
    #predicted = np.array(predicted)
    #actual = np.array(actual)

    # Delta
    errors = (actual - predicted)
    abs_errors = np.abs(errors)
    
    # Metrics
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(abs_errors)
    wape = mae / np.mean(np.abs(actual))

    # Wape with signs
    negative_errors = errors[errors < 0]
    positive_errors = errors[errors >= 0]
    neg_wape = np.mean(np.abs(negative_errors))/np.mean(np.abs(actual))
    pos_wape = np.mean(np.abs(positive_errors))/np.mean(np.abs(actual))

    print('--> MAE:', np.round(mae, 4), ' || RMSE:', np.round(rmse, 4), 
         f'\n--> WAPE: {np.round(wape * 100, 2)} %', f' --> [Pos] WAPE: {np.round(pos_wape * 100, 2)} %', 
         f' || [Neg] WAPE: {np.round(neg_wape * 100, 2)} %')    
    return mae, rmse, wape, pos_wape, neg_wape

def generate_sub_graph(fig, idk_plot, inputs, target_values, predicted_values, var_name, pov_elev = 5, pov_angle = -30,
                       visualize_actual_points = False, visualize_surface = True, hue_values = None):   
    
    if not visualize_surface and not visualize_actual_points:
        print("Hey, what are you doing? You MUST PICK at least one type of visualization!")
        return
        
    main_ax = fig.add_subplot(1, 2, idk_plot + 1, projection='3d')
    
    x_values = inputs[:, 0]
    y_values = inputs[:, 1]
    
    # Graph
    if visualize_surface:
        surf = main_ax.plot_trisurf(x_values, y_values, predicted_values, label = f"Predicted values",
                                cmap = 'cividis', shade = True, linewidth = 1, edgecolor = 'none')
    
        # Fix a library bug
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d
    
    # 3D: Point of view
    main_ax.view_init(elev=pov_elev, azim=pov_angle)
    
    if visualize_actual_points:
        if hue_values is not None:
            col_normalize = None
            cmap_name = 'viridis'
      
            unique_hue_values = sorted(list((set(hue_values))))
            if len(unique_hue_values) == 2:
                dim_class_a = round((len(np.argwhere(hue_values == unique_hue_values[0]))/len(hue_values))*100, 2)
                dim_class_b = round((len(np.argwhere(hue_values == unique_hue_values[1]))/len(hue_values))*100, 2)
                
                if (-10 in unique_hue_values) and (10 in unique_hue_values):
                    unique_hue_values = ["TRAIN", "TEST"]
                if (99 in unique_hue_values) and (101 in unique_hue_values):
                    unique_hue_values = ["Nominal obs.", "Failure events"]
                    
                colors = {f'CLASS: {unique_hue_values[0]} ({dim_class_a}%)':'black', 
                          f'CLASS:  {unique_hue_values[1]} ({dim_class_b}%)': 'yellow'}
            elif len(unique_hue_values) > 2:
                cmap_name = 'seismic'
                dim_class_pos = round((len(np.argwhere(hue_values >= 0))/len(hue_values))*100, 2)
                dim_class_neg = round((len(np.argwhere(hue_values < 0))/len(hue_values))*100, 2)
                colors = {f'Positive residuals ({dim_class_pos}%)': 'red', 
                          f'Negative residuals ({dim_class_neg}%)':'blue'}
                col_normalize = TwoSlopeNorm(vmin = hue_values.min(), vcenter = 0, vmax = hue_values.max())
            else:
                dim_class = round((len(np.argwhere(hue_values == unique_hue_values[0]))/len(hue_values))*100, 1)
                colors = {f'CLASS: {unique_hue_values[0]} ({dim_class}%)':'yellow'}

            main_ax.scatter(x_values, y_values, target_values, label = f"Target values", 
                            s = 2 ** int(np.round(np.log10(len(x_values)))), c = hue_values, cmap = cmap_name, norm = col_normalize)
            
            # Add a custom legend
            handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8,
                              label=key_name) 
                       for key_name, color in colors.items()]
            leged_pos = 'upper left' if idk_plot == 0 else 'upper right'
            main_ax.legend(title=f'Observations \n    (~ {int(len(x_values)/1000)}K)', handles=handles, 
                           loc = leged_pos, title_fontsize = 20, fontsize=13, 
                           markerscale = 2, shadow = True)
        else:
            main_ax.scatter(x_values, y_values, target_values, label = f"Target values", 
                            alpha = 0.5, marker = "x", s=20, color = "grey")
   
    # Graphical parameters
    main_ax.set_title(f'{var_name} = f(Amb. Temp., Solar Irr.)', fontsize = 30, y = 1.1)
    main_ax.set_xlabel('Temperature (°C)', fontsize = 20, labelpad = 10)
    main_ax.set_ylabel('Solar irradiance (w/mq)', fontsize = 20, labelpad = 10)
    main_ax.set_zlabel(var_name, fontsize = 20, labelpad = 10)
    #main_ax.legend()
    #main_ax.invert_yaxis()
    #main_ax.invert_xaxis()

def find_same_amb_conditions(df_row, theorethical_maxiumum_values, step_ambiental_temp, step_solar_irradiance):
    for idk, amb_cond in enumerate(theorethical_maxiumum_values):
        
        # Conditions
        temp_cond = amb_cond['amb_temp'] <= df_row['Cell Temp (°C)'] <= amb_cond['amb_temp'] + step_ambiental_temp
        irr_cond = amb_cond['solar_irr'] <= df_row['Irradiance (W/mq)'] <= amb_cond['solar_irr'] + step_solar_irradiance

        if temp_cond and irr_cond:
            return idk
        else:
            continue
    return -1