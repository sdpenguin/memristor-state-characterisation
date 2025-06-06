
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from tqdm import tqdm

from .models import get_f_proposed, get_f_mss, get_f_mss_mod
from .functions import preprocess_data

NUMBER = 7 # The number of points for each dimension of the grid (inclusive of the endpoints)
ZOOM_FACTOR = 4 # The reduction in (logarithmic) size of the grid width at each iteration
RANGE_LOWER_POWER_LIMIT = -12
RANGE_UPPER_POWER_LIMIT = 2.5 # ~314
RANGE_INITIAL_LOWER_POWER = -6
RANGE_INITIAL_UPPER_POWER = 1
CLUSTER_NUMBER = 8 # The number of initial clusters for KMeans

colors = [
'black',
'coral',
'blue',
'blueviolet',
'brown',
'cadetblue',
'chartreuse',
'chocolate',
'cornflowerblue',
'crimson',
'cyan',
'darkcyan',
'darkblue',
'darkgoldenrod',
'darkgray',
'darkgreen',
'darkgrey',
'darkkhaki',
'darkmagenta',
'darkolivegreen',
'darkorange',
'darkorchid',
'darkred',
'darksalmon',
'darkseagreen',
'darkslateblue',
'darkslategray',
'darkslategrey',
'darkturquoise',
'darkviolet',
'deeppink',
'deepskyblue',
'dimgray',
'dimgrey',
'dodgerblue',
'firebrick',
'forestgreen',
'fuchsia',
'gainsboro',
'gold',
'goldenrod',
'gray',
'green',
'greenyellow',
]
colors = np.repeat(colors, 4)

def zoom(centers, grid_size):
    limits = []
    for center in centers:
        factor = grid_size / ZOOM_FACTOR
        lower = max(center - factor, RANGE_LOWER_POWER_LIMIT) # We lower bound the minimum value so that it doesn't immediately get too close to 0.
        upper = min(center + factor, RANGE_UPPER_POWER_LIMIT) # We upper bound the parameters as well.
        limits.append([np.power(10.0,lower), np.power(10.0,upper)])

    return limits[0], limits[1], limits[2], limits[3], limits[4]

def get_extension(model):
    if model == 0:
        return ""
    elif model == 1:
        return "_GMMS"
    elif model == 2:
        return "_GMMS_mod"

def range_generator(a1, a2, b1, b2, Gp, number=NUMBER):
    # ~10000 parameter septets to try
    a1s = np.logspace(np.log10(a1[0]), np.log10(a1[1]), num=number)
    a2s = np.logspace(np.log10(a2[0]), np.log10(a2[1]), num=number)
    b1s = np.logspace(np.log10(b1[0]), np.log10(b1[1]), num=number)
    b2s = np.logspace(np.log10(b2[0]), np.log10(b2[1]), num=number)
    Gps = np.logspace(np.log10(Gp[0]), np.log10(Gp[1]), num=number)
    return a1s, a2s, b1s, b2s, Gps

def sort_by_error(septets, errors):
    indices = np.argsort(errors)
    sorted_septets = septets[indices]
    sorted_errors = errors[indices]
    return sorted_septets, sorted_errors

def get_parameter_generators(centres, grid_size, number=NUMBER):
    a1r, a2r, b1r, b2r, Gpr = zoom(np.log10(centres), grid_size)
    a1s, a2s, b1s, b2s, Gps = range_generator(a1r, a2r, b1r, b2r, Gpr, number=number)
    return a1s, a2s, b1s, b2s, Gps

# File read
def state_characterisation_file_reader(files, period):

    file_list = []
    for filename in files:
        with open(filename, 'r') as file:
            file_list_current = yaml.safe_load(file)
            file_list = file_list + file_list_current

    voltage_list = []
    current_list = []
    sigma_list = []
    center_list = []

    # Fit the KMeans algorithm to all voltages
    cluster = KMeans(n_init=CLUSTER_NUMBER)
    all_current = np.array([])
    for file in file_list:
        csv = pd.read_csv(file, header=1)
        voltage, current = preprocess_data(csv, period, optimize_offset=True)
        all_current = np.concatenate([all_current, current], axis=0)
    cluster = cluster.fit(all_current.reshape(-1, 1))

    for i, file in enumerate(file_list):
        csv = pd.read_csv(file, header=1)
        voltage, current = preprocess_data(csv, period, optimize_offset=True)

        labels = cluster.predict(current.reshape(-1, 1))

        # Number of elements in the clusters
        # Note that we concatenate with cluster.labels_ in order to ensure that every cluster has at least one entry initially
        cluster_num = np.unique(np.concatenate([labels, cluster.labels_]), return_counts=True)[1]
        cluster_num = cluster_num - 1 # Now subtract the extra entry

        # Data length
        data_length = len(current)

        # Declaration
        sigma = []

        # Populate sigmas (based on cluster counts)
        for j in range(data_length):
            # What is the label of the jth element
            current_label = labels[j]
            # How many elements there are in that cluster
            clus_num = cluster_num[current_label]
            # Sigma is the square root of the number of elements in that cluster
            # Therefore larger clusters have higher sigma
            sigma.append(np.sqrt(clus_num))

        sigma = np.array(sigma)

        center = []

        # Populate centers (based on corresponding cluster centres)
        for j in range(data_length):
            # What is the label of the xth element
            current_label = labels[j]
            # Center value for the corresponding value
            center.append(cluster.cluster_centers_[current_label][0])

        sigma_list.append(sigma)
        voltage_list.append(voltage)
        current_list.append(current)
        center_list.append(center)

    assert len(voltage_list) == len(current_list) == len(sigma_list) == len(file_list)

    return voltage_list, current_list, sigma_list, center_list


# State characterisation core functionality
def state_characterisation_core(f, voltage_list, current_list, center_list, sigma_list=None, plot_type='vi', suppress_output=False, params=None):
    '''
        f: The function to be fit. Should take only x (the state variable) as a parameter and other parameters should be hard coded.
        params: A list of the 6 parameters hard coded into f. We pass these for the purposes of debugging, so they can be printed if the fitting algorithm fails.
    '''
    all_Gp = []
    #all_Gp2 = []

    if sigma_list is None:
        sigma_list = [None]*len(current_list)

    errors = []
    errors2 = []
    errors3 = []
    errors4 = []
    for i in range(len(voltage_list)):
        voltage = voltage_list[i]
        current = current_list[i]
        sigma = sigma_list[i]
        center = center_list[i]

        sort_indices = np.argsort(voltage)
        voltage = voltage[sort_indices]
        current = current[sort_indices]
        if sigma is not None:
            sigma = sigma[sort_indices]

        voltage_subset = voltage
        current_subset = current
        center_subset = center

        def f_super(xdata, Gp):
            voltage, current, center = xdata
            pred_current = f(voltage, Gp)
            return abs((current - pred_current)) / (np.abs(center) + 1e-9)  # Relative error
        try:
            parameters_exp = curve_fit(f_super, [voltage_subset, current_subset, center_subset], np.zeros(sigma.shape), sigma=sigma, p0=[0.1], bounds=[[1e-8], [10.0]], check_finite=True)[0]
        except Exception as e:
            print("Curve fit failed with parameters:", params)
            raise e

        all_Gp.append(parameters_exp[0])

        fit_current_subset = f(voltage, all_Gp[-1])#, all_Gp2[-1])
        sigma = 1
        error = np.mean(((current - fit_current_subset)/sigma/1e5)**2) # MSE
        error2 = np.mean(np.abs(current - fit_current_subset)/sigma/1e5) # MAE
        error3 = np.mean(np.abs(current - fit_current_subset)/(np.abs(current)+1e-3)) # Relative error
        error4 = np.mean((current - fit_current_subset)**2/(current**2+1e-6)) # Squared relative error
        errors.append(error)
        errors2.append(error2)
        errors3.append(error3)
        errors4.append(error4)

        if not suppress_output and i in [8, 16, 44, 52]: #important: 8, 16, 44/48, 52, 56?
            print(all_Gp[-1], error)
            if plot_type == 'vi':
                current = current * 10 # divide by 1e5 for resistor and multiply 1e6 for uA scale
                fit_current_subset = fit_current_subset * 10 # divide by 1e5 for resistor and multiply 1e6 for uA scale
                plt.scatter(voltage, current, label='Data', color=colors[i], facecolors='none', alpha=0.2)
                plt.plot(voltage, fit_current_subset, label='Fit curve', color=colors[i], linestyle='dashed', alpha=1.0)
            elif plot_type == 'params':
                plt.scatter(all_Gp) #, all_Gp2)
            else:
                raise NotImplementedError()

    return errors, all_Gp, errors2, errors3, errors4

# State characterisation
def state_characterisation(f, files, period, model, plot_type='vi', output_name='curve_fit', suppress_output=False):

    voltage_list, current_list, sigma_list, center_list = state_characterisation_file_reader(files, period)

    errors, all_Gp, errors2, errors3, errors4 = state_characterisation_core(f, voltage_list, current_list, center_list, sigma_list=sigma_list, plot_type=plot_type, suppress_output=suppress_output)

    if not suppress_output:
        if plot_type == 'vi':
            plt.xlabel('Voltage (V)')
            plt.ylabel('Current (μA)')
        elif plot_type == 'params':
            plt.xlabel('G1')
            plt.ylabel('G2')
        plt.grid()
        plt.savefig(output_name)

    return errors, errors2, errors3, errors4

def state_characterisation_meta(files, period, current_iteration, max_iterations, output_path, model=2):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    voltage_list, current_list, sigma_list, center_list = state_characterisation_file_reader(files, period)

    if current_iteration > 0: # Load saved data from previous completed iteration
        septets = np.load(os.path.join(output_path, 'septets.npy'))
        errors = np.load(os.path.join(output_path, 'septet_errors.npy'))
        sorted_septets, sorted_errors = sort_by_error(septets, errors)
        a1r, a2r, b1r, b2r, Gpr = np.load(os.path.join(output_path, 'ranges.npy'))
    else:
        lower = RANGE_INITIAL_LOWER_POWER
        upper = RANGE_INITIAL_UPPER_POWER
        a1r = [np.power(10.0,lower), np.power(10.0,upper)]  # 5
        a2r = [np.power(10.0,lower), np.power(10.0,upper)]  # 5
        b1r = [np.power(10.0,lower), np.power(10.0,upper)]  # 5
        b2r = [np.power(10.0,lower), np.power(10.0,upper)]  # 5
        Gpr = [np.power(10.0,lower), np.power(10.0,upper)]  # 5
        septets = []
        errors = []

    # Hangover from previous results
    s1s = [1,]  # 1
    s2s = [1,]  # 1
    s3s = [1,]  # 1

    for i in range(current_iteration, max_iterations):
        print('Iteration {}/{}'.format(i, max_iterations-1)) # Zero-indexing means max_iterations-1 is the last and 0 is the first.
        a1s, a2s, b1s, b2s, Gps = range_generator(a1r, a2r, b1r, b2r, Gpr, number=NUMBER)
        # Make septets and errors lists, for the purposes of appending to them
        septets = list(septets)
        errors = list(errors)
        for a1 in tqdm(a1s):
            for a2 in a2s:
                for b1 in b1s:
                    for b2 in b2s:
                        for Gp in Gps:
                            for s1 in s1s:
                                for s2 in s2s:
                                    for s3 in s3s:
                                        #if s1 + s2 + s3 < 0.6 or s1 + s2 + s3 > 3.1:
                                        #    break
                                        septet = (a1, a2, b1, b2, Gp, s1, s2, s3)
                                        if model == 0:
                                            f = get_f_proposed(a1, a2, b1, b2, Gp, s1, s2, s3)
                                        if model == 1:
                                            f = get_f_mss(a1, a2, b1, b2, Gp, s1, s2, s3)
                                        if model == 2:
                                            f = get_f_mss_mod(a1, a2, b1, b2, Gp, s1, s2, s3)
                                        error, _, _, _, _ = state_characterisation_core(f, voltage_list, current_list, center_list, sigma_list=sigma_list, suppress_output=True, params=[a1, a2, b1, b2, Gp])
                                        mean_error = np.mean(error)
                                        errors.append(mean_error)
                                        septets.append(septet)
        # Make septets and errors arrays for ease of manipulation
        septets = np.array(septets)
        errors = np.array(errors)

        np.save(os.path.join(output_path, 'septets.npy'), np.array(septets))
        np.save(os.path.join(output_path, 'septet_errors.npy'), np.array(errors))

        sorted_septets, sorted_errors = sort_by_error(septets, errors)
        a1r, a2r, b1r, b2r, Gpr = zoom(np.log10(sorted_septets[0]), np.log10(a1r[1]) - np.log10(a1r[0])) # The zeroth sorted septet is the best and consequently, our centre.
        np.save(os.path.join(output_path, 'ranges.npy'), [a1r, a2r, b1r, b2r, Gpr])

        with open(os.path.join(output_path, 'iteration.txt'), 'w') as f:
            f.write(str(i + 1))

        print(sorted_septets[:20])
        print(sorted_errors[:20])
        print('New parameter ranges:\n', a1r, a2r, b1r, b2r, Gpr)

    print('Done')
    return np.array(septets), np.array(errors)
