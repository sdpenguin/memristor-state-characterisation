import pandas as pd
import numpy as np
import scipy
import math

def preprocess_data(data, period, average=True, crop=True, period_align=True, auto_offset=True, offset_v=0.0, offset_i=0.0, optimize_offset=False):
    if crop:
        # Detect and crop 0-start
        start_index = 0
        start = abs(data['Volt.1'][0])
        for i in range(1, period-1):
            if (np.abs(data['Volt.1'][i]) < start): # TODO: The second statement might be a bit unnecessary, and this was just to make sure that we started on a positive going cycle
                start = abs(data['Volt.1'][i])
                start_index = i
        data = data.drop([i for i in range(start_index)]).reset_index(drop=True)
    
    # data = data.drop([i for i in range(len(data)) if i%period >= (period//4)]).reset_index(drop=True)

    if period_align:
        # Possibly crop last partial period such that data length is divisible by period through cropping
        data = data[: int(len(data) // period * period)]

    voltage = (data['Volt.1'] - data['Volt'] + offset_v).to_numpy()
    current = (data['Volt'] + offset_i).to_numpy()

    if average:
        voltage = voltage.reshape(-1, period)
        current = current.reshape(-1, period)
        if auto_offset:
            voltage = voltage - np.expand_dims(voltage[:, 0], 1)
            current = current - np.expand_dims(current[:, 0], 1)
        voltage = voltage.mean(0)
        current = current.mean(0)
    elif auto_offset:
        voltage = voltage - voltage[0]
        current = current - current[0]

    # We minimise the magnitude of v*i for points in the upper left and lower right quadrants, where points should not exist.
    if optimize_offset:
        indices = np.argwhere(voltage[~np.isnan(voltage)])
        indices2 = np.argwhere(current[indices][~np.isnan(current[indices])])
        if len(indices2) > 0:
            def loss(x):
                offset_v, offset_c = x
                voltage_up = voltage[indices2] + offset_v
                current_up = current[indices2] + offset_c
                signs = voltage_up*current_up
                return np.sum(-signs[signs <= 0])

            result = scipy.optimize.least_squares(loss, [0.0, 0.0], method='dogbox', bounds=[[-.5*np.max(np.abs(voltage[indices2]))-1e-10, -.5*np.max(np.abs(current[indices2]))-1e-10], [.5*np.max(np.abs(voltage[indices2]))+1e-10, .5*np.max(np.abs(current[indices2]))+1e-10]])
            offset_v, offset_c = result.x

            voltage = voltage + offset_v
            current = current + offset_c

    return voltage, current
