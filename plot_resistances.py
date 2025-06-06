import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from argparse import ArgumentParser

from memristor_state_characterisation.functions import preprocess_data

def main(parsed):
    # If we only specify one value for the start index or the end_index, use this for all files, else use individual values
    if len(parsed.start_index) == 1:
        start_indices = [parsed.start_index[0] for _ in parsed.files]
    else:
        start_indices = parsed.start_index
    if len(parsed.end_index) == 1:
        end_indices = [parsed.end_index[0] for _ in parsed.files]
    else:
        end_indices = parsed.end_index
    for file_name, start_index, end_index in zip(parsed.files, start_indices, end_indices):
        csv = pd.read_csv('{}/{}'.format(parsed.directory, file_name), header=1)
        voltage, current = preprocess_data(csv, parsed.period, average=(not parsed.no_average), crop=(not parsed.no_zero_cropping), auto_offset=(not parsed.no_offset))
        if start_index < 0:
            voltage = np.concatenate([voltage[start_index:],voltage[:start_index]])
            current = np.concatenate([current[start_index:],current[:start_index]])
        else:
            voltage = voltage[start_index:]
            current = current[start_index:]
        if end_index is not None:
            voltage = voltage[:end_index]
            current = current[:end_index]
        resistances = voltage/current*parsed.series_resistance
        resistances[np.abs(resistances) > 1e6] = 1e6
        if parsed.plot_type == 'scatter':
            if parsed.phase_boundaries is None:
                plt.scatter(voltage, current, marker='x', alpha=0.5, label=file_name)
            else:
                prev_boundary = 0
                for x in parsed.phase_boundaries:
                    plt.scatter(voltage[prev_boundary:x], current[prev_boundary:x], marker='x', alpha=0.5, label='{}-{}'.format(file_name, x))
                    prev_boundary = x
                plt.scatter(voltage[x:end_index], current[x:end_index], marker='x', alpha=0.5, label='{}-{}'.format(file_name, end_index))
        elif parsed.plot_type == 'vi':
            plt.plot(voltage, label='{} V'.format(file_name))
            plt.plot(current, label='{} I'.format(file_name))
        elif parsed.plot_type == 'period_vi':
            for i in range(parsed.num_periods_to_plot):
                plt.plot(voltage[i*parsed.period:(i+1)*parsed.period], label='V {}'.format(i))
                plt.plot(current[i*parsed.period:(i+1)*parsed.period], label='I {}'.format(i))
        elif parsed.plot_type == 'period_power':
            for i in range(parsed.num_periods_to_plot):
                plt.plot(voltage[i*parsed.period:(i+1)*parsed.period]*current[i*parsed.period:(i+1)*parsed.period], label='P {}'.format(i))
        elif parsed.plot_type == 'period_i':
            for i in range(parsed.num_periods_to_plot):
                plt.plot(current[i*parsed.period:(i+1)*parsed.period], label='I {}'.format(i))
        elif parsed.plot_type == 'period_v':
            for i in range(parsed.num_periods_to_plot):
                plt.plot(current[i*parsed.period:(i+1)*parsed.period], label='V {}'.format(i))
        elif parsed.plot_type == 'period_scatter':
            for i in range(parsed.num_periods_to_plot):
                plt.scatter(voltage[i*parsed.period:(i+1)*parsed.period], current[i*parsed.period:(i+1)*parsed.period], label='Period {}'.format(i))
        else:
            raise ValueError('Plot type {} is unknown.'.format(parsed.plot_type))

    if parsed.plot_type == 'vi' and parsed.no_average:
        plt.xlabel('Time')
        plt.ylabel('Voltage or Current')
    elif parsed.plot_type == 'vi' and (not parsed.no_average):
        plt.xlabel('Time (relative to start of period)')
    elif parsed.plot_type == 'scatter' or parsed.plot_type == 'period_scatter':
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (A)')
    plt.legend()
    plt.grid()
    plt.savefig('{}.png'.format(parsed.output_name))

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--directory', default='data_01')
    parser.add_argument('--files', nargs='+')
    parser.add_argument('--period', default=160, type=int)
    parser.add_argument('--no-average', action='store_true')
    parser.add_argument('--no-zero-cropping', action='store_true')
    parser.add_argument('--no-offset', action='store_true', help='Do not attempt to automatically adjust the offset of the signals using the voltage offset at time 0.')
    parser.add_argument('--series_resistance', default=100e-3)
    parser.add_argument('--plot-type', nargs='?', default='scatter', choices=['vi', 'period_vi', 'period_i', 'period_v', 'period_power', 'scatter', 'period_scatter'])
    parser.add_argument('--start-index', nargs='+', type=int, default=[0,], help='Applied before end_index. Useful for moving the end of the array to the start if you use a negative index (circular/modulo shift is applied if the index is negative.)')
    parser.add_argument('--end-index', nargs='+', type=int, default=[-1,], help='The last value to read. Applied after start_index.')
    parser.add_argument('--output-name', default='resistance_plot')
    parser.add_argument('--phase-boundaries', default=None, nargs='+', type=int, help='Boundaries for colouring the scatter plot.')
    parser.add_argument('--num-periods-to-plot', default=1, type=int, help='The number of periods to plot for a period_scatter plot.')
    parsed = parser.parse_args()

    assert len(parsed.start_index) == 1 or len(parsed.start_index) == len(parsed.files)
    assert len(parsed.end_index) == 1 or len(parsed.end_index) == len(parsed.files)

    if parsed.phase_boundaries is not None:
        assert parsed.plot_type == 'scatter', "Phase boundaries are valid only for scatter plots."

    if parsed.end_index != [-1,] and np.sum([(end_index < parsed.period) for end_index in parsed.end_index]) and (not parsed.no_average or not parsed.no_zero_cropping):
        raise ValueError('Ensure you allow for enough values to cover the period since cropping/averaging is enabled.')

    main(parsed)
