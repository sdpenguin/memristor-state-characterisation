import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import yaml

from argparse import ArgumentParser
from tqdm import tqdm

from memristor_state_characterisation.functions import preprocess_data
from memristor_state_characterisation.state_estimation import get_state_estimate

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 25
sns.set_style('white')
plt.rcParams['lines.linewidth'] = 4

OPTIONS = ['', '2', '3', '4', '5', '6', '7', '8', '9']
COLOURS = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'grey', 'pink']


def main(files, model_params, output_dir='.', data_dir='.'):
    all_estimates = []
    for directory, option in zip(files, OPTIONS):
        estimates = []
        if not os.path.exists(os.path.join(output_dir, 'retention_states/estimate{}.npy'.format(option))):
            for i in tqdm(range(1, 60)):
                csv = pd.read_csv('{}/long_meas{}_{}.csv'.format(directory, option, i), header=1)
                voltage, current = preprocess_data(csv, 160, optimize_offset=True, average=False)
                state_estimate = 1e5*1/get_state_estimate(voltage, current, model_params, eps=0.3)
                if state_estimate == 0.0:
                    estimates.append(np.nan)
                else:
                    estimates.append(state_estimate)
            np.save(os.path.join(output_dir, 'retention_states/estimate{}.npy'.format(option)), estimates)
        else:
            print('Estimate npy exists for estimate {}. Loading.'.format(option))
            estimates = np.load(os.path.join(output_dir, 'retention_states/estimate{}.npy'.format(option)))
        all_estimates.append(estimates)
    all_estimates = np.stack(all_estimates, axis=0)
    np.save(os.path.join(data_dir, 'series.npy'), all_estimates) # Save the estimates for future use

    max_estimate = 1e7
    min_estimate = 1e6
    for option, color, directory in zip(OPTIONS, COLOURS, files):
        estimates = np.load(os.path.join(output_dir, 'retention_states/estimate{}.npy'.format(option)))
        window_size = 15
        moving_average = np.convolve(estimates, np.ones(window_size)/window_size, mode='valid')
        moving_avg_centered = np.concatenate([
            [np.nan] * int(window_size//2),
            moving_average,
            [np.nan] * (window_size - int(window_size//2))
        ])
        plt.plot(estimates, marker='x', linestyle='-', color=color)
        plt.plot(moving_avg_centered, linestyle='dotted', color=color)
        if min_estimate > min(estimates):
            min_estimate = min(estimates)
        if max_estimate < max(estimates):
            max_estimate = max(estimates)
    plt.ylim([10**(round(np.log10(min_estimate))-1), 10**(round(np.log10(max_estimate))+1)])
    plt.grid()
    plt.legend(handles=[Line2D([0], [0], color='black', linestyle='-', label='State estimate'),
    Line2D([0], [0], color='black', linestyle='dotted', label='Moving average')])
    plt.xlabel('Time (m)')
    plt.ylabel('Estimated State ($\Omega$)')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'state_estimation.pdf'), bbox_inches='tight')

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--list', type=str, default='data_lists/retention.yaml')
    parser.add_argument('--model-params', nargs=5, type=float, default=[2.622e-1, 6.597e-2, 1.37e1, 1.005e1, 8.679e0], help='[alpha1, alpha2, beta1, beta2, Gm] for the proposed model.')
    parser.add_argument('--output-dir', default='output')
    parser.add_argument('--data-dir', default='data')
    parsed = parser.parse_args()
    with open(parsed.list, 'r') as infile:
        file_list = yaml.safe_load(infile)
    if not os.path.exists(parsed.output_dir):
        os.mkdir(parsed.output_dir)
    if not os.path.exists(os.path.join(parsed.output_dir, 'retention_states')):
        os.mkdir(os.path.join(parsed.output_dir, 'retention_states'))
    main(file_list, parsed.model_params, parsed.output_dir, parsed.data_dir)
