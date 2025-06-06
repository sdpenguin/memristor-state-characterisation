import numpy as np
import os

from argparse import ArgumentParser

from memristor_state_characterisation.models import get_f_proposed, get_f_mss, get_f_mss_mod
from memristor_state_characterisation.model_fitting import get_extension, sort_by_error, state_characterisation

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--period', default=160)
    parser.add_argument('--files', nargs='+', default=['data_lists/state.yaml'], help='List of YAML files that contains experiment filenames for fitting and plotting.')
    parser.add_argument('--output-path', default='output')
    parser.add_argument('--model', type=int, default=0, help='0 for proposed model, 1 for GMMS, 2 for modified GMMS.')
    parser.add_argument('--output-name', default='curve_fit')
    parser.add_argument('--manual-extrapolation', type=float, default=None, nargs='+', help='Specify the max negative and max positive extrapolation values.')
    parser.add_argument('--plot-type', nargs='?', default='vi', choices=['vi', 'params'])
    parsed = parser.parse_args()
    parsed.output_name = parsed.output_name + get_extension(parsed.model)

    septets = np.load(os.path.join(parsed.output_path + get_extension(parsed.model), 'septets.npy'))
    errors = np.load(os.path.join(parsed.output_path + get_extension(parsed.model), 'septet_errors.npy'))
    sorted_septets, sorted_errors = sort_by_error(septets, errors)
    if parsed.model == 0:
        f = get_f_proposed(*sorted_septets[0])
    elif parsed.model == 1:
        f = get_f_mss(*sorted_septets[0])
    elif parsed.model == 2:
        f = get_f_mss_mod(*sorted_septets[0])

    errors, errors2, errors3, errors4 = state_characterisation(f, parsed.files, parsed.period, parsed.model, parsed.plot_type, parsed.output_name)
    print('Mean errors:')
    print(np.mean(errors))
    print(np.mean(errors2))
    print(np.mean(errors3))
    print(np.mean(errors4))
