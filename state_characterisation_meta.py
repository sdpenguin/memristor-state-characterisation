import numpy as np
import os

# To avoid problems associated with attempts to multithread for small data clustering operations, which can cause memory leaks
os.environ["OMP_NUM_THREADS"] = "1"

from argparse import ArgumentParser

from memristor_state_characterisation.model_fitting import get_extension, sort_by_error, state_characterisation_meta

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--period', default=160)
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--files', nargs='+', default=['data_lists/state.yaml'], help='List of YAML files that contains experiment filenames for fitting and plotting.')
    parser.add_argument('--max-iterations', type=int, default=10, help='The maximum number of meta iterations to run.')
    parser.add_argument('--model', type=int, default=0, help='0 for proposed model, 1 for GMMS, 2 for modified GMMS.')
    parsed = parser.parse_args()
    output_path = 'output' + get_extension(parsed.model)
    if os.path.exists(os.path.join(output_path, 'iteration.txt')):
        with open(os.path.join(output_path, 'iteration.txt'), 'r') as f:
            current_iterations = int(f.read())
    else:
        current_iterations = 0
    if parsed.reset:
        current_iterations = 0
    if current_iterations < parsed.max_iterations:
        state_characterisation_meta(parsed.files, parsed.period, current_iterations, parsed.max_iterations, output_path, parsed.model)
    septets = np.load(os.path.join(output_path, 'septets.npy'))
    errors = np.load(os.path.join(output_path, 'septet_errors.npy'))
    sorted_septets, sorted_errors = sort_by_error(septets,errors)
    print('Final septets and errors')
    print(sorted_septets[:20])
    print(sorted_errors[:20])
