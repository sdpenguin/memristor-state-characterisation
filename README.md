# Memristor State Characterisation

This repository contains code for the state characterisation of memristors, specifically applied to Self-Directed Channel (SDC) memristors. It includes for both fitting a general memristive state model and for estimating the state from data. It contains a dataset for SDC memristors, to which the state characterisation and estimation is applied.

## Dependencies

The following Python libraries are required:
- sklearn (for parameter optimisation in VI function fitting)
- hdbscan (for cluster identification in state characterisation meta)
- yaml
- scipy

## Installing the Python Module

To install `memristor_state_characterisation` as a Python module, run:

`` pip install . ``

This will allow `memristor_state_characterisation` to be imported as a module and used externally.

## Fitting Memristor Models

Please run the following to obtain the best parameters for a particular model:

`` python state_characterisation_meta.py --model [0, 1, 2]``

Models:
- 0: Proposed Model [""]
- 1: GMSM ["_GMSM]
- 2: GMSM Modified ["_GMSM_mod"]

The results will be saved by default in the ``output-folder`` directory, with the suffixes given above.

To produce a plot of the model fit, use the follwing once the results have been saved:

`` python state_characterisation.py --model [0, 1, 2]``

## State Estimation

To generate a figure demonstrating state estimation for retention data, run the following:

`` python state_estimation.py ``

Note that the parameters of the best fit proposed model (as of 21.04.2025) are set as defaults in the parameters of `state_estimation.py`.

## Plotting individual data

Use the following script to visiualise data files:

`` python plot_resistances.py [directory_containing_csvs] ``

Where `directory_containing_csvs` is a directory containing data files in the csv format.

## Data Descriptions

### State Data

data: `data/state_data/`
data_list: `data_lists/state.yaml`

Experiments involving a triangle wave function made to provide data to characterise the VI curve of the memristor.

N.B. `data_10` contains experiments at a higher voltage than `data_04`. These data directories are separated due to having been conducted on different days.

### Retention Data

data: `data/retention_data/`
data_lists: `data_lists/retention.yaml`

Contains 2 retention experiments.

In each retention directory, there are two kinds of CSV:
- The zeroth CSV is the initial programming, with triangle read applied before and after the programming to determine the starting state and initial state.
- Each subsequent CSV contains a triangle of an 8 second duration. The CSVs are reads taken 1 minute apart. There are 59 subsequent read CSVs.

N.B.
- Some parts of the second half of the read CSV may be missing in some cases, so all the reads should be cropped to half their original size to ensure consistency.
- There is no exact guarantee of the time between the 0th (the write) and the 1st (first read) CSV.
