# StarScope: Detecting Exoplanets from Stellar Light Curves

 The StarScope project implements a one-dimensional convolutional neural network (CNN) from scratch in NumPy and evaluates its ability to detect exoplanets from labeled Kepler Space Telescope light curve data.

 Labeled data is from the [Exoplanet Hunting in Deep Space](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data) dataset, by WΔ.

## Repository Structure

- `colab/` - primary StarScope implementation: the full project that runs in Google Colab
- `hpc/` - alternative StarScope implementation: a simplified version that runs in a miniconda environment on a high-performance computing (HPC) cluster
- `data/` - documentation of dataset hosting location
- `results/` - results of experiment
- `report/` - report and Overleaf files for experiment

## Environment Setup

### Using Colab

Upload the Jupyter Notebook containing the Colab implementation, `starscope_colab.ipynb`, to Google Colab. Google Colab is **required**, as our project uses the google.colab package to download the log of runs.

### Using HPC

If using miniconda, create the miniconda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate starscope-env
```

If using pip, install dependencies with pip:

```bash
pip install -r requirements.txt
```

## Running Experiments

The Colab implementation, in addition to running models, contains cells that give the user information without directly affecting experimentation. The HPC implementation is simplified and only contains experimentation. The experimentation in both implementations is identical, so the user should receive identical results with either.

### Using Colab

Run all cells sequentially in Google Colab.

The first cell contains the flag `run_all_experiments`. When `False`, it runs only the first hyperparameter combination. When `True`, it runs all four hyperparameter combinations. The flag is set to `False` by default, since individual models take a long time to train and the Google Colab runtime may time out before all runs are completed. However, the user may set this flag to `True` if they prefer.

The log of runs is downloaded to the user's machine and confusion matrix and ROC plot are displayed in Google Colab.

### Using HPC

The HPC implementation contains a Python script, `starscope_hpc.py`, that contains experimentation logic. The implementation also contains `kepler_cnn.sh`, a Slurm script used to run the Python script; this is provided for documentation purposes only, and should be modified according to the user's situation to work properly.

Log of runs, confusion matrix, and ROC plot are saved in the same directory as the Python script.

## Results

See report/ for analysis of the final experiment's results.
