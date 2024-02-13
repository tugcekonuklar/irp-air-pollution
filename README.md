# Forecasting PM2.5 Air Pollution in Potsdam, Germany

This project aims to forecast PM2.5 air pollutant levels in Potsdam, Germany, using European Environment Agency data. 
It consists of Jupyter notebooks and Python classes with methods dedicated to data processing, analysis, and forecasting models.


## Prerequisites

Before you begin, ensure you have the following installed:

- Anaconda or Miniconda
- Python 3.9
- Jupyter Notebook


## Conda Environment Setup

To create a conda environment with the required dependencies, run the command:

``` conda env export --no-builds | grep -v "prefix" > environment.yml```

activate the environment:

``` conda activate irp-air-pollution```


## Build and Run

To install required python libraries, run the command:  

``` pip install -r requirements.txt ```

To run the Jupyter notebooks, execute the command:

``` jupyter notebook ```

## Data

This project uses the European Environment Agency's air quality data for Potsdam, Germany. 
The data is available in the `data` directory and is organized into the following files:
* /parquet: Contains the raw data in parquet format
* /csv: Contains the raw data in csv format


## Usage
This project is structured into Jupyter notebooks and Python classes that perform various tasks related to the forecasting of PM2.5 levels. 

* Jupyter Notebooks: Provide step-by-step guides for data analysis, visualization, and model tuning and training. Navigate through the notebooks to understand the workflow.

* Python Classes and Methods: Contain reusable code for data preprocessing, model definition, training , tuning, evaluations, and forecasting. These are used within the notebooks but can also be imported into other Python scripts if needed.

## Running Notebooks
After launching Jupyter Notebook, navigate to the directory containing the notebooks. Open the desired notebook by clicking on its name.
Run the notebook cells in sequence by pressing **Shift + Enter** or using the **"Run"** button in the toolbar.