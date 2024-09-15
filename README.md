<h1>Sea Level Analysis - Final Project</h1>
This repository contains the final project for my statistics course, focusing on sea level changes and its relationship with various factors. Below is an overview of the structure and contents of this repository, along with instructions on how to use the code and data.

<h2>Project Overview</h2>
The main goal of this project is to analyze sea level data using statistical methods and predictive modeling. The analysis covers correlation testing and time series modeling, utilizing Python to perform computations and visualizations.

<h2>Repository Structure</h2>
```
├── Final_project-Sea_level.ipynb   # Main Jupyter Notebook for the analysis
├── sl_withmoon.csv                 # Dataset used for the analysis
├── raw_data/                       # Folder containing raw datasets used in preprocessing
├── correlation/                    # Folder containing the script for Spearman correlation analysis
│   └── spearmans.py
├── models/                         # Folder containing the models used for prediction and analysis
│   ├── sealevel-reg3.py            # Script for regression model on sea level data
│   └── sealevel-timeseries.py      # Script for time series analysis on sea level data
└── output/                         # Folder with the outputs from the Spearman correlation script
```


<h2>Files and Folders</h2>
1. Final_project-Sea_level.ipynb
This Jupyter notebook serves as the main file for the project. It runs the following analyses:

Spearman correlation analysis using the script correlation/spearmans.py.
Sea level regression model from the script models/sealevel-reg3.py.
Time series analysis using models/sealevel-timeseries.py.
You can run the notebook to reproduce all analyses and outputs.

2. sl_withmoon.csv
This is the primary dataset used for the analysis. It includes sea level data along with additional variables such as lunar information that may impact sea level changes.

3. raw_data/
This folder contains the raw datasets that were used for preprocessing and data cleaning before the main analysis.

4. correlation/spearmans.py
This Python script performs Spearman's correlation analysis to explore the relationship between various variables in the dataset. The output from this analysis is saved in the output/ folder.

5. models/
This folder contains the Python scripts used for modeling:

sealevel-reg3.py: A script implementing a regression model to predict sea level changes based on various factors.
sealevel-timeseries.py: A time series analysis script that forecasts future sea level changes based on historical data.
6. output/
This folder contains the outputs generated from running the Spearman correlation analysis in the spearmans.py script.

<h2>How to Run the Project</h2>
Clone the repository:

bash
git clone <repository-url>
cd <repository-folder>

Install the required dependencies:

pip install -r requirements.txt

Run the Jupyter notebook Final_project-Sea_level.ipynb to execute the analyses and generate outputs.
