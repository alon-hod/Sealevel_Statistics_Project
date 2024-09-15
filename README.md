<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sea Level Analysis - Final Project</title>
</head>
<body>
    <h1>Sea Level Analysis - Final Project</h1>

    <p>This repository contains the final project for my statistics course, focusing on sea level changes and its relationship with various factors. Below is an overview of the structure and contents of this repository, along with instructions on how to use the code and data.</p>

    <h2>Project Overview</h2>
    <p>The main goal of this project is to analyze sea level data using statistical methods and predictive modeling. The analysis covers correlation testing and time series modeling, utilizing Python to perform computations and visualizations.</p>

    <h2>Repository Structure</h2>
    <pre>
├── Final_project-Sea_level.ipynb   # Main Jupyter Notebook for the analysis
├── sl_withmoon.csv                 # Dataset used for the analysis
├── raw_data/                       # Folder containing raw datasets used in preprocessing
├── correlation/                    # Folder containing the script for Spearman correlation analysis
│   └── spearmans.py
├── models/                         # Folder containing the models used for prediction and analysis
│   ├── sealevel-reg3.py            # Script for regression model on sea level data
│   └── sealevel-timeseries.py      # Script for time series analysis on sea level data
└── output/                         # Folder with the outputs from the Spearman correlation script
    </pre>

    <h2>Files and Folders</h2>

    <h3>1. Final_project-Sea_level.ipynb</h3>
    <p>This Jupyter notebook serves as the main file for the project. It runs the following analyses:</p>
    <ul>
        <li><strong>Spearman correlation analysis</strong> using the script <code>correlation/spearmans.py</code>.</li>
        <li><strong>Sea level regression model</strong> from the script <code>models/sealevel-reg3.py</code>.</li>
        <li><strong>Time series analysis</strong> using <code>models/sealevel-timeseries.py</code>.</li>
    </ul>
    <p>You can run the notebook to reproduce all analyses and outputs.</p>

    <h3>2. sl_withmoon.csv</h3>
    <p>This is the primary dataset used for the analysis. It includes sea level data along with additional variables such as lunar information that may impact sea level changes.</p>

    <h3>3. raw_data/</h3>
    <p>This folder contains the raw datasets that were used for preprocessing and data cleaning before the main analysis.</p>

    <h3>4. correlation/spearmans.py</h3>
    <p>This Python script performs Spearman's correlation analysis to explore the relationship between various variables in the dataset. The output from this analysis is saved in the <code>output/</code> folder.</p>

    <h3>5. models/</h3>
    <p>This folder contains the Python scripts used for modeling:</p>
    <ul>
        <li><strong>sealevel-reg3.py</strong>: A script implementing a regression model to predict sea level changes based on various factors.</li>
        <li><strong>sealevel-timeseries.py</strong>: A time series analysis script that forecasts future sea level changes based on historical data.</li>
    </ul>

    <h3>6. output/</h3>
    <p>This folder contains the outputs generated from running the Spearman correlation analysis in the <code>spearmans.py</code> script.</p>

    <h2>How to Run the Project</h2>
    <ol>
        <li>Clone the repository:
            <pre><code>git clone &lt;repository-url&gt;
cd &lt;repository-folder&gt;</code></pre>
        </li>
        <li>Install the required dependencies:
            <pre><code>pip install -r requirements.txt</code></pre>
        </li>
        <li>Run the Jupyter notebook <code>Final_project-Sea_level.ipynb</code> to execute the analyses and generate outputs.</li>
    </ol>
</body>
</html>
