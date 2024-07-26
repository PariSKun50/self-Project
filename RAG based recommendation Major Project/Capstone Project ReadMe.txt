Capstone Project -Using Large Language Models to Automate Patent Citation Analysis
Overview
This project analyzes patents in Mechanical Engineering using two Jupyter notebooks.

Files
Pre Capstone Submission.ipynb: Processes patent_MECH.csv and saves concatenated_df.csv.
CapstoneFinalSubmission.ipynb: Analyzes concatenated_df.csv.
Instructions
Using Google Colab
Run Pre Capstone Submission.ipynb to process patent_MECH.csv and save concatenated_df.csv.
Run CapstoneFinalSubmission.ipynb to analyze concatenated_df.csv.
Using Local Environment
Remove gdown and google.colab imports.
Load CSV files traditionally:
import pandas as pd
'df = pd.read_csv('patent_MECH.csv')
concatenated_df = pd.read_csv('concatenated_df.csv')'
