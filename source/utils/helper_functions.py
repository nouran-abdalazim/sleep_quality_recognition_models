from dotenv import dotenv_values
import json
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, kendalltau, mannwhitneyu
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def get_dataset_path (dataset_name:str)-> (str):
   
    """
        The function returns the physical path of the specified dataset

        Args:
            dataset_name (string): represents the name of the desired dataset 

        Returns:
            dataset_path (string): represents the physical path of the dataset
    """

    keys =  dotenv_values() 
    dataset_path = [val for key,val in keys.items() if dataset_name+"_dataset_path" in key][0]
    return dataset_path

def get_self_reports_path (dataset_name:str)-> (str):
   
    """
        The function returns the physical path of the specified dataset

        Args:
            dataset_name (string): represents the name of the desired dataset 

        Returns:
            self_reports_file_path (string): represents the physical path of the self-reports file
    """

    keys =  dotenv_values() 
    self_reports_file_path = [val for key,val in keys.items() if dataset_name+"_self_reports_file_path" in key][0]
    return  self_reports_file_path

def get_dataset_participants_list (dataset_name:str)-> (list):
   
    """
        The function returns the participants list of the specified dataset

        Args:
            dataset_name (string): represents the name of the desired dataset 

        Returns:
            participants_list (list): represents the participants list of the dataset
    """

    keys =  dotenv_values() 
    participants_list = json.loads([val for key,val in keys.items() if dataset_name+"_participants_list" in key][0])
    return participants_list

def get_dataset_self_reports_list (dataset_name:str, self_reports: str)-> (list):
   
    """
        The function returns the self reports list of the specified dataset

        Args:
            dataset_name (string): represents the name of the desired dataset
            self_reports (string): represents the name of the desired self reports 

        Returns:
            participants_list (list): represents the participants list of the dataset
    """

    keys =  dotenv_values() 
    self_reports_list = json.loads([val for key,val in keys.items() if dataset_name+"_"+self_reports+"_self_reports_list" in key][0])
    return self_reports_list


def get_dataset_extracted_parameters_list (dataset_name:str, paramters_name: str)-> (list):
   
    """
        The function returns the self reports list of the specified dataset

        Args:
            dataset_name (string): represents the name of the desired dataset
            paramters_name (string): represents the name of the desired parameters 

        Returns:
            paramters_list (list): represents the participants list of the dataset
    """

    keys =  dotenv_values() 
    paramters_list = json.loads([val for key,val in keys.items() if dataset_name+"_"+paramters_name+"_parameters_list" in key][0])
    return paramters_list





def visualize_correlation_heatmap(data:pd.DataFrame, correlationMethod:str = "spearman", removeDuplicates:bool = True, title:str = "", xlabel:str = "", ylabel:str = "", xaxisParameters = [], yaxisParameters = []) -> (plt):
    """
        The function computes the spearman correlation coefficient along with the p-value between all the columns of the provided
        dataframe

        Args: 
            data (pd.DataFrame): represents the dataframe to compute the correlation for it
            correlationMethod (string): represent the name of the correlation method (spearman, kendall, pearson)
            removeDuplicates (bool): represent the option of removing duplicate values in heatmap

        Returns:   
            plt (matplotlib.pyplot): represents the heatmap of the correlation results along with the p-values
    """

    if correlationMethod == "spearman":
        corr_matrix, pvalues = spearmanr(data)
    elif correlationMethod == "kendal":
        corr_matrix, pvalues = kendalltau(data)
    else:
        corr_matrix, pvalues = pearsonr(data)


    # Select the columns to display
    correlation_df = pd.DataFrame(corr_matrix, columns=data.columns, index=data.columns)
    correlation_df = correlation_df[yaxisParameters] 
    correlation_df = correlation_df.loc[xaxisParameters]

    pvalues_df = pd.DataFrame(pvalues, columns=data.columns, index=data.columns)
    pvalues_df = pvalues_df[yaxisParameters] 
    pvalues_df = pvalues_df.loc[xaxisParameters] 

    # Create a Seaborn heatmap
    _, ax = plt.subplots(figsize=(6, 4))

    # Use vectorized operations to format and apply conditional logic
    vectorized_format = np.vectorize(lambda c, p: f'{c:.2f}*' if p < 0.05 else f'{c:.2f}')
    # Apply vectorized formatting to correlation matrix and p-value matrix
    annotated_labels = vectorized_format(correlation_df, pvalues_df)

    if removeDuplicates:
        mask = np.triu(np.ones_like(correlation_df, dtype=np.bool_))
        sns.heatmap(correlation_df, annot=annotated_labels, cmap='coolwarm', fmt='', ax=ax, mask=mask, vmin=-1, vmax=1)
    else:
        sns.heatmap(correlation_df, annot=annotated_labels, cmap='coolwarm', fmt='', ax=ax, vmin=-1, vmax=1)

    plt.title (title, wrap=True)
    plt.xlabel (xlabel)
    plt.ylabel (ylabel)

    return plt


def visualize_correlation_heatmap_per_participant(data:pd.DataFrame, correlationMethod:str = "spearman", participantsList = [], title:str = "", xlabel:str = "", ylabel:str = "", yaxisParameters = [], dataset_name = "", export_correlation_results  = False) -> (plt):
    """
        The function computes the spearman correlation coefficient along with the p-value between all the columns of the provided
        dataframe

        Args: 
            data (pd.DataFrame): represents the dataframe to compute the correlation for it, it must include a column for the participant_id
            correlationMethod (string): represent the name of the correlation method (spearman, kendall, pearson)
            removeDuplicates (bool): represent the option of removing duplicate values in heatmap

        Returns:   
            plt (matplotlib.pyplot): represents the heatmap of the correlation results along with the p-values
    """



    # Compute the correlation per participant

    sleep_quality_correlation_per_participant = pd.DataFrame(columns=yaxisParameters, index= participantsList)
    sleep_quality_correlation_pvalue_per_participant = pd.DataFrame(columns=yaxisParameters, index= participantsList)

    for participant in participantsList:
        participant_data = data.loc[participant]
        participant_sleep_quality = participant_data["sleep_quality"]
        participant_corr = []
        participant_pvalue = []

        for c in yaxisParameters:

            if correlationMethod == "spearman":
                corr, pvalue  = spearmanr(participant_data[c], participant_sleep_quality)
            elif correlationMethod == "kendal":
                corr, pvalue = kendalltau(participant_data[c], participant_sleep_quality)
            else:
                corr, pvalue  = pearsonr(participant_data[c], participant_sleep_quality)
    
            participant_corr.append(float(corr))
            participant_pvalue.append(float(pvalue))

        sleep_quality_correlation_per_participant.loc[participant] = participant_corr
        sleep_quality_correlation_pvalue_per_participant.loc[participant] = participant_pvalue
        
    sleep_quality_correlation_per_participant = sleep_quality_correlation_per_participant[ [x for i,x in enumerate(yaxisParameters) if x!="sleep_quality"]]
    sleep_quality_correlation_pvalue_per_participant = sleep_quality_correlation_pvalue_per_participant[ [x for i,x in enumerate(yaxisParameters) if x!="sleep_quality"]]

    if export_correlation_results:
        sleep_quality_correlation_per_participant.to_csv(f"../output_files/{dataset_name}_correlation_results.csv")


    # Create a Seaborn heatmap
    _, ax = plt.subplots(figsize=(35, 8))

    # Use vectorized operations to format and apply conditional logic
    vectorized_format = np.vectorize(lambda c, p: f'{c:.2f}*' if p < 0.05 else f'{c:.2f}')
    # Apply vectorized formatting to correlation matrix and p-value matrix
    annotated_labels = vectorized_format(sleep_quality_correlation_per_participant, sleep_quality_correlation_pvalue_per_participant)

    columns  =  sleep_quality_correlation_per_participant.select_dtypes(include='object').columns

    sleep_quality_correlation_per_participant[columns] = sleep_quality_correlation_per_participant[columns].astype("float")
    sns.heatmap(sleep_quality_correlation_per_participant, cmap='coolwarm', fmt='', annot=annotated_labels, ax=ax, vmin=-1, vmax=1)

    plt.title (title, wrap=True)
    plt.xlabel (xlabel)
    plt.ylabel (ylabel,)

    return plt


def visualize_models_results (title, preqacc_personalized_model_results, 
                              preqacc_user_specific_model_results, 
                              acc_population_model_results, 
                              acc_baseline_model_results,  
                              acc_biased_baseline_model_results,
                              acc_personalized_biased_baseline_model_results):

    # Combine lists into a dictionary
    data = {
        'Personalized': 100* preqacc_personalized_model_results,
        'One-User-Only': 100* preqacc_user_specific_model_results,
        'Population': 100* acc_population_model_results,
        'Biased Random Guess': 100* acc_biased_baseline_model_results,
        'Personalized Biased Random Guess': 100* acc_personalized_biased_baseline_model_results,
        'Random Guess': 100* acc_baseline_model_results,

    }


    # Convert the dictionary into a DataFrame
    df = pd.DataFrame(data)

    # Melt the DataFrame to make it long-form (for seaborn)
    df_melted = df.melt(var_name='Group', value_name='Value')

    # Set the plot size
    plt.figure(figsize=(8, 6))

    # Draw the horizontal boxplot using seaborn (swap x and y)
    sns.boxplot(y='Group', x='Value', data=df_melted, palette="Pastel2")

    # Annotate the mean values on the boxplot
    means = df.mean()  # Calculate the means of each list
    for group in means.index:
        mean_value = means[group]
        group_index = list(means.index).index(group)
        plt.text(mean_value, group_index, f'{mean_value:.2f}', color='black', ha='center', va='center', fontweight='bold')

    # Add title and labels
    plt.title(title)
    plt.xlabel("Accuracy")
    plt.ylabel("Models")

    # Show the plot
    return plt



