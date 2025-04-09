import pandas as pd

class Datasets:
    
    
    def __init__(self, contextual_features_list:list, 
                 hr_features_list:list, 
                 temp_features_list:list, 
                 acc_features_list:list, 
                 hrv_features_list:list, 
                labels_list:list,  
                BiheartS_dataset_path: str,
                M2sleep_dataset_path: str  ,
                datasets_names:list = ["BiheartS", "M2sleep"],
                seed:int = 42,  ):
        
        self.datasets_names = datasets_names
        self.contextual_features_list = contextual_features_list
        self.hr_features_list = hr_features_list
        self.temp_features_list = temp_features_list
        self.acc_features_list = acc_features_list
        self.hrv_features_list = hrv_features_list
        self.labels_list = labels_list
        self.seed = seed
        self.complete_features_list = self.contextual_features_list  + self.hr_features_list +  self.temp_features_list + self.acc_features_list + self.hrv_features_list
        self.BiheartS_dataset_path = BiheartS_dataset_path
        self.M2sleep_dataset_path = M2sleep_dataset_path

        self.load_datasets()
        self.combine_all_datasets()
       
    
    def load_datasets(self):

        self.Bihearts_dataset = pd.read_csv(self.BiheartS_dataset_path,  index_col=0)
        self.M2sleep_dataset = pd.read_csv(self.M2sleep_dataset_path,  index_col=0)

        self.Bihearts_dataset["sleep_quality_binary"] = self.Bihearts_dataset["sleep_quality"].map(lambda x: 1 if x >= 7 else 0)
        self.M2sleep_dataset["sleep_quality_binary"] = self.M2sleep_dataset["sleep_quality"].map(lambda x: 1 if x >= 4 else 0)


    def combine_all_datasets(self):
        
        self.combined_datasets = pd.concat([self.Bihearts_dataset, self.M2sleep_dataset])
        self.combined_datasets["dataset_name"] = self.combined_datasets["participant_id"].str[0]
    
    
    
