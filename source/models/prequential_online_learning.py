import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from source.models.personalized_dummy_classifer_model import PersonalizedDummyClassifier
import matplotlib.pyplot as plt
import numpy as np
import os, csv

class PrequentialOnlineLearning:
    
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
    
    
    

            

    def create_models(self, model_name, seed):
        if model_name == "PAC":
            
                # Create an PAC model
                personalized_model = PassiveAggressiveClassifier(loss='hinge',  max_iter=2000,  random_state=seed, shuffle=True, verbose=False)
                user_specific_model = PassiveAggressiveClassifier(loss='hinge', max_iter=2000, random_state=seed, shuffle=True, verbose=False)
                population_model = PassiveAggressiveClassifier(loss='hinge', max_iter=2000, random_state=seed, shuffle=True, verbose=False)
            
        elif model_name == "Xgboost":
            params = {
                    'objective': 'binary:logistic',  # for binary classification
                    'eval_metric': 'logloss',}
            personalized_model = PassiveAggressiveClassifier(loss='hinge',  max_iter=2000,  random_state=seed, shuffle=True, verbose=False)
            user_specific_model = PassiveAggressiveClassifier(loss='hinge', max_iter=2000, random_state=seed, shuffle=True, verbose=False)
            population_model = XGBClassifier(params = params, objective='binary:logistic', random_state=seed,  use_label_encoder = True)
    
        elif model_name == "SGD":
            # Create a SGD-based Logistic Regression model
            personalized_model = SGDClassifier(loss='log_loss', max_iter=2000, random_state=seed, shuffle=True, verbose=False)
            user_specific_model = SGDClassifier(loss='log_loss', max_iter=2000, random_state=seed, shuffle=True, verbose=False)
            population_model = SGDClassifier(loss='log_loss', max_iter=2000, random_state=seed, shuffle=True, verbose=False)
        
        else:

            # Create a MLP model
            personalized_model = MLPClassifier(max_iter=200,random_state=seed, shuffle=True, verbose=False)
            user_specific_model = MLPClassifier(max_iter=200, random_state=seed, shuffle=True, verbose=False)
            population_model = MLPClassifier(max_iter=200, random_state=seed, shuffle=True, verbose=False)
        
        return personalized_model, user_specific_model, population_model

   
    def prepare_training_set(self, mode, hold_out_participant, training_feature_vectors , testing_feature_vectors, featurs_list):

        if mode == "multiple":
            # extract the test set
            participant_test_set = testing_feature_vectors.loc[testing_feature_vectors.index.get_level_values(0) == hold_out_participant]
            participant_test_set = participant_test_set.sort_index()

            training_set = training_feature_vectors

            # sort the DataFrame by its MultiIndex (participant_id and Date of the session)
            training_set = training_set.sort_index()
            
            # prepare the feature vectors and the labels for the training
            X = training_set[featurs_list] #.drop(columns=["sleep_quality","sleep_quality_binary"])
            y = training_set ["sleep_quality_binary"]


        
        else:
            # extract the test set
            participant_test_set = training_feature_vectors.loc[training_feature_vectors.index.get_level_values(0) == hold_out_participant]
            participant_test_set = participant_test_set.sort_index()

            training_set = training_feature_vectors.loc[~(training_feature_vectors.index.get_level_values(0) == hold_out_participant)]

            # sort the DataFrame by its MultiIndex (participant_id and Date of the session)
            training_set = training_set.sort_index()
            
            # prepare the feature vectors and the labels for the training
            X = training_set[featurs_list] #.drop(columns=["sleep_quality","sleep_quality_binary"])
            y = training_set ["sleep_quality_binary"]
        
        return X, y, participant_test_set


    def prequential_evaluation_setup(self,
                                    training_feature_vectors, 
                                    testing_feature_vectors,
                                    featurs_list, 
                                    test_participants_list,
                                    seed, mode, modality, model, 
                                    ):
        # use dictionary to store the results for each model per each participant
        participants_results = {}
        for key in test_participants_list:
            participants_results[key] = {}
        
        for hold_out_participant in test_participants_list:

            participant_test_results = {}

            X, y, participant_test_set = self.prepare_training_set(mode, hold_out_participant, training_feature_vectors, testing_feature_vectors,  featurs_list)

           
            
            personalized_model, user_specific_model, population_model = self.create_models(model, seed)
        
             
            # intialize dummy classifier with the random
            random_baseline_dummy_clf = DummyClassifier(strategy="uniform", random_state=seed)
            # intialize dummy classifier with the biased random
            biased_random_baseline_dummy_clf = DummyClassifier(strategy="stratified", random_state=seed)
            # intialize dummy classifier with the personalized biased random
            personalized_biased_random_baseline_dummy_clf = PersonalizedDummyClassifier( random_state=seed)
            

            # initialize standardization
            scaler = StandardScaler()
            # transform the feature vectors of the training to the standardized format
            X = scaler.fit_transform(X)

            # Train the personalized model on the training batch using partial_fit() to allow incremental learning
            personalized_model.partial_fit(X, y, classes=[0,1])
            # Train the population model on the training data
            population_model.fit(X, y)
            # Train the baseline model (random guess) on the training data
            random_baseline_dummy_clf.fit(X, y)
            # Train the baseline model (biased random guess) on the training data
            biased_random_baseline_dummy_clf.fit(X, y)
            # Train the baseline model (personalized biased random guess) on the training data
            personalized_biased_random_baseline_dummy_clf.partial_fit(X, y, classes=[0,1])

          

            personalized_model_predictions = []    
            population_model_predictions = []
            user_specific_model_predictions = []
            random_baseline_model_predictions = []
            biased_random_baseline_model_predictions = []
            personalized_biased_random_baseline_model_predictions = []


            # iterarte over each session in the test set
            first_iteration = True
            for _, row in participant_test_set.iterrows():

                # prepare the new data 
                new_data_point_features = row[featurs_list].to_numpy().reshape(1, -1)
                y = row["sleep_quality_binary"]     
                new_data_point_features_normalized = scaler.transform(new_data_point_features)

                if first_iteration:
                    user_specific_model.partial_fit(new_data_point_features, [y], [0,1])       
                    first_iteration = False
                else:

                    # test using user_specific_model
                    user_specific_y_pred = user_specific_model.predict(new_data_point_features)
                    user_specific_model_predictions.append(user_specific_y_pred[0])
                    
                    # Incrementally update the user-specific model with the new datapoint
                    user_specific_model.partial_fit(new_data_point_features, [y])       


                # test using personalized_model
                y_pred = personalized_model.predict(new_data_point_features_normalized)
                personalized_model_predictions.append(y_pred[0])

                # test using personalized_model
                population_y_pred = population_model.predict(new_data_point_features_normalized)
                population_model_predictions.append(population_y_pred[0])

                # test using baseline
                random_baseline_y_pred = random_baseline_dummy_clf.predict(new_data_point_features_normalized)
                random_baseline_model_predictions.append(random_baseline_y_pred)

                # test using baseline
                bias_random_baseline_y_pred = biased_random_baseline_dummy_clf.predict(new_data_point_features_normalized)
                biased_random_baseline_model_predictions.append(bias_random_baseline_y_pred)

                # train using personalized baseline
            
                personalized_bias_random_baseline_y_pred = personalized_biased_random_baseline_dummy_clf.predict(new_data_point_features_normalized)
                personalized_biased_random_baseline_model_predictions.append(personalized_bias_random_baseline_y_pred)
                personalized_biased_random_baseline_dummy_clf.partial_fit(new_data_point_features_normalized,[y])

                
                # Incrementally update the personalized model with the new datapoint
                personalized_model.partial_fit(new_data_point_features_normalized, [y]) 

            
               
                
        
            participant_test_results["personalized_model_results"] = personalized_model_predictions
            participant_test_results["population_model_results"] = population_model_predictions
            participant_test_results["user_specific_model_results"] = user_specific_model_predictions
            participant_test_results["baseline_model_results"] = random_baseline_model_predictions
            participant_test_results["biased_random_baseline_model_results"] = biased_random_baseline_model_predictions
            participant_test_results["personalized_biased_random_baseline_model_results"] = personalized_biased_random_baseline_model_predictions
            participant_test_results["participant_ground_truth"] = participant_test_set["sleep_quality_binary"]
            
            participants_results[hold_out_participant] = participant_test_results

        
        return  participants_results

        

    def recognize_sleep_quality(self, training_dataset, seed_list, testing_dataset, modality, model, featurs_list = [],  mode="single"):
        preqacc_personalized_model_results = []
        acc_population_model_results = []
        acc_baseline_model_results = []
        acc_biased_baseline_model_results = []
        acc_personalized_biased_baseline_model_results = []
        keys = []

        training_dataset = training_dataset.set_index(["participant_id", "Date"])


        if mode == "multiple":
            testing_dataset = testing_dataset.set_index(["participant_id", "Date"])
            test_participants_list = testing_dataset.index.get_level_values(0).unique()
        else:
            test_participants_list = training_dataset.index.get_level_values(0).unique()


        for seed in tqdm(seed_list, desc='Processing Items'):
           participants_results = self.prequential_evaluation_setup(training_dataset, 
                                                                    testing_dataset, 
                                                                    featurs_list, 
                                                                    test_participants_list,
                                                                    seed=seed, 
                                                                    mode=mode, 
                                                                    modality = modality, 
                                                                    model = model)

           for key in participants_results.keys():
                personalized_model_results = participants_results[key]["personalized_model_results"]
                population_model_results = participants_results[key]["population_model_results"]
                baseline_model_results = participants_results[key]["baseline_model_results"]
                biased_random_baseline_model_results = participants_results[key]["biased_random_baseline_model_results"]
                ground_truth = participants_results[key]["participant_ground_truth"]
                personalized_biased_random_baseline_model_results = participants_results[key]["personalized_biased_random_baseline_model_results"]

                personalized_preq_acc = accuracy_score(ground_truth, personalized_model_results)
                personalied_biased_random_baseline_accuracy = accuracy_score(ground_truth, personalized_biased_random_baseline_model_results)

                population_accuracy = accuracy_score(ground_truth, population_model_results)
                baseline_accuracy = accuracy_score(ground_truth, baseline_model_results)
                biased_random_baseline_accuracy = accuracy_score(ground_truth, biased_random_baseline_model_results)

                preqacc_personalized_model_results.append(personalized_preq_acc)
                acc_population_model_results.append(population_accuracy)
                acc_baseline_model_results.append(baseline_accuracy)
                acc_biased_baseline_model_results.append(biased_random_baseline_accuracy)
                acc_personalized_biased_baseline_model_results.append(personalied_biased_random_baseline_accuracy)
                keys.append(key)
        
        return preqacc_personalized_model_results, acc_population_model_results, acc_baseline_model_results, acc_biased_baseline_model_results, acc_personalized_biased_baseline_model_results, keys
        
  

