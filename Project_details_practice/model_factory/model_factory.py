import os
import yaml
from collections import defaultdict
from typing import Any, List

from exception import CustomException
from logger import logging
from constants import *

import numpy as np
from dataclasses import dataclass, field

# models
import importlib


def get_sample_model_config_yaml_file(export_dir:str):
    
    try:
        # Sample configuration structure
        model_config = {
            GRID_SEARCH_KEY: {
                MODULE_KEY: "sklearn.model_selection",
                CLASS_KEY: "GridSearchCV",
                PARAM_KEY: {
                    "cv": 3,        # 3-fold cross-validation
                    "verbose": 1    # Show progress during search
                }
            },
            MODEL_SELECTION_KEY: {
                "module_0": {   # First model to test
                    MODULE_KEY: "module_of_model",      # Replace with actual module
                    CLASS_KEY: "ModelClassName",        # Replace with actual class
                    PARAM_KEY: {
                        "param_name1": "value1",
                        "param_name2": "value2",
                    },
                    SEARCH_PARAM_GRID_KEY: {
                        "param_name": ['param_value_1', 'param_value_2']
                    }
                },
            }
        }
        
        # make directory
        os.makedirs(export_dir, exist_ok=True)
        
        # config file path
        export_file_path = os.path.join(export_dir, MODEL_CONFIG_FILENAME)
        
        # save model dict in file
        with open(export_file_path, 'w') as file_obj:
            yaml.dump(model_config, file_obj, default_flow_style=False)
            
    except Exception as e:
        raise CustomException(e) from e



@dataclass
class Untuned_Model:
    """
    model: Instantiated scikit-learn model object
    model_detail : dict {'model_serial_number' : 'model_0', 'model_name' : "..."}
                    model_serial_number: ID like "model_0"
                    model_name: String like "sklearn.tree.DecisionTreeRegressor"
    grid_search_parameters: Dictionary of hyperparameters to search
    """
    model : Any
    model_detail : dict = field(default_factory= lambda: defaultdict(str))
    grid_search_parameters : dict = field(default_factory=dict)


@dataclass
class Grid_Searched_Model:
    """
    tuned_model : grid searched model with best parameters
    model_detail : dict {'model_serial_number' : 'model_0', 'model_name' : "..."}
                    model_serial_number: ID like "model_0"
                    model_name: String like "sklearn.tree.DecisionTreeRegressor"    
    best_parameters = grid searched best parameters for the model type (ex. decision tree)
    metrics = {'val_r2_score' : val, 'val_r2_std' : val, 'overfit_gap' : val}
    """
    tuned_model : Any
    model_detail : dict = field(default_factory= lambda: defaultdict(str))
    best_parameters : dict = field(default_factory=dict)
    metrics : dict = field(default_factory= lambda: defaultdict(float))

    
    

class Model_Factory:
    
    def __init__(self, model_config_file_path:str):
        try:
            self.model_config = self.read_config_yaml_file(file_path=model_config_file_path)
            
            # initialize grid search details
            self.grid_search_details: dict = self.model_config[GRID_SEARCH_KEY]
            
            # initalize untuned model details
            self.models_details: dict = self.model_config[MODEL_SELECTION_KEY]
            
            # initialize the lists 
            self.Grid_Searched_Models_List: List[Grid_Searched_Model] = []
            self.Untuned_Models_List: List[Untuned_Model] = []
             
        except Exception as e:
            raise CustomException(e) from e
        
    
    @staticmethod
    def read_config_yaml_file(file_path:str)->dict:
        try:
            if file_path is None:
                raise ValueError("Config path is given as None")
            
            with open(file_path, 'r') as yaml_file_obj:
                model_config = yaml.safe_load(file_path)
            
            return model_config
        
        except Exception as e:
            raise CustomException(e) from e
    
    
    @staticmethod
    def get_model_class_reference(module_name:str, class_name:str)->Any:
        """
        Dynamically import class from string

        Returns:
            Any: Example. <class sklearn.model_selection.DecisionTreeRegressor> class reference 
        """
        try:
            module = importlib.import_module(module_name)
            class_reference = getattr(module, class_name)
        
            return class_reference
        
        except Exception as e:
            raise CustomException(e) from e
        
    
    @staticmethod
    def set_model_class_properties(model_obj:Any, property_data:dict)-> Any:
        """
        Set the parameters for the model object (instance_ref)

        Args:
            model_obj (Any): Example. DecisionTreeRegressor()
            property_data (dict): {'criterion' : 'squared_error', 'min_samples_leaf' : 2,
            'max_depth' : [2,3,4,5,6,7,8,9]}
        """
        try:
            if not isinstance(property_data, dict):
                raise Exception("property_data parameter required to be dictionary")
            
            for property_name, property_value in property_data.items():
                setattr(model_obj, property_name, property_value)
            
            return model_obj
            
        except Exception as e:
            raise CustomException(e) from e
        

    def initiate_untuned_models_list(self)->None:
        try:
            for model_number in self.models_details:
                
                # initialize base_model: DecisionTreeRegressor()
                module_name = model_number[MODULE_KEY]
                class_name = model_number[CLASS_KEY]
                model_reference = self.get_model_class_reference(module_name=module_name, 
                                                                 class_name=class_name)
                base_model = model_reference()
                
                # set model parameters/property: DecisionTreeRegressor(criterion='...', 
                # min_samples_leaf=...)
                model_property = model_number[PARAM_KEY]  
                model = self.set_model_class_properties(model_obj=base_model, property_data=model_property)
                
                
                # grid search parameters
                model_grid_search_parameters = model_number[SEARCH_PARAM_GRID_KEY] 
                
                # model details
                model_name = model.__class__.__name__
                
                # initiate untuned model
                untuned_model = Untuned_Model(model=model)
                
                untuned_model.grid_search_parameters = model_grid_search_parameters
                untuned_model.model_detail[MODEL_NAME] = model_name
                untuned_model.model_detail[MODEL_NUMBER] = model_number
                
                
                # append to untuned models list
                self.Untuned_Models_List.append(untuned_model)
                
        except Exception as e:
            raise CustomException(e) from e  


    def initiate_model_factory(self):
        try:
            # set up the list of untuned models
            pass
        except Exception as e:
            raise CustomException(e) from e    