#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_factory.py - Comprehensive Model Factory Documentation
================================================================================
This module implements an automated machine learning model selection pipeline that:
1. Reads model configurations from YAML files
2. Dynamically imports and initializes multiple ML models from scikit-learn
3. Performs GridSearchCV hyperparameter tuning on each model
4. Selects the best performing model based on cross-validation scores

Key Features:
- YAML-based configuration (no hardcoding model parameters)
- Dynamic model loading via importlib
- Automated GridSearchCV for hyperparameter optimization
- Best model selection with configurable base accuracy threshold
- Comprehensive error handling with custom exceptions

Usage Example:
```python
factory = ModelFactory("model.yaml")
best_model = factory.get_best_model(X_train, y_train, base_accuracy=0.6)
```

Author: Machine Learning Aspirant
Date: April 10, 2026
"""

import importlib
import numpy as np
import yaml
import os
from typing import Optional, Any, List
from collections import namedtuple

# Assuming these are custom modules in your project
from exception import CustomException  # Custom exception handler
from logger import logging              # Custom logging utility

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
GRID_SEARCH_KEY = 'grid_search'
MODULE_KEY = 'module'
CLASS_KEY = 'class'
PARAM_KEY = 'params'
MODEL_SELECTION_KEY = 'model_selection'
SEARCH_PARAM_GRID_KEY = "search_param_grid"

# =============================================================================
# DATA STRUCTURES (NamedTuples for immutability and readability)
# =============================================================================
InitializedModelDetail = namedtuple(
    "InitializedModelDetail",
    ["model_serial_number", "model", "param_grid_search", "model_name"]
)
"""
Holds details of a single initialized model ready for grid search:
- model_serial_number: ID like "model_0"
- model: Instantiated scikit-learn model object
- param_grid_search: Dictionary of hyperparameters to search
- model_name: String like "sklearn.tree.DecisionTreeRegressor"
"""

GridSearchedBestModel = namedtuple(
    "GridSearchedBestModel", 
    ["model_serial_number", "model", "best_model", "best_parameters", "best_score"]
)
"""
Holds results after GridSearchCV completes:
- model_serial_number: Original model ID
- model: Original model (for reference)
- best_model: GridSearchCV.best_estimator_ (optimized model)
- best_parameters: GridSearchCV.best_params_ (optimal hyperparameters)
- best_score: GridSearchCV.best_score_ (CV score)
"""

# Note: BestModel namedtuple is defined but not used anywhere in the code
BestModel = namedtuple(
    "BestModel", 
    ["model_serial_number", "model", "best_model", "best_parameters", "best_score"]
)


def get_sample_model_config_yaml_file(export_dir: str) -> str:
    """
    HELPER FUNCTION: Creates a sample model.yaml configuration file
    
    Args:
        export_dir (str): Directory to save the sample YAML file
    
    Returns:
        str: Full path to the created YAML file
    
    Purpose:
        - Generates a template YAML file for users to understand the expected format
        - Useful during development/setup phase
        - Creates directory if it doesn't exist
    
    Example Usage:
        >>> get_sample_model_config_yaml_file("./config/")
        './config/model.yaml'
    """
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
        
        # Create directory and save YAML file
        os.makedirs(export_dir, exist_ok=True)
        export_file_path = os.path.join(export_dir, "model.yaml")
        with open(export_file_path, 'w') as file:
            yaml.dump(model_config, file, default_flow_style=False)
        return export_file_path
        
    except Exception as e:
        raise CustomException(e, "Error creating sample YAML config")


class ModelFactory:
    """
    MAIN CLASS: Factory pattern for automated model selection and hyperparameter tuning
    
    Workflow:
    1. Load YAML config → Extract GridSearchCV and model configurations
    2. Initialize models dynamically → Set fixed parameters
    3. Run GridSearchCV on each model → Find optimal hyperparameters
    4. Select best model → Return ready-to-use optimized model
    
    Attributes:
        config (dict): Complete YAML configuration
        grid_search_cv_module (str): GridSearchCV module path
        grid_search_class_name (str): GridSearchCV class name
        grid_search_property_data (dict): GridSearchCV parameters (cv, verbose)
        models_initialization_config (dict): All models to test
        initialized_model_list (List[InitializedModelDetail]): Initialized models
        grid_searched_best_model_list (List[GridSearchedBestModel]): Tuned models
    """
    
    def __init__(self, model_config_path:str):
        """
        Initialize ModelFactory with YAML configuration
        
        Args:
            model_config_path (str, optional): Path to model.yaml file
            
        Raises:
            CustomException: If config file is invalid or missing
        """
        try:
            # Load and parse YAML configuration
            self.config: dict = self.read_params(model_config_path)
            
            # Extract GridSearchCV configuration
            self.grid_search_cv_module: str = self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_class_name: str = self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_property_data: dict = dict(self.config[GRID_SEARCH_KEY][PARAM_KEY])
            
            # Extract model selection configuration
            self.models_initialization_config: dict = dict(self.config[MODEL_SELECTION_KEY])
            
            # Initialize empty lists for tracking
            self.initialized_model_list: Optional[List[InitializedModelDetail]] = None
            self.grid_searched_best_model_list: List[GridSearchedBestModel] = []
            
        except Exception as e:
            raise CustomException(e, "Error initializing ModelFactory") from e

    @staticmethod
    def update_property_of_class(instance_ref: Any, property_data: dict) -> Any:
        """
        DYNAMICALLY SET ATTRIBUTES ON ANY OBJECT
        
        Args:
            instance_ref (Any): Object to modify (model, GridSearchCV, etc.)
            property_data (dict): {parameter_name: value} dictionary
            
        Returns:
            Any: Modified object
            
        Example:
            >>> model = DecisionTreeRegressor()
            >>> update_property_of_class(model, {"max_depth": 5, "criterion": "squared_error"})
            # model now has max_depth=5 and criterion="squared_error"
        """
        try:
            if not isinstance(property_data, dict):
                raise Exception("property_data parameter required to be dictionary")
            
            # Use setattr() to dynamically set object attributes
            print(f"Setting properties: {property_data}")  # Debug output
            for key, value in property_data.items():
                setattr(instance_ref, key, value)
            return instance_ref
            
        except Exception as e:
            raise CustomException(e, "Error updating object properties") from e

    @staticmethod
    def read_params(config_path: Optional[str]) -> dict:
        """
        LOAD AND PARSE YAML CONFIGURATION FILE
        
        Args:
            config_path (str): Path to model.yaml
            
        Returns:
            dict: Parsed YAML configuration
            
        Raises:
            ValueError: If config_path is None
            CustomException: If YAML parsing fails
        """
        try:
            if config_path is None:
                raise ValueError("config path cannot be None")
            
            with open(config_path) as yaml_file:
                config = yaml.safe_load(yaml_file)
            return config
            
        except Exception as e:
            raise CustomException(e, f"Error reading config from {config_path}") from e

    @staticmethod
    def class_for_name(module_name: str, class_name: str) -> Any:
        """
        DYNAMICALLY IMPORT CLASS FROM MODULE STRING
        
        Args:
            module_name (str): Module path like "sklearn.tree"
            class_name (str): Class name like "DecisionTreeRegressor"
            
        Returns:
            Any: Class reference (not instance)
            
        Example:
            >>> class_ref = class_for_name("sklearn.tree", "DecisionTreeRegressor")
            >>> model = class_ref()  # Now create instance
        """
        try:
            # Dynamically import module (raises ImportError if module missing)
            module = importlib.import_module(module_name)
            # Get class from module (raises AttributeError if class missing)
            class_ref = getattr(module, class_name)
            return class_ref
            
        except Exception as e:
            raise CustomException(e, f"Error loading {class_name} from {module_name}") from e

    def execute_grid_search_operation(
        self, 
        initialized_model: InitializedModelDetail, 
        input_feature: np.ndarray,
        output_feature: np.ndarray
    ) -> GridSearchedBestModel:
        """
        CORE FUNCTION: Perform GridSearchCV hyperparameter tuning on single model
        
        Args:
            initialized_model (InitializedModelDetail): Pre-initialized model details
            input_feature (np.ndarray): Training features (X_train)
            output_feature (np.ndarray): Training target (y_train)
            
        Returns:
            GridSearchedBestModel: Best model with optimal parameters
            
        LINE-BY-LINE EXPLANATION:
        1. Log training start message
        2. Dynamically import GridSearchCV class
        3. Create GridSearchCV instance with model + param_grid
        4. Set additional GridSearchCV properties (cv, verbose)
        5. Fit on training data (performs grid search)
        6. Extract best model, params, and score
        7. Package results in namedtuple
        """
        try:
            # LINE 1: Training log message
            message = f"{'*' * 50} training {type(initialized_model.model).__name__} {'*' * 50}"
            logging.info(message)
            
            # LINE 2: Dynamically load GridSearchCV class
            grid_search_cv_ref = ModelFactory.class_for_name(
                module_name=self.grid_search_cv_module,
                class_name=self.grid_search_class_name
            )
            
            # LINE 3: Create GridSearchCV instance
            # estimator=base model, param_grid=hyperparameters to test
            grid_search_cv = grid_search_cv_ref(
                estimator=initialized_model.model,
                param_grid=initialized_model.param_grid_search
            )
            
            # LINE 4: Set GridSearchCV properties (cv=3, verbose=1)
            grid_search_cv = ModelFactory.update_property_of_class(
                grid_search_cv, self.grid_search_property_data
            )
            
            # LINE 5: Execute grid search (fits all parameter combinations)
            grid_search_cv.fit(input_feature, output_feature)
            
            # LINE 6: Extract best results from GridSearchCV
            grid_searched_best_model = GridSearchedBestModel(
                model_serial_number=initialized_model.model_serial_number,
                model=initialized_model.model,
                best_model=grid_search_cv.best_estimator_,    # Optimized model
                best_parameters=grid_search_cv.best_params_,  # Best hyperparameters
                best_score=grid_search_cv.best_score_         # Best CV score
            )
            return grid_searched_best_model
            
        except Exception as e:
            raise CustomException(e, "Error in grid search operation") from e

    def get_initialized_model_list(self) -> List[InitializedModelDetail]:
        """
        INITIALIZE ALL MODELS FROM CONFIG
        
        LINE-BY-LINE EXPLANATION:
        1. Loop through each model in config (model_0, model_1, etc.)
        2. Dynamically import and instantiate model class
        3. Set fixed parameters from config (if provided)
        4. Extract hyperparameter grid for GridSearchCV
        5. Create model detail record
        6. Store in class attribute and return list
        """
        try:
            initialized_model_list = []
            
            # LOOP THROUGH EACH MODEL IN CONFIG
            for model_serial_number in self.models_initialization_config.keys():
                # Get config for current model
                model_initialization_config = self.models_initialization_config[model_serial_number]
                
                # DYNAMICALLY CREATE MODEL INSTANCE
                model_obj_ref = ModelFactory.class_for_name(
                    module_name=model_initialization_config[MODULE_KEY],
                    class_name=model_initialization_config[CLASS_KEY]
                )
                model = model_obj_ref()  # Instantiate empty model
                
                # SET FIXED PARAMETERS (not searched)
                if PARAM_KEY in model_initialization_config:
                    model_obj_property_data = dict(model_initialization_config[PARAM_KEY])
                    model = ModelFactory.update_property_of_class(
                        instance_ref=model, property_data=model_obj_property_data
                    )
                
                # EXTRACT HYPERPARAMETER GRID (for GridSearchCV)
                param_grid_search = model_initialization_config[SEARCH_PARAM_GRID_KEY]
                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}"
                
                # PACKAGE MODEL DETAILS
                model_initialization_config = InitializedModelDetail(
                    model_serial_number=model_serial_number,
                    model=model,
                    param_grid_search=param_grid_search,
                    model_name=model_name
                )
                initialized_model_list.append(model_initialization_config)
            
            self.initialized_model_list = initialized_model_list
            return self.initialized_model_list
            
        except Exception as e:
            raise CustomException(e, "Error initializing model list") from e

    def initiate_best_parameter_search_for_initialized_model(
        self, 
        initialized_model: InitializedModelDetail,
        input_feature: np.ndarray, 
        output_feature: np.ndarray
    ) -> GridSearchedBestModel:
        """
        WRAPPER: Single model grid search (calls execute_grid_search_operation)
        """
        try:
            return self.execute_grid_search_operation(
                initialized_model=initialized_model,
                input_feature=input_feature,
                output_feature=output_feature
            )
        except Exception as e:
            raise CustomException(e, "Error in single model parameter search") from e

    def initiate_best_parameter_search_for_initialized_models(
        self,
        initialized_model_list: List[InitializedModelDetail],
        input_feature: np.ndarray, 
        output_feature: np.ndarray
    ) -> List[GridSearchedBestModel]:
        """
        BATCH PROCESS: Grid search all models
        
        LINE-BY-LINE:
        1. Loop through each initialized model
        2. Run grid search on individual model
        3. Store results in class attribute
        4. Return complete list
        """
        try:
            for initialized_model in initialized_model_list:
                grid_searched_best_model = self.initiate_best_parameter_search_for_initialized_model(
                    initialized_model=initialized_model,
                    input_feature=input_feature,
                    output_feature=output_feature
                )
                self.grid_searched_best_model_list.append(grid_searched_best_model)
            return self.grid_searched_best_model_list
            
        except Exception as e:
            raise CustomException(e, "Error in batch model parameter search") from e

    @staticmethod
    def get_model_detail(
        model_details: List[InitializedModelDetail], 
        model_serial_number: str
    ) -> Optional[InitializedModelDetail]:
        """
        UTILITY: Find specific model by serial number
        """
        try:
            for model_data in model_details:
                if model_data.model_serial_number == model_serial_number:
                    return model_data
            return None  # Not found
        except Exception as e:
            raise CustomException(e, "Error finding model detail") from e

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(
        grid_searched_best_model_list: List[GridSearchedBestModel],
        base_accuracy: float = 0.6
    ) -> GridSearchedBestModel:
        """
        SELECT BEST MODEL FROM GRID SEARCH RESULTS
        
        LINE-BY-LINE:
        1. Initialize best_model as None
        2. Loop through all grid-searched models
        3. Track model with highest CV score above base_accuracy
        4. Raise exception if no acceptable model found
        5. Log and return best model
        """
        try:
            grid_search_best_model: Optional[GridSearchedBestModel] = None
            for grid_searched_best_model in grid_searched_best_model_list:
                # Found better model than current best
                if base_accuracy < grid_searched_best_model.best_score:
                    logging.info(f"Acceptable model found: {grid_searched_best_model}")
                    base_accuracy = grid_searched_best_model.best_score
                    grid_search_best_model = grid_searched_best_model
            
            # No acceptable model found
            if not grid_search_best_model:
                raise Exception(f"None of Model has base accuracy: {base_accuracy}")
            
            logging.info(f"Best model: {grid_search_best_model}")
            return grid_search_best_model
            
        except Exception as e:
            raise CustomException(e, "Error selecting best model") from e

    def get_best_model(self, X: np.ndarray, y: np.ndarray, base_accuracy: float = 0.6) -> GridSearchedBestModel:
        """
        MAIN ENTRY POINT: Complete automated model selection pipeline
        
        COMPLETE WORKFLOW:
        1. Initialize all models from YAML config
        2. Run GridSearchCV on every model
        3. Select best model based on CV score
        4. Return production-ready optimized model
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training target
            base_accuracy (float): Minimum acceptable CV score
            
        Returns:
            GridSearchedBestModel: Best optimized model ready for predictions
        """
        try:
            logging.info("Started Initializing model from config file")
            
            # STEP 1: Initialize all models
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f"Initialized models: {len(initialized_model_list)} models")
            
            # STEP 2: Grid search all models
            grid_searched_best_model_list = self.initiate_best_parameter_search_for_initialized_models(
                initialized_model_list=initialized_model_list,
                input_feature=X,
                output_feature=y
            )
            
            # STEP 3: Select best model
            best_model = ModelFactory.get_best_model_from_grid_searched_best_model_list(
                grid_searched_best_model_list,
                base_accuracy=base_accuracy
            )
            
            logging.info("Best model selection completed successfully")
            return best_model
            
        except Exception as e:
            raise CustomException(e, "Error in complete model selection pipeline")


"""
COMPLETE USAGE EXAMPLE:
================================================================================

# 1. Create or use existing model.yaml (see provided example)
# 2. Initialize factory
factory = ModelFactory("model.yaml")

# 3. Get best model (automates everything!)
best_model_result = factory.get_best_model(X_train, y_train, base_accuracy=0.6)

# 4. Use best model for predictions
predictions = best_model_result.best_model.predict(X_test)

print(f"Best Model: {best_model_result.model_name}")
print(f"Best Parameters: {best_model_result.best_parameters}")
print(f"Best CV Score: {best_model_result.best_score:.4f}")
"""

# Save this file as 'model_factory_documented.py' in VSCode for reference!