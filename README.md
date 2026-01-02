# CycPeptMP3D
# Structure of the repo:
- `cpmp`, `egnn`, `se3_transformer` contains the source code from the original repo of the respective models.
- `src` contains the source code to run the training and inferences of the models.
# Structure of the `src` folder:
```
src/
├── __init__.py
├── arguments.py - Define the common arguments for the training process.
├── callbacks.py - Define the hooks to be called at common state in the training process.
├── data_module.py - Define the base class for the dataset.
├── dataset.py - Implement the specific datasets for each model.
├── gpu_affinity.py - Define the utilities for multi GPU training.
├── loggers.py - Define logging.
├── main.py - Main script to run the training, validation, and testing
├── models.py - Define a light wrapper around the model to be compatible with the `Trainer` class.
├── test.py
├── training.py - Define the `Trainer` class.
└── utils.py - Define the utilities for the training process.
```
# Design choices
- The `Trainer` class must be independent of the specific data module and model used.
- The common parameters that define an experiment should be an argument in the `arguments.py` file. If it is an argument specific to a model or a dataset, then it must be put in the `add_argparse_args` method of the specific data model or model definition. **Note**: The new argument cannot contain the same name as the previous argument.
- The scripts are initialized as module with the `__init__.py` file in all the folders. This is to ensure that all the functions and classes are accessible anywhere in the repo and to avoid the anti-pattern of directly modifying the system interpreter path environment variable. **Note**: For notebooks not in the root folder modify the system interpreter but is still necessary. This is a feature/bug of the Jupyter notebook.
# How to run
Refer to the `scripts` folder for the minimally required arguments to be set for each model and dataset. All the models and datasets's specific arguments have the original research parameters as the default value. **Note:** (If one wants to experiment with the parameters of the model or the dataset) For models, the specific arguments are set in their respective original folders. For datasets, the arguments are set in the `dataset.py` file.