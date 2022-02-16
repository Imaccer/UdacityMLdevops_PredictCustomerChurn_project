# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This is a machine learning project to predict customer churn for a credit card company.

Best coding practices have been used for this project including testing, logging, [PEP8 formatting](https://www.python.org/dev/peps/pep-0008/), and linting with pylint.

## Create Virtual Environment
To create a virtual environment with another Python version, take the following steps:

  *  Download the Python version that you need (in this case, Python 3.6.3)
  *  Install the Python executable
  * To create a new virtual env using python 3.6 run this:

```
</path/to/python/you/want/to/use> -m venv <path/to/new/virtualenv/>

e.g.
C:\Users\imcnally\AppData\Local\Programs\Python\Python36\python -m venv .venv
```
## Windows/Linux/Mac
To activate the virtual environment, open a git bash terminal (or a terminal in Linux/Mac) and run:
```
source ./.venv/Scripts/activate
```
If on Windows, you may need to execute this in powershell as admin to ensure git bash has permission to run commands from the activated environment:
```
Add-MpPreference -AttackSurfaceReductionOnlyExclusions "C:\Users\imcnally\source\project_name\.venv\Scripts"
```

Once the environment is activated, run:
```
pip install -r requirements.txt
```

## Running files

### Data science process

To train the machine learning models and output EDA figures, run:
```
python churn_library.py
```
This file completes the process for solving the data science process:

1. EDA
    * EDA plots stored in ./images/eda/
2. Feature Engineering (including encoding of categorical variables)
3. Model Training
    * Model files stored in .pkl format in ./models/
4. Prediction
5. Model Evaluation
    * Results plots stored in ./images/results/

### Testing and logging 

To test and log the result of each function, run:
```
python churn_script_logging_and_testing.py
```
The logging output is stored in ./logs/churn_library.log

Note, the functions classification_report_image, feature_importance_plot, and roc_curve_plot do not have corresponding test functions. However, their testing is included in the function test_train_models.

### Code formatting and linting

The following command was run to format the code according to PEP8 guidelines:
```
autopep8 --in-place --aggressive --aggressive churn_library.py
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
```

Then Pylint can be run with:
```
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```

Note, have added list of good names to .pylintrc file to preserve machine learning naming conventions (e.g., X_train, X_test etc.) without lowering pylint score for variable names not meeting pylint requirements.

### Note to reviewer

Accidentally committed to the repo with my work github account (imcnally20), hence, why there are 2 contributors to the repo. Both account (Imaccer and imcnally20) belong to me.


