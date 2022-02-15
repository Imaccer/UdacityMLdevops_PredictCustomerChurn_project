'''
Test suite for library of functions for calculating customer churn

Author: Ian McNally
Date: Feb 14th 2022
'''

import os
import logging
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s %(name)s - %(levelname)s - %(message)s')

# @pytest.fixture
# def df():
# 	return cl.import_data("./data/bank_data.csv")

# @pytest.fixture
# def df2(df):
#     return cl.encoder_helper(df,cl.cat_columns,'_Churn')

# @pytest.fixture
# def traintest(df2):
#     X_train,X_test,y_train,y_test,X = cl.perform_feature_engineering(df2)
#     return X_train,X_test,y_train,y_test,X


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data - SUCCESS: Testing import_data")
    except FileNotFoundError as err:
        logging.error("Testing import_data - ERROR: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info(
            "Testing import_data - SUCCESS:\
			There are %i rows in your dataframe",df.shape[0])
        return df
    except AssertionError as err:
        logging.error(
            "Testing import_data - ERROR: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda, df):
    '''
    test perform eda function
    '''
    try:
        df = perform_eda(df)
        assert os.path.isfile("./images/eda/churn_distribution.png")
        assert os.path.isfile(
            "./images/eda/credit_limit_vs_total_transaction.png")
        assert os.path.isfile("./images/eda/customer_age_distribution.png")
        assert os.path.isfile("./images/eda/heatmap.png")
        assert os.path.isfile("./images/eda/marital_status_distribution.png")
        assert os.path.isfile(
            "./images/eda/total_transaction_distribution.png")
        logging.info(
            "Testing perform_eda - SUCCESS: All expected files created.")
        return df
    except AssertionError as err:
        logging.error("Testing perform_eda - ERROR: Missing file/s")
        raise err


def test_encoder_helper(encoder_helper, cat_columns, df, response):
    '''
    test encoder_helper
    '''
    df = encoder_helper(df, cat_columns, response)
    df_cols = df.columns

    try:
        for col in cat_columns:
            assert col in df_cols
        logging.info(
            "Testing encoder_helper - SUCCESS: Expected columns exist")
        return df
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper - ERROR: Dataframe is missing categorical column/s")
        raise err


def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    test perform_feature_engineering
    '''
    X_train, X_test, y_train, y_test, X = perform_feature_engineering(df)

    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info(
            'Testing perform_feature_engineering - SUCCESS:\
				 X_train,X_test,y_train,y_test populated with data')
        return X_train, X_test, y_train, y_test, X
    except AssertionError as err:
        logging.error(
            'Testing perform_feature_engineering - ERROR: One or more objects not populated.')
        raise err


def test_train_models(train_models, X_train, X_test, y_train, y_test, X):
    '''
    test train_models
    '''
    train_models(X_train, X_test, y_train, y_test, X)
    try:
        assert os.path.isfile("./images/results/feature_importances.png")
        assert os.path.isfile("./images/results/logistic_results.png")
        assert os.path.isfile("./images/results/rf_results.png")
        assert os.path.isfile("./images/results/roc_curve_result.png")
        logging.info('Testing train_models - SUCCESS: Result plots created')
    except AssertionError as err:
        logging.error('Testing train_models - ERROR: Result file/s missing')
        raise err

    try:
        assert os.path.isfile("./models/logistic_model.pkl")
        assert os.path.isfile("./models/rfc_model.pkl")
        logging.info('Testing train_models - SUCCESS: Models created')
    except AssertionError as err:
        logging.error('Testing train_models - ERROR: Model/s missing')
        raise err


def main():
    '''
    main  function for testing script
    '''
    df_import = test_import(cl.import_data)
    df_eda = test_eda(cl.perform_eda, df_import)
    df_helper = test_encoder_helper(
        cl.encoder_helper, cl.cat_columns, df_eda, '_Churn')
    X_train, X_test, y_train, y_test, X = test_perform_feature_engineering(
        cl.perform_feature_engineering, df_helper)
    test_train_models(cl.train_models, X_train, X_test, y_train, y_test, X)


if __name__ == "__main__":
    main()
    # pytest.main(args=['-s', os.path.abspath(__file__)])
