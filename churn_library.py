'''
Library of functions for calculating customer churn

Author: Ian McNally
Date: Jan 21st 2022
'''

# import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

quant_columns = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth)


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    fig1 = plt.figure(figsize=(20, 10))
    sns.histplot(df['Churn'].astype('category'), stat="probability", shrink=.2)
    plt.xticks([0, 1])
    plt.title("Churn Distribution")
    plt.savefig('./images/eda/churn_distribution.png', bbox_inches='tight')
    plt.close(fig1)

    fig2 = plt.figure(figsize=(20, 10))
    sns.histplot(df['Customer_Age'])
    plt.title("Customer Age Distribution")
    plt.xlabel('Customer Age')
    plt.savefig(
        './images/eda/customer_age_distribution.png',
        bbox_inches='tight')
    plt.close(fig2)

    fig3 = plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title("Heat Map")
    plt.savefig('./images/eda/heatmap.png', bbox_inches='tight')
    plt.close(fig3)

    fig4 = plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title("Marital Status Distribution")
    plt.xlabel('Marital Status')
    plt.savefig(
        './images/eda/marital_status_distribution.png',
        bbox_inches='tight')
    plt.close(fig4)

    fig5 = plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Amt'])
    plt.title("Total Transaction Distribution")
    plt.xlabel('Total Transaction Amt')
    plt.savefig(
        './images/eda/total_transaction_distribution.png',
        bbox_inches='tight')
    plt.close(fig5)

    fig6 = plt.figure(figsize=(20, 10))
    sns.scatterplot(
        data=df,
        x='Credit_Limit',
        y='Total_Trans_Amt',
        hue='Gender',
        alpha=0.3,
        size=0.1)
    plt.savefig(
        './images/eda/credit_limit_vs_total_transaction.png',
        bbox_inches='tight')
    plt.close(fig6)


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
            be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    tmp_lst = []
    tmp_series = pd.Series()
    for col in category_lst:
        tmp_series = df.groupby(col).mean()['Churn']
        for val in df[col]:
            tmp_lst.append(tmp_series.loc[val])
        df[col + response] = tmp_lst
        tmp_lst = []

    return df


def perform_feature_engineering(df):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could
              be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = df['Churn']
    X = pd.DataFrame()

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X = df[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test, X


def classification_report_image(y_train,
                                y_test,
                                predictions):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            predictions['y_train_preds_lr']: training predictions from logistic regression
            predictions['y_train_preds_rf']: training predictions from random forest
            predictions['y_test_preds_lr']: test predictions from logistic regression
            predictions['y_test_preds_rf']: test predictions from random forest

    output:
             None
    '''
    # Random Forest Results
    fig1 = plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, predictions['y_test_preds_rf'])), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, predictions['y_train_preds_rf'])), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/rf_results.png', bbox_inches='tight')
    plt.close(fig1)

    # Logistic Regression Results
    fig2 = plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, predictions['y_train_preds_lr'])), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_test, predictions['y_test_preds_lr'])), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/logistic_results.png', bbox_inches='tight')
    plt.close(fig2)


def feature_importance_plot(model, X, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X.columns[i] for i in indices]

    # Create plot
    fig1 = plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X.shape[1]), names, rotation=90)

    # Save figure and close
    plt.savefig(output_pth, bbox_inches='tight')
    plt.close(fig1)


def roc_curve_plot(rfc_model, lr_model, X_test, y_test, output_pth):
    '''
    creates and stores the roc curves in pth
    input:
            model: model object containing best_estimator_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    # plots
    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
    fig1 = plt.figure(figsize=(20, 10))
    ax = plt.gca()
    plot_roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.title("ROC Curve")
    plt.savefig(output_pth, bbox_inches='tight')
    plt.close(fig1)


def train_models(X_train, X_test, y_train, y_test, X):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
              X : complete data to pass to feature importance function
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(max_iter=500)

    param_grid = {
        'n_estimators': [5, 10],  # reduced number estimators to speed up
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # make predictions
    predictions = {
        'y_train_preds_rf': cv_rfc.best_estimator_.predict(X_train),
        'y_test_preds_rf': cv_rfc.best_estimator_.predict(X_test),
        'y_train_preds_lr': lrc.predict(X_train),
        'y_test_preds_lr': lrc.predict(X_test)
    }

    classification_report_image(y_train,
                                y_test,
                                predictions)

    roc_curve_plot(cv_rfc.best_estimator_, lrc, X_test, y_test,
                   './images/results/roc_curve_result.png')
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X,
        './images/results/feature_importances.png')


def main():
    '''
    Create main() function which be holding all the main logic etc.
    every new function instance creates a new local scope.
    '''
    df = import_data("./data/bank_data.csv")
    perform_eda(df)
    df2 = encoder_helper(df, cat_columns, '_Churn')
    X_train, X_test, y_train, y_test, X = perform_feature_engineering(df2)
    train_models(X_train, X_test, y_train, y_test, X)


if __name__ == "__main__":
    main()
