import numpy as np 
import uproot as ur
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score, roc_auc_score, recall_score, precision_score, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.inspection import permutation_importance

import xgboost as xgb
from xgboost import plot_importance
import shap

# ================================================================================
# DATA Prepareation
# ================================================================================
def sig_bkg_data_loading(sig_file, bkg_file, features, sample_weights, preselection):
    '''
    Loading the .root datasets for signal and backgrouund with events features and preselection
    '''
    sig_tree = ur.open(sig_file)['mytree']
    bkg_tree = ur.open(bkg_file)['mytree']
    
    sig_dict = sig_tree.arrays(features, library='np', cut=preselection)
    bkg_dict = bkg_tree.arrays(features, library='np', cut=preselection)

    signal = np.stack(list(sig_dict.values()))
    backgr = np.stack(list(bkg_dict.values()))

    sig_weights = sig_tree.arrays(sample_weights, library='np', cut=preselection)[sample_weights]
    bkg_weights = np.ones(backgr.shape[1])

    return signal, backgr, sig_weights, bkg_weights

# def prepare_combined_data(sig_file, bkg_file, features, sample_weights, preselection, test_size=0.2, random_seed=2001):
def prepare_combined_data(sig_dataset, bkg_dataset, sig_weights, bkg_weights, test_size=0.2, random_seed=2001):

    '''
    Combining the signal and baclground into X, y
    '''
    ### TODO: account for the unbalanced calsses using stratified KFold
    
    # Loading the dataset
    # sig, bkg, sig_weights, bkg_weights = sig_bkg_data_loading(sig_file, bkg_file, features, sample_weights, preselection)


    # combining sig+bkg
    X = np.transpose(np.concatenate((sig_dataset, bkg_dataset), axis=1))
    y = np.concatenate((np.ones(sig_dataset.shape[1]), np.zeros(bkg_dataset.shape[1])))
    weights = np.concatenate((sig_weights, bkg_weights))

    # Train-test Split
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=test_size, random_state=random_seed)

    return X_train, X_test, y_train, y_test, weights_train, weights_test


# ================================================================================
# Metrics Displays
# ================================================================================
def display_cf_matrix(y_test, y_pred, class_names=None, normalization=False):
    '''
    Display the confusion matrix
    '''
    if class_names is None:
        class_names = ['Background', 'Signal'] # 0 = bkg and 1 = sig
    
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred = (y_pred >= 0).astype(int)  # bdt_score = 0 is the threshold between background and signal
    # need to invert the bdt_score into label
    # NOTE: no need if y_pred = bdt.predict(X_test)


    cm = confusion_matrix(y_test, y_pred)
    if normalization:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Signal vs Background)')
    plt.show()

def display_roc_curve(y_test, y_pred):
    '''
    Plotting Roc Curve
    '''
    # computing fpr, tpr, ts baseline
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    # roc_auc score
    roc_auc = auc(fpr, tpr)
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def display_pr_curve(y_test, y_pred):
    '''
    Plotting PR Curve
    '''
    # computing fpr, tpr, ts baseline
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    # pr_auc score
    pr_auc = average_precision_score(y_test, y_pred) # taking (y_test, y_pred) not (precision, recall)
    # Plot ROC curve
    plt.figure()
    plt.plot(precision, recall, color='darkorange', lw=2, label='Precision-Recall curve (AP = %0.4f)' % pr_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.show()

# ================================================================================
# Features Analysis
# ================================================================================
def plot_features_importance(model, feature_names=None, importance_type='gain', max_num_features=15):
    '''
    Ploting the features importance in the trained model
    note: this is soecific to the xgbbost classifier (not generic) becasue we using xgb api for convenience
    https://xgboosting.com/xgboost-get_booster/
    '''
    booster = model.get_booster() # get_booster() helps accessing the underlying native features
    importance = booster.get_score(importance_type=importance_type)

    # Sort by importance value and select top N
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_importance[:max_num_features]

    # Prepare labels and scores
    keys = [k for k, _ in top_features]
    scores = [v for _, v in top_features]
    
    # Map f0, f1, ... to actual feature names
    if feature_names:
        labels = [feature_names[int(k[1:])] if k.startswith('f') else k for k in keys]
    else:
        labels = keys

    # Plotting manually for more control
    plt.figure(figsize=(8, max(4, len(labels) * 0.5)))
    plt.barh(labels[::-1], scores[::-1])  # reverse for top-to-bottom
    plt.xlabel(importance_type.capitalize(), loc='right')
    plt.ylabel('Features', loc='top')
    plt.title(f"Top {max_num_features} Feature Importances ({importance_type})")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    return

def shap_summary(model, X, feature_names=None):
    '''
    Investigating the individual predictions using game theory
    This could be really helpful for non-linear model.
    https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Basic%20SHAP%20Interaction%20Value%20Example%20in%20XGBoost.html
    
    It tells us the predictive power of each feature --> how much each feature chaanges the mdoel's prediction.
    So, it does not concern about the true label, but compare the model prdiction to some sort of baseline (average prediction).
    '''
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, features=X, feature_names=feature_names)
    return

def plot_permutation_importance(model, X, y, feature_names=None, n_repeats=10, scoring='roc_auc', random_state=2001):
    '''
    NOTE: make sure that X is transform the way it was trained!
    This could be generic for any model type.
    https://scikit-learn.org/stable/modules/permutation_importance.html
    '''
    result = permutation_importance(model, X, y, 
                                    n_repeats=n_repeats,
                                    scoring=scoring,
                                    random_state=random_state)
    
    sorted_idx = result.importances_mean.argsort()[::-1]  # Descending order
    sorted_features = np.array(feature_names)[sorted_idx] if feature_names is not None else sorted_idx

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)),
             result.importances_mean[sorted_idx],
             xerr=result.importances_std[sorted_idx],
             align='center')
    plt.yticks(range(len(sorted_idx)), sorted_features)
    plt.ylabel('Features')
    plt.xlabel('Permutation Importance (mean decrease in score)')
    plt.title('Permutation Feature Importance')
    plt.grid()
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.tight_layout()
    plt.show()