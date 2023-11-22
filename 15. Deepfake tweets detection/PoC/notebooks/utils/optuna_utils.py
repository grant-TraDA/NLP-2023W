import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from optuna import Trial
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier


def instantiate_lgbm(trial: Trial) -> LGBMClassifier:
    """
    Instantiate a LGBMClassifier with the parameters suggested by Optuna.
    :param trial: Optuna Trial object
    :return: LGBMClassifier with suggested parameters
    """
    params = {
        "boosting_type": trial.suggest_categorical('lgbm_boosting_type', ['gbdt', 'dart']),
        "max_depth": trial.suggest_int('lgbm_max_depth', 1, 15),
        "n_estimators": trial.suggest_int('lgbm_n_estimators', 10, 500, log=True),
        "subsample": trial.suggest_float('lgbm_subsample', 0.6, 1),
        'n_jobs': 6
    }
    return LGBMClassifier(**params)


def instantiate_xgb(trial: Trial) -> XGBClassifier:
    """
    Instantiate a XGBClassifier with the parameters suggested by Optuna.
    :param trial: Optuna Trial object
    :return: XGBClassifier with suggested parameters
    """
    params = {
        "booster": trial.suggest_categorical('xgb_booster', ['gbtree', 'dart']),
        "max_depth": trial.suggest_int('xgb_max_depth', 1, 15),
        "n_estimators": trial.suggest_int('xgb_n_estimators', 10, 500, log=True),
        "subsample": trial.suggest_float('xgb_subsample', 0.6, 1),
        'n_jobs': 6
    }
    return XGBClassifier(**params)


def instantiate_rf(trial: Trial) -> RandomForestClassifier:
    """
    Instantiate a RandomForestClassifier with the parameters suggested by Optuna.
    :param trial: Optuna Trial object
    :return: RandomForestClassifier with suggested parameters
    """
    params = {
        "max_depth": trial.suggest_int('rf_max_depth', 1, 15),
        "n_estimators": trial.suggest_int('rf_n_estimators', 10, 500, log=True),
        "criterion": trial.suggest_categorical('rf_criterion', ['gini', 'entropy', 'log_loss']),
        "min_samples_split": trial.suggest_float('rf_min_samples_split', 0.01, 0.1),
        'n_jobs': 6
    }
    return RandomForestClassifier(**params)


def instantiate_svc(trial: Trial) -> SVC:
    """
    Instantiate a SVC with the parameters suggested by Optuna.
    :param trial: Optuna Trial object
    :return: SVC with suggested parameters
    """
    params = {
        "kernel": trial.suggest_categorical('svc_kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        "C": trial.suggest_float('svc_C', 1e-2, 1e2, log=True)
    }
    return SVC(**params)


def instantiate_lr(trial: Trial) -> LogisticRegression:
    """
    Instantiate a LogisticRegression with the parameters suggested by Optuna.
    :param trial: Optuna Trial object
    :return: LogisticRegression with suggested parameters
    """
    params = {
        "solver": 'saga',
        "penalty": trial.suggest_categorical('lr_penalty', ['l1', 'l2']),
        "C": trial.suggest_float('lr_C', 1e-2, 1e2, log=True),
        "max_iter": 1000,
        "n_jobs": 6
    }
    return LogisticRegression(**params)


# Mapping from ngram name to ngram range
NGRAMS_MAPPING = {
    "unigram": (1, 1),
    "digram": (1, 2),
    "trigram": (1, 3)
}


def instantiate_tfidf(trial: Trial) -> TfidfVectorizer:
    """
    Instantiate a TfidfVectorizer with the parameters suggested by Optuna.
    :param trial: Optuna Trial object
    :return: TfidfVectorizer with suggested parameters
    """
    ngram = trial.suggest_categorical('tfidf_ngram_range', ["unigram", "digram", "trigram"])
    ngram_range = NGRAMS_MAPPING[ngram]
    params = {
        "max_features": trial.suggest_int('tfidf_max_features', 1000, 10000, log=True),
        "ngram_range": ngram_range,
        "max_df": trial.suggest_float('tfidf_max_df', 0.8, 1.0),
        "min_df": trial.suggest_float('tfidf_min_df', 0.0, 0.2),
    }
    return TfidfVectorizer(**params)


# Mapping from model name to model instantiation function
MODELS_MAPPING = {
    'LGBM': instantiate_lgbm,
    'XGB': instantiate_xgb,
    'RF': instantiate_rf,
    'SVC': instantiate_svc,
    'LR': instantiate_lr
}

# Mapping from encoder name to encoder instantiation function
ENCODERS_MAPPING = {
    'TFIDF': instantiate_tfidf
}

# Mapping from model name to model class
MODELS = {
    'LGBM': LGBMClassifier,
    'XGB': XGBClassifier,
    'RF': RandomForestClassifier,
    'SVC': SVC,
    'LR': LogisticRegression
}

# Mapping from encoder name to encoder class
ENCODERS = {
    'TFIDF': TfidfVectorizer
}


def extract_model_encoder(params: dict) -> (str, str, dict, dict):
    """
    Extract the model and encoder from the parameters' dictionary.
    :param params: parameters dictionary
    :return: model name, model parameters, encoder name (if present), encoder parameters (if present)
    """
    model_params = {}
    encoder_params = {}
    for key in params.keys():
        begin, end = key.split('_', 1)
        begin = begin.upper()
        if begin in MODELS.keys():
            model = begin
            model_params[end] = params[key]
        elif begin in ENCODERS.keys():
            encoder = begin
            encoder_params[end] = params[key]
    if encoder is None:
        return model, model_params
    return model, model_params, encoder, encoder_params


def get_best_model(params: dict, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """
    Get the model from the parameters' dictionary.
    :param params: parameters dictionary
    :param X: predictors
    :param y: target
    :return: fitted model with the given parameters
    """
    model, model_params, encoder, encoder_params = extract_model_encoder(params)
    type_of_model = MODELS[model.upper()]
    if model in ('LGBM', 'XGB', 'RF', 'LR'):
        model_params['n_jobs'] = 6
    if model == 'LR':
        model_params['solver'] = 'saga'
        model_params['max_iter'] = 1000
    model = type_of_model(**model_params)
    if encoder is not None:
        type_of_encoder = ENCODERS[encoder.upper()]
        if "ngram_range" in encoder_params:
            encoder_params["ngram_range"] = NGRAMS_MAPPING[encoder_params["ngram_range"]]
        encoder = type_of_encoder(**encoder_params)
        pipeline = Pipeline([
            ('encoder', encoder),
            ('model', model)
        ])
    else:
        pipeline = Pipeline([
            ('model', model)
        ])
    pipeline.fit(X, y)
    return pipeline


def objective(trial: Trial, X: pd.DataFrame, y: pd.Series, n_splits: int, model: str, encoder: str,
              random_state: int = 42) -> float:
    """
    Objective function for Optuna.
    :param trial: Optuna Trial object
    :param X: predictors
    :param y: target
    :param n_splits: number of splits for cross-validation
    :param model: name of the model
    :param encoder: name of the encoder
    :param random_state: Random state for cross-validation
    :return: cross-validation score of the model
    """
    type_of_model = MODELS_MAPPING[model]
    type_of_encoder = ENCODERS_MAPPING[encoder]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = type_of_model(trial)
    encoder = type_of_encoder(trial)
    pipeline = Pipeline([
        ('encoder', encoder),
        ('model', model)
    ])
    scorer = make_scorer(balanced_accuracy_score)
    scores = cross_val_score(pipeline, X, y, scoring=scorer, cv=skf)

    return np.mean(scores)


def get_score(retrained_model, x_test, y_test):
    """
    Get the score of the retrained model on the test set.
    :param retrained_model: retrained model
    :param x_test: predictors for the test set
    :param y_test: target for the test set
    :return: dictionary of metrics for the retrained model (balanced accuracy, f1-score, precision, recall)
    """
    y_pred = retrained_model.predict(x_test)
    results = {
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }
    return results
