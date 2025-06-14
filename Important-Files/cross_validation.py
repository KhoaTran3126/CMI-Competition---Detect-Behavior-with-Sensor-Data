from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import pandas
import numpy
import lightgbm
import xgboost
import catboost

def cv_evaluate(model, model_kind, X, y, lbl_encoder, n_splits=5, 
                random_state=SEED, stopping_rounds=100, min_delta=.0005):

    """
    model: The initialized model
    model_kind: Name of model. Could be "lgbm", "xgb", "catboost", or None to customize model training
    X: Training samples
    y: Training labels
    lbl_encoder: Label encoder
    n_splits=5: Number of splits on training data
    random_state: Seed for controlling KFold process 
    stopping_rounds: Maximum allowable additional iterations before halting
    min_delta: Minimum improvement in loss function required to continue
    """
                  
    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros_like(y)
    binary_scores   = []
    macro_scores    = []
    weighted_scores = []
    history = {}
    
    for fold_num,(train_fold, val_fold) in enumerate(skfold.split(X, y)):
        print(f"\nFold {fold_num + 1}/{n_splits}")
        X_train, y_train = X.iloc[train_fold], y[train_fold]
        X_val, y_val     = X.iloc[val_fold], y[val_fold]

        cloned_model = clone(model)

        if model_kind=="lgbm":
            cloned_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lightgbm.early_stopping(stopping_rounds=stopping_rounds, min_delta=min_delta)]
            )
        else:
            cloned_model.fit(X_train, y_train)
        ## Stores out-of-fold predictions
        y_pred = cloned_model.predict(X_val)
        oof_preds[val_fold] = y_pred
        
        ## Store cv scores
        binary, macro, weighted_score = F1_score(y_val, y_pred, lbl_encoder, choice=None)
        binary_scores.append(binary)
        macro_scores.append(macro)
        weighted_scores.append(weighted_score)
    
    ## Store cv results inside dict
    history["oof_preds"] = oof_preds
    history["binary_scores"] = binary_scores
    history["macro_scores"]  = macro_scores
    history["weighted_scores"] = weighted_scores

    ## Store oof prediction scores inside dict
    binary, macro, weighted_score = F1_score(y, oof_preds, lbl_encoder, choice=None)
    history["full_binary_score"] = binary
    history["full_macro_score"] = macro
    history["full_weighted_score"] = weighted_score
    return history
