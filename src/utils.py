import os
import sys
import random
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from functools import reduce
from tqdm.notebook import tqdm
from typing import Tuple
import matplotlib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

def seed_everything(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
def sampling_from_group(curve_id, group_df, N, M, SEED, min_distance):
    random_state_base = SEED * (curve_id + 1)
    tmp = pd.DataFrame(data=0.0, index=range(N//M), columns=['Curve_id', 'Q', 'q1', 'V_oc1', 'q2', 'V_oc2', 'q(V_min)'])
    i = 0
    j = 1
    while i < N//M:
        random_state = random_state_base * j
        x1 = group_df.sample(n=1, random_state=random_state)
        
        if len(group_df[group_df['q'] >= x1['q'].item() + min_distance]) > 0:
            x2 = group_df[group_df['q'] >= x1['q'].item() + min_distance].sample(n=1, random_state=random_state)
            
            assert(x1['Curve_id'].item() == x2['Curve_id'].item())
            assert(x1['Q'].item() == x2['Q'].item())
            assert(x1['q(V_min)'].item() == x2['q(V_min)'].item())
            assert((x2['q'].item() - x1['q'].item()) >= min_distance)
            
            tmp.loc[i, 'Curve_id'] = x1['Curve_id'].item()
            tmp.loc[i, 'Q'] = x1['Q'].item()
            tmp.loc[i, 'q1'] = x1['q'].item()
            tmp.loc[i, 'V_oc1'] = x1['V_oc'].item()
            tmp.loc[i, 'q2'] = x2['q'].item()
            tmp.loc[i, 'V_oc2'] = x2['V_oc'].item()
            tmp.loc[i, 'q(V_min)'] = x1['q(V_min)'].item()
            
            i+=1
        j+=1
    return tmp

def process_mc_run(M_train, df_grouped, N, N_MC, random_state, test_sampling_strategy, M_test, min_distance, target, params, num_folds):
    rng = np.random.default_rng(random_state)

    # Comb creation
    df_list = Parallel(n_jobs=1, verbose=0)(delayed(sampling_from_group)(curve_id, group_df, N, M_train, random_state, min_distance) for curve_id, group_df in df_grouped)
    comb = pd.concat(df_list, ignore_index=True)

    # Train-test split
    if test_sampling_strategy == 'random':
        test_curves = sorted(rng.choice(range(1, M-1), size=M_test, replace=False).tolist())
    elif test_sampling_strategy == 'equispaced':
        test_curves = sorted(np.linspace(1, M-2, M_test, endpoint=True, dtype='int'))
    elif test_sampling_strategy == 'percentiles':
        targets = sorted(comb[target].unique())
        test_curves = find_percentile_indices(targets, np.linspace(1, 99, M_test, dtype=int).tolist())
    else:
        raise ValueError(f"Test sampling strategy {test_sampling_strategy} not recognised.")

    train_curves = list(set(range(62)).difference(set(test_curves)))
    train_curves = sorted(rng.choice(train_curves, size=M_train, replace=False).tolist())
    assert(set(train_curves).intersection(test_curves) == set())
    train_idx = comb[comb['Curve_id'].isin(train_curves)].index
    test_idx = comb[comb['Curve_id'].isin(test_curves)].index
    comb['Dataset'] = ''
    comb.loc[train_idx, 'Dataset'] = 'train'
    comb.loc[test_idx, 'Dataset'] = 'test'

    # Feature engineering
    index_cols = ['Curve_id', 'Q', 'Dataset']
    features = ['q1', 'V_oc1', 'q2', 'V_oc2']
    comb, computed_features = add_features(comb)
    train = comb[comb['Dataset'] == 'train'].reset_index(drop=True).copy()
    test = comb[comb['Dataset'] == 'test'].reset_index(drop=True).copy()

    res = cross_validate(train=train,
                         test=test,
                         features=features + computed_features,
                         target=target,
                         params=params,
                         num_folds=num_folds,
                         refit=True,
                         refit_multiplier=1.2,
                         log=0,
                         verbose=False,
                         feval=None,
                         compute_oof_importance=False,
                         compute_test_importance=False)
    res_df = print_results(res, display_metrics=False, return_metrics=True)
    return res_df

def add_features(df):
    df = df.copy()
    x1, y1 = df['q1'].copy(), df['V_oc1'].copy()
    x2, y2 = df['q2'].copy(), df['V_oc2'].copy()
    # Delta of x and y coordinates
    df['dy'] = y2 - y1
    df['dx'] = x2 - x1
    # Slope of the line segment
    df['slope'] = df['dy'] / df['dx']
    # Harmonic Mean of Y Coordinates
    df['harmonic_mean_y'] = 2 / (1/y1 + 1/y2)
    # Euclidean Distance from Fixed Points
    fixed_x, fixed_y = 6, 4
    df['distance_from_fixed_point_2'] = np.sqrt((fixed_x - x2)**2 + (fixed_y - y2)**2)
    # Midpoint coordinates
    df['midpoint_y'] = (y1 + y2) / 2
    features = ['slope', 'harmonic_mean_y', 'distance_from_fixed_point_2', 'midpoint_y']
    return df, features

def cross_validate(train,
                   test,
                   features,
                   target,
                   params,
                   num_folds=None,
                   refit=True,
                   refit_multiplier=1.0,
                   log=10,
                   verbose=True,
                   feval=None,
                   other_callbacks=None,
                   compute_oof_importance=True,
                   compute_test_importance=True):
    
    train = train.copy()
    test = test.copy()
    assert(np.array_equal(train.index.values, np.arange(train.shape[0])))

    group_kfold = GroupKFold(n_splits=num_folds)
    custom_cv = []
    for fold, (train_fold_idx, valid_fold_idx) in enumerate(group_kfold.split(X=train[features], y=train[target], groups=train['Curve_id'])):
        custom_cv.append((train_fold_idx, valid_fold_idx))

    train_lgb = lgb.Dataset(train[features].values, train[target].values, feature_name=features, free_raw_data=False, categorical_feature=[])
    callbacks = [lgb.log_evaluation(period=log, show_stdv=True),
                 lgb.early_stopping(stopping_rounds=params['early_stopping_round'], first_metric_only=False, verbose=verbose)]
    if other_callbacks is not None:
        callbacks.append(other_callbacks)
    cv_results = lgb.cv(params=params,
                        train_set=train_lgb,
                        folds=custom_cv,
                        metrics=params['metric'],
                        num_boost_round=params['num_iterations'],
                        stratified=False,
                        callbacks=callbacks,
                        eval_train_metric=True,
                        return_cvbooster=True,
                        feval=feval
                       )
    best_iteration = cv_results['cvbooster'].best_iteration

    results = {'score': [], 'best_iteration': best_iteration, 'cv_models': cv_results['cvbooster'], 'refit_model': None, 'train_preds': None, 'test_preds':None,
               'cv_feature_importance': [], 'test_feature_importance': []}

    train['preds'] = 0.0
    test['preds_ensemble'] = 0.0
    test['preds_refit'] = 0.0
    #for fold, (train_fold_idx, valid_fold_idx) in tqdm(enumerate(custom_cv), total=num_folds):
    for fold, (train_fold_idx, valid_fold_idx) in enumerate(custom_cv):
        train_fold = train.loc[train_fold_idx].copy()
        valid_fold = train.loc[valid_fold_idx].copy()
        model = cv_results['cvbooster'].boosters[fold]
        train.loc[valid_fold_idx, 'preds'] = model.predict(valid_fold[features], num_iteration=best_iteration)
        test['preds_ensemble'] += model.predict(test[features], num_iteration=best_iteration) / num_folds
        score_fold = compute_metrics(true=train.loc[valid_fold_idx, target].values,
                             preds=train.loc[valid_fold_idx, 'preds'].values, 
                             fold=fold+1)
        results['score'].append(score_fold)
        if compute_oof_importance:
            results['cv_feature_importance'].append(model.predict(valid_fold[features], num_iteration=best_iteration, pred_contrib=True))
        if compute_test_importance:
            results['test_feature_importance'].append(model.predict(test[features], num_iteration=best_iteration, pred_contrib=True))

    results['score'].append(compute_metrics(true=test[target].values, preds=test['preds_ensemble'].values, fold='Test ensemble'))

    if refit:
        train_lgb = lgb.Dataset(train[features], train[target], feature_name=features, free_raw_data=False, categorical_feature=[])
        test_lgb = lgb.Dataset(test[features], test[target], reference=train_lgb)
        learning_curves = dict()
        callbacks = [lgb.log_evaluation(period=log, show_stdv=True),
                     lgb.record_evaluation(learning_curves)]
        if params['boosting_type'] == 'dart':
            model = lgb.train(params=params,
                              train_set=train_lgb,
                              valid_sets=[train_lgb, test_lgb],
                              callbacks=callbacks,
                              #num_boost_round=best_iteration,
                              feval=feval)
        else:
            model = lgb.train(params=dict(params, **{'num_iterations': int(best_iteration*refit_multiplier), 'early_stopping_round': None}),
                              train_set=train_lgb,
                              valid_sets=[train_lgb, test_lgb],
                              callbacks=callbacks,
                              num_boost_round=best_iteration,
                              feval=feval)

        test['preds_refit'] = model.predict(test[features], num_iteration=int(best_iteration*refit_multiplier))
        results['score'].append(compute_metrics(true=test[target].values, preds=test['preds_refit'].values, fold='Test refit'))
        results['refit_model'] = model

    results['train_preds'] = train
    results['test_preds'] = test
    
    return results

def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def compute_metrics(true, preds, fold, compute_metrics=True, mae_value=0.0, rmse_value=0.0, r2_value=0.0):
    res = pd.DataFrame(data=0.0, index=[fold], columns=['MAPE', 'MAE', 'RMSE', 'R2'])
    res['MAPE'] = mean_absolute_percentage_error(true, preds) * 100
    res['MAE'] = mean_absolute_error(true, preds)
    res['RMSE'] = root_mean_squared_error(true, preds)
    res['R2'] = r2_score(true, preds)
    return res

def print_results(results, display_metrics=True, return_metrics=False):
    if isinstance(results, list):
        tmp = pd.concat([pd.concat(res['score'])for res in results])
        tmp = tmp.groupby(tmp.index).mean()
    else:
        tmp = pd.concat(results['score'])
    tmp.index.names = ['Fold']
    if 'OOF' not in tmp.index:
        tmp.loc['OOF'] = tmp.loc[[i for i in tmp.index if isinstance(i, int)]].mean(0)
    if display_metrics:
        try:
            display(tmp.loc[['OOF', 'Test ensemble', 'Test refit']])
        except:
            display(tmp.loc[['OOF', 'Test ensemble']])
    if return_metrics:
        return tmp
    
def plot_importance(results, features=None, max_features=None, show=True, return_imps=False):
    oof_feature_importance = results['cv_feature_importance']
    oof_imps = [pd.DataFrame(oof_feature_importance[f], columns=features + ['expected_values'])\
                [features].abs().mean(axis=0).to_frame(name=f'fold{f+1}') for f in range(len(oof_feature_importance))]
    oof_imps = reduce(lambda df1,df2: pd.merge(df1,df2,left_index=True, right_index=True), oof_imps)
    oof_imps = oof_imps.agg(['mean', 'std'], axis=1).sort_values(by='mean', ascending=False)
    
    test_feature_importance = results['test_feature_importance']
    test_imps = [pd.DataFrame(test_feature_importance[f], columns=features + ['expected_values'])\
                [features].abs().mean(axis=0).to_frame(name=f'fold{f+1}') for f in range(len(test_feature_importance))]
    test_imps = reduce(lambda df1,df2: pd.merge(df1,df2,left_index=True, right_index=True), test_imps)
    test_imps = test_imps.agg(['mean', 'std'], axis=1).sort_values(by='mean', ascending=False)
    
    if max_features is not None:
        oof_imps = oof_imps.iloc[:max_features]
        test_imps = test_imps.iloc[:max_features]
        
    if show:
        fig, axes = plt.subplots(1, 2, figsize=(10,5), sharex=True)
        axes[0].barh(y=oof_imps.index.values, width=oof_imps['mean'].values, xerr=oof_imps['std'].values, capsize=3, edgecolor='black', linewidth=0.5)
        axes[0].set_title('OOF Feature Importance')
        axes[0].invert_yaxis()
        axes[0].set_axisbelow(True)
        axes[1].barh(y=test_imps.index.values, width=test_imps['mean'].values, xerr=test_imps['std'].values, capsize=3, edgecolor='black', linewidth=0.5)
        axes[1].set_title('Test Feature Importance')
        axes[1].invert_yaxis()
        axes[1].set_axisbelow(True)
        plt.tight_layout()
        plt.show()
    if return_imps:
        if test_results is not None:
            return oof_imps, test_imps
        else:
            return oof_imps
        
        
def evaluate_candidate(train,
                       test,
                       selected_features,
                       candidate,
                       target,
                       params,
                       num_folds,
                       scoring):
    features = selected_features + [candidate]
    res = cross_validate(train=train,
                         test=test,
                         features=features,
                         target=target,
                         params=params,
                         num_folds=num_folds,
                         refit=True,
                         refit_multiplier=1+round(len(test)/len(train),1),
                         log=0,
                         verbose=False,
                         feval=None,
                         compute_oof_importance=False,
                         compute_test_importance=False)
    
    tmp = pd.concat(res['score'])
    tmp.index.names = ['Fold']
    tmp.loc['OOF'] = tmp.loc[[i for i in tmp.index if isinstance(i, int)]].mean(0)
    oof_score = tmp.loc['OOF', scoring]
    test_ensemble_score = tmp.loc['Test ensemble', scoring]
    test_refit_score = tmp.loc['Test refit', scoring]
    return oof_score, test_ensemble_score, test_refit_score, candidate


def forward_feature_selection_parallel(train,
                                       test,
                                       features,
                                       target,
                                       params,
                                       num_folds,
                                       selected_features_init=None,
                                       min_features_to_select=1,
                                       max_features_to_select=None,
                                       scoring='MAPE',
                                       fold_to_optimize='oof',
                                       n_jobs=-1,
                                       verbose=0):
    if max_features_to_select is None:
        max_features_to_select = len(features)

    selected_features = selected_features_init if selected_features_init is not None else []
    remaining_features = [f for f in features if f not in selected_features]
    current_score = np.inf
    res = pd.DataFrame(columns=['selected_features', f'oof_{scoring}', f'test_ensemble_{scoring}', f'test_refit_{scoring}'])

    while remaining_features and len(selected_features) < max_features_to_select:
        scores_with_candidates = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(evaluate_candidate)(train=train,
                                        test=test,
                                        selected_features=selected_features,
                                        candidate=candidate,
                                        target=target,
                                        params=params,
                                        num_folds=num_folds,
                                        scoring=scoring)
            for candidate in remaining_features)
        tmp = pd.DataFrame(scores_with_candidates, columns=[f'oof_{scoring}', f'test_ensemble_{scoring}', f'test_refit_{scoring}', 'candidate'])
        best_score_oof, best_score_test_ensemble, best_score_test_refit, best_candidate = tmp.sort_values(f'{fold_to_optimize}_{scoring}').iloc[0].values
        best_score, best_candidate = tmp.sort_values(f'{fold_to_optimize}_{scoring}').iloc[0][[f'{fold_to_optimize}_{scoring}', 'candidate']].values

        if best_score < current_score or len(selected_features) < min_features_to_select:
            remaining_features.remove(best_candidate)
            selected_features.append(best_candidate)
            current_score = best_score
        elif len(selected_features) >= min_features_to_select:
            break

        res.loc[len(selected_features), 'selected_features'] = selected_features.copy()
        res.loc[len(selected_features), f'oof_{scoring}'] = best_score_oof
        res.loc[len(selected_features), f'test_ensemble_{scoring}'] = best_score_test_ensemble
        res.loc[len(selected_features), f'test_refit_{scoring}'] = best_score_test_refit

        if verbose == 1:
            sys.stderr.write("\rFeatures: %d/%s" % (len(selected_features), max_features_to_select))
            sys.stderr.flush()
        elif verbose > 1:
            sys.stderr.write("\n[%s] Features: %d/%s -- selected: %s -- score: %.4f" % (
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                len(selected_features),
                max_features_to_select,
                best_candidate,
                current_score))
            sys.stderr.flush()
    return res, selected_features


def add_features_all(df):
    df = df.copy()
    x1, y1 = df['q1'].copy(), df['V_oc1'].copy()
    x2, y2 = df['q2'].copy(), df['V_oc2'].copy()
    # Delta of x and y coordinates
    df['dy'] = y2 - y1
    df['dx'] = x2 - x1
    # Slope and intercept of the line segment
    df['slope'] = df['dy'] / df['dx']
    df['intercept'] = (x1*y2 - x2*y1) / (x1 - x2)
    # Distance between points
    df['distance'] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    # Distance from origin
    df['distance_from_origin_1'] = np.sqrt(x1**2 + y1**2)
    df['distance_from_origin_2'] = np.sqrt(x2**2 + y2**2)
    # Midpoint coordinates
    df['midpoint_x'] = (x1 + x2) / 2
    df['midpoint_y'] = (y1 + y2) / 2
    # Angle with horizontal
    #df['angle'] = np.arctan2(y2 - y1, x2 - x1)
    # Area under the line segment (Trapezoidal approximation)
    df['area_under_curve'] = 0.5 * (y1 + y2) * np.abs(x2 - x1)
    # Relative positioning of points
    df['ratio_x1_x2'] = x1 / x2
    df['ratio_y1_y2'] = y1 / y2
    # Sum of coordinates
    df['x1_plus_y1'] = x1 + y1
    df['x2_plus_y2'] = x2 + y2
    df['x1_plus_x2'] = x1 + x2
    df['y1_plus_y2'] = y1 + y2
    # Product of coordinates
    df['x1_times_y1'] = x1 * y1
    df['x2_times_y2'] = x2 * y2
    df['x1_times_x2'] = x1 * x2
    df['y1_times_y2'] = y1 * y2
    # Difference in y/x ratios
    df['difference_in_y_x_ratio'] = (y1/x1) - (y2/x2)
    # Squared Distances
    df['dx_squared'] = df['dx']**2
    df['dy_squared'] = df['dy']**2
    # Cubic Distances
    df['dx_cubed'] = df['dx']**3
    df['dy_cubed'] = df['dy']**3
    # Harmonic Mean of X and Y Coordinates
    df['harmonic_mean_x'] = 2 / (1/x1 + 1/x2)
    df['harmonic_mean_y'] = 2 / (1/y1 + 1/y2)
    # Geometric Mean of X and Y Coordinates
    df['geometric_mean_x'] = np.sqrt(x1 * x2)
    df['geometric_mean_y'] = np.sqrt(y1 * y2)
    # Logarithm of Absolute Distances
    df['log_dx'] = np.log(np.abs(df['dx']) + 1)
    df['log_dy'] = np.log(np.abs(df['dy']) + 1)
    # Normalized Coordinates
    df['normalized_x1'] = x1 / (x1 + x2)
    df['normalized_y1'] = y1 / (y1 + y2)
    df['normalized_x2'] = x2 / (x1 + x2)
    df['normalized_y2'] = y2 / (y1 + y2)
    df['normalized_dy'] = df['normalized_y2'] - df['normalized_y1']
    df['normalized_dx'] = df['normalized_x2'] - df['normalized_x1']
    df['normalized_slope'] = df['normalized_dy'] / df['normalized_dx']
    df['normalized_intercept'] = (df['normalized_x1']*df['normalized_y2'] - df['normalized_x2']*df['normalized_y1']) / (df['normalized_x1'] - df['normalized_x2'])
    # Ratio of Distances to Sum of Coordinates
    df['distance_to_sum_ratio'] = df['distance'] / (df['x1_plus_x2'] + df['y1_plus_y2'])
    # Euclidean Distance from Fixed Points
    fixed_x, fixed_y = 6, 4
    df['distance_from_fixed_point_1'] = np.sqrt((fixed_x - x1)**2 + (fixed_y - y1)**2)
    df['distance_from_fixed_point_2'] = np.sqrt((fixed_x - x2)**2 + (fixed_y - y2)**2)
    # Polar Coordinates
    df['r1'] = np.sqrt(x1**2 + y1**2)
    df['theta1'] = np.arctan2(y1, x1)
    df['r2'] = np.sqrt(x2**2 + y2**2)
    df['theta2'] = np.arctan2(y2, x2)
    # Cross Ratios
    df['cross_ratio_x1_y2'] = x1 / y2
    df['cross_ratio_y1_x2'] = y1 / x2
    # Relative Position to Average Points
    average_x = df[['q1', 'q2']].mean(axis=1)
    average_y = df[['V_oc1', 'V_oc2']].mean(axis=1)
    df['relative_position_x1'] = x1 - average_x
    df['relative_position_y1'] = y1 - average_y
    df['relative_position_x2'] = x2 - average_x
    df['relative_position_y2'] = y2 - average_y
    features = ['dy', 'dx', 'slope', 'intercept', 'distance', 'distance_from_origin_1',
                'distance_from_origin_2', 'midpoint_x', 'midpoint_y', #'angle',
                'area_under_curve', 'ratio_x1_x2', 'ratio_y1_y2', 'x1_plus_y1',
                'x2_plus_y2', 'x1_plus_x2', 'y1_plus_y2', 'x1_times_y1', 'x2_times_y2',
                'x1_times_x2', 'y1_times_y2', 'difference_in_y_x_ratio', 'dx_squared',
                'dy_squared', 'dx_cubed', 'dy_cubed', 'harmonic_mean_x',
                'harmonic_mean_y', 'geometric_mean_x', 'geometric_mean_y', 'log_dx',
                'log_dy', 'normalized_x1', 'normalized_y1', 'normalized_x2',
                'normalized_y2', 'normalized_dy', 'normalized_dx', 'normalized_slope',
                'normalized_intercept', 'distance_to_sum_ratio',
                'distance_from_fixed_point_1', 'distance_from_fixed_point_2', 'r1',
                'theta1', 'r2', 'theta2', 'cross_ratio_x1_y2', 'cross_ratio_y1_x2',
                'relative_position_x1', 'relative_position_y1', 'relative_position_x2',
                'relative_position_y2']
    return df, features


def plot_diagnostic(results, target, fold='test', bins=50, percentiles=(.05, .95), figsize=(8,4)):
    if fold == 'test':
        df = results['test_preds'].copy()
        preds_column = 'preds_ensemble'
    elif fold == 'valid':
        df = results['train_preds'].copy()
        preds_column = 'preds'
    else:
        df = results.copy()
        preds_column = 'preds'
        
    grouped = df.groupby(target)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.errorbar(grouped.groups.keys(), grouped[preds_column].median().values,
                     yerr=[grouped[preds_column].median()-grouped[preds_column].quantile(percentiles[0]),
                           grouped[preds_column].quantile(percentiles[1])-grouped[preds_column].median()], marker='.', linestyle='--', markersize=18, capsize=5, color='b')
    bounds = min([ax1.get_xlim()[0], ax1.get_ylim()[0]]), max([ax1.get_xlim()[1], ax1.get_ylim()[1]])
    ax1.set_xlim(bounds)
    ax1.set_ylim(bounds)
    ax1.set_aspect('equal', adjustable='box')
    ax1.plot(bounds, bounds, lw=2, ls='--', color='k', alpha=0.5)
    ax1.set_xlabel(r'$q(V_{min})$')
    ax1.set_ylabel(r'$\hat{q}(V_{min})$')

    df['APE'] = 100 * (df[target] - df[preds_column]).abs() / df[target]
    df['APE'].hist(bins=bins, ax=ax2, color='b')
    ax2.set_xlabel('Absolute percentage error [%]')
    ax2.axvline(df['APE'].median(), linestyle='--', color='k', label=f'Median APE = {df["APE"].median().round(2)}%', lw=2, alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    return fig
    
    
def ecdf_values(x):
    """
    Generate values for empirical cumulative distribution function
    
    Params
    --------
        x (array or list of numeric values): distribution for ECDF
    
    Returns
    --------
        x (array): x values
        y (array): percentile values
    """
    
    # Sort values and find length
    x = np.sort(x)
    n = len(x)
    # Create percentiles
    y = np.arange(1, n + 1, 1) / n
    return x, y

def ecdf_plot(x, name = 'Value', plot_normal=False, log_scale=False, save=False, save_name='Default', ps=None, figsize = (10, 6)):
    """
    ECDF plot of x

    Params
    --------
        x (array or list of numerics): distribution for ECDF
        name (str): name of the distribution, used for labeling
        plot_normal (bool): plot the normal distribution (from mean and std of data)
        log_scale (bool): transform the scale to logarithmic
        save (bool) : save/export plot
        save_name (str) : filename to save the plot
    
    Returns
    --------
        none, displays plot
    
    """
    xs, ys = ecdf_values(x)
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)
    plt.step(xs, ys, linewidth = 2., c= 'b');
    
    plot_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    fig_sizex = fig.get_size_inches()[0]
    data_inch = plot_range / fig_sizex
    right = 0.6 * data_inch + max(xs)
    gap = right - max(xs)
    left = min(xs) - gap
    
    if log_scale:
        ax.set_xscale('log')
        
    if plot_normal:
        gxs, gys = ecdf_values(np.random.normal(loc = xs.mean(), 
                                                scale = xs.std(), 
                                                size = 100000))
        plt.plot(gxs, gys, 'g');

    plt.vlines(x=min(xs), 
               ymin=0, 
               ymax=min(ys), 
               color = 'b', 
               linewidth = 2.5)
    
    # Add ticks
    plt.xticks(size = 16)
    plt.yticks(size = 16)
    # Add Labels
    plt.xlabel(f'{name}', size = 18)
    plt.ylabel('Probability', size = 18)

    plt.vlines(x=min(xs), 
               ymin = min(ys), 
               ymax=0.065, 
               color = 'r', 
               linestyle = '-', 
               alpha = 0.8, 
               linewidth = 1.7)
    
    plt.vlines(x=max(xs), 
               ymin=0.935, 
               ymax=max(ys), 
               color = 'r', 
               linestyle = '-', 
               alpha = 0.8, 
               linewidth = 1.7)

    # Add Annotations
    plt.annotate(s = f'{min(xs):.2f}', 
                 xy = (min(xs), 
                       0.065),
                horizontalalignment = 'center',
                verticalalignment = 'bottom',
                size = 15)
    plt.annotate(s = f'{max(xs):.2f}', 
                 xy = (max(xs), 
                       0.935),
                horizontalalignment = 'center',
                verticalalignment = 'top',
                size = 15)
    if ps is None:
        ps = [0.25, 0.5, 0.75]

    for p in ps:

        ax.set_xlim(left = left, right = right)
        ax.set_ylim(bottom = 0)

        value = xs[np.where(ys > p)[0][0] - 1]
        pvalue = ys[np.where(ys > p)[0][0] - 1]

        plt.hlines(y=p, xmin=left, xmax = value,
                    linestyles = ':', colors = 'r', linewidth = 1.4);

        plt.vlines(x=value, ymin=0, ymax = pvalue, 
                   linestyles = ':', colors = 'r', linewidth = 1.4)
        
        plt.text(x = p / 3, y = p - 0.01, 
                 transform = ax.transAxes,
                 s = f'{int(100*p)}%', size = 15,
                 color = 'r', alpha = 0.7)

        plt.text(x = value, y = 0.01, size = 15,
                 horizontalalignment = 'left',
                 s = f'{value:.2f}', color = 'r', alpha = 0.8);

    # fit the labels into the figure
    plt.title(f'eCDF of {name}', size = 20)
    plt.tight_layout()
    

    if save:
        plt.savefig(save_name + '.png')
        
        
        
def find_percentile_indices(arr, percentiles):
    # Calculate the percentile values
    percentile_values = np.percentile(arr, percentiles, interpolation='nearest')
    # Find indices of the closest matches for these percentile values
    indices = [np.abs(arr - value).argmin() for value in percentile_values]
    return indices


### Figure saving function taken from:
### https://zhauniarovich.com/post/2022/2022-09-matplotlib-graphs-in-research-papers/#saving-figures
def save_fig(
        fig: matplotlib.figure.Figure, 
        fig_name: str, 
        fig_dir: str, 
        fig_fmt: str,
        fig_size: Tuple[float, float] = [6.4, 4], 
        save: bool = True, 
        dpi: int = 300,
        transparent_png = True,
    ):
    """This procedure stores the generated matplotlib figure to the specified 
    directory with the specified name and format.

    Parameters
    ----------
    fig : [type]
        Matplotlib figure instance
    fig_name : str
        File name where the figure is saved
    fig_dir : str
        Path to the directory where the figure is saved
    fig_fmt : str
        Format of the figure, the format should be supported by matplotlib 
        (additional logic only for pdf and png formats)
    fig_size : Tuple[float, float]
        Size of the figure in inches, by default [6.4, 4] 
    save : bool, optional
        If the figure should be saved, by default True. Set it to False if you 
        do not want to override already produced figures.
    dpi : int, optional
        Dots per inch - the density for rasterized format (png), by default 300
    transparent_png : bool, optional
        If the background should be transparent for png, by default True
    """
    if not save:
        return
    
    fig.set_size_inches(fig_size, forward=False)
    fig_fmt = fig_fmt.lower()
    fig_dir = os.path.join(fig_dir, fig_fmt)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    pth = os.path.join(
        fig_dir,
        '{}.{}'.format(fig_name, fig_fmt.lower())
    )
    if fig_fmt == 'pdf':
        metadata={
            'Creator' : '',
            'Producer': '',
            'CreationDate': None
        }
        fig.savefig(pth, bbox_inches='tight', metadata=metadata)
    elif fig_fmt == 'png':
        alpha = 0 if transparent_png else 1
        axes = fig.get_axes()
        fig.patch.set_alpha(alpha)
        for ax in axes:
            ax.patch.set_alpha(alpha)
        fig.savefig(
            pth, 
            bbox_inches='tight',
            dpi=dpi,
        )
    else:
        try:
            fig.savefig(pth, bbox_inches='tight')
        except Exception as e:
            print("Cannot save figure: {}".format(e))