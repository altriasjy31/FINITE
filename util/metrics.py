import typing as T
import numpy as np
from scipy import interpolate
from sklearn.metrics import auc, precision_recall_curve, precision_recall_fscore_support, roc_curve
import pandas as pd

def f1_max(targs: np.ndarray, preds: np.ndarray,
               no_empty_labels: bool = False,
               no_zero_classes: bool = False,
               need_threshold: bool = False,
               auprc: bool = False,
               curve: bool = False):
    n = 100
    if no_empty_labels:
        idx = np.where(targs.sum(1) != 0)
        targs = targs[idx]
        preds = preds[idx]
    
    if no_zero_classes:
        idx = targs.sum(0) > 0
        targs = targs[:, idx]
        preds = preds[:, idx]

    fmax_value = 0.
    max_thres = 0.

    pr, rc = [], []
    for t in range(n+1):
        threshold = t / n
        preds_label = np.where(preds > threshold, 1, 0)
        tp = (preds_label * targs).sum(1)
        tp_fp = preds_label.sum(1)
        tp_fn = targs.sum(1)

        idx= np.where(tp_fp != 0)
        tp_fp = tp_fp[idx]
        precision = (tp[idx] / tp_fp
        if tp_fp.shape[0] != 0
        else np.asarray([0.])).mean()

        # when no_empty_label is False
        idx = np.where(tp_fn != 0)
        tp_fn = tp_fn[idx]
        recall = (tp[idx] / tp_fn
        if tp_fn.shape[0] != 0
        else np.asarray([0.])).mean()

        if ((denom := precision + recall) == 0.0):
            denom = 1.0
        if (score := 2 * precision * recall / denom) > fmax_value:
          fmax_value = score
          max_thres = threshold
        
        if t == 0:
          pr.append(0.)
        elif t < n:
          pr.append(precision)
        else: # t == n
          pass
        
        if t == n:
          rc.append(0.)
          pr.append(1.)
        elif pr[-1] <= 0:
          rc.append(1.)
        else:
          rc.append(recall)

    # fmax_value = reduce(_calculate_fmax, np.arange(n+1))
    # fmax_value = max(map(_calculate_fmax, np.arange(n+1)))
    pr = np.array(pr)
    rc = np.array(rc)
    index = np.argsort(rc)

    if need_threshold and curve:
      return (fmax_value, max_thres), (pr, rc)
    elif need_threshold and auprc:
      return (fmax_value, max_thres), np.trapz(pr[index], rc[index])
    elif need_threshold:
      return fmax_value, max_thres
    elif curve:
      return fmax_value, (pr, rc)
    elif auprc:
      return fmax_value, np.trapz(pr[index], rc[index])
    else:
      return fmax_value


def macro_f1_max(targs, preds, 
                 auprc: bool = False, 
                 need_threshold: bool = False, 
                 curve: bool = False,
                 no_empty_labels: bool = True,
                 no_zero_classes: bool = True):
    """
    Calculate macro-averaged F1-maximum score by first creating macro-averaged PR curve.
    
    Parameters:
    -----------
    targs : array-like of shape (n_samples, n_classes)
        True binary labels matrix
    preds : array-like of shape (n_samples, n_classes)
        Target scores (probability estimates or decision function values)
    auprc : bool, default=False
        Whether to include Area Under Precision-Recall Curve
    need_threshold : bool, default=False
        Whether to include optimal threshold
    curve : bool, default=False
        Whether to include precision-recall curves
    no_empty_labels : bool, default=True
        If True, removes samples with no positive labels
    no_zero_classes : bool, default=True
        If True, removes classes with no positive labels
    
    Returns:
    --------
    Various combinations based on the boolean flags as specified in the match statement
    """
    
    # Apply filtering based on flags
    if no_empty_labels:
        idx = np.where(targs.sum(1))[0]
        targs = targs[idx]
        preds = preds[idx]
    
    if no_zero_classes:
        idx = targs.sum(0) > 0
        targs = targs[:, idx]
        preds = preds[:, idx]
    
    n_classes = preds.shape[1]
    
    # Step 1: Calculate PR curve for each class
    per_class_precision = []
    per_class_recall = []
    per_class_thresholds = []
    
    for i in range(n_classes):
        precision, recall, threshold = precision_recall_curve(
            targs[:, i], preds[:, i]
        )
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_thresholds.append(threshold)
    
    # Step 2: Create common recall grid for interpolation
    # Use a fine-grained recall grid for smooth macro-averaged curve
    common_recall = np.linspace(0, 1, 1000)
    
    # Step 3: Interpolate precision for each class at common recall points
    interpolated_precisions = []
    
    for i in range(n_classes):
        recall = per_class_recall[i]
        precision = per_class_precision[i]
        
        # Sort by recall (sklearn returns them in reverse order)
        sorted_indices = np.argsort(recall)
        recall_sorted = recall[sorted_indices]
        precision_sorted = precision[sorted_indices]
        
        # Handle edge case where recall might have duplicate values
        # Remove duplicates while keeping the highest precision for each recall
        unique_recall, unique_indices = np.unique(recall_sorted, return_index=True)
        unique_precision = precision_sorted[unique_indices]
        
        # Interpolate precision at common recall points
        f_interp = interpolate.interp1d(
            unique_recall, unique_precision, 
            kind='linear', bounds_error=False, 
            fill_value=(unique_precision[0], unique_precision[-1])
        )
        
        interpolated_precision = f_interp(common_recall)
        interpolated_precisions.append(interpolated_precision)
    
    # Step 4: Calculate macro-averaged precision-recall curve
    macro_precision = np.mean(interpolated_precisions, axis=0)
    macro_recall = common_recall
    
    # Step 5: Calculate F1 scores from macro-averaged curve
    f1_scores = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall + 1e-8)
    
    # Step 6: Find maximum F1 score and corresponding point
    max_f1_idx = np.nanargmax(f1_scores)  # Use nanargmax to handle any NaN values
    fmax_macro = f1_scores[max_f1_idx]
    
    # Find the corresponding threshold (approximate)
    optimal_recall = macro_recall[max_f1_idx]
    optimal_precision = macro_precision[max_f1_idx]
    
    # Estimate threshold by finding the closest point in the original curves
    threshold_estimates = []
    for i in range(n_classes):
        recall = per_class_recall[i]
        precision = per_class_precision[i]
        thresholds = per_class_thresholds[i]
        
        # Find closest point to optimal recall/precision
        distances = np.sqrt((recall - optimal_recall)**2 + (precision - optimal_precision)**2)
        closest_idx = np.argmin(distances)
        
        if closest_idx < len(thresholds):
            threshold_estimates.append(thresholds[closest_idx])
        else:
            threshold_estimates.append(thresholds[-1] if len(thresholds) > 0 else 0.5)
    
    # Use mean of threshold estimates
    thr = np.mean(threshold_estimates)
    
    # Calculate AUPRC from macro-averaged curve if needed
    auprc_macro = None
    if auprc:
        auprc_macro = auc(macro_recall, macro_precision)
    
    # Prepare curve data if needed
    if curve:
        pr_curves = macro_precision
        rc_curves = macro_recall
    else:
        pr_curves = None
        rc_curves = None
    
    # Prepare return values based on flags
    match (auprc, need_threshold, curve):
        case (True, True, True):
            return (fmax_macro, thr), auprc_macro, (pr_curves, rc_curves)
        case (True, True, False):
            return (fmax_macro, thr), auprc_macro
        case (True, False, True):
            return fmax_macro, auprc_macro, (pr_curves, rc_curves)
        case (True, False, False):
            return fmax_macro, auprc_macro
        case (False, True, True):
            return (fmax_macro, thr), (pr_curves, rc_curves)
        case (False, True, False):
            return fmax_macro, thr
        case (False, False, True):
            return fmax_macro, (pr_curves, rc_curves)
        case _:
            return fmax_macro

def f1_support(targs: np.ndarray, preds: np.ndarray,
               no_empty_labels: bool = True,
               no_zero_classes: bool = True,
               average: str = 'macro',
               auprc: bool = False,
               ):
    if no_empty_labels:
        idx = np.where(targs.sum(1))[0]
        targs = targs[idx]
        preds = preds[idx]
    
    if no_zero_classes:
        idx = targs.sum(0) > 0
        targs = targs[:, idx]
        preds = preds[:, idx]

    precision_macro, recall_macro, fbeta, _ = precision_recall_fscore_support(
        targs, preds, average=average
    )

    return fbeta

# fmax score
def fmax_score(targs: np.ndarray, preds: np.ndarray,
               auprc: bool = False,
               curve: bool = False,
               need_threshold: bool = False,
               no_empty_labels: bool = False,
               no_zero_classes: bool = False,
               drop_intermediate: bool = False):
    """
    Calculate the Fmax score for given targets and predictions.
    
    Parameters:
    - targs: True labels (targets).
    - preds: Predicted scores.
    - need_threshold: If True, return the threshold used for Fmax.
    - curve: If True, return the precision-recall curve.
    
    Returns:
    - Fmax score or precision-recall curve based on parameters.
    """
    if no_empty_labels:
        idx = np.where(targs.sum(1))[0]
        targs = targs[idx]
        preds = preds[idx]
    if no_zero_classes:
        # Filter out classes with zero targets
        non_zero_indices = targs.sum(axis=0) > 0
        targs = targs[:, non_zero_indices]
        preds = preds[:, non_zero_indices]
    pr, rc, ths = precision_recall_curve(targs.ravel(), preds.ravel(),
                                         drop_intermediate=drop_intermediate)
    # fmax_value = np.max(2 * pr * rc / (pr + rc + 1e-10))
    values = 2 * pr * rc / (pr + rc + 1e-10)
    index = np.argmax(values)
    fmax_value = values[index]
    threshold = ths[index]
    # sorted_indices = np.argsort(rc)[::-1]
    # pr = pr[sorted_indices]
    # rc = rc[sorted_indices]
    auprc_value = auc(rc, pr)

    # using an efficient way to return results
    # only 3.10 and later support match-case
    match (auprc, need_threshold, curve):
        case (True, True, True):
            return (fmax_value, threshold), auprc_value, (pr, rc)
        case (True, True, False):
            return (fmax_value, threshold), auprc_value
        case (True, False, True):
            return fmax_value, auprc_value, (pr, rc)
        case (True, False, False):
            return fmax_value, auprc_value
        case (False, True, True):
            return (fmax_value, threshold), (pr, rc)
        case (False, True, False):
            return fmax_value, threshold
        case (False, False, True):
            return fmax_value, (pr, rc)
        case _:
            return fmax_value

    # if auprc and need_threshold and curve:
    #     return fmax_value, auprc_value, threshold, (pr, rc)
    # elif auprc:
    #     return fmax_value, auprc_value
    # elif need_threshold:
    #     return fmax_value, threshold
    # else:
    #     return fmax_value

# AuPRC score
def auprc_score(targs: np.ndarray, preds: np.ndarray):
    """
    Calculate the Area under the Precision-Recall Curve (AuPRC) score.
    
    Parameters:
    - targs: True labels (targets).
    - preds: Predicted scores.
    
    Returns:
    - AuPRC score.
    """
    pr, rc, _ = precision_recall_curve(targs, preds)
    return np.trapz(pr, rc)

def roc_auc_score(targs: np.ndarray, preds: np.ndarray,):
    """
    class-centric roc-auc socre
    """

    idx = targs.sum(0) > 0
    if not idx.any(): return 0.
    targs = targs[:, idx]
    preds = preds[:, idx]

    # remove all targs are positive
    idx = targs.sum(0) < targs.shape[0]
    targs = targs[:, idx]
    preds = preds[:, idx]

    def single_class(y_true: np.ndarray, 
                     y_score: np.ndarray):
        fpr, tpr, _ = roc_curve(y_true.ravel(), y_score.ravel())
        return np.trapz(tpr, fpr)
    
    return np.mean([single_class(targs[:, i], preds[:, i])
                        for i in range(targs.shape[1])])

def fscore_singe(targs: np.ndarray, pred_bi: np.ndarray,
               no_empty_labels: bool = False,
               no_zero_classes: bool = False,
               ):
    if no_empty_labels:
        idx = np.where(targs.sum(1))[0]
        targs = targs[idx]
        pred_bi = pred_bi[idx]
    
    if no_zero_classes:
        idx = targs.sum(0) > 0
        targs = targs[:, idx]
        pred_bi = pred_bi[:, idx]

    pred_mask = pred_bi > 0
    targ_mask = targs > 0
    pred_bi = pred_bi
    tpM = pred_bi * targs

    tp_sum = tpM.sum()
    pred_sum = pred_bi.sum()
    true_sum = targs.sum()

    # control zero division
    precision = tp_sum / pred_sum if pred_sum != 0.0 else 0.0
    recall = tp_sum / true_sum if true_sum != 0.0 else 0.0

    # fmax
    denom = precision + recall
    if denom == 0.0: denom = 1
    return 100 * 2 * precision * recall / denom


def evaluate_by(nspace_ti: T.Dict[str, T.Dict[str, int]],
                term_count_ic: pd.DataFrame,
                by: str,
                key: str,
                ont_pred: np.ndarray,
                low: T.Union[float, int, None]=None, 
                high: T.Union[float, int, None]=None):
  targs, preds = ont_pred
  if low is None and high is None:
    return fmax_score(targs, preds, need_threshold=True), auprc_score(targs, preds)
  
  assert low is not None or high is not None
  if low is not None and high is not None:
    assert high > low

  assert key in term_count_ic.columns

  index = index_of_term(nspace_ti, term_count_ic, by, key, low, high)
  
  return fmax_score(targs[:, index], preds[:, index], need_threshold=True), \
      auprc_score(targs[:, index], preds[:, index])


def index_of_term(nspace_ti: T.Dict[str, T.Dict[str, int]],
                  term_count_ic: pd.DataFrame,
                  by: str,
                  key: str,
                  low: float | int | None = None,
                  high: float | int | None = None):
    if high is None:
        index = term_count_ic[term_count_ic[key] >= low]["gos"] \
            .transform(lambda x: nspace_ti[by].get(x, pd.NA)) \
            .dropna().to_numpy(dtype=int)
    elif low is None:
        index = term_count_ic[term_count_ic[key] < high]["gos"] \
            .transform(lambda x: nspace_ti[by].get(x, pd.NA)) \
            .dropna().to_numpy(dtype=int)
    else:
        index = term_count_ic[(term_count_ic[key] >= low) &
                              (term_count_ic[key] < high)]["gos"] \
            .transform(lambda x: nspace_ti[by].get(x, pd.NA)) \
            .dropna().to_numpy(dtype=int)
    return index