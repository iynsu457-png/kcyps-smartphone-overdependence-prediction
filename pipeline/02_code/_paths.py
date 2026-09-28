import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(CODE_DIR)

DATA_DIR = os.path.join(BUNDLE, "01_data")
RESULTS_DIR = os.path.join(BUNDLE, "03_results")
FIG_DIR = os.path.join(BUNDLE, "04_figures")

RERUN_DIR = os.path.join(BUNDLE, "_rerun")
RERUN_FIGS = os.path.join(RERUN_DIR, "figs")
os.makedirs(RERUN_FIGS, exist_ok=True)

TRANSITIONS_PKL = os.path.join(DATA_DIR, "transitions.pkl")
TEST_PREDICTIONS = os.path.join(DATA_DIR, "test_predictions.csv")
THRESHOLDS_JSON = os.path.join(RESULTS_DIR, "thresholds.json")
BEST_PARAMS_JSON = os.path.join(RESULTS_DIR, "best_hyperparameters.json")
TRANSITION_PERF = os.path.join(RESULTS_DIR, "transition_performance.csv")

def load_transitions():
    import numpy as np
    import pandas as pd
    from pandas.core.arrays.string_ import StringDtype, StringArray

    _orig_init = StringDtype.__init__

    def _patched_init(self, storage=None, *a, **k):
        _orig_init(self, storage)

    def _patched_setstate(self, state):
        if isinstance(state, tuple) and len(state) == 2 and not isinstance(state[0], np.ndarray):
            arr = state[1]
        elif isinstance(state, tuple):
            arr = state[0]
        else:
            arr = state
        self.__init__(np.asarray(arr, dtype=object), copy=False)

    StringDtype.__init__ = _patched_init
    StringArray.__setstate__ = _patched_setstate
    return pd.read_pickle(TRANSITIONS_PKL).reset_index(drop=True)
