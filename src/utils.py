import os
import sys
import dill

import numpy as np
import pandas as pd

from src.exception import CustomException

def save_object(file_path, obj):
    '''
    Save a Python object to disk so another run can load it later.

    Data transformation calls this with the fitted preprocessor.
    The file is artifacts/preprocessor.pkl.
    dill can store that object, including the imputer, encoder, and scaler inside it.
    A normal pickle dump is less reliable for some of those fitted steps.
    '''
    try:
        # file_path looks like artifacts/preprocessor.pkl.
        # dirname is the folder part: artifacts.
        dir_path = os.path.dirname(file_path)
        # Create the folder when this function is used before ingestion has created it.
        # exist_ok=True lets the folder already exist.
        os.makedirs(dir_path, exist_ok = True)

        # "wb" means write binary. The file holds bytes, not readable text.
        # dill.dump writes the whole object, including values it learned from the training rows.
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        # Report the original error together with this file and line number.
        raise CustomException(e, sys)
