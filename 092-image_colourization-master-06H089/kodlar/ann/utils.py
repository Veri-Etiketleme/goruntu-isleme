from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, scale
import numpy as np

def create_missing_data(row):
    chosen_feature = np.random.randint(0, len(row))
    chosen_value = np.random.standard_normal()
    row[chosen_feature] = chosen_value
    return row


def create_dataset(data, p=0.05):
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=10)
    missing_X_test = X_test.copy()
    mask = np.random.binomial(1, 1-p, X_test.shape)
    missing_X_test = np.multiply(missing_X_test, mask)
    # missing_X_test = missing_X_test.apply(create_missing_data, axis=1, result_type="expand")
    return X_train, X_test, missing_X_test, mask

def mean_centring(data):
    # Mean centring
    data = (data - np.mean(data, axis=0))
    return data