import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class PercentileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_percentile=0.05, upper_percentile=0.95):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.clipping_values_ = {}

    def fit(self, X, y=None):
        tmp = X.select_dtypes(include="number")
        for col in tmp.columns:
            l = tmp[col].quantile(self.lower_percentile)
            u = tmp[col].quantile(self.upper_percentile)
            self.clipping_values_[col] = (l, u)
        return self

    def transform(self, X):
        tmp = X.copy(deep=True)
        
        # 1. Get the list of columns we actually learned bounds for . AI HELPED WITH THIS! REMEMBER IN FUTURE
        cols_to_clip = list(self.clipping_values_.keys())

        lower = pd.Series({k: v[0] for k, v in self.clipping_values_.items()})
        upper = pd.Series({k: v[1] for k, v in self.clipping_values_.items()})
    
        tmp[cols_to_clip] = tmp[cols_to_clip].clip(lower=lower, upper=upper, axis=1)
        return tmp


if __name__ == "__main__":
    # 1. The Mock Dataset (with crazy outliers like a 150-year-old and a $9999 transaction)
    data = {
        'transaction_amount': [15.50, 22.00, 18.75, 9999.00, 12.00, -50.00, 25.50, 19.99, 21.00, 30.00],
        'user_age': [25, 34, 28, 150, 22, 45, 31, 19, 0, 40],
        'user_city': ['NY', 'LA', 'NY', 'SF', 'LA', 'NY', 'SF', 'NY', 'LA', 'NY'] # Should be untouched
    }
    df = pd.DataFrame(data)

    print("--- ORIGINAL DATA ---")
    print(df)
    print("\n")

    # 2. Instantiate your class (targeting the bottom 10% and top 10%)
    clipper = PercentileClipper(lower_percentile=0.10, upper_percentile=0.90)

    # 3. Fit and Transform
    clipper.fit(df)
    clean_df = clipper.transform(df)

    print("--- LEARNED CLIPPING BOUNDARIES ---")
    for col, bounds in clipper.clipping_values_.items():
        print(f"{col}: Lower={bounds[0]}, Upper={bounds[1]}")
    print("\n")

    print("--- CLIPPED DATA ---")
    print(clean_df)