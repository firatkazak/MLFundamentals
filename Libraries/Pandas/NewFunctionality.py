import pandas as pd

# PyArrow backend
results_arrow = pd.read_csv(filepath_or_buffer='C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Pandas/Data/Inputs/results.csv', engine='pyarrow', dtype_backend='pyarrow')
results_arrow.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 308408 entries, 0 to 308407
# Data columns (total 11 columns):
#  #   Column      Non-Null Count   Dtype
# ---  ------      --------------   -----
#  0   year        305807 non-null  double[pyarrow]
#  1   type        305807 non-null  string[pyarrow]
#  2   discipline  308407 non-null  string[pyarrow]
#  3   event       308408 non-null  string[pyarrow]
#  4   as          308408 non-null  string[pyarrow]
#  5   athlete_id  308408 non-null  int64[pyarrow]
#  6   noc         308407 non-null  string[pyarrow]
#  7   team        121714 non-null  string[pyarrow]
#  8   place       283193 non-null  double[pyarrow]
#  9   tied        308408 non-null  bool[pyarrow]
#  10  medal       44139 non-null   string[pyarrow]
# dtypes: bool[pyarrow](1), double[pyarrow](2), int64[pyarrow](1), string[pyarrow](7)
# memory usage: 37.5 MB
