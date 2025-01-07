import pandas as pd

coffee = pd.read_csv(filepath_or_buffer='C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Pandas/Data/Inputs/coffee.csv')  # Load data from CSV
results = pd.read_parquet(path='C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Pandas/Data/Inputs/results.parquet')  # Load data from Parquet
olympics_data = pd.read_excel(io='C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Pandas/Data/Inputs/olympics-data.xlsx', sheet_name="results")  # Load data from Excel
