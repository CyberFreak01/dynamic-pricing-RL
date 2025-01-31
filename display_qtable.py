import numpy as np
import pandas as pd

# Load the .npy file with allow_pickle=True
data = np.load('q_table.npy', allow_pickle=True)

# Check the shape and type of the loaded data
print("Data shape:", data.shape)
print("Data type:", type(data))

# Print the data based on its shape
if data.ndim == 0:
    print("Scalar value:", data.item())
elif data.ndim == 1:
    # If it's a 1D array, convert it to a DataFrame with one column
    df = pd.DataFrame(data, columns=['Value'])
    print(df)
elif data.ndim == 2:
    # If it's a 2D array, directly convert it to a DataFrame
    df = pd.DataFrame(data)
    print(df)
else:
    print("Data is too complex to display.")
