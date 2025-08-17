# Import NumPy
import numpy as np

# Load specific columns from CSV into separate arrays
brokered_by, price, acre_lot, city, house_size = np.genfromtxt('DataSetForPractice/RealEstate-USA.csv',delimiter=',',usecols=(0, 2, 5, 7, 10),unpack=True,dtype=None,skip_header=1)

# Print loaded arrays
print(brokered_by)  # Broker names
print(house_size)   # House sizes
print(price)        # Prices
print(city)         # City names
print(acre_lot)     # Acre lot sizes

# Descriptive statistics on price
print(np.mean(price))    # Mean
print(np.average(price)) # Average
print(np.std(price))     # Standard deviation

# Addition of price & house size
total = (price + house_size)
print(total)

# Subtraction of price & house size
total = (price - house_size)
print(total)

# Multiplication of price & house size
total = (price * house_size)
print(total)

# Create 2D array from price & house size
darray = np.array([price, house_size])
print(darray)

# Create 3D array from price, house size & acre lot
earray = np.array([price, house_size, acre_lot])
print(earray)

# Iterate price array using nditer
for item in np.nditer(price):
    print(item)

# Iterate price array with index using ndenumerate
for item in np.ndenumerate(price):
    print(item)

# 7 common NumPy array properties
print(np.ndim(price))   # Number of dimensions
print(np.shape(price))  # Shape of array
print(np.size(price))   # Total number of elements
print(price.dtype)      # Data type of array elements
print(price.itemsize)   # Size of one element in bytes
print(price.nbytes)     # Total bytes consumed by array
print(price.T)          # Transpose of array

# Slice rows 0-2, columns 1-3
price_slice = price[0:3]
house_size_slice = house_size[1:4]
slice = np.array([price_slice, house_size_slice]).T
print(slice)

# Trigonometric operations on darray
print(np.sin(darray))   # Sine
print(np.cos(darray))   # Cosine
print(np.tan(darray))   # Tangent

# Normalize data for inverse trig functions
normalized = np.clip(darray / np.max(darray), -1, 1)

# Inverse trig functions
print(np.arcsin(normalized))  # Inverse sine
print(np.arccos(normalized))  # Inverse c
