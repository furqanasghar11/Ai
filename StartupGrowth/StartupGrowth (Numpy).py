import numpy as np

# CSV file se data load (sirf required columns liye gaye hain)
fund, investment, value, rating = np.genfromtxt("C:\\Users\\ECON\\OneDrive\\Documents\\GitHub\\FULLSTACK-WITH-AI-BOOTCAMP-B1-MonToFri-2.5Month-Explorer\\DataSetForPractice\\startup_growth_investment_data.csv", delimiter=",", usecols=(2,3,4,8), unpack=True, skip_header=1)

# Print basic arrays
print(fund)
print(investment)
print(value)
print(rating)

#Statistics on rating
print("Mean:", np.mean(rating))
print("Average:", np.average(rating))
print("Std Dev:", np.std(rating))
print("Median:", np.median(rating))
print("25th Percentile:", np.percentile(rating, 25))
print("Min:", np.min(rating))
print("Max:", np.max(rating))
print("Square:", np.square(rating))
print("Square root:", np.sqrt(rating))
print("Absolute:", np.abs(rating))

#Basic Math operations
addition = investment + rating
substraction = investment - rating
multiplication = investment * rating
division = investment / rating

print("Addition:", addition)
print("Subtraction:", substraction)
print("Multiplication:", multiplication)
print("Division:", division)

#Trigonometric Functions
ratingPie = (rating / np.pi) + 1
print("sin:", np.sin(ratingPie))
print("cos:", np.cos(ratingPie))
print("tan:", np.tan(ratingPie))

# Exponential
print("Exponential:", np.exp(ratingPie))

#Logarithmic Functions
print("Natural log:", np.log(ratingPie))
print("Base-10 log:", np.log10(ratingPie))

#2D Array
D2InvestRate = np.array([investment, rating])
print("2D Array:", D2InvestRate)
print("Dimensions:", D2InvestRate.ndim)
print("Total elements:", D2InvestRate.size)
print("Shape:", D2InvestRate.shape)
print("Data type:", D2InvestRate.dtype)

# Slicing & Indexing 
D2slicing = D2InvestRate[:1, :5]   # First row, first 5 values
print("Slice:", D2slicing)

D2InvestRateSliceItemOnly = D2slicing[0, 1]  # Single element
print("Indexed item:", D2InvestRateSliceItemOnly)

#Iteration
for elem in np.nditer(D2InvestRate):
    print(elem)

for index, elem in np.ndenumerate(D2InvestRate):
    print(index, elem)

#Reshape
D2InvestRate1TO298 = np.reshape(D2InvestRate, (1, 298))
print("Reshaped Array:", D2InvestRate1TO298)
print("Reshaped size:", D2InvestRate1TO298.size)

#Trigonometric on whole 2D array
print("sin:", np.sin(D2InvestRate))
print("cos:", np.cos(D2InvestRate))
print("tan:", np.tan(D2InvestRate))
