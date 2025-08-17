import numpy as np

address,city,country,longitude, latitude, name,postalcode= np.genfromtxt('C:\\Users\\ECON\\OneDrive\\Documents\\GitHub\\FULLSTACK-WITH-AI-BOOTCAMP-B1-MonToFri-2.5Month-Explorer\\DataSetForPractice\\FastFoodRestaurants.csv.csv', delimiter=',', usecols=(0, 1,2,4,5,6,7), unpack=True , dtype=None , skip_header=1)

print(address)
print(city)
print(country)
print(name)
print(longitude)
print(latitude)
print(postalcode)

print("FastFoodRestaurants: " , np.mean(postalcode))
print("FastFoodRestaurants: " , np.average(postalcode))
print("FastFoodRestaurants: " , np.std(postalcode))
print("FastFoodRestaurants: " , np.median(postalcode))
print("FastFoodRestaurants: " , np.percentile(postalcode,25))
print(" FastFoodRestaurants : " , np.percentile(postalcode,75))
print("FastFoodRestaurants: " , np.percentile(postalcode,3))
print("FastFoodRestaurants: " , np.min(postalcode))
print("FastFoodRestaurants : " , np.max(postalcode))

print("FastFoodRestaurants square: " , np.square(postalcode))
print("FastFoodRestaurants Price sqrt: " , np.sqrt(postalcode))
print("FastFoodRestaurants Price pow: " , np.power(postalcode,postalcode))
print("FastFoodRestaurants abs: " , np.abs(postalcode))

addition = longitude + latitude
subtraction = longitude - latitude
multiplication = longitude * latitude
division = longitude / latitude

print(" FastFoodRestaurants Long - lat - Addition:", addition)
print(" FastFoodRestaurants long - lat - Subtraction:", subtraction)
print(" FastFoodRestaurants long - lat - Multiplication:", multiplication)
print(" FastFoodRestaurants long - lat - Division:", division)



postalcodePie = (postalcode/np.pi) +1

sine_values = np.sin(postalcodePie)
cosine_values = np.cos(postalcodePie)
tangent_values = np.tan(postalcodePie)

print(" FastFoodRestaurants postalcode - div - pie  - Sine values:", sine_values)
print(" FastFoodRestaurants postalcode - div - pie Cosine values:", cosine_values)
print(" FastFoodRestaurants postalcode - div - pie Tangent values:", tangent_values)

print("FastFoodRestaurants postalcode - div - pie  - Exponential values:", np.exp(postalcodePie))


log_array = np.log(postalcodePie)
log10_array = np.log10(postalcodePie)

print("FastFoodRestaurants postalcode - div - pie  - Natural logarithm values:", log_array)
print("FastFoodRestaurants posstalcode - div - pie  = Base-10 logarithm values:", log10_array)

sinh_values = np.sinh(postalcodePie)
print("FastFoodRestaurants PostalCode - div - pie   - Hyperbolic Sine values:", sinh_values)

#Hyperbolic Cosine Using cosh() Function
# Calculate the hyperbolic cosine of each element
cosh_values = np.cosh(postalcodePie)
print("FastFoodRestaurants PostalCode - div - pie   - Hyperbolic Cosine values:", cosh_values)

#Example: Hyperbolic Tangent
# Calculate the hyperbolic tangent of each element
tanh_values = np.tanh(postalcodePie)
print("FastFoodRestaurants PostalCode - div - pie   -Hyperbolic Tangent values:", tanh_values)

#Example: Inverse Hyperbolic Sine
# Calculate the inverse hyperbolic sine of each element
asinh_values = np.arcsinh(postalcodePie)
print("FastFoodRestaurants PostalCode - div - pie   -Inverse Hyperbolic Sine values:", asinh_values)

#Example: Inverse Hyperbolic Cosine
# Calculate the inverse hyperbolic cosine of each element
acosh_values = np.arccosh(postalcodePie)
print("FastFoodRestaurants PostalCode - div - pie   -Inverse Hyperbolic Cosine values:", acosh_values)

#FastFoodRestaurants Long Plus Lat - 2 dimentional arrary
D2LongLat = np.array([longitude,
                  latitude])

print ("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - " ,D2LongLat)

# check the dimension of array1
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - dimension" , D2LongLat.ndim) 
# Output: 2

# return total number of elements in array1
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - total number of elements" ,D2LongLat.size)
# Output: 6

# return a tuple that gives size of array in each dimension
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - gives size of array in each dimension" ,D2LongLat.shape)
# Output: (2,3)

# check the data type of array1
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - data type" ,D2LongLat.dtype) 
# Output: int64

# Splicing array
D2LongLatSlice=  D2LongLat[:1,:5]
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - Splicing array - D2LongLat[:1,:5] " , D2LongLatSlice)
D2LongLatSlice2=  D2LongLat[:1, 4:15:4]
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - Splicing array - D2LongLat[:1, 4:15:4] " , D2LongLatSlice2)


D2LongLatSliceItemOnly=  D2LongLatSlice[0,1]
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - Index array - D2LongLatSlice[0,1] " , D2LongLatSliceItemOnly)
D2LongLatSlice2ItemOnly=  D2LongLatSlice2[0, 2]
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - index array - D2LongLatSlice2[0, 2] " , D2LongLatSlice2ItemOnly)

for elem in np.nditer(D2LongLat):
    print(elem)

for index, elem in np.ndenumerate(D2LongLat):
    print(index, elem)



D2LongLat1TO50 = np.reshape(D2LongLat, (1, 50))
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - np.reshape(D2LongLat, (1, 298)) : " , D2LongLat1TO50)
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - np.reshape(D2LongLat, (1, 298)) : Size " , D2LongLat1TO50.size)
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - np.reshape(D2LongLat, (1, 298)) : ndim " , D2LongLat1TO50.ndim)
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - np.reshape(D2LongLat, (1, 298)) : shape " , D2LongLat1TO50.shape)
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - np.reshape(D2LongLat, (1, 298)) : ndim " , D2LongLat1TO50.ndim)

print()