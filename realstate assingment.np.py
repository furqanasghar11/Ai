#RealEstate-USA.csv
import numpy as np

brokered_by,price,acre_lot,city,house_size= np.genfromtxt('DataSetForPractice/RealEstate-USA.csv', delimiter=',', usecols=(0,2,5,7,10), unpack=True,dtype=None,skip_header=1)


print(brokered_by)
print(house_size)
print(price)
print(city)
print(acre_lot)


#realstate-USA statics operation
print(np.mean(price))
print(np.average(price))
print(np.std(price))


#addition
total=(price + house_size)
print(total)
#subtraction
total=(price - house_size)
print(total)
#multiplication
total=(price * house_size)
print(total)

#2d array
darray = np.array([price,house_size])
print(darray)

#3d array 
earray=np.array([price,house_size,acre_lot])
print(earray)

#nditer
for item in np.nditer(price):
    print(item)

#ndenumerate
for  item in np.ndenumerate(price):
    print(item)

#7 common properties of array
print(np.ndim(price))
print(np.shape(price))
print(np.size(price))
print(price.dtype)
print(price.itemsize)
print(price.nbytes)
print(price.T)


# Slicing: rows 0 to 2 (1st to 3rd), columns 1 to 3 (2nd to 4th)
price_slice = price[0:3]
house_size_slice = house_size[1:4]

slice= np.array([price_slice,house_size_slice]).T
print(slice)

# Slicing: rows 2 to 8 , columns 3 to 5
# darray = np.column_stack((price, house_size))
# sliced = darray[1:8, 2:5]
# print("Sliced Array:")
# print(sliced)

#trignometery function

print(np.sin(darray))
print(np.cos(darray))
print(np.tan(darray))

normalized = np.clip(darray / np.max(darray), -1, 1)

print(np.arcsin(normalized))
print(np.arccos(normalized))
print(np.arctan(normalized))

 