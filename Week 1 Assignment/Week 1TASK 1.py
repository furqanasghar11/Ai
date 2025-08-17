#Question No 1

celcius = input("enter temprature in celcius: ")
celcius = float(celcius)
fahrenhiet = (celcius * 9/5 ) +32
print("fahrenhiet :", fahrenhiet)

#Question No 2

lenght = int(input("enter your lenght:"))
width = int(input("enter your width:"))
area = lenght * width
print("area of rectangle is:", area)

#uestion No 3

P = input("enter principal")
R = input("enter rate")
T = input("enter time")
P = float(P)
R = float(R)
T = float(T)
# Simple Interest Formula: CI = P * (1 + R/100)**T - P
CI = P * (1 + R/100)**T - P
print(CI)

#Question No 4

lenght = int(input("enter your lenght:"))
width = int(input("enter your width:"))
parimeter = 2 * (lenght + width)
print(parimeter)

#Question No 5

number1 = int(input("enter your first number:"))
number2 = int (input("enter your second numer:"))
number3 = int(input("enter your third number:"))
Average = (number1 + number2 + number3) / 3
print(Average)

#Question No6

number = int(input("enter your number:"))
square = number * number
cube = number * number * number
print(square,cube)

#Question No 7

n = int(input("enter your candies:"))
k = int(input("enter your students:"))
distribution = n // k
remainder = n % k
print(distribution, remainder)

#Question No 8

cost = float(input("enter your cost:"))
price = float(input("enter your price:"))
if cost < price:
    profit = price-cost
    print(f"profit:{profit}")
elif cost > price:
    loss = cost-price
    print(f"loss: {loss}")
else:
    print("no profit no loss")

#Question No 9

subject1 = int(input("enter your first subject number:"))
subject2 = int(input("enter your second subject number:"))
subject3 = int(input("enter your third subject number:"))
subject4 = int(input("enter your fourth subject number:"))
subject5 = int(input("enter your fifth subject number:"))

total_marks = subject1+subject2+subject3+subject4+subject5
print(total_marks)

average = total_marks/5
print(average)

percentage = total_marks/500*100
print(percentage)


#Question No 10

basic_salary = float(input("enter your basic salary:"))


HRA = basic_salary * 0.20

DA  = basic_salary * 0.15

Total_Salary = basic_salary + HRA + DA

print(f"Total_Salary:{Total_Salary}")

#Question No 11

age = float(input("enter your age:"))

age_in_months = age * 12
age_in_days = age * 365

print(f"age_in_days:{age_in_days}")
print(f"age_in_months:{age_in_months}")

#Question No 12

convert = int(input("enter your number:"))
convert = float(convert)
convert = convert * 280
print("your amount in rupees is:",convert)


#Question No 13

n = int(input("enter your n natural number:"))
sum_of_natural_number = n * (n + 1) / 2
print("sum of first n",sum_of_natural_number)

#Question No 14

total_question = int(input("enter your total quesstion:"))
correct_awnser = int(input("enter your correct awnser:"))
percentage = correct_awnser/total_question *100
print(percentage)

#Question No 15

distance = int(input("enter your ditance in kn:"))
time = int(input("enter your time in hours:"))
speed = distance / time
print("your speed is:", speed, "kn/h")

#Question No16

weight = int(input("enter your weight:"))
height = float(input("enter your height:"))
bmi = weight / (height ** 2)
print("your bmi is:", bmi)

# Question No 17

minutes = int(input("enter your minutes:"))
hours = minutes/60 
remainder = minutes % 60
minutes = remainder
hours = int(hours)
print("your hours is:",hours,"and minutes is:",minutes)