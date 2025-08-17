
# STRING MANIPULATION

# 1. First, middle, last character
s = input("Enter a string: ")
mid = len(s) // 2
new_str = s[0] + s[mid] + s[-1]
print("New String:", new_str)

# 2. Count occurrences of all characters
s = input("Enter a string: ")
char_count = {}
for ch in s:
    char_count[ch] = char_count.get(ch, 0) + 1
print("Occurrences:", char_count)

# 3. Reverse string
s = input("Enter a string: ")
print("Reversed:", s[::-1])

# 4. Split on hyphens
s = input("Enter a string with hyphens: ")
print("Splitted:", s.split('-'))

# 5. Remove punctuation
import string
s = input("Enter a string with punctuation: ")
clean = "".join(ch for ch in s if ch not in string.punctuation)
print("Clean String:", clean)


# ================================
# 🔹 LIST MANIPULATION
# ================================

# 1. Reverse list
lst = [1,2,3,4,5]
lst.reverse()
print("Reversed List:", lst)

# 2. Squares of list
lst = [1,2,3,4,5]
squares = [x**2 for x in lst]
print("Squares:", squares)

# 3. Remove empty strings
lst = ["apple", "", "banana", "", "cherry"]
lst = [x for x in lst if x]
print("No Empty Strings:", lst)

# 4. Add new item after specific item
lst = [1,2,3,4,5]
item, new_item = 3, 99
if item in lst:
    index = lst.index(item)
    lst.insert(index+1, new_item)
print("Updated List:", lst)

# 5. Replace item if found
lst = [10,20,30,40]
old, new = 20, 200
lst = [new if x == old else x for x in lst]
print("Replaced List:", lst)


# ================================
# 🔹 DICTIONARY MANIPULATION
# ================================

d = {"a": 10, "b": 5, "c": 30}

# 1. Check value exists
print(5 in d.values())

# 2. Get key of minimum value
min_key = min(d, key=d.get)
print("Key of min value:", min_key)

# 3. Delete keys
keys_to_delete = ["a","c"]
for k in keys_to_delete:
    d.pop(k, None)
print("After Deletion:", d)


# ================================
# 🔹 TUPLE MANIPULATION
# ================================

# 1. Reverse tuple
t = (1,2,3,4)
print("Reversed Tuple:", t[::-1])

# 2. Access value 20
t = (10,20,30)
print("Value:", t[1])

# 3. Swap tuples
t1, t2 = (1,2), (3,4)
t1, t2 = t2, t1
print("Swapped:", t1, t2)


# ================================
# 🔹 LOOP MANIPULATION
# ================================

# 1. First 10 natural numbers
i = 1
while i <= 10:
    print(i, end=" ")
    i += 1

# 2. Even numbers till n
n = int(input("\nEnter number: "))
for i in range(2, n+1, 2):
    print(i, end=" ")

# 3. Odd numbers till n
n = int(input("\nEnter number: "))
for i in range(1, n+1, 2):
    print(i, end=" ")

# 4. Prime numbers till n
n = int(input("\nEnter number: "))
for num in range(2, n+1):
    for j in range(2, int(num**0.5)+1):
        if num % j == 0:
            break
    else:
        print(num, end=" ")

# 5. Multiplication table
num = int(input("\nEnter a number: "))
for i in range(1, 11):
    print(num, "x", i, "=", num*i)


# ================================
# 🔹 STORY BASED ASSIGNMENTS
# ================================

story = """The year was 2147. Humanity had long since ceded control of its daily functions 
to artificial intelligence... (cut short for brevity) ... decided what to do next."""

#  1: List all words with vowels
words = story.split()
vowel_words = [w for w in words if any(v in w.lower() for v in "aeiou")]
print("Words with vowels:", vowel_words)

#  2a: List of nouns (just sample nouns picked manually for demo)
nouns = ["year","humanity","control","intelligence","Athena-9","Dr.Voss","Council","freedom"]
print("Nouns List:", nouns)

#  2b: Add numbers
numbers = ["2147","9"]
nouns_with_numbers = nouns + [numbers]
print("Nouns + Numbers:", nouns_with_numbers)

#  3: Tuples
nouns_tuple = tuple(nouns)
print("Nouns Tuple:", nouns_tuple)

#  3b: Tuples + Numbers
nouns_tuple_with_numbers = tuple(nouns) + (tuple(numbers),)
print("Tuple + Numbers:", nouns_tuple_with_numbers)

#  4: Sets
nouns_set = set(nouns)
print("Nouns Set:", nouns_set)

#  4b: Set + Numbers
nouns_set_with_numbers = nouns_set.union({tuple(numbers)})
print("Set + Numbers:", nouns_set_with_numbers)

#  5: Dictionary
nouns_dict = {i: nouns[i] for i in range(len(nouns))}
nouns_dict["numbers"] = numbers
print("Nouns Dictionary:", nouns_dict)