from numpy import *
import math

students_count = 1000
rating = 4.99
is_published = True
course_name = "Python"
multi_line = """
Multiple
Lines
"""

x = 1
y = 2
z, a = 1, 2   # the same as above

x = y = 1

print(type(students_count))
print(type(1.1))
print(type(multi_line))

# Python is dynamic language

age: int = 20
age = "Python"
print(age)

arr = array([1, 2, 3, 4])
print(arr)

########################

i = 1
print(id(i))  # returns address of memory location
i += 1
print(id(i))

l = [1, 2, 3]
print(id(l))

l.append(4)
print(id(l))
print(l)

# Strings

course = "Python Programming"
print(len(course))

print(course[0])
print(course[-1])  # negative index
print(course[3:6])
print(course[:10])
print(course[7:])
print(course[:])

print(id(course))
print(id(course[0]))

# Escape sequences

message = "Programming \n in \"Python \\ \'"
print(message)

another_message = """
Python 
Programming
"""

print(another_message)

first = "Marta"
last = "Budnik"
full_name = first + " " + last
print(full_name)
full = f"{first} {last}"  # it will be replaced in run time
print(full)

sth = f"{len(first)} {2 + x}"
print(sth)

project_name = "   Python Fundamentals"
print(project_name.upper())
print(project_name.lower())
print(project_name.title())
print(project_name.lstrip())
print(project_name.find("Fun"))
print(project_name.find("fun"))
print(project_name.replace("F", "*"))
print("Python" in project_name)
print("Python" not in project_name)

# Numbers

x = 10
x = 0b10
print(bin(x))

x = 0x12c
print(x)
print(hex(x))

x = 1 + 2j
print(x)

x = 10//3
print(x)

x = 10 % 3
x= 10 ** 3
print(x)

# no ++ or --

PI = -3.14
print(round(PI))
print(abs(PI))

print(math.floor(PI))

x = input("x: ")

print(int(x))
print(float(x))
print(bool(x))