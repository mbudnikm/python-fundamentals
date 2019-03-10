# Conditional Statements

age = 22

if age >= 18:
    print("Adult")
elif age >= 13:
    print("Teenager")
else:
    print("Child")

print("All done!")

x = 5

if x > 1:
    pass
else:
    pass

# Logical operators

name = " "
# Falsy values: 0, "", None, []
if not name.strip():
    print("Name is empty")
else:
    print(name)

if 18 <= age < 65:
    print("Eligible")

if age >= 18:
    message = "Eligible"
else:
    message = "Not eligible"

# Ternary operator
message = "Eligible" if age >= 18 else "Not eligible"

print(message)

# Loops 1:12



