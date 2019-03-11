# loops
for x in "Python":
    print(x)

for x in ['a', 'b', 'c']:
    print(x)

for x in range(0, 10, 2):
    print(x)

print(type(range(0, 5)))
print([1, 2, 3, 4, 5])

names = ["John", "Mary"]
for name in names:
    if name.startswith("J"):
        print("Found")
        break
else:
    print("Not found")

# While

guess = 0
answer = 5

while answer != guess:
    guess = int(input("Guess: "))


def increment(number: int, by: int = 1) -> tuple:
    return (number, number + by)


print(increment(2, 3))


def multiply(*list):
    total = 1
    for number in list:
        total *= number
    return total


print(multiply(2, 3, 4, 5))


def save_user(**user):
    print(user)


save_user(id=1, name="admin")  # dictionary

