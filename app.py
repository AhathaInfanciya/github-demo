a=5
b=30
def add_numbers(x, y):
    return x + y
def subtract_numbers(x, y):
    return x - y
def multiply_numbers(x, y):
    return x * y
def divide_numbers(x, y):
    if y == 0:
        return "Error: Division by zero"
    return x / y
print("Addition:", add_numbers(a, b))
