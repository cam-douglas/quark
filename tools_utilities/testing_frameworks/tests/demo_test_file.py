#!/usr/bin/env python3
"""
Demo test file for autonomous code editing system
Simple calculator implementation for testing

Purpose: Testing autonomous editing capabilities
Inputs: Calculator operations (add, subtract, multiply, divide)
Outputs: Calculated results and test outputs
Seeds: N/A (deterministic calculations)
Dependencies: None (pure Python)
"""

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def calculate(operation, a, b):
    if operation == "add":
        return add(a, b)
    elif operation == "subtract":
        return subtract(a, b)
    elif operation == "multiply":
        return multiply(a, b)
    elif operation == "divide":
        return divide(a, b)
    else:
        raise ValueError("Unknown operation")

def main():
    print("Simple Calculator Demo")
    print("=====================")
    
    # Test basic operations
    print(f"5 + 3 = {calculate('add', 5, 3)}")
    print(f"10 - 4 = {calculate('subtract', 10, 4)}")
    print(f"6 * 7 = {calculate('multiply', 6, 7)}")
    print(f"15 / 3 = {calculate('divide', 15, 3)}")

if __name__ == "__main__":
    main()

