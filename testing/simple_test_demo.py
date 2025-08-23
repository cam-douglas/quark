#!/usr/bin/env python3
"""
Simple Test Demo - Demonstrates Live 3D Streaming with Pytest
"""

import pytest
import time
import numpy as np

def test_simple_calculation():
    """Simple test that will show live 3D streaming."""
    # Simulate some work
    time.sleep(0.1)
    
    # Simple calculation
    result = 2 + 2
    assert result == 4
    
    # Simulate more work
    time.sleep(0.1)
    
    # Test passed
    assert True

def test_array_operations():
    """Test array operations with live streaming."""
    # Create test data
    arr = np.array([1, 2, 3, 4, 5])
    
    # Simulate processing
    time.sleep(0.2)
    
    # Test array operations
    assert np.sum(arr) == 15
    assert np.mean(arr) == 3.0
    assert len(arr) == 5
    
    # Simulate more work
    time.sleep(0.1)

def test_string_operations():
    """Test string operations."""
    # Simulate work
    time.sleep(0.15)
    
    # Test string operations
    text = "Hello, World!"
    assert len(text) == 13
    assert "Hello" in text
    assert text.upper() == "HELLO, WORLD!"
    
    # Simulate more work
    time.sleep(0.1)

def test_boolean_logic():
    """Test boolean logic operations."""
    # Simulate work
    time.sleep(0.1)
    
    # Test boolean operations
    assert True and True
    assert not False
    assert True or False
    assert 1 == 1
    assert 2 != 3
    
    # Simulate more work
    time.sleep(0.1)

def test_list_operations():
    """Test list operations."""
    # Simulate work
    time.sleep(0.2)
    
    # Create and test list
    my_list = [1, 2, 3, 4, 5]
    assert len(my_list) == 5
    assert sum(my_list) == 15
    assert 3 in my_list
    
    # Simulate more work
    time.sleep(0.1)

def test_dictionary_operations():
    """Test dictionary operations."""
    # Simulate work
    time.sleep(0.15)
    
    # Create and test dictionary
    my_dict = {"a": 1, "b": 2, "c": 3}
    assert len(my_dict) == 3
    assert my_dict["a"] == 1
    assert "b" in my_dict
    
    # Simulate more work
    time.sleep(0.1)

def test_numerical_operations():
    """Test numerical operations."""
    # Simulate work
    time.sleep(0.25)
    
    # Test various numerical operations
    assert 10 + 5 == 15
    assert 10 - 5 == 5
    assert 10 * 5 == 50
    assert 10 / 5 == 2
    assert 10 ** 2 == 100
    assert 10 % 3 == 1
    
    # Simulate more work
    time.sleep(0.1)

def test_comparison_operations():
    """Test comparison operations."""
    # Simulate work
    time.sleep(0.1)
    
    # Test comparisons
    assert 5 > 3
    assert 3 < 5
    assert 5 >= 5
    assert 3 <= 5
    assert 5 == 5
    assert 5 != 3
    
    # Simulate more work
    time.sleep(0.1)
