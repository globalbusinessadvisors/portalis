"""
External Package Support Demo
Shows transpilation of popular PyPI packages to Rust equivalents
"""

# NumPy → ndarray (Full WASM compatible)
import numpy as np

# Create arrays
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))

# Array operations
result = np.dot(arr, arr)

# Pandas → Polars (Partial WASM - I/O needs WASI)
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'score': [95.5, 87.3, 92.1]
})

# Operations
summary = df.describe()
top_rows = df.head()

# Requests → reqwest (Requires JS interop in WASM)
import requests

response = requests.get('https://api.example.com/data')
data = response.json()

# Pillow → image crate (Full WASM for processing, WASI for I/O)
from PIL import Image

img = Image.new('RGB', (100, 100), color='red')
# img.save('output.png')  # Needs WASI

# Scikit-learn → linfa (Full WASM compatible)
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# ML models
model = LinearRegression()
# model.fit(X, y)

clustering = KMeans(n_clusters=3)
# clustering.fit(data)

# Other popular packages

# SciPy → nalgebra + statrs
from scipy import stats
from scipy.linalg import inv

# Matplotlib → plotters
import matplotlib.pyplot as plt

# Pydantic → serde
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# Click → clap
import click

@click.command()
@click.option('--name', default='World')
def hello(name):
    print(f'Hello {name}!')
