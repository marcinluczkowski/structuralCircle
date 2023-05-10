import sys
import pandas as pd
import matplotlib.pyplot as plt

num1 = sys.argv[1]
num2 = sys.argv[2]
df = pd.DataFrame({'A': [0, 1, 2], 'B' : [2, 3, 4]})
#result = float(num1) * float(num2)
result1 = num1
print("Two input numbers are multiplied and result is seen below.")
sys.stdout.flush()
print(f'Result: {result1}')
print(df)

