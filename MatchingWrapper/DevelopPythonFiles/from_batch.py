import sys # import system to read arguments
import os
import numpy as np # import numpy for calculating the periods


amplitude = float(sys.argv[1]) # first input number
periods = float(sys.argv[2]) # second input number

x_array = np.arange(0, 6 * np.pi, step = 0.1)
y_array = amplitude * np.sin(x_array * periods)
path = "C:\\Users\\sverremh\\source\\repos\\MatchingWrapper\\MatchingWrapper\\DevelopPythonFiles"
#print(path)

# set working dir
print("Working dir: {0}".format(path))
sys.stdout.flush()

with open('{0}\\csv_file.csv'.format(path), 'w') as f:
    f.write('x_coordinate,y_coordinate')
    for x, y in zip(x_array, y_array):
        f.write('{0},{1}\n'.format(x,y))    


