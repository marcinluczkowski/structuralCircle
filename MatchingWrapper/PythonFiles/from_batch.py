import sys # import system to read arguments
import os
import numpy as np # import numpy for calculating the periods


amplitude = float(sys.argv[1]) # first input number
periods = float(sys.argv[2]) # second input number

x_array = np.arange(0, 6 * np.pi, step = 0.1)
y_array = amplitude * np.sin(x_array * periods)

# This path should change as well
path = "C:\\Users\\sverremh\\AppData\\Roaming\\Grasshopper\\Libraries\\MatchingWrapper\\PythonFiles"
#print(path)

# set working dir


filepath = '{0}\\csv_file.csv'.format(path)
with open(filepath, 'w') as f:
    f.write('x_coordinate,y_coordinate\n')
    for x, y in zip(x_array, y_array):
        f.write('{0},{1}\n'.format(x,y))    

print("CSV-file named {0} created".format(filepath))
