import sys # import system to read arguments

num1 = sys.argv[1] # first input number
num2 = sys.argv[2] # second input number

sys.stdout.flush()
print(f'Added inputs: {int(num1) + int(num2)}')