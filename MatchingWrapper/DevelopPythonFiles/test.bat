::===============================================================
:: The below batch file runs the matching script from the terminal. 
:: It also installs a new local virtual environmen for the folder we need
:: and activates this
::===============================================================
@echo off 

:: Start the script by creating a virutal environment if it does not already exist.

echo "This is running"

if exist "matching_env" (
	echo "Virtual environment already created. Proceed to matching."

	echo "activating environment"
	call matching_env\Scripts\activate.bat :: need to use call keyword 
	echo "Activated local virtual environemnt for matching scripts"
)
if not exist "matching_env" (
	echo "Creating new virtual environment."
	python -m venv matching_env
	echo "Virtual environment named 'matching_env' created"
	
	echo "activating environment"
	call matching_env\Scripts\activate.bat :: need to use call keyword 
	
    echo "Installing necessary packages."
	deactivate
)

	
:: create a csv file with coordinates
echo "Create a csv file with coordinates"
python from_batch.py %1 %2




:: deacivate local environment
deactivate
