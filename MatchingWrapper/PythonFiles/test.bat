::===============================================================
:: The below batch file runs the matching script from the terminal. 
:: It also installs a new local virtual environmen for the folder we need
:: and activates this
::===============================================================
@echo off

:: Set the "PythonFiles" as the working directory.
(echo Current dir: & echo %cd% & echo ----)
cd PythonFiles
(echo Dir after changing: & echo %cd% & echo ------)
:: Start the script by creating a virutal environment if it does not already exist.

echo "This is running"
:: set variables for the script
:: ------------------------------
set packageList=sitePackages.txt
set environment=matching_env
set packageName=elementmatcher
set packageLocation=elementmatcher-0.0.1.tar.gz
::-------------------------------

:: test if the matching environment already exist
if exist %environment% (
	echo Environment already exists. Checking for packages...
	
	:: check if package exist in environnment site-packages
	dir "matching_env\Lib\site-packages" > %packageList%
	setlocal enabledelayedexpansion
	set count=0
	for /F %%G in ('type %packageList% ^| findstr /R /I /C:"%packageName%"') do (
		set /A count+=1
	)
	:: if count > 0 package exists.
	if !count! gtr 0 (
		echo Package exists. Activating environment.
		call matching_env\Scripts\activate.bat
	) else (
		echo Package not installed. Activating environment and installing package with dependencies.
		call matching_env\Scripts\activate.bat
		pip install %packageLocation%
	)
	del %packageList%
	
	
) else (
	echo "Creating new virtual environment."
	python -m venv matching_env
	echo "Virtual environment named 'matching_env' created"
	
	echo "activating environment"
	call matching_env\Scripts\activate.bat :: need to use call keyword 
	
    echo "Installing necessary packages."
	pip install %packageLocation% :: install the necessary package
)

	
:: Run the python relevant python file
(echo "==================" & echo "			Run matching script" & echo "==================")
:: echo %cd%
:: First input is amplitude, seond is number of periods.
:: Echo input
echo %1
echo %2
echo %3
echo %4
python -u from_batch.py %1 %2 %3 %4

deactivate :: deactivat local environment

