@echo off

echo Checking if the midas-py310 environment exists...
call conda info --envs | findstr /B /C:"midas-py310" 1>nul && (
    echo midas-py310 environment found, updating...
    call conda env update -n midas-py310 -f environment.yaml
) || (
    echo midas-py310 environment not found, creating...
    call conda env create -f environment.yaml
)

echo Activating the environment...
call conda activate midas-py310

echo Installing OpenCV and additional packages...
call conda install opencv numpy scipy scikit-image -c conda-forge -y

echo Done.
pause
