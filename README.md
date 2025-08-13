# ML_Batch_Run
This will create Random batches from seed INPs to elaborate the dataset


# Create virtual env 
1. pip install virtualenv
2. python -m venv env
3. env\Scripts\activate
4. install required packages
5. to verify 'pip list'
6. pip freeze > requirements.txt

# setup env
1. pip install virtualenv
2. python -m venv env
3. env\Scripts\activate  
4. pip install -r requirements.txt

# To run code each time  
5. To activate env - "env\Scripts\activate"  
6. to run code - python main.py  
7. to deactivate run command  - deactivate  

# bug fixing
Error while running script
if facing error : 'Activate.ps1 cannot be loaded because running scripts is disabled on this system. For more 
information, see about_Execution_Policies'

Run following command:
 'Set-ExecutionPolicy -Scope CurrentUser'
 input 
 'unrestricted'
