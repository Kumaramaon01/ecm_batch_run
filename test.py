import os
import subprocess
from pathlib import Path
from helper import *
from report_ext import *
from src.masterFile import *

""" inp_folder = "E:/09_TestCode/00_ml_batch_run/20250630_3/seedinp" #input("Enter the INP folder path: ")
weather_path = "IND_Ahmedabad.426470_ISHRAE_EDS" #input("Enter the weather file name (without extension): ")
output_inp_folder = "E:/09_TestCode/00_ml_batch_run/20250630_3/modifiedInp" #input("Enter the output folder path: ")


bat_file_path = os.path.join(output_inp_folder, "script.bat")
print(bat_file_path)

print("calling script")
subprocess.call([bat_file_path, output_inp_folder, weather_path], shell=True) """

output_inp_folder= "E:/09_TestCode/00_ml_batch_run/20250630_3/modifiedInp/nikunj_Batch_1"
log_file_path = "E:/09_TestCode/00_ml_batch_run/20250630_3/modifiedInp/nikunj_Batch_1/nikunj_Batch1_Log_File.xlsx"
#get_failed_simulation_data(output_inp_folder, log_file_path)

#get_files_for_data_extraction(output_inp_folder,log_file_path, 7)

print("Data extraction complete")

loc_data = locationInfo.get_locInfo(output_inp_folder)
print(loc_data)
combined_data = masterFile.get_all_calculated_values(loc_data, output_inp_folder)
print(combined_data)
combined_data.to_csv(os.path.join(output_inp_folder, 'combined_data.csv'), index=False)