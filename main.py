import os
import random
import time
from datetime import datetime
import pandas as pd
from shutil import copyfile
import glob as gb
import subprocess
import numpy as np
import shutil
from pathlib import Path
from collections import defaultdict
from src import insertWall, insertConst, orient, lighting, equip, windows, insertRoof, wwr
import sys
import traceback
import re
from src import lv_b, ls_c, lv_d, pv_a_loop, sv_a, beps, bepu, lvd_summary, sva_zone, locationInfo, masterFile, sva_sys_type, pv_a_pump, pv_a_heater, pv_a_equip, pv_a_tower, ps_e, inp_shgc

# function to get report in csv and save in specific folders
def get_report_and_save(report_function, name, file_suffix, folder_name, path):
    try:
        print("calling report function: ", report_function, name, path)
        report = report_function(name, path)
    except:
        print("Skipping...")
    # get file path name as .csv
    file_path = os.path.join(path, f'{x[0]}_{file_suffix}.csv')
    # if that file already exist, replace with other file.
    if os.path.isfile(file_path):
        os.remove(file_path)
    # writing csv file with headers and no index column
    print("Done-", name)
    with open(file_path, 'w', newline='') as f:
        report.to_csv(f, header=True, index=False, mode='wt') 

def remove_section_content(content, start_marker, end_marker):
    """
    Function to remove content between start_marker and end_marker.
    Deletes lines between start_marker_index + 3 and end_marker_index - 3.
    """
    new_content = []
    
    start_index = -1
    end_index = -1
    for i, line in enumerate(content):
        if start_marker in line and start_index == -1:
            start_index = i
        if end_marker in line and end_index == -1:
            end_index = i
    
    if start_index != -1 and end_index != -1:
        new_content.extend(content[:start_index + 1])
        new_content.extend(content[start_index + 1:start_index + 3])
        new_content.extend(content[end_index - 3:end_index])
        new_content.append(content[end_index])
        new_content.extend(content[end_index + 1:])
    else:
        new_content = content
    
    return new_content

def delete_glass_type_codes(content):
    return remove_section_content(content, "$              Glass Type Codes", "$              Glass Types")

def modify_glass_types(content, start_marker, end_marker):
    """
    Modify GLASS-TYPE names by adding 'ML_' as a prefix.
    """
    start_index = None
    end_index = None

    # Identify the start and end of the Glass Types section
    for i, line in enumerate(content):
        if start_marker in line and start_index is None:
            start_index = i
        if end_marker in line and start_index is not None:
            end_index = i
            break

    if start_index is None or end_index is None:
        # Markers not found, return content unchanged
        return content

    # Modify GLASS-TYPE names in the Glass Types section
    for i in range(start_index + 1, end_index):
        if "= GLASS-TYPE" in content[i]:
            parts = content[i].split("=")
            if len(parts) > 1:
                glass_type_name = parts[0].strip().strip('"')
                modified_name = f'ML_{glass_type_name}'
                content[i] = content[i].replace(glass_type_name, modified_name)

    return content, glass_type_name

def delete_window_layers(content):
    return remove_section_content(content, "$              Window Layers", "$              Lamps / Luminaries / Lighting Systems")

def remove_window_sections(content, start_marker, end_marker):
    """
    Removes all = WINDOW sections between the start_marker and end_marker.
    """
    start_index = None
    end_index = None
    for i, line in enumerate(content):
        if start_marker in line and start_index is None:
            start_index = i
        if end_marker in line and start_index is not None:
            end_index = i
            break
    
    if start_index is None or end_index is None:
        # Markers not found, return content unchanged
        return content

    # Extract the content between the markers
    pre_marker_content = content[:start_index + 1]
    between_marker_content = content[start_index + 1:end_index]
    post_marker_content = content[end_index:]
    
    # Filter out = WINDOW sections
    filtered_content = []
    skip_window_section = False

    for line in between_marker_content:
        if "= WINDOW" in line:
            skip_window_section = True
        if skip_window_section and line.strip() == "..":
            skip_window_section = False
            continue
        if not skip_window_section:
            filtered_content.append(line)

    return pre_marker_content + filtered_content + post_marker_content

def include_window_sections(content, start_marker, end_marker, df, glass_type_name, height):
    # Initialize indices to None
    start_index = None
    end_index = None

    # Search for the markers in the content (which is a list of lines)
    for i, line in enumerate(content):
        if start_marker in line and start_index is None:
            start_index = i 
        if end_marker in line and start_index is not None:
            end_index = i
            break

    if start_index is None or end_index is None:
        return content  # If markers are not found, return the original content

    # Extract the content between the markers
    section = content[start_index:end_index]

    # Prepare the modified section
    modified_section = []

    # Iterate through the section to process EXTERIOR-WALL entries
    current_wall_name = None
    include_window = False

    for line in section:
        modified_section.append(line)

        if "= EXTERIOR-WALL" in line:
            current_wall_name = line.split("=")[0].strip().strip('"')  # Extract wall name
            include_window = True  # Allow processing unless LOCATION says otherwise

        if "LOCATION" in line and current_wall_name:
            location_value = line.split("=")[1].strip()
            if location_value in ["TOP", "BOTTOM"]:
                include_window = False  # Skip if LOCATION is TOP or BOTTOM
                current_wall_name = None  # Reset wall name for safety

        if include_window and line.strip() == "..":
            if current_wall_name in df['EXTERIOR-WALL'].values:
                row = df[df['EXTERIOR-WALL'] == current_wall_name].iloc[0]
                # Construct window name
                base_name = current_wall_name.split("Wall")[0].strip()  # Ensure "Wall" is excluded
                identifier = current_wall_name.split("(")[1].strip(")")  # Extract the identifier
                window_name = f"{base_name} Win ({identifier}.W1)"  # Construct the correct name
                window_section = f'''"{window_name}" = WINDOW
   GLASS-TYPE       = "ML_{glass_type_name}"
   FRAME-WIDTH      = 0
   X                = {row['X']}
   Y                = {row['Y']}
   HEIGHT           = {row[f'HEIGHT{height + 1}']}
   WIDTH            = {row[f'WIDTH{height + 1}']}
   FRAME-CONDUCT    = 2.781
   ..
'''
                modified_section.append(window_section)  # Insert window section
                include_window = False  # Reset flag

    # Reassemble content with the modified section
    modified_content = content[:start_index] + modified_section + content[end_index:]
    return modified_content

def process_sections(content, df, height):
    # df.to_csv("window_coordinates.csv", index=False)
    df = df[df['SH2'].isna() | (df['SH2'] == '')]
    # with open(file_path, 'r') as file:
    #     content = file.readlines()
    
    content = delete_glass_type_codes(content)
    content, glass_type_name = modify_glass_types(content, "$              Glass Types", "$              Window Layers")
    content = delete_window_layers(content)
    content = remove_window_sections(content, "$ **      Floors / Spaces / Walls / Windows / Doors      **",
        "$ **              Electric & Fuel Meters                 **")
    content = include_window_sections(content, "$ **      Floors / Spaces / Walls / Windows / Doors      **",
    "$ **              Electric & Fuel Meters                 **", df, glass_type_name, height)  # or any height value you want

    
    # dir_name, file_name = os.path.split(file_path)
    # modified_file_name = 'Purged_90%_' + file_name
    # modified_file_path = os.path.join(dir_name, modified_file_name)

    return content

def process_all_inp_files_in_folder(inp_path, df, height):
    # print(inp_path)
    """Process all .inp files in a folder, modifying each one using the process_sections function."""
    process_sections(inp_path, df, height)

def extract_polygons(flist):
    # with open(inp_file) as f:
    #     # Read all lines from the file and store them in a list named flist
    #     flist = f.readlines()
        
    # Initialize an empty list to store line numbers where 'Polygons' occurs
    polygon_count = [] 
    # Iterate through each line in flist along with its line number
    for num, line in enumerate(flist, 0):
        if 'Polygons' in line:
            polygon_count.append(num)
        if 'Wall Parameters' in line:
            numend = num
    # Store the line number of the first occurrence of 'Polygons'
    numstart = polygon_count[0] if polygon_count else None
    if not numstart:
        # print("No 'Polygons' section found in the file.")
        return pd.DataFrame()  # Return an empty dataframe if no polygons section is found
    
    # Slice flist from the start of 'Polygons' to the line before 'Wall Parameters'
    polygon_rpt = flist[numstart:numend]
    
    # Initialize an empty dictionary to store polygon data
    polygon_data = {}
    current_polygon = None
    vertices = []
    
    # Iterate through the lines in polygon_rpt
    for line in polygon_rpt:
        if line.strip().startswith('"'):  # This indicates a new polygon
            if current_polygon:
                polygon_data[current_polygon] = vertices
            current_polygon = line.split('"')[1].strip()  # Extract the polygon name
            vertices = []
        elif line.strip().startswith('V'):  # This is a vertex line
            try:
                vertex = line.split('=')[1].strip()
                vertex = tuple(map(float, vertex.strip('()').split(',')))
                vertices.append(vertex)
            except ValueError:
                pass  # Handle any lines that don't match the expected format
    if current_polygon:
        polygon_data[current_polygon] = vertices  # Add the last polygon

    # Debugging: Print the extracted polygon data
    # print("Extracted Polygon Data:")
    # print(polygon_data)
    
    # If polygon_data is empty, return an empty DataFrame
    if not polygon_data:
        # print("No polygons data extracted.")
        return pd.DataFrame()
    
    # Get the maximum number of vertices in any polygon
    max_vertices = max(len(vertices) for vertices in polygon_data.values())

    # Create a DataFrame to store the polygon data
    result = []
    for polygon_name, vertices in polygon_data.items():
        # Fill missing vertex data with blanks
        vertices = list(vertices) + [''] * (max_vertices - len(vertices))
        result.append([polygon_name] + vertices)
    
    # Create the DataFrame and assign column names
    polygon_df = pd.DataFrame(result)
    column_names = ['Polygon'] + [f'V{i+1}' for i in range(max_vertices)]
    polygon_df.columns = column_names

    # Add a new column 'Total Vertices' to count non-empty vertices
    polygon_df['Total Vertices'] = polygon_df.iloc[:, 1:].apply(lambda row: sum(1 for v in row if v != ''), axis=1)

    return polygon_df

def extract_floor_space_wall_data(lines):
    import pandas as pd

    # Initialize variables
    floor_data = []
    current_floor = None
    current_fh = None
    current_sh = None
    current_space = None
    current_polygon = None
    current_space_height = None
    walls_details = []
    inside_space_block = False  # Flag to track if inside a SPACE block

    # Helper function to append data
    def append_wall_data():
        for wall in walls_details:
            floor_data.append({
                'FLOOR': current_floor,
                'FLOOR-HEIGHT': current_fh,
                'SPACE-HEIGHT': current_sh,
                'SPACE': current_space,
                'SH2': current_space_height,
                'POLYGON': current_polygon,
                'EXTERIOR-WALL': wall.get('name'),
                'LOCATION': wall.get('location')
            })

    # Read input file
    # with open(inp_file, 'r') as file:
    #     lines = file.readlines()

    for line in lines:
        line = line.strip()

        # Start FLOOR block
        if line.startswith('"') and '= FLOOR' in line:
            if walls_details:
                append_wall_data()
            # Reset for new floor
            current_floor = line.split('=')[0].strip().strip('"')
            current_fh = None
            current_sh = None
            current_space = None
            current_polygon = None
            current_space_height = None
            walls_details = []
            inside_space_block = False  # Reset SPACE block flag

        elif "FLOOR-HEIGHT" in line:
            current_fh = float(line.split('=')[1].strip())

        elif "SPACE-HEIGHT" in line:
            current_sh = float(line.split('=')[1].strip())

        # Start SPACE block
        elif line.startswith('"') and '= SPACE' in line:
            if walls_details:
                append_wall_data()
            # Reset for new space
            current_space = line.split('=')[0].strip().strip('"')
            current_polygon = None
            current_space_height = None
            walls_details = []
            inside_space_block = True  # Set SPACE block flag

        elif inside_space_block:
            # Capture SPACE block details
            if "HEIGHT" in line:
                current_space_height = float(line.split('=')[1].strip())
            elif "POLYGON" in line:
                current_polygon = line.split('=')[1].strip().strip('"')
            elif line == "..":  # End of SPACE block
                inside_space_block = False

        # Start EXTERIOR-WALL block
        elif line.startswith('"') and '= EXTERIOR-WALL' in line:
            wall_name = line.split('=')[0].strip().strip('"')
            walls_details.append({'name': wall_name, 'location': None})

        elif "LOCATION" in line and walls_details:
            walls_details[-1]['location'] = line.split('=')[1].strip()

        # End of a block
        elif line == "..":
            if walls_details:
                append_wall_data()
            # Reset after appending
            walls_details = []

    # Append remaining data
    if walls_details:
        append_wall_data()

    # Convert to DataFrame
    df = pd.DataFrame(floor_data)

    # Debug: Display extracted data
    # print("Extracted data preview:")
    # print(df.head())

    # Remove rows with LOCATION as 'TOP' or 'BOTTOM'
    df = df[~df['LOCATION'].isin(['TOP', 'BOTTOM'])]

    return df

def calculate_corr(row):
    try:
        # Attempt to split and calculate
        diff = row["Diff"].split(" - ")
        if len(diff) != 2:
            raise ValueError(f"Invalid Diff format: {row['Diff']}")
        col1, col2 = diff[0], diff[1]
        val1, val2 = row[col1], row[col2]
        return tuple(a - b for a, b in zip(val1, val2))
    except Exception as e:
        # print(f"Error in row: {row.name}, Diff: {row['Diff']}, Error: {e}")
        return None  # Or a default value (e.g., (0, 0))

# Handle different data types in the Cordinate column
def calculate_distance(coord):
    if isinstance(coord, str) and coord.strip() != '':
        # If coord is a string and not blank, process it
        try:
            return np.sqrt(sum(float(num.strip())**2 for num in coord.strip('()').split(',')))
        except ValueError:
            return np.nan  # Handle invalid numeric values
    elif isinstance(coord, tuple):
        # If coord is a tuple, calculate the distance directly
        try:
            return np.sqrt(sum(float(num)**2 for num in coord))
        except ValueError:
            return np.nan
    else:
        # For other cases, return NaN
        return np.nan

def get_next_column(row):
    # Handle NaN or invalid cases
    if pd.isna(row["LOCATION"]) or not isinstance(row["LOCATION"], str):
        return np.nan
    
    if "V" in row["LOCATION"]:
        # Extract numeric part from 'SPACE-V1' or similar
        current_v = int(row["LOCATION"].split("-V")[1])
        total_vertices = row["Total Vertices"]
        
        if current_v < total_vertices:
            next_v = current_v + 1
            return f"SPACE-V{next_v}"
        elif current_v == total_vertices:
            return f"SPACE-V1"
    
    return ""

def create_ext_win(row):
    if isinstance(row, str):  # Process only strings
        # Replace 'Wall' with 'Win'
        new_value = row.replace('Wall', 'Win')
        # Append '.W1' after 'E<number>'
        new_value = re.sub(r'(E\d+)', r'\1.W1', new_value)
        return new_value
    return row  # Return the original value if it's not a string

def process_inp_file(inp_file):
    polygon_df = extract_polygons(inp_file)  # Polygon DataFrame
    df = extract_floor_space_wall_data(inp_file)  # Floor, Space, and Wall DataFrame
    df = pd.merge(df, polygon_df[['Polygon', 'Total Vertices']], left_on='POLYGON', right_on='Polygon', how='left')
    df.drop(columns=['Polygon'], inplace=True)
    return df

from helper import *
from report_ext import *


location_id = ''
####user Input 1. Name of user
user_nm = input("Provide name of user running the code: ")
user_nm = user_nm.replace(" ", "_")


####user Input 2. location
# Read the CSV file
weather_df = pd.read_csv('database/Simulation_locations.csv')

# Show available locations to the user
print("Available Locations:")
print(weather_df['Sim_location'].str.replace('IND_', '').tolist())
print()

# Get the location from the user, loop untill get correct location
while True:
    # Ask user to enter a location
    user_input = input("Enter the location name from above list: ").strip().lower()

    # Filter row where 'Sim_location' contains the user input
    matched_row = weather_df[weather_df['Sim_location'].str.lower().str.contains(user_input)]

    # Show result
    if not matched_row.empty:
        for index, row in matched_row.iterrows():
            weather_path = row['Weather_file']
            location_id = row['Location_ID']
            print(f"\nLocation ID: {row['Location_ID']}")
            print(f"Simulation Location: {row['Sim_location']}")
            print(f"Weather File: {row['Weather_file']}")
        break
    else:
        print("❌ Location not found. Please enter a valid option.")

####user Input 3. input folder path and output folder path
print("** Note: Paths should not contains any spaces **")
inp_folder = input("Enter the INP folder path: ") #"E:/09_TestCode/00_ml_batch_run/20250630_3/seedinp" #
output_inp_folder = input("Enter the output folder path: ") #"E:/09_TestCode/00_ml_batch_run/20250630_3/modifiedInp" #


# set Paths
db_path = 'database/AllData.xlsx'
output_csv = "Randomized_Sheet.csv"
run_cnt=int(input("Enter the number of runs you want to generate: ") ) #10 # set run number here


# Ensure output folder exists if not create output folder
os.makedirs(output_inp_folder, exist_ok=True)

# Step 1: Get all INP files from the input folder
inp_files = [f for f in os.listdir(inp_folder) if f.endswith(".inp")]

# Step 2: Read all sheets from the Excel file to load all data set
xlsx_data = pd.ExcelFile(db_path)
ignore_sheets = {"Wall_New","Roof_New","Material_DB", "Material_DB_IP"}  # sheets to skip

sheet_data = {
    sheet: pd.read_excel(db_path, sheet_name=sheet)
    for sheet in xlsx_data.sheet_names
    # if sheet not in ignore_sheets
}

# Step 3: Get row counts
sheet_row_counts = {sheet: len(df) for sheet, df in sheet_data.items()}

# # Step 4: Determine the new Batch_ID
if os.path.exists(output_csv):
    
    existing_df = pd.read_csv(output_csv)
    last_batch_id = existing_df["Batch_ID"].max() if "Batch_ID" in existing_df.columns else 0
else:
    last_batch_id = 0
new_batch_id = last_batch_id + 1  # Increment Batch_ID

# Step 5: Generate 600 random selections
batch_data = []
timestamp = datetime.now().strftime("%d-%m-%Y %H:%M")


for run_id in range(1, run_cnt):  # Generate 600 rows
    inp_file = random.choice(inp_files)  # Random INP file
    
    # Select unique row numbers from each sheet
    selected_rows = {}
    for sheet, row_count in sheet_row_counts.items():
        if row_count > 0:
            # selected_rows[sheet] = random.randint(1, row_count)  # Random row number (1-based index)
            selected_rows[sheet] = random.randint(0, row_count - 1)
        else:
            selected_rows[sheet] = None  # If the sheet is empty

    # Append data
    row_data = [new_batch_id, run_id, timestamp, inp_file] + [selected_rows[sheet] for sheet in sheet_row_counts.keys()] + [user_nm]
    batch_data.append(row_data)

# Step 6: Convert to DataFrame
columns = ["Batch_ID", "Run_ID", "Timestamp", "Selected_INP"] + list(sheet_row_counts.keys()) + ["RunningUser"]
batch_df = pd.DataFrame(batch_data, columns=columns)

# Step 7: Append to CSV
if os.path.exists(output_csv):
    batch_df.to_csv(output_csv, mode='a', header=False, index=False)
else:
    batch_df.to_csv(output_csv, index=False)

# batch_df = pd.read_csv('Randomized_Sheet.csv')
# print(batch_df)
# new_batch_id = 1 # delete this
# run_cnt = 2
#exit()

print(f"Processing complete! Batch_ID {new_batch_id} with {run_cnt} runs appended to:", output_csv)

# ----------------------------- MODIFY INP FILES BASED ON GENERATED DATA -----------------------------
# Step 8: Read the latest batch from CSV
updated_df = pd.read_csv(output_csv)
#print(updated_df)

# Step 2: Extract the prefix from Selected_INP (before first '_')
updated_df['prefix'] = updated_df['Selected_INP'].str.split('_').str[0]

# Step 3: Create 'CheckUnique' column
updated_df['CheckUnique'] = (
    updated_df['prefix'] + '_' +
    updated_df['Wall'].astype(str) + '_' +
    updated_df['Roof'].astype(str) + '_' +
    updated_df['Glazing'].astype(str) + '_' +
    updated_df['Orient'].astype(str) + '_' +
    updated_df['Light'].astype(str) + '_' +
    updated_df['WWR'].astype(str) + '_' +
    updated_df['Equip'].astype(str) +
    '.inp'
)

# Step 4: Drop duplicates based on CheckUnique
updated_df = updated_df.drop_duplicates(subset='CheckUnique')

# Step 5: Drop the helper column 'prefix' (optional)
updated_df = updated_df.drop(columns='prefix')
updated_df = updated_df.drop(columns='CheckUnique')

# Step 9: Select the last 600 rows
selected_rows =updated_df[updated_df['Batch_ID'] == new_batch_id] #updated_df.iloc[:10]
#print(selected_rows)

output_inp_folder = os.path.join(output_inp_folder, user_nm+"_Batch_"+str(new_batch_id))
os.makedirs(output_inp_folder, exist_ok=True)

num = 1
for index, row in selected_rows.iterrows():
    selected_inp = row["Selected_INP"]
    new_inp_name = f"{0}_{0}_{0}_{0}_{0}_{0}_{0}_{selected_inp}"
    new_inp_path = os.path.join(output_inp_folder, new_inp_name)

    inp_file_path = os.path.join(inp_folder, selected_inp)
    if not os.path.exists(inp_file_path):
        print(f"File {inp_file_path} not found. Skipping modification.")
        continue

    print(f"Modifying INP file{num}: {selected_inp} -> {new_inp_name}")
    num = num + 1
    inp_content = wwr.process_window_insertion_workflow(inp_file_path, 4 + 1)
    print("Modified WWR")
    inp_content = orient.updateOrientation(inp_content, 2)
    print("Modified Orientation")
    inp_content = lighting.updateLPD(inp_content, 0)
    print("Modified Light")
    inp_content = insertWall.update_Material_Layers_Construction(inp_content, 0)
    print("Modified Wall")
    inp_content = insertRoof.update_Material_Layers_Construction(inp_content, 0)
    inp_content = insertRoof.removeDuplicates(inp_content)
    print("Modified Roof")
    inp_content = equip.updateEquipment(inp_content, 0)
    print("Modified Equipment")
    inp_content = windows.insert_glass_types_multiple_outputs(inp_content, 0)
    print("Modified Glazing\n")
    
    with open(new_inp_path, 'w') as file:
        file.writelines(inp_content)

    print(f"Successfully modified and saved: {new_inp_name}\n")

print(f"Batch {new_batch_id} generated, and INP files modified successfully!\n\n")

####### copy script.bat file to output folder and running simulations
script_dir = os.path.dirname(os.path.abspath(__file__))  # current .py file directory

source_file = os.path.join(script_dir, "script.bat")
destination_folder = output_inp_folder
shutil.copy(source_file, destination_folder)

bat_file_path = os.path.join(destination_folder, "script.bat")
subprocess.call([bat_file_path, output_inp_folder, weather_path], shell=True)

#### Extracting result

## checking for missing section in sim file and log it
required_sections = ['BEPS', 'BEPU', 'LS-C', 'LV-B', 'LV-D', 'PS-E', 'SV-A']
log_file_path = check_missing_sections(output_inp_folder, required_sections, new_batch_id, user_nm)

# Step 12: Clean up - Delete all files except .inp and .sim
get_failed_simulation_data(output_inp_folder, log_file_path)
clean_folder(output_inp_folder)


######################################################################################
# Step = Organize matching inp and sim to folders
#organize_file_2_folder(output_inp_folder)


get_files_for_data_extraction(output_inp_folder,log_file_path,new_batch_id, location_id,user_nm)
exit()

# 
""" next_script = "file.py"
try:
    print(f"\nExecuting: {next_script}")
    result = subprocess.run(['python', next_script], check=True, capture_output=True, text=True)
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Script {next_script} failed with exit code {e.returncode}")
    print("Error Output:\n", e.stderr)
except Exception as e:
    print(f"Failed to run script {next_script}: {e}")
    traceback.print_exc()
finally:
    print("Moving to next step...") """