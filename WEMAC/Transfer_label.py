"""
                                                    ⠀⠀⠀⠀⠀⠀⠀⢠⣤⣀⠀⠀⠀⠀⢀⣀⣤⣤⠀⠀⠀⠀⠀⠀⠀
                                                    ⠀⠀⢀⢀⠀⠀⠀⢸⡿⠛⠛⠛⠛⠛⠉⠛⢿⣿⠀⠀⠀⠀⠀⠀⠀
 _   _       _     _____         _             		⠀⠠⣿⣿⣿⣄⠀⣼⠀⠀⠉⣍⣀⣀⡍⠁⠀⢹⡀⠀⠀⠀⠀⠀⠀
| \ | | ___ | |_  |_   _|__   __| | __ _ _   _ 		⠀⢸⣿⣿⣿⣿⡷⠋⠈⠀⠀⠀⠀⠀⠀⠀⠈⠘⠣⡀⠀⠀⠀⠀⠀
|  \| |/ _ \| __|   | |/ _ \ / _` |/ _` | | | |		⠀⠈⣿⣿⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣷⣦⡀⠀⠀
| |\  | (_) | |_    | | (_) | (_| | (_| | |_| |		⠀⠀⢹⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣿⣿⣦⠀
|_| \_|\___/ \__|   |_|\___/ \__,_|\__,_|\__, |		⠀⠀⣸⣿⣿⣶⣶⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇
                                         |___/ 		⠀⣤⡟⠛⠋⠉⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠉⠈⠋⠈⢿⣿⡿
                                                    ⢀⡉⠀⠀⣀⣤⣄⢈⣿⣿⣿⣿⣿⣿⣿⣿⣿⢀⣤⣤⣄⠀⠀⣴⡄
                                                    ⠘⢇⠀⠰⣿⣿⢟⢼⣿⣿⣿⣿⣿⣿⣿⣿⡿⢜⠿⠿⠿⠀⡀⠀⠀
@Author: Junjiao Sun                               
@Time : Created in 4:39 PM 2024/02/20   
@FileName: Transfer_label.py                           
@Software: PyCharm


Introduction of this File:
Because the former training process are based on IT07. Here we need to transfer the labels of IT06 to fit the project.
"""

import pandas as pd
import csv
import csv
import xlwt
import numpy as np
# Specify the CSV file path
file_path = 'Labels_BBDDLab_IT06.csv'

# Create an empty list to store the data
data_list = []

# Read the CSV file and store data in the list
with open(file_path, 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        data_list.append(row)

# Display the data list
Voluntaria = []
Video = []
Tanda = []
ReportadaBinarizado = []
TargetBinarizado = []
PADBinarizado = []
Arousal = []
Valencia = []
Dominancia = []
for row in data_list:
    # print(row)
    row_split = row[0].split(';')
    if row_split[0] != 'Voluntaria':
        Voluntaria.append(int(row_split[0].replace('V', '')))
        Video.append(int((row_split[1].replace('VIDEO ', '')).replace('"', '')))
        Tanda.append(int((row_split[2].replace('T', '')).replace('"', '')))
        ReportadaBinarizado.append(int((row_split[3]).replace('"', '')))
        TargetBinarizado.append(int((row_split[4]).replace('"', '')))
        PADBinarizado.append(int((row_split[5]).replace('"', '')))
        Arousal.append(int(row_split[6]))
        Valencia.append(int(row_split[7]))
        Dominancia.append(int(row_split[8]))

arrayVoluntaria = np.array(Voluntaria)
arrayVideo = np.array(Video)
arrayTanda = np.array(Tanda)
arrayReportadaBinarizado = np.array(ReportadaBinarizado)
arrayTargetBinarizado = np.array(TargetBinarizado)
arrayPADBinarizado = np.array(PADBinarizado)
arrayArousal = np.array(Arousal)
arrayValencia = np.array(Valencia)
arrayDominancia = np.array(Dominancia)

array_label = np.vstack((arrayVoluntaria, arrayVideo))
array_label = np.vstack((array_label, arrayTanda))
array_label = np.vstack((array_label, arrayReportadaBinarizado))
array_label = np.vstack((array_label, arrayTargetBinarizado))
array_label = np.vstack((array_label, arrayPADBinarizado))
array_label = np.vstack((array_label, arrayArousal))
array_label = np.vstack((array_label, arrayValencia))
array_label = np.vstack((array_label, arrayDominancia))

array_label = np.transpose(array_label)
array_label = array_label.astype('int')

df = pd.DataFrame(array_label)

excel_file_path = 'Labels_TabLab_IT06.xls'

df.to_excel(excel_file_path, index=False, header=False)


# # Create a new workbook and add a worksheet
# workbook = xlwt.Workbook()
# worksheet = workbook.add_sheet('Sheet1')
#
# # Write the NumPy array to the worksheet
# for row_index, row_data in enumerate(array_label):
#     for col_index, value in enumerate(row_data):
#         worksheet.write(row_index, col_index, value)
#
# # Save the workbook to a file
# workbook.save('Labels_TabLab_IT06.xls')



