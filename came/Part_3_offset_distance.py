import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import csv
import time
from Part_2_functions_for_eachstep import create_folder

def file_names(inputpath):
    """
       This function is for looping over files

       Args:
           inputpath: The folder path  to be looped over

       Returns:
           namelist: The files list
           
    """
    namelist = []
    filePath = inputpath
    for i, j, k in os.walk(filePath):
        namelist.append([i, j, k])
    return namelist

def savecsvs(path,item,model = 'a'):
    """
        This function is for saving csv files

       Args:
           path: The  path for saving csv files
           item: The data to be saved
           model: The default parameter

       Returns:
           True: Omitted
           
    """
    while True:
        # try:
        with open(path, model, encoding='utf_8_sig', newline='') as f:
        #with open(path, model, encoding='gb18030', newline='') as f:
            w = csv.writer(f)
            w.writerows(item)
            return True

def savecsv(path,item,model = 'a'):
    """
       This function is for saving csv files

       Args:
           path: The  path for saving csv files
           item: The data to be saved
           model: The default parameter

       Returns:
           True: Omitted
           
    """
    while True:
        try:
            with open(path, model, encoding='utf_8_sig', newline='') as f:
            #with open(path, model, encoding='gb18030', newline='') as f:
                w = csv.writer(f)
                w.writerow(item)
                return True
        except:
            print('Close the table or the program cannot write')
            time.sleep(1)

def get_file_list(path):
    """
       This function is for getting file lists

       Args:
           path: The  path for files

       Returns:
            file_list: The file list
            
    """

    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file != '.DS_Store':
                file_list.append(os.path.join(root, file))
    return file_list


def get_distance_point2line(point, line):
    """
       This function is for calculating the distance from points to a line

       Args:
           point: The observation points
           line: A line

       Returns:
           distance: The distance from points to the line
           
    """
    line_point1, line_point2 = np.array(line[0:2]), np.array(line[2:])
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
    return distance

#Index calculation of offset distance
def calculate_distance(path_list):
    """
       This function is for getting migration axis and  offset distance

      Args:
         path_list: The files in the path

      Returns:
          list11: The offset distances
          
    """
    list_distance = []
    for l in path_list:
        name1 = os.path.basename(l)
        if name1[0] == 'r':
            tmp = pd.read_excel(l, header=None)
            list_distance.append(tmp)

    mean_x = []
    mean_y = []
    for q in list_distance:
        mean_x.append(float(q.iloc[1, 0]))
        mean_y.append(float(q.iloc[1, 1]))

    mean_x1 = np.mean(mean_x)
    mean_y1 = np.mean(mean_y)
    print(mean_x1, mean_y1)


    dict1 = {}
    list1 = []
    max_list_x = []
    max_list_y = []
    for r in path_list:
        name1 = os.path.basename(r)
        if name1[0:2] == 're':
            tmp = pd.read_excel(r, header=None)
            for l in range(1,365):
                x1 = float(tmp.iloc[1, 0])
                y1 = float(tmp.iloc[1, 1])

                x = float(tmp.iloc[l, 0])
                y = float(tmp.iloc[l, 1])
                distance1 = ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
                dict1[distance1] = l
                list1.append(distance1)
            max_distance = max(list1)
            max_x_1 = float(tmp.iloc[dict1[max_distance], 0])
            max_y_1 = float(tmp.iloc[dict1[max_distance], 1])
            max_list_x.append(max_x_1)
            max_list_y.append(max_y_1)
            list1 = []
            dict1 = {}

    mean_x2 = np.mean(max_list_x)
    mean_y2 = np.mean(max_list_y)
    print(mean_x2, mean_y2)


    line = [mean_x1, mean_y1, mean_x2, mean_y2]

    list11 = []
    for r in path_list:
        list_tmp = []
        name1 = os.path.basename(r)
        if name1[0:2] == 're':
            tmp = pd.read_excel(r, header=None)
            for p in range(1,365):
                x = float(tmp.iloc[p, 0])
                y = float(tmp.iloc[p, 1])
                distance2 = get_distance_point2line((x, y), line)
                list_tmp.append(distance2)
            list11.append(list_tmp)
    return list11

def off_distance_main(path):
    """
       This function is for saving and drawing the results

       Args:
           path: The file path

       Returns:
           True: Omitted
           
    """
    path1 = get_file_list(path)
    result = calculate_distance(path1)
    count1 = 0
    tmp = 0
    index = 1
    for j in result:
        list_averange = []
        result2 = []
        items = [[datas] for datas in j]
        savecsvs(os.path.join(path, 'off_distance', 'd{}.csv'.format(str(index))),items)
        index = index + 1
        for b in range(len(j)):
            for l in result:
                tmp += l[b]
            averange = int(tmp/len(result))
            tmp = 0
            result2.append(averange)
    items = [[datas] for datas in result2]
    #savecsvs(os.path.join(path, 'off_distance', 'da.csv'), items)

    count = 1
    for l in result:
        name = os.path.basename(path1[count])
        #plt.xlim(0,360)
        plt.plot(l,label='d{}'.format(count))
        plt.xlabel("date")
        plt.ylabel("distance")
        #plt.title('%s'%l)
        count += 1
        #plt.title('%s'%name)

    #plt.legend()
    #plt.savefig('q2/OffsetDistancePreDay.png')
    #plt.show()

    #plt.plot(result2,label='aveVar')
    #plt.xlim(0,420)
    #plt.ylim()
    plt.xlabel("date")
    plt.ylabel("distance")
    #plt.title('averangeVar')
    plt.legend()
    plt.savefig(os.path.join(path, 'off_distance', 'OffsetDistancePerDay.png'))
    plt.show()


    #plt.savefig(path + '\\variance\\VarPreDay.png')
    plt.show()

def off_distance(data_path):
    """
        This function is for saving the results in the folder

        Args:
            data_path: The data path

        Returns:
            True: Omitted
            
    """
    # --------------------------------------------
    excel_list = file_names(data_path)

    for i in range(len(excel_list)):
        folder_attribute = excel_list[i]  #
        index = 1

        if len(folder_attribute[2]) > 0:
            for fileName in folder_attribute[2]:
                if fileName[0:2] == 're':
                    path = folder_attribute[0]
                    create_folder(os.path.join(path, 'speed'))
                    create_folder(os.path.join(path, 'off_distance'))

                    # --------------------------------------------
                    off_distance_main(path)

                    # q4('result')
                    print('{}Result found'.format(path))
                    break

