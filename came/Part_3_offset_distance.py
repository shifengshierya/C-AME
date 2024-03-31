import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from Part_2_functions_for_eachstep import create_folder
from data_writes import data_write,savecsvs,readcsv

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
        if name1[:2] == 'fi':
            tmp = pd.read_csv(l, header=None)
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
        if name1[0:2] == 'fi':
            tmp = pd.read_csv(r, header=None)
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
        if name1[0:2] == 'fi':
            tmp = pd.read_csv(r, header=None)
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
    xx = [i for i in range(43101, 43465)]
    for l in result:
        name = os.path.basename(path1[count])
        #plt.xlim(0,360)
        plt.plot(xx,l,label='d{}'.format(count))
        #plt.plot(xx, l)
        plt.xlabel("date")
        plt.ylabel("offset distance(meter)")
        #plt.title('%s'%l)
        count += 1
        #plt.title('%s'%name)


    plt.legend()
    plt.savefig(os.path.join(path, 'off_distance', 'OffsetDistancePerDay.png'))
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
                if fileName[0:2] == 'fi':
                    path = folder_attribute[0]
                    create_folder(os.path.join(path, 'speed'))
                    create_folder(os.path.join(path, 'off_distance'))

                    # --------------------------------------------
                    off_distance_main(path)

                    # q4('result')
                    print('{}Result found'.format(path))
                    break

if __name__ == '__main__':
    data_path = r'D:\python\20240314\came\ResultFiles\Anthus_spragueii1'
    off_distance(data_path)