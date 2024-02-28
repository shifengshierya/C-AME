import os
import pandas as pd
from matplotlib import pyplot as plt
import csv
import time
from Part_2_functions_for_eachstep import create_folder

def file_names(inputpath):
    """
       This function is for looping over files

       Args:
           inputpath: The folder path  to be looped over

       Returns:
           namelist:The files list
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
           item:The data to be saved
           model:The default parameter

        Returns:
           True:Omitted
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
            item:The data to be saved
            model:The default parameter

        Returns:
            True:Omitted
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
            file_list:The file list
    """
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file != '.DS_Store':
                file_list.append(os.path.join(root, file))
    return file_list

#Index calculation of speed
def calculate_speed(path):
    """
        This function is for calculating the speed

        Args:
            path : The file path

        Returns:
            list_speed: The speed results
    """
    df = pd.read_excel(path, header=None)
    list_speed = [0]
    for i in range(1,364):
        # x1 = float(df.iloc[i + 1, 0])
        x = float(df.iloc[i + 1, 0]) - float(df.iloc[i, 0])
        y = float(df.iloc[i + 1, 1]) - float(df.iloc[i, 1])
        distance = (x ** 2 + y ** 2) ** 0.5
        list_speed.append(distance)
    return list_speed

def speed_main(path):
    """
       This function is for saving and drawing the results

       Args:
           path : The file path

       Returns:
           True: Omitted
    """
    path1 = get_file_list(path)
    list_averange = []
    index = 1
    for i in path1:
        name1 = os.path.basename(i)
        if name1[0:2] == 're':
            #print(name1)
            result = calculate_speed(i)
            result1 = [[datas] for datas in result ]
            savecsvs(os.path.join(path, 'speed', 's{}.csv'.format(str(index))),result1)
            list_averange.append(result)

            #plt.xlim(0,600)
            plt.plot(result,label='s{}'.format(str(index)))
            plt.xlabel("date")
            plt.ylabel("speed")
            #plt.title('%s'%i)
            index = index + 1

    result2 = []
    for m in range(len(list_averange[0])):
        sum = 0
        for n in range(len(list_averange)):
            sum = sum + float(list_averange[n][m])
        #savecsv(os.path.join(path, 'speed', 'sa.csv'),[sum/len(list_averange)])
        result2.append(sum/len(list_averange))


    #plt.legend()
    #plt.savefig('q1/SpeedPreDay.png')
    #plt.show()
    #plt.close()


    #plt.xlim(0,500)
    plt.ylim(0,110000)
    #plt.plot(result2,label='averange')
    #plt.xlabel("date")
    #plt.ylabel("speed")
    #plt.title('averangeSpeed')

    plt.legend()
    plt.savefig(os.path.join(path, 'speed', 'SpeedPerDay.png'))
    plt.show()
    plt.close()

def speed(data_path):
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
                    speed_main(path)

                    # q4('result')
                    print('{}Result found'.format(path))
                    break

