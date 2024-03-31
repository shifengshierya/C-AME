import os
import pandas as pd
from matplotlib import pyplot as plt
from Part_2_functions_for_eachstep import create_folder
from data_writes import data_write,savecsvs,readcsv,savecsv

def file_names(inputpath):
    """
       This function is for looping over files

       Args:
           inputpath: The folder path to be looped over

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

#Index calculation of speed
def calculate_speed(path):
    """
       This function is for calculating the speed

       Args:
           path: The file path

       Returns:
           list_speed: The speed results
            
    """
    df = pd.read_csv(path, header=None)
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
           path: The file path

       Returns:
           True: Omitted
           
    """
    path1 = get_file_list(path)
    list_averange = []
    index = 1
    xx = [i for i in range(43101,43465)]
    for i in path1:
        name1 = os.path.basename(i)
        if name1[0:2] == 'fi':
            #print(name1)
            result = calculate_speed(i)
            result1 = [[datas] for datas in result]
            # result2 = []
            # for k in range(len(result)):
            #     result2.append([43101 + k,result[k]])
            savecsvs(os.path.join(path, 'speed', 's{}.csv'.format(str(index))),result1)
            list_averange.append(result)

            #plt.xlim(0,600)
            plt.plot(xx,result,label='s{}'.format(str(index)))
            #plt.plot(xx, result)
            plt.xlabel("date")
            plt.ylabel("speed(meter per day)")
            #plt.title('%s'%i)
            index = index + 1

    result2 = []
    for m in range(len(list_averange[0])):
        sum = 0
        for n in range(len(list_averange)):
            sum = sum + float(list_averange[n][m])
        #savecsv(os.path.join(path, 'speed', 'sa.csv'),[sum/len(list_averange)])
        result2.append(sum/len(list_averange))



    plt.legend()
    plt.savefig(os.path.join(path, 'speed', 'SpeedPerDay.png'))
    plt.show()


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
                if fileName[0:2] == 'fi':
                    path = folder_attribute[0]
                    create_folder(os.path.join(path, 'speed'))
                    create_folder(os.path.join(path, 'off_distance'))

                    # --------------------------------------------
                    speed_main(path)

                    # q4('result')
                    print('{}Result found'.format(path))
                    break

if __name__ == '__main__':
    data_path = r'D:\python\20240314\came\ResultFiles\Anthus_spragueii1'
    speed(data_path)