# -*- coding: utf-8 -*-
import csv
import time
import os
from datetime import datetime
import pandas as pd
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from Part_2_main_paths_estiamtion import part2


# beginDate, endDate are datetime format like"20180101"
def datelist(beginDate, endDate):
    """
        This function is for getting the datelist

        Args:
            beginDate: The beginning date in the datelist
            endDate:The end date in the datelist

        Returns:
            date_l:The datelist result
    """
    date_l=[datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_l

def readcsv(path):
    """
       This function is for reading csv files

       Args:
           path: The file  path

       Returns:
           rows:The file data
    """
    try:
        with open(path, 'r',encoding='utf_8_sig') as f:
        #with open(path, 'r',encoding='gb18030') as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        return rows
    except:
        #with open(path, 'r',encoding='utf_8_sig') as f:
        with open(path, 'r',encoding='gb18030') as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        return rows

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
            print('Close')
            time.sleep(1)

#Read the names of all files and folders within the directory
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

#Show the raw data
def map_2(save_path,csv_name):
    """
       This function is for showing the raw data on the map

       Args:
           save_path : Path for storing the raw data figures
           csv_name: Name of the species to be processed

       Returns:
           True:Omitted
    """
    m = Basemap(llcrnrlat=-60, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=-20)  # Instantiate a map
    m.drawcoastlines()  # Draw the coastline
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(lake_color='white')  # Draw the continents and fill them in white

    parallels = np.arange(-90., 90., 10.)  # Draw latitudes with ranges [-90,90] and intervals of 10
    m.drawparallels(parallels, labels=[False, True, True, False], color='none')
    meridians = np.arange(-180., 180., 20.)  # Draw the longitude with a range of [-180,180] and an interval of 10
    m.drawmeridians(meridians, labels=[True, False, False, True], color='none')

    # plt.rcParams['figure.figsize'] = (28, 8)
    # plt.show()

    datalist = readcsv(save_path + csv_name)
    datalist = datalist[1:]

    LON = []
    LAT = []
    for i in range(1):
        Lat = [[float(data[0]),int(data[2])] for data in datalist]
        Lon = [[float(data[1]),int(data[2])] for data in datalist]
        LON.append(Lon)
        LAT.append(Lat)

    for doc in range(1):
        colorMap = ['red', 'darkorange', 'gold', 'greenyellow', 'pink', 'limegreen', 'mediumturquoise',
                    'dodgerblue',
                    'navy', 'blue', 'mediumorchid', 'fuchsia']
        # Show labels
        label = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                 'November', 'December']

        marker = ['x', 'o', '.', '+', '<', '_', '^', 'v', 'H', '|', 's', '*']
        j = 0
        # print(len(lon))
        flag = True

        col1 = 43101
        for i in range(13):
            # print(i)
            if doc == 0:
                # m.plot(LON[doc][i:i + 30], LAT[doc][i:i + 30], marker=marker[doc], linewidth=0.4,
                #        color=colorMap[j],
                #        markersize=0.5, label=label[
                #         j])
                LON1 = [data[0] for data in LON[doc] if data[1]>=col1 and data[1]<col1 + 30]
                LAT1 = [data[0] for data in LAT[doc] if data[1]>=col1 and data[1]<col1 + 30]
                col1 = col1 + 30
                m.scatter(LON1, LAT1, color=colorMap[j],s=1, label=label[j])
                # plt.show()
                j += 1
                if j == 12:
                    j = 0
                    if flag:
                        plt.legend(loc='lower left', shadow=True)
                        flag = False
                    continue

    plt.xlabel('Lon', labelpad=10)
    plt.ylabel('Lat')
    plt.savefig(save_path + '{}.jpg'.format(csv_name.replace('.csv', '')), dpi=1000)
    # plt.show()
    plt.close()

def main(path1='./RawData/', path2='./source/', save_path='./target/', obs_count=3,lat_col=8,obs_date=11):
    """
       This function is for datetime format standardization

       Args:
           path1:The directory of input data mentioned above (support for multiple files processing)
           path2:Data format standardization results folder
           save_path: Final results storage folder
           obs_count:The column number for observation count
           lat_col:The column number for latitude
           obs_date:The column number for observation date

       Returns:
           True:Omitted
     """
    #Absolute path
    # path1 = './2018/'
    # path2 = './source/'
    #Terminal
    # path1 = input('Please set part1 input folder ')
    # path1 = path1 + '/'
    # path2 = input('Please set part1 output folder')
    # path2 = path2 + '/'

    year = '2018'
    date_list = datelist(f"{year}-01-01", f"{year}-12-31")

#Datetime format standardization: 20180101 corresponds to 43101
    date1 = {}
    num = 43101
    for date in date_list:
        date1[date] = num
        num = num + 1

#Extract three columns :LONGITUDE,LATITUDE, and OBS_DATE and name them as:LATITUDE', 'LONGITUDE', 'OBSERVATION DATE'
    csv_list = file_names(path1)[0][2]
    csv_list = [name for name in csv_list if '.csv' in name]
    for i in range(len(csv_list)):
        csv_path = path1 + csv_list[i]
        datalist = readcsv(csv_path)
        datalist = [[data[obs_count]] + data[lat_col:obs_date] for data in datalist]
        datalist = datalist[1:]

#Perform quantity correction according to observation count
        data2 = []
        for datas in datalist:
            ctime = '2018' + datas[3][4:]
            ctime = date1[ctime]
            datas[3] = ctime
            if datas[0] == 'X':
                num = 1
            else:
                num = int(datas[0])
            for j in range(num):
                data2.append(datas)
        data2 = [data[1:] for data in data2]
#Sort by time
        sorted_data = sorted(data2, key=lambda x: x[2])
        sorted_data = [['LATITUDE', 'LONGITUDE', 'OBSERVATION DATE']] + sorted_data


        for data in sorted_data:
            os.makedirs(path2, exist_ok=True)
            savecsv(path2 + csv_list[i],data)
        map_2(path2, csv_list[i])
        print(csv_list[i])
    print('Part 1 finished')
    part2(path2, save_path)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='./RawData/')
    parser.add_argument("--data_dir", type=str, default='./source/')
    parser.add_argument("--save_dir", type=str, default='./target/')
    opt = parser.parse_args()
    main(path1=opt.input_dir, path2=opt.data_dir, save_path=opt.save_dir)
