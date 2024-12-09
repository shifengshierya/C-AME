import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import csv
import time
import os
from data_writes import savecsv,savecsvs,readcsv

def dist_eclud(data1, data2):
    """
       This function is for calculating the Euclidean distance

       Args:
           data1: One data
           data2: The other data

       Returns:
           data: The Euclidean distance
           
    """
    #print(data2[0], data1[0], type(data2[0]), type(data1[0]))
    x = data2[0] - data1[0]
    y = data2[1] - data1[1]
    data = math.sqrt(x*x + y*y)
    return data

def get_date_list(datelist):
    """
       This function is for getting the date list

       Args:
           datelist: The date list

       Returns:
           date1: The date list
           
    """
    date1 = []
    for dates in datelist:
        if dates[2] not in date1:
            date1.append(dates[2])
    return date1

def get_same_date_list(date,excel_list):
    """
       This function is for getting the same dates

       Args:
           date: The date
           excel_list: The data to be processed

       Returns:
           same_datas: The same dates
           
    """
    same_datas = []
    for datas in excel_list:
        if datas[2] == date:
            same_datas.append(datas)
    return same_datas

#Avarage distance 
def get_avg(datalist):
    """
       This function is for obtaining the average distance

       Args:
           datalist: The data to be processed

       Returns:
           avg: The average distance
            
    """
    data1 = [datalist[0][0],datalist[0][1]]
    avg_datalist = []
    for i in range(1,len(datalist)):
        datas = [datalist[i][0],datalist[i][1]]
        avg_datalist.append(dist_eclud(data1,datas))
    if avg_datalist:
        avg = sum(avg_datalist)/(len(datalist) - 1)
    else:
        avg = 0
    return avg
    
#Calculate the average Euclidean distance for daily centroids
def avg_distance(save_path):
    """
       This function is for calculating the average distance for daily centroids

       Args:
           save_path: The path of files

       Returns:
           True: Omitted
           
    """
    df=pd.read_csv(os.path.join(save_path, 'centroids.csv'))
    train_data = np.array(df)
    excel_list = train_data.tolist()  # list
    # excel_list = excel_list[1:]

    datelist = get_date_list(excel_list)

    avg_list = []
    for date in datelist:
        same_date = get_same_date_list(date,excel_list)
        avg_date = get_avg(same_date)
        avg_list.append(avg_date)
        savecsv(os.path.join(save_path, 'avg.csv'),[date,avg_date])

    plt.plot(datelist, avg_list, label="avg_distance")
    plt.xlabel("date")
    plt.ylabel("avg_distance(meter)")
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    plt.legend(loc="best")
    plt.gca().ticklabel_format(axis="x", useOffset=False)
    plt.savefig(os.path.join(save_path, 'avg.png'))
    plt.show()

if __name__ == '__main__':
    save_path = ''
    avg_distance(save_path)
