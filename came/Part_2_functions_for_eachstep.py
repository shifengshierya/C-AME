import numpy as np
import xlwt
import math
import pandas as pd
from pygam import LinearGAM
import os
from tqdm import tqdm
from pyproj import Transformer
from pyproj import CRS
from sklearn.cluster import MeanShift
from itertools import cycle
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap



# WGS84 geographic coordinate system and Web Mercator coordinate system
crs_WGS84 = CRS.from_epsg(4326)
crs_WebMercator = CRS.from_epsg(3857)
cell_size = 0.009330691929342804
origin_level = 24
EarthRadius = 6378137.0
tile_size = 256

def create_folder(inputpath):
    """
       This function is for creating folders

       Args:
           inputpath: The folder path

        Returns:
           True:Omitted
    """
    if not os.path.exists(inputpath):
        os.makedirs(inputpath)

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

# Get the file names in the source folder
def get_all_csv(data_path):
    """
       This function is for getting all csv files

       Args:
           data_path: The folder path of storing the csv files

       Returns:
           all_csv:All the csv files
    """

    excel_list = file_names(data_path)
    all_csv = []
    for i in range(len(excel_list)):
        folder_attribute = excel_list[i]
        if len(folder_attribute[2])>0:
            for fileName in folder_attribute[2]:
                if fileName[-1] == 'v':
                    all_csv.append(fileName)
    return all_csv

#sldf() for outlier detection
def sldf(x):
    """
       This function is for outlier detection based on sldf values

       Args:
           x : The input data

       Returns:
           result:The result after sldf outlier detection
    """
    n = len(x)
    column = len(x[0])
    x_max = np.max(x)
    x_min = np.min(x)
    x_ = (x - x_min) / (x_max - x_min)
    k = 50
    lens = 1 / k
    position_x = np.ceil(x_ / lens)

    for i in range(len(position_x)):
        for j in range(len(position_x[0])):
            if position_x[i][j] == 0:
                position_x[i][j] = 1

    B = np.lexsort([position_x[:, 1], position_x[:, 0]])
    A = position_x[B, :]
    A = A.astype(int)
    count = np.zeros((k, k))
    for i in range(n):
        count[A[i][0] - 1][A[i][1] - 1] += 1
    max_count = np.max(count)

    q = 2
    q = q * max_count
    w = [0.5, 0.5]
    dist = np.zeros((n, n))
    for i in range(n):
        dist[:, i] = w[0] * ((x_[:, 0] - x_[i, 0]) ** 2) + w[1] * ((x_[:, 1] - x_[i, 1]) ** 2)
    dist = np.sqrt(dist)
    max_dist = np.max(dist)
    k = max_dist
    N = []
    for i in range(len(dist)):
        for j in range(len(dist[0])):
            Ni, Nj = j, i
            N.append((Ni, Nj))

    N = np.array(N)
    u = np.zeros(n)
    SLDR = np.zeros(n)
    N_i = N[:, 0]
    N_j = N[:, 1]

    for i in range(n):
        tmp = np.argwhere(N_j == i)
        tmp_E = int(max(tmp))
        tmp_S = int(min(tmp))
        tmp_N = N[tmp_S: tmp_E + 1, :]
        tmp_D = []
        for j in range(len(tmp_N)):
            a, b = tmp_N[j]
            tmp_D.append(dist[a, b])
        tmp_ji = tmp_E - tmp_S + 1
        u[i] = sum(tmp_D) / tmp_ji
        tmp_c = (tmp_D - u[i]) ** 2
        SLDR[i] = sum(tmp_c) / tmp_ji

    SLDIR = np.zeros(n)
    for i in range(n):
        tmp = np.argwhere(N_j == i)
        tmp_E = int(max(tmp))
        tmp_S = int(min(tmp))
        tmp = SLDR[N_i[tmp_S: tmp_E + 1]]
        SLDIR[i] = sum(tmp) / tmp_ji

    SLDF = SLDR / SLDIR
    # print(SLDF.shape, x.shape)
    selected_index = np.argsort(SLDF)[:int(0.8 * len(SLDF) + 1)]
    # print(selected_index.shape)
    SLDF_new = SLDF[selected_index]
    result = np.concatenate((x[selected_index], SLDF_new[:, np.newaxis]), axis=1)
    return result

def data_write(file_path, datas):
    """
       This function is for saving files

       Args:
           file_path : The path for saving files
           datas:The data to be saved

       Returns:
           True:Omitted
    """
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # Create a sheet
    i = 0
    for data in datas:
        for j in range(len(data)):
            sheet1.write(i, j, str(data[j]))
        i = i + 1
    f.save(file_path)


#Coordinate conversion from Web Mercator to wgs84
def webmercator2wgs84(a, b):
    """
       This function is for Web Mercator coordinates conversion into  geographical ones

       Args:
           a: The Web Mercator coordinates of longitude
           b: The Web Mercator coordinates of latitude

       Returns:
           lon: The geographical coordinates of longitude
           lat:The geographical coordinates of latitude
    """
    lon = a / 20037508.34 * 180
    lat = b / 20037508.34 * 180
    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180)) - math.pi / 2)
    return lon, lat

# def gam_pic(gam,data,save_path,key):
def gam_pic(gam,save_path,csv_name,key,x_y):
    """
       This function is for drawing the gam figures

       Args:
           gam: The usage of gam model
           save_path: The path for saving the result file
           csv_name:The species name being processed
           key:The number for file naming
           x_y:The choice for Lat or Lon when file naming

       Returns:
           True:Omitted
    """
    XX = gam.generate_X_grid(term=0, n=365)
    # plt.scatter([data for data in range(364)], data, color='b', marker='o')
    plt.plot(XX[:, 0], gam.partial_dependence(term=0, X=XX))
    plt.plot(XX[:, 0], gam.partial_dependence(term=0, X=XX, width=.95)[1], c='r', ls='--')
    plt.xlabel('Date', labelpad=10)
    plt.ylabel(x_y)

    # plt.show()
    plt.savefig(os.path.join(save_path, csv_name.replace('.csv', ''), 'gam{}{}.jpg'.format(x_y,key + 1)))
    plt.close()

#gam() for fitting longitude and latitude in Web Mercator system with time; Coordinate conversion from WebMercator to wgs84 with webmercator2wgs84() 
def gam(save_path, csv_name,key):
    """
       This function is for gam algorithm

       Args:
           save_path: The path for saving the result file
           csv_name:The species name being processed
           key:The number for file naming

       Returns:
           Lon:The longitude after gam algorithm
           Lat:The latitude after gam algorithm

    """
    df = pd.read_excel(os.path.join(save_path, csv_name.replace('.csv',''), 'ni_traj{}.xlsx'.format(key + 1)), sheet_name='Sheet1')
    date = df["date"]
    x = df["X"]
    y = df["Y"]
    gam_model = LinearGAM().fit(date, x)
    gam_pic(gam_model,save_path,csv_name,key,'Lat')
    predictions_x = gam_model.predict(date)
    gam_model = LinearGAM().fit(date, y)
    gam_pic(gam_model, save_path,csv_name, key, 'Lon')
    predictions_y = gam_model.predict(date)
    datas = [["X*", "Y*","date"]]
    # datas = []
    for i in range(len(x)):
        datas.append([predictions_x[i], predictions_y[i],date[i]])
    data_write(os.path.join(save_path, csv_name.replace('.csv',''), 'result_{}.xls'.format(key + 1)), datas)
    
#Coordinate conversion from WebMercator to wgs84 
    Lat = []
    Lon = []
    for i in range(len(x)):
        res = webmercator2wgs84(predictions_y[i], predictions_x[i])
        Lon.append(res[0])
        Lat.append(res[1])
    return Lon, Lat

#Coordinate conversion from wgs84 to WebMercator
def wgs84towebmercator_single(lat, lon):
    """
       This function is for coordinates conversion into Web Mercator ones

       Args:
           lat: Latitude data
           lon: Longitude data

       Returns:
           m: The Web Mercator coordinates of latitude
           n:The Web Mercator coordinates of longitude
    """
    transformer = Transformer.from_crs(crs_WGS84, crs_WebMercator)
    m, n = transformer.transform(lat, lon)
    return n, m


# Data preprocessing: convert original latitude and longitude data to Web Mercator coordinates with wgs84towebmercator_Single()
def get_initial_data(x,y,date,length):
    """
       This function is for converting the geographical coordinates into Web Mercator ones

       Args:
           x: Latitude data
           y: Longitude data
           date: The observation date
           length:The length of the data

       Returns:
           initial_data: The Web Mercator coordinates after converting
    """
    initial_data = []
    initial_data.append(["LATITUDE", "LONGITUDE", "OBSERVATION DATE"])
    for i in range(length):
        result = wgs84towebmercator_single(x[i], y[i])
        initial_data.append([result[0], result[1], date[i]])

    return initial_data

#Data preprocessing: Interpolation for missing data
def interpolation(date,length,initial_data):
    """
       This function is to finish interpolation for missing data

       Args:
           date: The observation date
           length:The length of the data
           initial_data:The data after coordinates conversion

       Returns:
           initial_data: The data results after interpolation
    """
    k = 43101
    lose_date = []
    now_date = []
    all_date = [i for i in range(k, k + 365)]
    for i in range(length):
        now_date.append(date[i])
    for i in all_date:
        if i not in now_date:
            lose_date.append(i)
    # print(lose_date)


    for lose in lose_date:
        x = []
        y = []
        for data in initial_data:
            if data[2] == "OBSERVATION DATE":
                continue
            if 0 < lose - int(data[2]) <= 2:
                x.append(data[0])
                y.append(data[1])
        initial_data.append([sum(x) / len(x), sum(y) / len(y), lose])
    return initial_data

##Data preprocessing:Smooth the data by rolling window algorithm with window width=7
def rolling_window(initial_data,save_path,csv_name):
    """
       This function is for rolling_window algorithm

       Args:
           initial_data: The data after interpolation
           save_path:The path for saving the result file
           csv_name:The species name being processed

       Returns:
           rolling_window_data_df: The data results after rolling_window
    """
    Rolling_window_data = []
    Rolling_window_data.append(["LATITUDE", "LONGITUDE", "OBSERVATION DATE"])

    k = 43101
    for i in range(k, k + 365):
        for data in initial_data:
            if data[2] == "OBSERVATION DATE":
                continue
            if -3 < data[2] - i <= 3:
                Rolling_window_data.append([data[0], data[1], i])

    # window_data_df = pd.DataFrame(window_data, columns=False)
    # window_data_df.to_csv('426.csv', index=False)
    Rolling_window_data_df = pd.DataFrame(Rolling_window_data[1:], columns=Rolling_window_data[0])
    Rolling_window_data_df.to_csv(os.path.join(save_path, csv_name.replace('.csv',''), 'Rolling_window_data.csv'), index=False)
    return Rolling_window_data_df

#Data preprocessing:Outlier detectiobn through SLDF
def get_sldf(window_data_df,save_path,csv_name):
    """
       This function is for saving the results after sldf outlier detection

       Args:
           window_data_df: The data after rolling_window
           save_path:The path for saving the result file
           csv_name:The species name being processed

       Returns:
           sldf_df: The data results after sldf outlier detection
    """
    xall = window_data_df.values.astype(float)
    SLDF_all = np.zeros((0, 4))

    day_index = dict()
    day_index[43101] = 0
    for index, i in enumerate(xall[:, 2]):
        if i not in day_index:
            day_index[i] = index

    for day in range(1, 366):
        date = day + 43100
        if day != 365:
            temp = xall[day_index[date]:day_index[date + 1], :2]
        else:
            temp = xall[day_index[date]:, :2]
        outl = sldf(temp)
        t = date * np.ones((outl.shape[0], 1))
        outl = np.concatenate((outl, t), axis=1)
        SLDF_all = np.concatenate((SLDF_all, outl), axis=0)
    new_columns = ["LATITUDE", "LONGITUDE", "SLDF", "OBSERVATION DATE"]

    SLDF_df = pd.DataFrame(SLDF_all, columns=new_columns)
    SLDF_df.to_csv(os.path.join(save_path, csv_name.replace('.csv',''), 'sldf.csv'), index=False)
    return SLDF_df

#Trajectory estimation: Get daily population centroids by Meanshift algorithm
def mean_shift(SLDF_df,save_path,csv_name):
    """
       This function is for getting daily population centroids by Meanshift algorithm

       Args:
           SLDF_df: The data after sldf outlier detection
           save_path:The path for saving the result file
           csv_name:The species name being processed

        Returns:
           result: The data results after meanshift clustering
    """
    # datas = pd.read_excel('data/clean_window_data.xlsx')
    datas = SLDF_df.drop(['SLDF'], axis=1)
    result = []
    result.append(["LATITUDE", "LONGITUDE", "OBSERVATION DATE"])
    for date in tqdm(range(43101, 43466)):
    #for date in range(43101, 43119):
        #print(date)
        data = datas.loc[date == datas['OBSERVATION DATE']]  # .values.tolist()#["answer"]
        data = data.iloc[:, :2]
        data = np.array(data)
        if len(data) == 0:
            continue

        ms = MeanShift()
        ms.fit(data)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        labels_unique = np.unique(labels)
        n_clusters = len(labels_unique)

        for c in cluster_centers:
            result.append([float(c[0]), float(c[1]), date])

        colors = cycle('bcmyk')
        if date % 20 == 0:
            for k, color in zip(range(n_clusters), colors):
                # current_member indicates true if the label is k and false if not
                current_member = labels == k
                cluster_center = cluster_centers[k]
                # Draw plots
                plt.plot(data[current_member, 0], data[current_member, 1], color + '.')
                # Draw circles
                plt.plot(cluster_center[0], cluster_center[1], 'o',
                         markerfacecolor=color,
                         markeredgecolor='k',
                         markersize=14)

                plt.xlabel('Latitude in Mercator system(meter)')

                plt.ylabel('Longitude in Mercator system(meter)')
            # plt.show()
            plt.savefig(os.path.join(save_path, csv_name.replace('.csv', ''), 'shift_{}.jpg'.format(date)), dpi=1000)
            plt.close()
    return result

#Trajectory estimation: Group the daily population centroids according to the minimum distance principle
def group(csv_path,save_path,csv_name):
    """
       This function is for grouping the daily population centroids

       Args:
           csv_path: The path for the data file after Meanshift algorithm
           save_path:The path for saving the result file
           csv_name:The species name being processed

       Returns:
            True: Omitted

    """
    A1 = np.array([[0], [0]])
    A3 = np.array([[0], [0]])

    datas = pd.read_excel(csv_path)
    #datas = datas.iloc[1:, :]

    result_list = []
    for date in range(43101, 43466):
        data = datas.loc[date == datas['OBSERVATION DATE']]
        data = data.iloc[:, :2].values.tolist()
        A1 = np.hstack((A1, np.array(list(data)).T))
        A3 = np.hstack((A3, np.array([[int(len(data))], [A3[1, -1] + int(len(data))]])))

    A1 = np.delete(A1, 0, axis=1)
    A3 = np.delete(A3, 0, axis=1)

    np.save("A1.npy", A1)
    np.save("A3.npy", A3)

    # A1 = np.load('A1.npy')
    # A3 = np.load('A3.npy')

    p2 = 0
    N3 = A3.shape[1]
    LL4 = 0
    dddd = 0
    LL5 = 0
    LL6 = 0

    zhongjian = {}
    KKK2 = np.zeros((1, 10000))
    KKK3 = np.zeros((1, 10000))
    new1 = np.zeros((2, 1000))
    guiji = {}
    abc = np.zeros((10000, 10000), dtype=int)

#Traversal calculations of the centroid distance between two adjacent days in the annual circle
    for b1 in range(2, N3):
        if LL6 > 0:
            LL1 = A3[1, b1 - 2]
            LL2 = A3[1, b1 - 1]
            LL3 = A3[1, b1]
            O1 = A3[0, b1 - 2]
            O2 = new2
            O3 = A3[0, b1]
            KK1 = A1[:, np.arange(LL4, LL1)]
            KK2 = new1
            KK3 = A1[:, np.arange(LL2, LL3)]
            LL4 = LL1
            LL5 = LL5 + 1
        if LL6 == 0:
            LL1 = A3[1, b1 - 2]
            LL2 = A3[1, b1 - 1]
            LL3 = A3[1, b1]
            O1 = A3[0, b1 - 2]
            O2 = A3[0, b1 - 1]
            O3 = A3[0, b1]
            KK1 = A1[:, np.arange(LL4, LL1)]
            KK2 = A1[:, np.arange(LL1, LL2)]
            KK3 = A1[:, np.arange(LL2, LL3)]
            LL4 = LL1
            LL5 = LL5 + 1

#Store centroid coordinates
        guodu1 = KK2.shape[1]
        for pp1 in range(1, guodu1 + 1):
            kk2 = KK2[:, pp1 - 1]
            kk2 = np.transpose(kk2)
            KKK2[:, np.arange(pp1 * 2 - 2, pp1 * 2)] = kk2
        guodu2 = KK3.shape[1]
        for pp1 in range(1, guodu2 + 1):
            kk3 = KK3[:, pp1 - 1]
            kk3 = np.transpose(kk3)
            KKK3[:, np.arange(pp1 * 2 - 2, pp1 * 2)] = kk3

#Perform centroid distance traversal calculation when the number of centroids on the later day is greater than the previous day
#The calculation order needs to consider from O2 to O3 and from O3 to O2 to ensure that all centroids in O3 can be connected.
        jl = np.zeros((100, 100))
        if O3 > O2:
            new2 = O3
            dddd = dddd + 1
            for t1 in range(1, O2 + 1):
                for t2 in range(1, O3 + 1):
                    jl[t1 - 1, t2 - 1] = np.sqrt(
                        (KK2[0, t1 - 1] - KK3[0, t2 - 1]) ** 2 + (KK2[1, t1 - 1] - KK3[1, t2 - 1]) ** 2)
            index = np.argmin(jl[:O2, :O3], axis=1)
            for t3 in range(1, O2 + 1):
                aaa = int(index[t3 - 1] + 1)
                if dddd == 1:
                    shuju1 = np.vstack(
                        (KKK2[:, np.arange(t3 * 2 - 2, t3 * 2)], KKK3[:, np.arange(aaa * 2 - 2, aaa * 2)]))
                    new1[:, t3 - 1] = KK3[:, aaa - 1]
                    LL6 = LL6 + 1
                if dddd > 1:
                    shuju1 = np.vstack((guiji[t3 - 1], KKK3[:, np.arange(aaa * 2 - 2, aaa * 2)]))
                    new1[:, t3 - 1] = KK3[:, aaa - 1]
                zhongjian[t3 - 1] = shuju1
                # zhongjian[t3-1][np.all(shuju1 == 0, axis=1),:].fill(0)
            jl = np.zeros((100, 100))
            for t1 in range(1, O3 + 1):
                for t2 in range(1, O2 + 1):
                    jl[t1 - 1, t2 - 1] = np.sqrt(
                        (KK3[0, t1 - 1] - KK2[0, t2 - 1]) ** 2 + (KK3[1, t1 - 1] - KK2[1, t2 - 1]) ** 2)
            index = np.argmin(jl[:O3, :O2], axis=1)
            XX = np.unique(index)
            nnp = O3 - O2
            nnn = 1
            for i in range(1, len(XX) + 1):
                m = (index == XX[i - 1]).nonzero()[0]
                if len(m) >= 2:
                    abc[nnn - 1, 0] = XX[i - 1] + 1
                    abc[nnn - 1, 1] = len(m)
                    nnn = nnn + 1
                if len(m) >= 2:
                    for nnc in range(1, nnp + 1):
                        if nnn - 1 > 0:
                            aaa = int(index[abc[nnn - 2, 0]] + 1)
                        if nnn - 1 > 1:
                            aaa = int(index[abc[nnn - 2, 0]] + 1)
                            nnn = nnn - 1
                        if dddd > 1:
                            shuju1 = np.vstack((guiji[abc[nnn - 2, 0]], KKK3[:, np.arange(aaa * 2 - 2, aaa * 2)]))
                        if dddd == 1:
                            shuju1 = np.vstack((KKK2[:, np.arange(abc[nnn - 2, 0] * 2 - 2, abc[nnn - 2, 0] * 2)],
                                                KKK3[:, np.arange(aaa * 2 - 2, aaa * 2)]))
                        new1[:, O2 + nnc - 1] = KK3[:, aaa - 1]
                        zhongjian[O2 + nnc - 1] = shuju1
                        # zhongjian[O2 + nnc-1][np.all(shuju1 == 0,axis=1),:].fill(0)
            for t10 in range(0, O3):
                guiji[t10] = zhongjian[t10]

#Perform centroid distance traversal calculation when the number of centroids on the later day is less than the previous day

        if O3 <= O2:
            new2 = O2
            dddd = dddd + 1
            for t1 in range(1, O2 + 1):
                for t2 in range(1, O3 + 1):
                    jl[t1 - 1, t2 - 1] = np.sqrt(
                        (KK2[0, t1 - 1] - KK3[0, t2 - 1]) ** 2 + (KK2[1, t1 - 1] - KK3[1, t2 - 1]) ** 2)
            index = np.argmin(jl[:O2, :O3], axis=1)
            for t3 in range(1, O2 + 1):
                aaa = int(index[t3 - 1] + 1)
                if dddd == 1:
                    shuju1 = np.vstack(
                        (KKK2[:, np.arange(t3 * 2 - 2, t3 * 2)], KKK3[:, np.arange(aaa * 2 - 2, aaa * 2)]))
                    new1[:, t3 - 1] = KK3[:, aaa - 1]
                    LL6 = LL6 + 1
                if dddd > 1:
                    shuju1 = np.vstack((guiji[t3 - 1], KKK3[:, np.arange(aaa * 2 - 2, aaa * 2)]))
                    new1[:, t3 - 1] = KK3[:, aaa - 1]
                    LL6 = LL6 + 1
                guiji[t3 - 1] = shuju1
                # guiji[t3][np.all(shuju1 == 0, axis=1),:].fill(0)

    for key, value in guiji.items():
        value = np.hstack((np.array(value), np.arange(1, len(value) + 1).reshape((len(value), 1))))
        df = pd.DataFrame(value)
        names = ['X', 'Y', 'date']
        df.columns = names
        df.to_excel(os.path.join(save_path, csv_name.replace('.csv',''), 'ni_traj{}.xlsx'.format(key + 1)), sheet_name='Sheet1', index=False)

#Trajectory estimation:Show the estimation results on the map
def map_1(save_path,csv_name):
    """
       This function is for showing the result figure on the map

       Args:
           save_path : The path for storing the result figures
           csv_name: The name of the species to be processed

       Returns:
           True:Omitted
    """
    # plt.rcParams['figure.figsize'] = (28, 8)
    # plt.show()

    excel_list = os.listdir(os.path.join(save_path, csv_name.replace('.csv', '')))
    excel_list1 = []
    for csv_excel in excel_list:
        if 'ni_traj' in csv_excel:
            excel_list1.append(csv_excel)

    LON = []
    LAT = []
    for i in range(len(excel_list1)):
        Lon, Lat = gam(save_path, csv_name, i)
        LON.append(Lon)
        LAT.append(Lat)

    m = Basemap(llcrnrlat=-60, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=-20)  # Instantiate a map
    m.drawcoastlines()  # Draw the coastline
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(lake_color='white')  # Draw the continents and fill them in white

    parallels = np.arange(-90., 90., 10.)  # Draw latitudes with ranges [-90,90] and intervals of 10
    m.drawparallels(parallels, labels=[False, True, True, False], color='none')
    meridians = np.arange(-180., 180., 20.)  # Draw the longitude with a range of [-180,180] and an interval of 10
    m.drawmeridians(meridians, labels=[True, False, False, True], color='none')
    for doc in range(0, len(LON)):
        colorMap = ['red', 'darkorange', 'gold', 'greenyellow', 'pink', 'limegreen', 'mediumturquoise',
                    'dodgerblue',
                    'navy', 'blue', 'mediumorchid', 'fuchsia']
        # Show labels
        label = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                 'November', 'December']

        marker = ['x', 'o', '.', '+', '<', '_', '^', 'v', 'H', '|', 's', '*','x', 'o', '.', '+', '<', '_', '^', 'v','x', 'o', '.', '+', '<', '_', '^', 'v']
        j = 0
        # print(len(lon))
        flag = True


        for i in range(0, len(LON[doc]) - 30, 30):
            # print(i)
            if doc == 0:
                m.plot(LON[doc][i:i + 30], LAT[doc][i:i + 30], marker=marker[doc], linewidth=0.4,
                       color=colorMap[j],
                       markersize=0.5, label=label[
                        j])
                # plt.show()
                j += 1
                if j == 12:
                    j = 0
                    if flag:
                        plt.legend(loc='lower left', shadow=True)
                        flag = False
                    continue
            else:
                m.plot(LON[doc][i:i + 30], LAT[doc][i:i + 30], marker=marker[doc], linewidth=0.4,
                       color=colorMap[j],
                       markersize=0.5)
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
    plt.savefig(os.path.join(save_path, csv_name.replace('.csv', ''), 'map.jpg'), dpi=1000)
    # plt.show()
    plt.close()


