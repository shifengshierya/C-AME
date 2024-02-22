from Part_2_functions_for_eachstep import get_all_csv, get_initial_data, interpolation, rolling_window, get_sldf, mean_shift, data_write, group,  create_folder,map_1
from Part_3_speed import speed
from Part_3_offset_distance import off_distance
from data_writes import data_write
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from Part_3_avg_distance import avg_distance


def part2(data_path='./source/', save_path='./target/'):
#Run by input paths from the terminal
    # data_path = input('Please set part2 input folder ')
    # data_path = data_path + '/'
    # save_path = input('Please set part2 output folder')
    # save_path = save_path + '/'
#Your absolute paths
    # data_path = './source/'
    # save_path = './target/'
    os.makedirs(save_path, exist_ok=True)

    all_csv = get_all_csv(data_path)
    print('Get all csv files')


    for csv_num in range(len(all_csv)):
        csv_name = all_csv[csv_num]
        # var(save_path + '\\' + csv_name.replace('.csv', ''))
        create_folder(os.path.join(save_path, csv_name.replace('.csv', '')))


        df = pd.read_csv(os.path.join(data_path, csv_name))
        x = df["LATITUDE"]
        y = df["LONGITUDE"]
        date = df["OBSERVATION DATE"]
        length = len(x)
# Data preprocessing:convert original latitude and longitude data to WebMercator coordinates by WGS84ToWebMercator_Single()
        initial_data = get_initial_data(x, y, date, length)
        initial_data_df = pd.DataFrame(initial_data[1:], columns=initial_data[0])
        initial_data_df.to_csv(os.path.join(save_path, csv_name.replace('.csv', ''), 'initial_data.csv'), index=False)
        print('File {}/{},Step 1 finished'.format(csv_num + 1, len(all_csv)))

#Data preprocessing: Interpolation for missing data by interpolation()
        initial_data = interpolation(date, length, initial_data)
        print('{},Step2 finished'.format(csv_name))

#Data preprocessing:Smooth the data by rolling_window()
        Rolling_window_data_df = rolling_window(initial_data, save_path, csv_name)
        print('{},Step3 finished'.format(csv_name))

#Data preprocessing:Outlier detectiobn by get_sldf()
        SLDF_df = get_sldf(Rolling_window_data_df, save_path, csv_name)
        print('{}, Get SLDF_df'.format(csv_name))


#Path estimation: Get daily population centroids by mean_shift()
        result = mean_shift(SLDF_df,save_path,csv_name)
        print('{},Get Meanshift results'.format(csv_name))
        data_write(os.path.join(save_path, csv_name.replace('.csv', ''), "shift.xlsx"), result)

#Path estimation: Group the daily population centroids according to the minimum distance principle by group()
        group(os.path.join(save_path, csv_name.replace('.csv', ''), "shift.xlsx"), save_path, csv_name)
        print('File{}/{},Group finished'.format(csv_num + 1, len(all_csv)))


#Path estimation:Show the estimation results on the map by map_1() after GAM fitting and converting  WebMercator to wgs84
        map_1(save_path,csv_name)


#Index calculation of speed and offset distance by speed(),offset_distance()
        speed(os.path.join(save_path, csv_name.replace('.csv', '')))
        off_distance(os.path.join(save_path, csv_name.replace('.csv', '')))
        print('File {}/{},speed and offset distance finished'.format(csv_num + 1, len(all_csv)))
#Index calculation of avarage distance of daily centroids by  avg_distance()
        avg_distance(os.path.join(save_path, csv_name.replace('.csv', '')))
        print('File {}/{},Avg_distance finished'.format(csv_num + 1, len(all_csv)))


        print('Finish,{}/{}'.format(csv_num + 1, len(all_csv)))

if __name__ == '__main__':
    part2()
