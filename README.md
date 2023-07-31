# C-AMEv1.1
C-AME is an open source Python software application for avian migration paths estimation based on observation data from eBird platform.

## C-AME Application Usage
![57ad48a6a345251dad021074217d808](https://github.com/shifengshierya/C-AMEv1.1/assets/131585037/8645d155-c98f-46e3-85e6-73dc140a9aaf)
1.	Download the code in the folder named application, and extract all files.
2.	Navigate into the cameMain folder, and determine the folder to store the raw data files, the folder to save the data format standardization results, and the folder to store the final results. As an example, we provide the following folder paths as a reference:./application/ cameMain/Raw Data; ./application/cameMain/ Process Files; ./application/cameMain/Result Files.
3.	Double click the Application file (.exe) named cameMain, which has a blue bird as an icon. Then you will see a terminal window open.
4.	Set the three paths you have determined in Step 2, and make sure you select the correct column: Latitude ,Longitude, Obs_date(we have set the Column Latitude : Column_Obs_date default as 8 : 11, and the observation count(set Column_Obs_count default as 3)
5.	Run the application file by clicking RUN button.

## Source-code Usage
This part is recommended for developers.
### Environment
We use Python 3.10 for all the experiments. Install the dependency via
```
pip install requirements.txt
```

### Data Preparation
We collect all the data from [ebird](https://ebird.org/science/status-and-trends). The input data structure is
![image](https://github.com/shifengshierya/C-AMEv1.1/assets/131585037/06a86fe7-e1ce-452e-9776-648cc1d2102a)

### Usage
1.	Install Pycharm.
2.	Download the code in the folder named `source`. Create the environment with a terminal or command prompt window  by referring to the requirements.txt file. 
3.	Navigate into the source folder, and run the file Part_1_data_format_standardization.py. 
Then, you can get the results in the corresponding folders：
#### Result Files
- map.jpg: The migration paths estimation results
- off_distance folder:The offset distance calculation results and its figure(d1.csv-d13.csv,the offset distance for each trajectory;da.csv,the average offset distance for the species)
- speed folder:The speed calculation results and its figure(s1.csv-s13.csv,the speed for each trajectory;da.csv,the average speed for the species)
- ave_disatance.csv&avg_distance.png:The average distance of daily centroids  and its figure
#### Processing Files
- initial_data.csv: The data after data format standardization and interpolation
- Rolling_window_data.csv: The data after rolling window
- sldf.csv:The data after SLDF outlier detection
- shift.xls:The data after Meanshift algorithm
- shift_43120.jpg-shift_43460.jpg:The figures showing the centroids during clustering
- ni_traj1.xls-ni_traj13.xls:The centroids coordinates files after grouping
- gamLat1.jpg-gamLon13.jpg:The process figures for Gam algorithm
- result1.xls-result13.xls:The results after Gam algorithm
