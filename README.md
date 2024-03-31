# C-AMEv1.1 <a href="https://colab.research.google.com/drive/1kOmRemx4p2Wqa2JtFeZtZNlCNiVo8zEc?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>
C-AME is an open-source Python software tool for avian migration trajectory estimation based on observation data.
We offer both a **Windows-only** application and the corresponding source code with a Google Colab demo ready.
## C-AME Application Usage (Windows only)
![came](./assets/came-win.png)
1. **Download the Code**
   - Download the code located in the "application" folder and extract all files.
2. **Navigate and Determine Folder Paths**
   - Navigate into the `cameMain` folder.
   - Determine the following folder paths to organize your data:
     - Raw data folder (support for multiple files processing): Part 1 Input Folder
     - Data format standardization results folder: Part 1 Output Folder
     - Processing results storage folder: Part 2 output Folder
   - Note that we have set the default folder 'Raw Data' as the Part 1 Input Folder with the raw observation data in it to reproduce our method, you need to set two empty folders, such as 'Process Files' and 'Result Files', as Part 1 Output Folder and Part 2 Output Folder  before you run it. 
3. **Execute the Application**
   - Double click on the Application file named `cameMain.exe`, which is represented by a blue bird icon. This action will open the software terminal.
4. **Parameter Settings**
   - Set the three paths you determined in Step 2 within the terminal window.
   - Ensure you select the correct columns for the following data:
     - Latitude
     - Longitude
     - Observation date
     - Observation count
   - Note that we have set the default column numbers as follows:
     - Col# Latitude : Col# Observation Date = 8 : 11
     - Col# Observation Count= 3.
   -  Set the EPSG Code to define projection model:
     - The default value of EPSG: 3857.
   -  Choose the fitting model for grouped centroids:
     -  Three alternative models to choose from: GAM, Random Forests, and K-NN
5. **Run the Application**
   - Click on the "RUN" button to execute the application.

## C-AME Source-code Usage 
[[API Documentation](https://shifengshierya.github.io/C-AME/)]

This part is recommended for developers. 
### Quick Start on Colab
We provide a [Colab Demo](https://colab.research.google.com/drive/1kOmRemx4p2Wqa2JtFeZtZNlCNiVo8zEc?usp=sharing) of this code for a quick try.
### Environment
We use Python 3.10 for all the experiments. Install other dependencies via
```bash
pip install -r requirements.txt
```
### Data Preparation
The raw observation data can be organized into lists, including the species with observation counts, observation dates and locations (latitude and longitude). In our case, raw data for Anthus spragueii, a small songbird in North America, is acquired from eBird and saved in the Part 1 Input Folder for our case study. It includes 2169 records from 2018/1/1 to 2018/12/31, and organized into lists, including common name, scientific name, observation count, country, state, county, latitude, longitude, observation date, and so on. The software tool can also be applied to other time-series latitude and longitude observation data of migratory species from different observation databases, such as GBIF, iNaturalist, the researchers themselves, or integrated data from multiple data sources. 

### Usage
Navigate into the source folder, and run 
```bash
python Part_1_data_format_standardization.py --input_dir {INPUT_DIR} --data_dir {DATA_DIR} --save_dir {SAVE_DIR}
```
where 
- `INPUT_DIR` refers to the directory of input data mentioned above (support for multiple files processing)
- `DATA_DIR` refers to data format standardization results folder
- `SAVE_DIR` refers to process results storage folder

Then, you can get the results in the corresponding folders：
#### Result Files
| Item                               | Description                                                                               |
|:------------------------------------:|-------------------------------------------------------------------------------------------|
| `map.jpg`                            | The migration trajectory estimation results                                                  |
| `off_distance` folder                | The offset distance calculation results and its figure <br> - `d1.csv`-`d13.csv`: the offset distance for each trajectory <br> - `da.csv`: the average offset distance for the species|
| `speed` folder                       | The speed calculation results and its figure <br> - `s1.csv`-`s13.csv`: the speed for each trajectory <br> - `da.csv`: the average speed for the species|
| `ave_disatance.csv` & `avg_distance.png`| The average distance of daily centroids and its figure                                    |

#### Processing Files
|           Items           |                               Description                               |
|:-------------------------:|:-----------------------------------------------------------------------:|
|  initial_data.csv         | The data after data format standardization and interpolation          |
|  Rolling_window_data.csv  |                The data after rolling window                            |
|  sldf.csv                 |             The data after SLDF outlier detection                        |
|  shift.xls                |                  The data after Meanshift algorithm                      |
|  shift_43110.jpg-shift_43460.jpg | The figures showing the centroids during clustering                |
|  ni_traj1.xls-ni_traj13.xls     | The centroids coordinates files after grouping                      |
|  gamLat1.jpg-gamLon13.jpg       | The process figures for Gam algorithm                                |
|  result1.xls-result13.xls       | The results after Gam algorithm                                      |

