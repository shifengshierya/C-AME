# C-AMEv1.1 <a href="https://colab.research.google.com/drive/1kOmRemx4p2Wqa2JtFeZtZNlCNiVo8zEc?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>
C-AME is an open source Python software application for avian migration trajectory estimation based on observation data from eBird platform.
We offer both a **Windows-only** application and the corresponding source code with a Google Colab demo ready.
## C-AME Application Usage (Windows only)
![came](https://github.com/shifengshierya/C-AME/assets/50764534/ccc84a8f-b4ca-42d4-998d-38d8996ab5b6)
1. **Download the Code**
   - Download the code located in the "application" folder and extract all files.
2. **Navigate and Determine Folder Paths**
   - Navigate into the `cameMain` folder.
   - Determine the following folder paths to organize your data:
     - Raw data folder (support for multiple files processing): Part 1 Input path
     - Data format standardization results folder: Part 1 Output path
     - Final results storage folder: Part 2 output path
   - Note that we have set the default folder as:Raw Data,Process Files,Result Files in the cameMain.
3. **Execute the Application**
   - Double click on the Application file named `cameMain.exe`, which is represented by a blue bird icon. This action will open a terminal window.
4. **Configure Column Settings**
   - Set the three paths you determined in Step 2 within the terminal window.
   - Ensure you select the correct columns for the following data:
     - Latitude
     - Longitude
     - Obs_date
     - Observation_count
   - Note that we have set the default column mappings as follows:
     - Column Latitude : Column_Obs_date = 8 : 11
     - Default observation count (Column_Obs_count) is set to 3.
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
We collect all the data from [ebird](https://ebird.org/data/download). The input data structure is
![data2](https://github.com/shifengshierya/C-AME/assets/50764534/70cac4c9-09ef-4b35-8a9a-dfa65cec966b)

### Usage
Navigate into the source folder, and run 
```bash
python Part_1_data_format_standardization.py --input_dir {INPUT_DIR} --data_dir {DATA_DIR} --save_dir {SAVE_DIR}
```
where 
- `INPUT_DIR` refers to the directory of input data mentioned above (support for multiple files processing)
- `DATA_DIR` refers to data format standardization results folder
- `SAVE_DIR` refers to final results storage folder

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
|  shift_43120.jpg-shift_43460.jpg | The figures showing the centroids during clustering                |
|  ni_traj1.xls-ni_traj13.xls     | The centroids coordinates files after grouping                      |
|  gamLat1.jpg-gamLon13.jpg       | The process figures for Gam algorithm                                |
|  result1.xls-result13.xls       | The results after Gam algorithm                                      |

