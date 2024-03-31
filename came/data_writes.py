import xlwt
import csv
import time

#Function for file storage
def data_write(file_path, datas):
    """
        This function is for saving files

        Args:
            file_path: The path for saving files
            datas: The data to be saved

        Returns:
            True: Omitted
    """

    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)
    
    i = 0
    for data in datas:
        for j in range(len(data)):
            sheet1.write(i, j, str(data[j]))
        i = i + 1
    f.save(file_path)

def savecsv(path, item, model='a'):
    """
       This function is for saving csv files

       Args:
           path: The  path for saving csv files
           item: The data to be saved
           model: The default parameter

       Returns:
           True: Omitted
    """
    while True:
        try:
            with open(path, model, encoding='utf_8_sig', newline='') as f:
                # with open(path, model, encoding='gb18030', newline='') as f:
                w = csv.writer(f)
                w.writerow(item)
                return True
        except:
            print('Close the table or the program cannot write')
            time.sleep(1)


def savecsvs(path,item,model = 'a'):
    """
       This function is for saving csv files

       Args:
           path: The  path for saving csv files
           item: The data to be saved
           model: The default parameter

       Returns:
           True: Omitted
    """
    while True:
        try:
            with open(path, model, encoding='utf_8_sig', newline='') as f:
            #with open(path, model, encoding='gb18030', newline='') as f:
                w = csv.writer(f)
                w.writerows(item)
                return True
        except Exception as e:
            print(e)
            print('请关闭表格，否则程序无法写入')
            time.sleep(1)


def readcsv(path):
    """
       This function is for reading csv files

       Args:
           path: The file  path

       Returns:
           rows: The file data
    """
    try:
        with open(path, 'r',encoding='utf_8_sig') as f:
        #with open(path, 'r',encoding='gb18030') as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        return rows
    except:
        #with open(path, 'r',encoding='utf_8_sig') as f:
        with open(path, 'r',encoding='gb18030',errors='ignore') as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        return rows