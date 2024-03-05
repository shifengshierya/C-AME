import xlwt

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
