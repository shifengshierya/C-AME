install.packages("auk")
install.packages("magrittr")

library(auk)
# path to the ebird data file, here a sample included in the package
# get the path to the example data included in the package
# in practice, provide path to ebd, e.g. f_in <- "data/ebd_relFeb-2018.txt

#The 2023 version is used here as an example
f_in <- "F:/ebd_relDec-2023/ebd_relDec-2023.txt/ebd_relDec-2023.txt"

file_path <- "F:/ebd_relDec-2023/ebd_relDec-2023.txt/ebd_relDec-2023.txt"
#file_path <- "D:/BCRCodes.txt"
permissions <- file.access(file_path)
print(permissions)

# output text file
f_out <- "ebd_filtered_grja.txt"
ebird_data <- f_in %>% 
  # 1. reference file
  auk_ebd() %>% 
  # 2. define filters
  auk_species(species = "Sprague's Pipit") %>% 

  auk_date(date = c("2018-01-01", "2018-12-31")) %>%
  # 3. run filtering
  auk_filter(file = f_out,overwrite = TRUE) %>% 
  # 4. read text file into r data frame
  read_ebd() 

filtered_data <- ebird_data
csv_file_path <- "D:/R/Anthus_spragueii.csv"
write.csv(filtered_data, file = csv_file_path, row.names = FALSE)

