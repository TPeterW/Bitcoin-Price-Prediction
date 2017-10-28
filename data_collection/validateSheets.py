from os import listdir,rename
from os.path import isfile,join
import csv
import sys


MYPATH = "coin_historical data/"
CORRECT_FILES_PATH = "correct_historical_data"
INCORRECT_FILES_PATH = "incorrect_historical_data"

csv_files = [f for f in listdir(MYPATH) if isfile(join(MYPATH, f))]

length_ = None

coin_dict = {}
clean_coin_dict = {}

for file in csv_files[1:]:
    file_path = MYPATH + file 
    with open(file_path,"r") as f:
        reader = csv.reader(x.replace('\0', '') for x in f) #replace null byte in files        
        i = 0
        for row in reader:
            i += 1
        coin_dict[file] = i #number of rows in file

        if file == "Bitcoin.csv":
            length_ = i 


print length_


count = 0

print "\nCORRECT SHEETS\n"
for k,i in coin_dict.items():
    if i == length_:
        # print k + " " + str(i) + " off by " + str(length_ - i)
        count += 1
        try:
            rename(join(MYPATH,k), join(CORRECT_FILES_PATH,k))
            print "moved " + k + " to " + CORRECT_FILES_PATH
        except OSError as e:
            continue

print "\n"+str(count) + " sheets are correct ".upper()

count = 0
print "\nINCORRECT SHEETS\n"
for k,i in coin_dict.items():
    if i != length_:
        # print k + " " + str(i) + " off by " + str(length_ - i)
        count += 1
        try:
            rename(join(MYPATH,k), join(INCORRECT_FILES_PATH,k))
            print "moved " + k + " to " + INCORRECT_FILES_PATH
        except OSError as e:
            continue
print "\n"+str(count) + " sheets are incorrect ".upper()



