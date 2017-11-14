import sys,csv
from os import listdir,rename
from os.path import isfile,join
from datetime import datetime
from collections import OrderedDict



MYPATH = "raw_data_incomplete/"
BITCOIN_FILE = "correct_historical_data/Bitcoin.csv"
DEST_PATH = "raw_data_complete/"


def read_from_file_csv(filename):
    csv_data = []
    with open(filename,'rU') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            try:
                if row[5] is '':
                    continue 
            except IndexError as e:
                pass
            csv_data.append(row)
    return csv_data

def write_to_csv(filename,header,data):
    print filename
    with open(filename,'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        for row in data:
            csv_writer.writerow(row)


if __name__=='__main__':

    # date_object = datetime.strptime(date_input, '%b %d, %Y') %b %d, %Y
    #datetime.date.strftime(d, "%b %d, %Y")

    raw_bitcoin_data = read_from_file_csv(BITCOIN_FILE)
    bitcoin_dates = [row[0] for row in raw_bitcoin_data[1:]]

    csv_files = [f for f in listdir(MYPATH) if isfile(join(MYPATH, f))]
    files_1 = [csv_files[0],csv_files[-1]] #files with date format '%Y-%m-%d'
    files_2 = [csv_files[1:4]] #files with dateformat '%m/%d/%y'

    data_dict = OrderedDict()

    # change date format in files
    for file in csv_files:
        file_path = MYPATH + file
        raw_data = read_from_file_csv(file_path)

        if file in files_1:
            headers =  raw_data[0]
            for row in raw_data[1:]:
                date_old = row[0]
                date_object = datetime.strptime(date_old, '%Y-%m-%d')
                date_object = date_object.strftime("%b %d, %Y")
                row[0] = date_object #change date 
            data_dict[file] = raw_data
        else:
            headers =  raw_data[0]
            for row in raw_data[1:]:
                date_old = row[0]
                date_object = datetime.strptime(date_old.replace(' ',''), '%m/%d/%y')
                date_object = date_object.strftime("%b %d, %Y")
                row[0] = str(date_object) #change date 
            data_dict[file] = raw_data

        # print "Done with " + file
        # print len(data_dict.keys()) == len(bitcoin_dates)

    # fill in missing rows and write to destination
    for file in csv_files:
        file_data_old = data_dict[file]

        # store file data in dict with date as key
        headers = file_data_old[0]
        header_length = len(headers)
        date_data_dict_old = OrderedDict()
        date_data_dict_new = OrderedDict()

        for data_row in file_data_old[1:]:
            date_data_dict_old[data_row[0]] = data_row


        # cross reference against bitcoin dates and fill with dummy data
        for date in bitcoin_dates:
            if date in date_data_dict_old.keys():
                date_data_dict_new[date] = date_data_dict_old[date]
            else:
                dummy_elems = ['*?' for _ in list(range(header_length-1))]
                date_data_dict_new[date] = [date] + dummy_elems

        # print "Done with " + file
        # print len(date_data_dict_new.keys()) == len(bitcoin_dates)

        # write back to csv
        data_to_write = [date_data_dict_new[key] for key in date_data_dict_new.keys()]
        write_to_csv(DEST_PATH + file,headers,data_to_write)



   


 

 
