from selenium import webdriver
import csv

def write_to_csv(coin_data,coin_name,header):
    with open("coin_historical data/" + coin_name +'.csv','w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        for row in coin_data[coin_name]:
            csv_writer.writerow(row)

url = "https://coinmarketcap.com/historical/20131229/"

url2 = "https://coinmarketcap.com/currencies/%s/historical-data/?start=20131231&end=20171001"

# 20131227
driver = webdriver.Chrome()

driver.get(url)


table_rows = driver.find_elements_by_class_name("currency-name")

coin_names = []
coin_symbols = []

for table_row in table_rows:
    coin_names += table_row.find_elements_by_tag_name("img")



for coin in coin_names:
    coin_symbols += [coin.get_attribute("alt")]

coin_hist_data_dict = {} 

coin_symbols = coin_symbols[27:]

print coin_symbols

for symbol in coin_symbols:
    try:
        hist_data_url = url2 % (symbol)
        driver.get(hist_data_url)
        # get header
        table_head = driver.find_elements_by_class_name("table")[0].find_elements_by_tag_name("thead")
        headers = table_head[0].find_elements_by_tag_name("th")
        headers_list = []
        for header in headers:
            headers_list += [header.text]
        # print headers_list
        # get historical data
        table_body = driver.find_elements_by_class_name("table")[0].find_elements_by_tag_name("tbody")
        data_rows = table_body[0].find_elements_by_tag_name("tr")
        coin_hist_values = []
        for data_row in data_rows:
            data = data_row.find_elements_by_tag_name("td")
            temp_store = []
            for value in data:
                temp_store += [value.text]
                # print value.text
            # store as dict
            # temp_dict = {}
            # temp_dict[headers_list[0]] = temp_store[0]
            # temp_dict[headers_list[1]] = temp_store[1]
            # temp_dict[headers_list[2]] = temp_store[2]
            # temp_dict[headers_list[3]] = temp_store[3]
            # temp_dict[headers_list[4]] = temp_store[4]
            # temp_dict[headers_list[5]] = temp_store[5]
            # temp_dict[headers_list[6]] = temp_store[6]

            coin_hist_values.append(temp_store)
        coin_hist_data_dict[symbol] = coin_hist_values
        # print coin_hist_data_dict
        write_to_csv(coin_hist_data_dict,symbol,headers_list)
    except IndexError as e:
        print "data for " + symbol + " not available at " + hist_data_url 

driver.quit()

