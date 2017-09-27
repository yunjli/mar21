# import the necessary packages
import requests as rq
from bs4 import BeautifulSoup
import re
import csv

# build empty lists to store target information
Carrier = list()
Rate_plan = list()
Monthly_service = list()
Additional_charge = list()
Airtime = list()
Data = list()
Messaging = list()
Bonus = list()
Features = list()
Data_usage = list()
Contract = list()
Activation = list()
Emergency = list()
Cancellation = list()
Notes = list()
Information = list()
web_id = list()



def _get_carrier(img_tag): # img_tag is the input line
    pattern = re.compile('img alt=\"(.+?)\" ') # read lines that are in certain format 
    result = re.search(pattern, img_tag) # pick out target information from img_tag line
    if result:
        return result.group(1) # return the target information
    return "Not found" # give error sentence when the scrape failed


def get_carriers_names(soup): # soup is the website address
    carriers = list() # build a new list to store the target information
    for i in soup.find_all('img', width = "80"): # find out all satisfied lines in the soup website 
        carriers.append(_get_carrier(str(i))) # input satisfied line into the helper function and store target information into the list
    return carriers # return the list that contains all the target information

## =====repeat the same method============

def _parse_rate_plan(span_tag):
    pattern = re.compile('>(.+?)</span>') # format changes when pick different target information
    result = re.search(pattern, span_tag)
    if result:
        return result.group(1)
    return "Not found"


def get_rate_plan(soup):
    plans = list()
    for span_tag in soup.find_all('span', style = "font-weight: bold;"):
        plans.append(_parse_rate_plan(str(span_tag)))
    return plans


def _parse_fee(td_tage):
    pattern = re.compile('<td>(.*?)</td>')
    result = re.search(pattern, td_tage)
    if result:
        return result.group(1)
    return "Not found"


def get_fee(soup):
    fees = list()
    results = soup.find("td", string = re.compile("Monthly Service") ) # pick out all lines contains words "Monthly Service"
    if results:
    	for td_tag in results.find_next_siblings("td"): # use the find_next_siblings function to find out all satisfied lines
            fees.append(_parse_fee(str(td_tag)))
    return fees


def _parse_tab_charge(td_tag):
    pattern = re.compile('>(.*?)</td>')
    result = re.search(pattern, td_tag)
    if result:
        return result.group(1)
    return "Not found"


def get_tab_charge(soup):
    tab = list()
    results = soup.find("td", string = re.compile("Additional Monthly"))
    if results:
    	for td_tag in results.find_next_siblings("td"):
        	tab.append(_parse_tab_charge(str(td_tag)))
    return tab


def _parse_airtime(td_tag):
    pattern = re.compile('>(.*?)</td>')
    result = re.search(pattern, td_tag)
    if result:
        result1 = re.sub("\xc2\xb2", " ", result.group(1)) # use sub function to replace the first string with the second string which are in the third object
        result2 = re.sub("\xc2\xb9", " ", result1)
        result3 = re.sub("\xc2\xb3", " ", result2)
        result4 = re.sub("<br/>", " ", result3)
        result5 = re.sub("</s>", "change to", result4)
        return result5
    return "Not found"


def get_airtime(soup):
    airtime = list()
    results = soup.find("td", string = re.compile("Airtime"))
    if results:
    	for td_tag in results.find_next_siblings("td"):
        	airtime.append(_parse_airtime(str(td_tag)))
    return airtime


def _parse_data(td_tag):
    pattern = re.compile('>(.*?)</td>')
    result = re.search(pattern, td_tag)
    if result:
        result1 = re.sub("\xc2\xb2", " ", result.group(1))
        result2 = re.sub("\xc2\xb9", " ", result1)
        result3 = re.sub("\xc2\xb3", " ", result2)
        result4 = re.sub("<br/>", " ", result3)
        result5 = re.sub("</s>", " changes to", result4)
        result6 = re.sub("<s>", " ", result5)
        return re.sub("<br>", " ", result6)
    return "Not found"


def get_data(soup):
    data = list()
    results = soup.find("td", string = re.compile("Data Included"))
    if results:
    	for td_tag in results.find_next_siblings("td"):
        	data.append(_parse_data(str(td_tag)))
    return data


def _parse_messaging(td_tag):
    pattern = re.compile('>(.*?)</td>')
    result = re.search(pattern, td_tag)
    if result:
        return result.group(1)
    return "Not found"


def get_messaging(soup):
    messaging = list()
    results = soup.find("td", string = re.compile("Messaging"))
    if results:
    	for td_tag in results.find_next_siblings("td"):
        	messaging.append(_parse_messaging(str(td_tag)))
    return messaging


def _parse_bonus(td_tag):
    pattern = re.compile('>(.*?)</td>')
    result = re.search(pattern, td_tag)
    if result:
        return re.sub("\xc2\xb2", " ", result.group(1))
    return "Not found"


def get_bonus(soup):
    bonus = list()
    results = soup.find("td", string = re.compile("Bonus"))
    if results:
    	for td_tag in results.find_next_siblings("td"):
        	bonus.append(_parse_bonus(str(td_tag)))
    return bonus


def _parse_features(td_tag):
    pattern = re.compile('>(.*?)</td>')
    result = re.search(pattern, td_tag)
    if result:
        return result.group(1)
    return "Not found"


def get_features(soup):
    features = list()
    results = soup.find("td", string = re.compile("Features"))
    if results:
    	for td_tag in results.find_next_siblings("td"):
        	features.append(_parse_features(str(td_tag)))
    return features


def _parse_data_rate(td_tag):
    pattern = re.compile('>(.*?)</td>')
    result = re.search(pattern, td_tag)
    if result:
        return re.sub("\xc2\xa2", " ", result.group(1))
    return "Not found"


def get_data_rate(soup):
    data_rate = list()
    results = soup.find("td", string = re.compile("Additional Data"))
    if results:
    	for td_tag in results.find_next_siblings("td"):
        	data_rate.append(_parse_data_rate(str(td_tag)))
    return data_rate


def _parse_contract(td_tag):
    pattern = re.compile('>(.*?)</td>')
    result = re.search(pattern, td_tag)
    if result:
        return result.group(1)
    return "Not found"


def get_contract(soup):
    contract = list()
    results = soup.find("td", string = re.compile("Contract"))
    if results:
    	for td_tag in results.find_next_siblings("td"):
        	contract.append(_parse_contract(str(td_tag)))
    return contract


def _parse_activation(td_tag):
    pattern = re.compile('>(.*?)</td>')
    result = re.search(pattern, td_tag)
    if result:
        return result.group(1)
    return "Not found"


def get_activation(soup):
    activation = list()
    results = soup.find("td", string = re.compile("Activation"))
    if results:
    	for td_tag in results.find_next_siblings("td"):
        	activation.append(_parse_activation(str(td_tag)))
    return activation


def _parse_emergency(td_tag):
    pattern = re.compile('>(.*?)</td>')
    result = re.search(pattern, td_tag)
    if result:
        result1 = re.sub("\xc2\xb2", " ", result.group(1))
        result2 = re.sub("\xc2\xb9", " ", result1)
        return re.sub("\xc2\xb3", " ", result2)
    return "Not found"


def get_emergency(soup):
    emergency = list()
    results = soup.find("td", string = re.compile("911 Emergency"))
    if results:
    	for td_tag in results.find_next_siblings("td"):
        	emergency.append(_parse_emergency(str(td_tag)))
    return emergency


def _parse_cancellation(td_tag):
    pattern = re.compile('>(.*?)</td>')
    result = re.search(pattern, td_tag)
    if result:
        return result.group(1)
    return "Not found"


def get_cancellation(soup):
    cancellation = list()
    results = soup.find("td", string = re.compile("Early"))
    if results:
    	for td_tag in results.find_next_siblings("td"):
        	cancellation.append(_parse_cancellation(str(td_tag)))
    return cancellation


def _parse_notes(td_tag):
    pattern = re.compile('>(.*?)</td>')
    result = re.search(pattern, td_tag)
    if result:
        result1 = re.sub("\xc2\xae", " ", result.group(1))
        result2 = re.sub("\xc2\xb9", " ", result1)
        result3 = re.sub("\xc2\xb2", " ", result2)
        result4 = re.sub("<br/>", " ", result3)
        result5 = re.sub("\xc2\xb3", " changes to", result4)
        result6 = re.sub("\xc2\xa2", " ", result5)
        return re.sub("<br>", " ", result6)
    return "Not found"


def get_notes(soup):
    notes = list()
    results = soup.find("td", string = re.compile("Notes"))
    if results:
    	for td_tag in results.find_next_siblings("td"):
        	notes.append(_parse_notes(str(td_tag)))
    return notes


def _parse_information(td_tag):
    pattern = re.compile('>(.*?)</td>')
    result = re.search(pattern, td_tag)
    if result:
        return result.group(1)
    return "Not found"


def get_information(soup):
    information = list()
    results = soup.find("td", string = re.compile("More"))
    if results:
    	for td_tag in results.find_next_siblings("td"):
        	information.append(_parse_information(str(td_tag)))
    return information


file = open("ONXapple.txt", "r") #read the txt file contains all website addresses
weblst = file.readlines()#store all web addresses strings in the list
for i in weblst:
	web_id.append(i)


for url in web_id:
	response = rq.get(url) # go to the url website
	html_doc = response.text # read the original code of the url website
	soup = BeautifulSoup(html_doc, 'html.parser')  # fixed functions for BeautifulSoup method

	Carrier += get_carriers_names(soup)
	Rate_plan += get_rate_plan(soup)
	Information += get_information(soup)
	
	if get_fee(soup):
		Monthly_service += get_fee(soup)
	else:
		Monthly_service.append("Empty")
	
	if get_tab_charge(soup):
		Additional_charge += get_tab_charge(soup)
	else:
		Additional_charge.append("Empty")
		
	if get_airtime(soup):
		Airtime += get_airtime(soup)
	else:
		Airtime.append("Empty")
		
	if get_data(soup):
		Data += get_data(soup)
	else:
		Data.append("Empty")
	
	if get_messaging(soup):
		Messaging += get_messaging(soup)
	else:
		Messaging.append("Empty")
		
	if get_bonus(soup):
		Bonus += get_bonus(soup)
	else:
		Bonus.append("Empty")
	
	if get_features(soup):
		Features += get_features(soup)
	else:
		Features.append("Empty")
		
	if get_data_rate(soup):
		Data_usage += get_data_rate(soup)
	else:
		Data_usage.append("Empty")
		
	if get_contract(soup):
		Contract += get_contract(soup)
	else:
		Contract.append("Empty")
		
	if get_activation(soup):
		Activation += get_activation(soup)
	else:
		Activation.append("Empty")
		
	if get_emergency(soup):
		Emergency += get_emergency(soup)
	else:
		Emergency.append("Empty")
		
	if get_cancellation(soup):
		Cancellation += get_cancellation(soup)
	else:
		Cancellation.append("Empty")
		
	if get_notes(soup):
		Notes += get_notes(soup)
	else:
		Notes.append("Empty")
		
	

path = "/Users/apple/Desktop/work-study/report/week1/ONXapple2.csv" # give the path to store the file you will write out
with open(path, "w+") as output:
    writer = csv.writer(output, lineterminator = "\n") # function used to write new csv file
    writer.writerow(Carrier) # write row information 
    writer.writerow(Rate_plan)
    writer.writerow(Monthly_service)
    writer.writerow(Additional_charge)
    writer.writerow(Airtime)
    writer.writerow(Data)
    writer.writerow(Messaging)
    writer.writerow(Bonus)
    writer.writerow(Features)
    writer.writerow(Data_usage)
    writer.writerow(Contract)
    writer.writerow(Activation)
    writer.writerow(Emergency)
    writer.writerow(Cancellation)
    writer.writerow(Notes)
    writer.writerow(Information)



