import re
import time
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium import webdriver

base_url = "https://simba.isr.umich.edu/"
result_page_url = base_url + "cb.aspx?vList="

data = pd.read_excel('Copy of PSID Data Mapping Document - July 2023.xlsx')
vars2011 = data[2011].tolist()

# Remove column title
vars2011.pop(0)

driver = webdriver.Chrome()

years = ['2013', '2014', '2015', '2016',
         '2017', '2018', '2019', '2020', '2021']

# Create dictionaries to store variables for each year
vars_by_year = {year: [] for year in years}

for var2011 in vars2011:
    try:
        driver.get(result_page_url + var2011)

        time.sleep(0.5)

        # find other years
        years_available = driver.find_element(By.XPATH,
                                              '//b[contains(text(), "Years")]/ancestor::tr/td[2]').text

        # Loop through each year and extract values
        for year in years:
            pattern = fr'\[{year[2:]}](\w+)'
            match = re.search(pattern, years_available)
            if match:
                vars_by_year[year].append(match.group(1))
            else:
                vars_by_year[year].append('N/A')
    except:
        for year in years:
            vars_by_year[year].append('N/A')
        print("No result found for: " + var2011)

# Close the web driver
driver.quit()

# Save variables to text files
desktop_path = "/Users/student/Desktop/"
for year in years:
    file_path = f'{desktop_path}vars{year}.txt'
    with open(file_path, 'w') as f:
        for var in vars_by_year[year]:
            f.write("%s\n" % var)
