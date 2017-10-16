from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import Process64 as P64

##### config
loadingtime = 8
#####

''' sample of selenium
driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")
driver.get("http://www.python.org")
assert "Python" in driver.title,"Not Found Target Text in Web Title" 
elem = driver.find_element_by_name("q")
elem.clear()
elem.send_keys("pycon")
elem.send_keys(Keys.RETURN)
assert "No results found." not in driver.page_source,"No results found."
'''

#    Open Web browser
driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")

#    go to Google image
driver.get("https://images.google.com/")

#    Check the website is your target , should enter your title of target site
assert "Google" in driver.title,"Not Found Target Text in Web Title"

#    operation that you want on WebSite
elem = driver.find_element_by_name("q")
elem.clear()
elem.send_keys("man photo")
elem.send_keys(Keys.RETURN)

PicIndex = 0
for q in range(5):
    driver.execute_script("window.scrollTo(100, document.body.scrollHeight);")
    time.sleep(loadingtime)
    elem = driver.find_elements_by_class_name("rg_ic")
    for i in range(PicIndex,len(elem)):        
        Base64Straing = elem[i].get_attribute("src")    
        P64.Dl_Google(Base64Straing,"W_"+str(PicIndex))
        PicIndex+=1

    try:
        elem = driver.find_element_by_class_name("_kvc")
        elem.click()
        time.sleep(loadingtime)
    except:
        pass
print "Ending Process after %d sec" % (loadingtime)
time.sleep(loadingtime)
driver.close()
