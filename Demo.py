import Detect
import Image

a = ''
c = ''

data = Image.open('./TraingData/0/1.jpg') 
a,c = Detect.DetectAPic(data)
print a,c

data = Image.open('./TraingData/0/6.jpg') 
a,c = Detect.DetectAPic(data)
print a,c

data = Image.open('./TraingData/0/11.jpg') 
a,c = Detect.DetectAPic(data)
print a,c

data = Image.open('./TraingData/0/16.jpg') 
a,c = Detect.DetectAPic(data)
print a,c

data = Image.open('./TraingData/0/21.jpg') 
a,c = Detect.DetectAPic(data)
print a,c
