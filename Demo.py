import Detect
import Image

a = ''
c = ''



for i in range(10):
    data = Image.open('./Test/0/W_'+str(i)+'.jpeg') 
    a,c = Detect.DetectAPic(data)
    print a,c
print '--'
for i in range(10):
    data = Image.open('./Test/1/W_'+str(i)+'.jpeg') 
    a,c = Detect.DetectAPic(data)
    print a,c
