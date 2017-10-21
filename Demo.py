import Detect
import Image
import os
a = ''
c = ''

'''

for i in range(300):
    data = Image.open('./Test/0/W_'+str(i)+'.jpeg') 
    a,c = Detect.DetectAPic(data)
    #print a,c
    if a == 1:count+=1
'''

count =0
NumPic = 0.0

for dirPath, dirNames, fileNames in os.walk('./Test/1/'):
    NumPic = len(fileNames)
    for f in fileNames:#os.path.join(dirPath, f)
        data = Image.open(os.path.join(dirPath, f)) 
        a,c = Detect.DetectAPic(data)
        print 'pic src:',dirPath,f,'prediction:',a,'confident:',c,'%'
        if a == 1:
            count+=1

for dirPath, dirNames, fileNames in os.walk('./Test/0/'):
    NumPic += len(fileNames)
    for f in fileNames:#os.path.join(dirPath, f)
        data = Image.open(os.path.join(dirPath, f)) 
        a,c = Detect.DetectAPic(data)
        print 'pic src:',dirPath,f,'prediction:',a,'confident:',c,'%'
        if a == 0:
            count+=1

print 'successful:',float(count)/float(NumPic),'%'
'''
print '--'
for i in range(10):
    data = Image.open('./Test/1/W_'+str(i)+'.jpeg') 
    a,c = Detect.DetectAPic(data)
    print a,c
'''
