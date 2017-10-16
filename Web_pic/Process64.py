import base64
import requests
import shutil

#For Google search

def Dl_Google(GoogleString,FileName):
    FileType = None
    #print GoogleString
    if str(GoogleString) == "None":
        pass
    elif GoogleString[16:22] == "base64":
        s1 = GoogleString[:22]
        s2 = GoogleString[23:]
        if "jpeg" in s1:
            FileType = "jpeg"
        elif "png" in s1:
            FileType = "png"
        else:
            pass

        Dl(s2,FileName,FileType)
    else:
        Dl_From_Url(GoogleString,FileName)

def Show_Google(GoogleString):
    s1 = GoogleString[:22]
    s2 = GoogleString[23:]
    Show(s2)


##base function
def DlNoticeText(FileName):
    print "DL: "+FileName

def Dl(String64,FileName,FileType):
    DecodeStr = base64.decodestring(String64)
    f = open("image/W/"+FileName+"."+FileType, "w")
    f.write(DecodeStr)
    f.close()
    DlNoticeText(FileName)

def Dl_From_Url(GString,FileName):
    r = requests.get(GString, stream=True)
    print(r.status_code)
    if r.status_code == 200:
        FlieType = "None"
        with open("image/W/"+FileName+".jpeg", 'w') as f:
            r.raw.decode_content = True
            print type(r.raw)
            shutil.copyfileobj(r.raw, f)    
            DlNoticeText(FileName)
def Show(String64):
    DecodeStr = base64.decodestring(String64)
    import PIL.Image
    from io import BytesIO
    img = PIL.Image.open(BytesIO(DecodeStr))
    img.show()
