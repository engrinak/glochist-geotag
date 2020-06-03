import pandas as pd
import numpy as np
import json
from exif import Image
from datetime import datetime
import os
from GPSPhoto import gpsphoto


with open ('History.json') as f:
    d = json.load(f)

data = pd.json_normalize(d['locations'])

def timefmt(x):
    return datetime.fromtimestamp(int(x))

output = pd.DataFrame()
output['timestampMs'] = data['timestampMs'].astype('float') / 1000
output['timestamp'] = output['timestampMs'].apply(timefmt)
output['latitudeE7'] = data['latitudeE7'].astype('float') / 10000000
output['longitudeE7'] = data['longitudeE7'].astype('float') / 10000000
output['altitude'] = data['altitude'].astype('double')

root = ".\\img"
file_list = []

for path, subdirs, files in os.walk(root):
     for name in files:
        #print(name)
        file_list.append(os.path.join(path, name))

for file in file_list:
    with open(file, 'rb') as image_file:
        my_image = Image(image_file)
    
    dt = datetime.strptime(my_image.datetime_original, '%Y:%m:%d %H:%M:%S')
    
    #get_loc requires values to be sorted and without duplicates
    output = output.sort_values(by=['timestamp'], axis=0)
    output = output.drop_duplicates(subset=['timestamp'], keep='first')

    idx = pd.Index(output['timestamp'])
    n = idx.get_loc(dt, method='nearest')
    lat = output.iloc[n]['latitudeE7']
    lon = output.iloc[n]['longitudeE7']
    altd = output.iloc[n]['altitude']
    
    #Filter out some bad values in the altitude after getting an error.   
    #This would have been better done in the source data but for now this works.
    if altd != np.nan:
        if altd == 'NaN':
            altd = 0
        else:
            altd = int(np.int_(altd))
            if not 0 < altd < 5000:  #sometimes the altitude in google is weird
                altd = 0          
    else:
        altd = 0
            
    photo = gpsphoto.GPSPhoto(file)
    info = gpsphoto.GPSInfo((lat, lon), alt=altd, timeStamp=dt)
    photo.modGPSData(info, file)
    
    print('Modified image: ' + file + ' with lat=' + str(lat) + ' long=' + str(lon) + ' alt=' + str(altd))