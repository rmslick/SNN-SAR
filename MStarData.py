#https://blog.katastros.com/a?ID=01650-3e933b21-f810-42d1-a0a9-d0a2e1ba5c37

#./mstar2jpeg -i /home/rmslick/SpykeTorch/MSTAR/TARGETS/TRAIN/17_DEG/BMP2/SN_9563/HB03993.000 -o test.jpeg
import os
#https://www.sdms.afrl.af.mil/index.php?collection=mstar&page=targets
#https://www.sdms.afrl.af.mil/index.php?collection=tools_mstar&page=mstar2jpeg
import glob
#os.chdir('mstar2jpeg')
import cv2
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
#https://stackoverflow.com/questions/39785970/speckle-lee-filter-in-python
import findpeaks

# filters parameters
# window size
winsize = 15
# damping factor for frost
k_value1 = 2.0
# damping factor for lee enhanced
k_value2 = 1.0
# coefficient of variation of noise
cu_value = 0.25
# coefficient of variation for lee enhanced of noise
cu_lee_enhanced = 0.523
# max coefficient of variation for lee enhanced
cmax_value = 1.73


for t in glob.glob('MSTAR/TARGETS/TRAIN/17_DEG/BMP2/SN_9563/*'):
    #print(t)
    fName = ((t.split('/')[-1])).split('.')[0] +'.jpeg'
    #print(fName)
    os.system('mstarutils/mstar2jpeg/mstar2jpeg -i '+t+' -o mstardataset/train/SN_9563/'+fName)
    # load in for 64x64
    img = cv2.imread('mstardataset/train/SN_9563/'+fName)
    #print(img)
    up_points = (64,64)
    resized_up = cv2.resize(img, up_points, interpolation= cv2.INTER_LINEAR)
    resized_up = findpeaks.stats.scale(resized_up).copy()
    #resized_up = cv2.bilateralFilter(resized_up, 15, 75, 75)
    resized_up = findpeaks.stats.togray(resized_up).copy()
    resized_up = findpeaks.frost_filter(resized_up.copy(), damping_factor=k_value1, win_size=winsize,)
    cv2.imwrite( 'mstardataset64/train/SN_9563/'+fName, resized_up)

for t in glob.glob('MSTAR/TARGETS/TEST/15_DEG/BMP2/SN_9563/*'):
    fName = ((t.split('/')[-1])).split('.')[0] +'.jpeg'
    #print(fName)
    os.system('mstarutils/mstar2jpeg/mstar2jpeg -i '+t+' -o mstardataset/test/SN_9563/'+fName)
    # load in for 64x64
    img = cv2.imread('mstardataset/test/SN_9563/'+fName)
    up_points = (64,64)
    resized_up = cv2.resize(img, up_points, interpolation= cv2.INTER_LINEAR)
    #resized_up = cv2.bilateralFilter(resized_up, 15, 75, 75)
    resized_up = findpeaks.stats.scale(resized_up).copy()
    resized_up = findpeaks.stats.togray(resized_up).copy()
    resized_up = findpeaks.frost_filter(resized_up.copy(), damping_factor=k_value1, win_size=winsize)

    cv2.imwrite( 'mstardataset64/test/SN_9563/'+fName, resized_up)

for t in glob.glob('MSTAR/TARGETS/TRAIN/17_DEG/BTR70/SN_C71/*'):
    #print(t)
    fName = ((t.split('/')[-1])).split('.')[0] +'.jpeg'
    #print(fName)
    os.system('mstarutils/mstar2jpeg/mstar2jpeg -i '+t+' -o mstardataset/train/SN_C71/'+fName)
        # load in for 64x64
    img = cv2.imread('mstardataset/train/SN_C71/'+fName)
    up_points = (64,64)
    resized_up = cv2.resize(img, up_points, interpolation= cv2.INTER_LINEAR)
    resized_up = findpeaks.stats.scale(resized_up).copy()
    #resized_up = cv2.bilateralFilter(resized_up, 15, 75, 75)
    resized_up = findpeaks.stats.togray(resized_up).copy()
    resized_up = findpeaks.frost_filter(resized_up.copy(), damping_factor=k_value1, win_size=winsize)

    cv2.imwrite( 'mstardataset64/train/SN_C71/'+fName, resized_up)

for t in glob.glob('MSTAR/TARGETS/TEST/15_DEG/BTR70/SN_C71/*'):
    fName = ((t.split('/')[-1])).split('.')[0] +'.jpeg'
    #print(fName)
    os.system('mstarutils/mstar2jpeg/mstar2jpeg -i '+t+' -o mstardataset/test/SN_C71/'+fName)
        # load in for 64x64
    img = cv2.imread('mstardataset/test/SN_C71/'+fName)
    up_points = (64,64)
    resized_up = cv2.resize(img, up_points, interpolation= cv2.INTER_LINEAR)
    resized_up = findpeaks.stats.scale(resized_up).copy()
    #resized_up = cv2.bilateralFilter(resized_up, 15, 75, 75).copy()
    resized_up = findpeaks.stats.togray(resized_up).copy()
    resized_up = findpeaks.frost_filter(resized_up.copy(), damping_factor=k_value1, win_size=winsize)

    cv2.imwrite( 'mstardataset64/test/SN_C71/'+fName, resized_up)
    