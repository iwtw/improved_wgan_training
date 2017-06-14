import numpy as np
import scipy.misc
import sys
def batch_mse_psnr(dbatch):
    im1,im2=np.split(dbatch,2)
    mse=((im1-im2)**2).mean(axis=(1,2))
    psnr=np.mean(20*np.log10(255.0/np.sqrt(mse)))
    return np.mean(mse),psnr
def batch_y_psnr(dbatch):
    r,g,b=np.split(dbatch,3,axis=3)
    y=np.squeeze(0.3*r+0.59*g+0.11*b)
    im1,im2=np.split(y,2)
    mse=((im1-im2)**2).mean(axis=(1,2))
    psnr=np.mean(20*np.log10(255.0/np.sqrt(mse)))
    return psnr
def batch_ssim(dbatch):
    im1,im2=np.split(dbatch,2)
    imgsize=im1.shape[1]*im1.shape[2]
    avg1=im1.mean((1,2),keepdims=1)
    avg2=im2.mean((1,2),keepdims=1)
    std1=im1.std((1,2),ddof=1)
    std2=im2.std((1,2),ddof=1)
    cov=((im1-avg1)*(im2-avg2)).mean((1,2))*imgsize/(imgsize-1)
    avg1=np.squeeze(avg1)
    avg2=np.squeeze(avg2)
    k1=0.01
    k2=0.03
    c1=(k1*255)**2
    c2=(k2*255)**2
    c3=c2/2
    return np.mean((2*avg1*avg2+c1)*2*(cov+c3)/(avg1**2+avg2**2+c1)/(std1**2+std2**2+c2))

DATA_PATH="data.test"
data_list  = open(DATA_PATH).read().split('\n')
data_list.pop( len(data_list) - 1  )
#TEST_NUM = len(data_list)
TEST_NUM = 6400
name = sys.argv[1]

dbatch = np.zeros((2*TEST_NUM,112,96,3))
for i in xrange(TEST_NUM):
    dbatch[i] = scipy.misc.imread(data_list[i]  )
for i in xrange(TEST_NUM):
    j = TEST_NUM + i 
    dbatch[j]  = scipy.misc.imread( "/home/wtw/improved_wgan_training/test_output/" + name + "/" + data_list[i].split('/')[-2]+'/'+data_list[i].split('/')[-1])
mse , psnr = batch_mse_psnr(dbatch)
ssim = batch_ssim(dbatch)
print("name %s : mse %.4f , psnr %.4f , ssim %.4f "%(name,mse,psnr,ssim))
