import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import time
import scipy.ndimage as ndimg
from sklearn.cluster import KMeans

img = cv2.imread("texture.png")
NROT = 6
NPER = 8
NFILT = NROT*NPER
FILTSIZE = 49
NCLUSTERS = 4
TEXELSIZE = 4

pathName = "C:\\Data\\Software"

def gaussian1d(sigma, mean, x, ord):
    x = np.array(x)
    x_ = x - mean
    var = sigma**2

    # Gaussian Function
    g1 = (1/np.sqrt(2*np.pi*var))*(np.exp((-1*x_*x_)/(2*var)))

    if ord == 0:
        g = g1
        return g
    elif ord == 1:
        g = -g1*((x_)/(var))
        return g
    else:
        g = g1*(((x_*x_) - var)/(var**2))
        return g

def gaussian2d(sup, scales):
    var = scales * scales
    shape = (sup,sup)
    n,m = [(i - 1)/2 for i in shape]
    x,y = np.ogrid[-m:m+1,-n:n+1]
    g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
    return g

def log2d(sup, scales):
    var = scales * scales
    shape = (sup,sup)
    n,m = [(i - 1)/2 for i in shape]
    x,y = np.ogrid[-m:m+1,-n:n+1]
    g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
    h = g*((x*x + y*y) - var)/(var**2)
    return h

def makefilter(scale, phasex, phasey, pts, sup):

    gx = gaussian1d(3*scale, 0, pts[0,...], phasex)
    gy = gaussian1d(scale,   0, pts[1,...], phasey)

    image = gx*gy

    image = np.reshape(image,(sup,sup))
    return image

def makeLMfilters():
    sup     = 49
    scalex  = np.sqrt(2) * np.array([1,2,3])
    norient = 6
    nrotinv = 12

    nbar  = len(scalex)*norient
    nedge = len(scalex)*norient
    nf    = nbar+nedge+nrotinv
    F     = np.zeros([sup,sup,nf])
    hsup  = (sup - 1)/2

    x = [np.arange(-hsup,hsup+1)]
    y = [np.arange(-hsup,hsup+1)]

    [x,y] = np.meshgrid(x,y)

    orgpts = [x.flatten(), y.flatten()]
    orgpts = np.array(orgpts)

    count = 0
    for scale in range(len(scalex)):
        for orient in range(norient):
            angle = (np.pi * orient)/norient
            c = np.cos(angle)
            s = np.sin(angle)
            rotpts = [[c+0,-s+0],[s+0,c+0]]
            rotpts = np.array(rotpts)
            rotpts = np.dot(rotpts,orgpts)
            F[:,:,count] = makefilter(scalex[scale], 0, 1, rotpts, sup)
            F[:,:,count+nedge] = makefilter(scalex[scale], 0, 2, rotpts, sup)
            count = count + 1

    count = nbar+nedge
    scales = np.sqrt(2) * np.array([1,2,3,4])

    for i in range(len(scales)):
        F[:,:,count]   = gaussian2d(sup, scales[i])
        count = count + 1

    for i in range(len(scales)):
        F[:,:,count] = log2d(sup, scales[i])
        count = count + 1

    for i in range(len(scales)):
        F[:,:,count] = log2d(sup, 3*scales[i])
        count = count + 1

    return F

F = makeLMfilters()
print(F.shape)
def saveFilters(img):
    (height, width, depth) = img.shape
    count = 0
    for row in range(NPER):
        for col in range(NROT):
            tempImg = img[:, :, count]
            filename = "Filters\\LM_" + str(row) + "_" + str(col)
            normedFilter = normImg(tempImg)
            a = pathName + "\\" + filename + ".png"
            cv2.imwrite(a, normedFilter)
            count = count + 1
    return

def normImg(img):
    tempImg = np.zeros_like(img)
    tempImg = (cv2.normalize(img, tempImg, 0.0, 127.0, cv2.NORM_MINMAX))
    res = (tempImg+128.0).astype(np.uint8)
    return res

def makeMosaic(img):
    (height, width, depth) = img.shape
    res = np.zeros((height*8, width*6), np.float64)
    count = 0
    for row in range(8):
        for col in range(6):
            res[row*height:(row+1)*height, col*width:(col+1)*width] = \
            normImg(img[:, :, count])
            count = count + 1
    cv2.imwrite("LM_Filters.png", res)

def applyLMFilters(img, filters):
    img = np.asarray(img)
    img = img.astype('float32')
    mean, std = img.mean(), img.std()
    img = (img - mean)/std
    row, col = img.shape
    responses = np.zeros((row,col,NFILT))
    for i in range(NFILT):
        b = filters[:,:,i]
        #b = normalize(b, axis = 1, norm = 'l1')
        res = cv2.filter2D(img, -1, b)
        #res = np.mean(res)
        responses[0:row, 0:col, i] = res
        #plt.imshow(res, cmap = 'gray')
        #plt.show()
    return responses

def formTexels(R, sz):
    h, w, c = R.shape
    l = int(h/sz)
    k = int(w/sz)
    i, j = 0, 0
    texels = np.zeros((h,w,NFILT))
    for z in range(NFILT):
        b = R[:,:,z]
        c = texels[:,:,z]
        while i < h:
            if i + sz == h:
                break
            while j < w:
                if j + sz =
                = w:
                    break
                window = b[i:i + sz, j:j + sz]
                mean = np.mean(window)
                c[i:i + sz, j:j + sz] = mean
                j = j + sz
                #print(j)
            i = i + sz
            j = 0
        i = 0
        print(z)
        '''print(len(c))
        c = np.asarray(c)
        c = np.reshape(c, (l+1,k+1))
        print(c.shape)
        cv2.imshow(str(z), c)'''
        
        texels[0:h, 0:w, z] = c

    return texels

        
        
def saveImage(img, name):
    cv2.imwrite(pathName + name + ".png", img)
    return

saveFilters(F)
print(F.dtype)
fig = plt.figure(figsize=(8,6))
for i in range(0,48):
    plt.axis('off')
    a = F[:,:,i]
    a = np.asarray(a)
    fig.add_subplot(8, 6, i+1)
    plt.imshow(a, cmap = 'gray')
plt.axis('off')
plt.savefig('LM_Filters1.png')
plt.show()

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
a = applyLMFilters(img, F)
print(a.shape)
#b = a[:,:,3]
#print(a[5,5,:])
#cv2.imshow('jjjj', b)
c = formTexels(a, TEXELSIZE)
cv2.imshow('jjjj', c[:,:,3])

h, w, d = c.shape
c = np.reshape(c,(h*w,d))
print(c.shape)

kmeans = KMeans(n_clusters = NCLUSTERS, random_state=0).fit(c)
print(kmeans.labels_.shape)
print(kmeans.cluster_centers_)
print(np.unique(kmeans.labels_))
e = np.reshape(kmeans.labels_, (h, w))
e = e/3
cv2.imshow('eeee', e)
print(e[0,0])
e = 255*e
e = np.array(e, dtype = np.uint8)
f = cv2.applyColorMap(e, cv2.COLORMAP_JET)
cv2.imshow('ffff', f)
cv2.imwrite('Segment_AH.png', f)
#print(c[5,5,:])

#print(a[0,25,:])

#response_vector = np.concatenate(responses)
#print(response_vector.shape)

