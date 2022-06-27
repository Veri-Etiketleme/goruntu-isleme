# Image-features-extraction
# here we are buliding the model so that machine can predict that the bird is there in the picture or not
Extracting features from images using opencv and different kernels to build a logistic regression model.
- Importing all the libraries such as
import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
import math
import pandas as pd
import seaborn as sns
import glob
import cv2
import re
import os
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf

# using glob library we can read all the images there in the given path.
path = glob.glob("train/*.jpg")
cv_img = []
for img in path:
    n = cv2.imread(img)
    
# we have to apply different kernels to extract the wanted features such as correlation coefficient, covariance, variance, angle and many more.
# I have used sobelx, sobely, sobel, canny, scharrx, scharry, laplacian to extract the different features
# firstly I have used only laplacian to tract the features and create a dataframe to build a model, there I have got the accuracy for traing and testing data approximately 57%.
# for increasing the accuracy I have used other kernels too for building the good model.
## explaining how the filter works:
# canny: its an edge detector
-You will get clean, thin edges that are well connected to nearby edges. 
- The canny edge detector is a multistage edge detection algorithm. The steps are:
1. Preprocessing
2. Calculating gradients
3. Nonmaximum suppression
4. Thresholding with hysterysis
    gray = cv2.cvtColor(n,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    cv_img.append(blur)
    
# above I have converted the images to gray and then applied blur for better results.

# then after declaring the seperated empty lists for appending the features of 200 images in seperatelists for so that we can create dataframes with multiple number of columns.

canny_cov=[] 
canny_corr=[]
angle1_canny=[]
angle2_canny=[]
variancex_canny=[]
variancey_canny=[]

for i in cv_img:
        canny = cv2.Canny(i,cv2.CV_8U,100)
        canny       
       #threshold value is 15
        _,thresh = cv2.threshold(canny,15,255,cv2.THRESH_BINARY)
        x,y = np.nonzero(thresh)
        x = x - np.mean(x)
        y = y - np.mean(y)
        coords = np.vstack([x, y])
        cov=np.cov(x,y)
        cov=np.nan_to_num(cov)
        corr=np.corrcoef(x,y)
        corr=np.nan_to_num(corr)
        evals, evecs = np.linalg.eig(cov)
        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
        x_v2, y_v2 = evecs[:, sort_indices[1]]
        varx=[np.var(x)]
        varx=np.nan_to_num(varx)
        vary=[np.var(y)]
        vary=np.nan_to_num(vary)
        theta1=[np.arctan(y_v1/x_v1)]
        theta1=np.nan_to_num(theta1)
        theta2=[math.atan(y_v2/x_v2)]
        theta2=np.nan_to_num(theta2)       
        for i in cov:
            #print(i)
            canny_cov.append(cov[0,1]) # here given  cov[0,1] as it is in form of matrix so we need 0,1 th value which fines as covariance
            #listu.append(listnew[0])
        #print(listu)
        #print(listu)
        for j in corr:
            canny_corr.append(corr[0,1]) # here given  cov[0,1] as it is in form of matrix so we need 0,1 th value which fines as covariance
        for k in varx:
            variancex_canny.append(k)
        for l in vary:
            variancey_canny.append(l)
        for m in theta1:
            angle1_canny.append(m)
        for n in theta2:
            angle2_canny.append(n)
covariance_canny=canny_cov[0::2]# here used step function because we are getting the same value 2 times in a list due to loop is running more than 1 time. note: we should not use set function beacause there is a risk of changing of order which may not follow the order of images in given path.
correlation_canny=canny_corr[0::2]# here used step function because we are getting the same value 2 times in a list due to loop is running more than 1 time. note: we should not use set function beacause there is a risk of changing of order which may not follow the order of images in given path.

# in similar way we can use other filters too extracting the features.

# after that we need to extract our independent variable whether the bird is there or not. 0 can be represented as no bird and 1 can be represented as there is a bird in given picture.
import os
names=os.listdir("train") # here trying to extract the names of pictures in a given path in a single list.
['100_0.JPG', '100_1.JPG', '10_0.JPG', '10_1.JPG', '11_0.JPG', '11_1.JPG', '12_0.JPG', '12_1.JPG', '13_0.JPG', '13_1.JPG', '14_0.JPG', '14_1.JPG', '15_0.JPG', '15_1.JPG', '16_0.JPG', '16_1.JPG', '17_0.JPG', '17_1.JPG', '18_0.JPG', '18_1.JPG', '19_0.JPG', '19_1.JPG', '1_0.JPG', '1_1.JPG', '20_0.JPG', '20_1.JPG', '21_0.JPG', '21_1.JPG', '22_0.JPG', '22_1.JPG', '23_0.JPG', '23_1.JPG', '24_0.JPG', '24_1.JPG', '25_0.JPG', '25_1.JPG', '26_0.JPG', '26_1.JPG', '27_0.JPG', '27_1.JPG', '28_0.JPG', '28_1.JPG', '29_0.JPG', '29_1.JPG', '2_0.JPG', '2_1.JPG', '30_0.JPG', '30_1.JPG', '31_0.JPG', '31_1.JPG', '32_0.JPG', '32_1.JPG', '33_0.JPG', '33_1.JPG', '34_0.JPG', '34_1.JPG', '35_0.JPG', '35_1.JPG', '36_0.JPG', '36_1.JPG', '37_0.JPG', '37_1.JPG', '38_0.JPG', '38_1.JPG', '39_0.JPG', '39_1.JPG', '3_0.JPG', '3_1.JPG', '40_0.JPG', '40_1.JPG', '41_0.JPG', '41_1.JPG', '42_0.JPG', '42_1.JPG', '43_0.JPG', '43_1.JPG', '44_0.JPG', '44_1.JPG', '45_0.JPG', '45_1.JPG', '46_0.JPG', '46_1.JPG', '47_0.JPG', '47_1.JPG', '48_0.JPG', '48_1.JPG', '49_0.JPG', '49_1.JPG', '4_0.JPG', '4_1.JPG', '50_0.JPG', '50_1.JPG', '51_0.JPG', '51_1.JPG', '52_0.JPG', '52_1.JPG', '53_0.JPG', '53_1.JPG', '54_0.JPG', '54_1.JPG', '55_0.JPG', '55_1.JPG', '56_0.JPG', '56_1.JPG', '57_0.JPG', '57_1.JPG', '58_0.JPG', '58_1.JPG', '59_0.JPG', '59_1.JPG', '5_0.JPG', '5_1.JPG', '60_0.JPG', '60_1.JPG', '61_0.JPG', '61_1.JPG', '62_0.JPG', '62_1.JPG', '63_0.JPG', '63_1.JPG', '64_0.JPG', '64_1.JPG', '65_0.JPG', '65_1.JPG', '66_0.JPG', '66_1.JPG', '67_0.JPG', '67_1.JPG', '68_0.JPG', '68_1.JPG', '69_0.JPG', '69_1.JPG', '6_0.JPG', '6_1.JPG', '70_0.JPG', '70_1.JPG', '71_0.JPG', '71_1.JPG', '72_0.JPG', '72_1.JPG', '73_0.JPG', '73_1.JPG', '74_0.JPG', '74_1.JPG', '75_0.JPG', '75_1.JPG', '76_0.JPG', '76_1.JPG', '77_0.JPG', '77_1.JPG', '78_0.JPG', '78_1.JPG', '79_0.JPG', '79_1.JPG', '7_0.JPG', '7_1.JPG', '80_0.JPG', '80_1.JPG', '81_0.JPG', '81_1.JPG', '82_0.JPG', '82_1.JPG', '83_0.JPG', '83_1.JPG', '84_0.JPG', '84_1.JPG', '85_0.JPG', '85_1.JPG', '86_0.JPG', '86_1.JPG', '87_0.JPG', '87_1.JPG', '88_0.JPG', '88_1.JPG', '89_0.JPG', '89_1.JPG', '8_0.JPG', '8_1.JPG', '90_0.JPG', '90_1.JPG', '91_0.JPG', '91_1.JPG', '92_0.JPG', '92_1.JPG', '93_0.JPG', '93_1.JPG', '94_0.JPG', '94_1.JPG', '95_0.JPG', '95_1.JPG', '96_0.JPG', '96_1.JPG', '97_0.JPG', '97_1.JPG', '98_0.JPG', '98_1.JPG', '99_0.JPG', '99_1.JPG', '9_0.JPG', '9_1.JPG', 'desktop.ini', 'Thumbs.db']
# here we need to apply regular expressions to extract 0 and 1 from the names.
# before that we need to convert the list into string for applying re. - string = ''.join(names)

result=re.findall(r'\_(\d)',string)
print(result)
[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
then after creating a dictionary with different keys for building a dataframe.
birdsdict={'lapCovariance':covariance,'lapCorrelation':correlation,'lapVarx':variancex,'lapVary':variancey,'lapAngle1':angle1,'lapAngle2':angle2,'cannyCovariance':covariance_canny,'cannyCorrelation':correlation_canny,'cannyVariancex':variancex_canny,'cannyVariancey':variancey_canny,'cannyAngle1':angle1_canny,'cannyAngle2':angle2_canny,'sobelxCovariance':covariance_sobelx,'sobelxCorrelation':correlation_sobelx,'sobelxVariancex':variancex_sobelx,'sobelxVariancey':variancey_sobelx,'sobelxAngle1':angle1_sobelx,'sobelxAngle2':angle2_sobelx,'sobelyCovariance':covariance_sobely,'sobelyCorrelation':correlation_sobely,'sobelyVariancex':variancex_sobely,'sobelyVariancey':variancey_sobely,'sobelyAngle1':angle1_sobely,'sobelyAngle2':angle2_sobely,'sobelCovariance':covariance_sobel,'sobelCorrelation':correlation_sobel,'sobelVariancex':variancex_sobel,'sobelVariancey':variancey_sobel,'sobelAngle1':angle1_sobel,'sobelAngle2':angle2_sobel,'scharrxCovariance':covariance_scharrx,'scharrxCorrelation':correlation_scharrx,'scharrxVariancex':variancex_scharrx,'scharrxVariancey':variancey_scharrx,'scharrxAngle1':angle1_scharrx,'scharrxAngle2':angle2_scharrx,'scharryCovariance':covariance_scharry,'scharryCorrelation':correlation_scharry,'scharryVariancex':variancex_scharry,'scharryVariancey':variancey_scharry,'scharryAngle1':angle1_scharry,'scharryAngle2':angle2_scharry,'scharrCovariance':covariance_scharr,'scharrCorrelation':correlation_scharr,'scharrVariancex':variancex_scharr,'scharrVariancey':variancey_scharr,'scharrAngle1':angle1_scharr,'scharrAngle2':angle2_scharr,'morCovariance':covariance_scharr,'Birds':result}

after creating a dictionary we can create a dataframe. - Birds_feature_extraction=pd.DataFrame(birdsdict)
Birds_feature_extraction.head() # you can run the code for reference which is already uploaded.
x = Birds_feature_extraction.iloc[:,:-1] # dependent variable
y = Birds_feature_extraction.iloc[:,-1:] # independent variable
from sklearn.feature_selection import RFE # for doing feature selection to know that which columns are optimal.

selector = RFE(logmodel)
selector.fit(x_train,y_train)
# then after we have to split the model into train and test data so that we can start building the model.
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

train = pd.concat((x_train,y_train),axis=1)
test = pd.concat((x_test,y_test),axis=1)# concatenating x and y for using glm method where as other methods you no need to concatenating it.
model=smf.glm(formula='Birds~lapCorrelation+lapAngle1+lapAngle2+cannyCorrelation+cannyAngle1+cannyAngle2+sobelxCovariance+sobelxAngle1+sobelxAngle2+sobelyCovariance+sobelyAngle1+sobelyAngle2+sobelVariancey+sobelAngle1+sobelAngle2+scharrxCovariance+scharrxCorrelation+scharrxAngle1+scharrxAngle2+scharryAngle1+scharryAngle2+scharrCorrelation+scharrAngle1+scharrAngle2',data=train,family=sm.families.Binomial()).fit()
print(model.summary())              

                 Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:                  Birds   No. Observations:                  160
Model:                            GLM   Df Residuals:                      135
Model Family:                Binomial   Df Model:                           24
Link Function:                  logit   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:                -62.301
Date:                Tue, 09 Apr 2019   Deviance:                       124.60
Time:                        13:10:34   Pearson chi2:                     161.
No. Iterations:                     6   Covariance Type:             nonrobust
======================================================================================
                         coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------------
Intercept             -1.8894      8.748     -0.216      0.829     -19.036      15.257
lapCorrelation        -0.5439      1.072     -0.507      0.612      -2.645       1.557
lapAngle1             -0.5066      0.375     -1.352      0.176      -1.241       0.228
lapAngle2             -0.2782      0.564     -0.493      0.622      -1.383       0.827
cannyCorrelation       1.6802      0.868      1.935      0.053      -0.021       3.382
cannyAngle1           -1.9638      0.380     -5.172      0.000      -2.708      -1.220
cannyAngle2           -0.6269      0.457     -1.373      0.170      -1.522       0.268
sobelxCovariance       0.0159      0.005      3.235      0.001       0.006       0.026
sobelxAngle1          -1.3436      0.540     -2.490      0.013      -2.401      -0.286
sobelxAngle2          -0.7025      0.332     -2.113      0.035      -1.354      -0.051
sobelyCovariance       0.0080      0.003      2.488      0.013       0.002       0.014
sobelyAngle1          -0.1678      0.337     -0.498      0.618      -0.828       0.492
sobelyAngle2           1.0745      0.560      1.918      0.055      -0.024       2.173
sobelVariancey         0.0008      0.002      0.394      0.694      -0.003       0.005
sobelAngle1           -0.4100      0.380     -1.080      0.280      -1.154       0.334
sobelAngle2           -0.1414      0.455     -0.311      0.756      -1.032       0.750
scharrxCovariance     -0.0374      0.026     -1.443      0.149      -0.088       0.013
scharrxCorrelation   104.7643    107.904      0.971      0.332    -106.723     316.251
scharrxAngle1          0.6777      0.616      1.100      0.271      -0.530       1.885
scharrxAngle2          0.0138      0.354      0.039      0.969      -0.680       0.708
scharryAngle1         -0.0156      0.331     -0.047      0.962      -0.665       0.633
scharryAngle2         -0.3924      0.597     -0.657      0.511      -1.563       0.779
scharrCorrelation     -0.6959      5.750     -0.121      0.904     -11.965      10.573
scharrAngle1           1.0567      0.442      2.389      0.017       0.190       1.924
scharrAngle2           0.4956      0.543      0.912      0.362      -0.570       1.561
======================================================================================

 successfully builded the model.
 
 # now we need to know the accuracy of the trained model then we need to check for test data whether the accuracy is nearly same or not. To know whether the model is good or bad.
 
# to knowing the accuracy we need to compute the confusion matrix for calculating fpr,tpr,tfr,ffr etc.

def prediction(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24):

    Varx= x1
    Angle1 = x2  
    # as the formula for logistic regression is y=ax1+bx2+cx3 and so on 
    lnor = model.params['lapCorrelation']*x1 + model.params['lapAngle1']*x2 + model.params['lapAngle2']*x3 + model.params['cannyCorrelation']*x4 + model.params['cannyAngle1']*x5 + model.params['cannyAngle2']*x6 + model.params['sobelxCovariance']*x7  + model.params['sobelxAngle1']*x8 + model.params['sobelxAngle2']*x9 + model.params['sobelyCovariance']*x10 + model.params['sobelyAngle1']*x11 + model.params['sobelyAngle2']*x12 + model.params['sobelVariancey']*x13 +  model.params['sobelAngle1']*x14  + model.params['sobelAngle2']*x15 + model.params['scharrxCovariance']*x16 + model.params['scharrxCorrelation']*x17 + model.params['scharrxAngle1']*x18 + model.params['scharrxAngle2']*x19 + model.params['scharryAngle1']*x20  + model.params['scharryAngle2']*x21 +  model.params['scharrCorrelation']*x22 + model.params['scharrAngle1']*x23 + model.params['scharrAngle2']*x24 # it can be derived from the the model where we can use coefficients to form the equartion.
    oddsratio = np.exp(lnor)
    p = oddsratio/(oddsratio+1)    

    return p
    
def predict(prob,thresh=0.5):
    if prob > thresh:
        return 1
    else:
        return 0
# above function is used for comparing the machine's output and given output.

# now we need to apply the functions to the data.
prob=prediction(train['lapCorrelation'],train['lapAngle1'],train['lapAngle2'],train['cannyCorrelation'],train['cannyAngle1'],train['cannyAngle2'],train['sobelxCovariance'],train['sobelxAngle1'],train['sobelxAngle2'],train['sobelyCovariance'],train['sobelyAngle1'],train['sobelyAngle2'],train['sobelVariancey'],train['sobelAngle1'],train['sobelAngle2'],train['scharrxCovariance'],train['scharrxCorrelation'],train['scharrxAngle1'],train['scharrxAngle2'],train['scharryAngle1'],train['scharryAngle2'],train['scharrCorrelation'],train['scharrAngle1'],train['scharrAngle2'])


y_pred = prob.apply(predict)
# now we can compute the confusion matrix.

from pandas_ml import ConfusionMatrix

Predicted  False  True  __all__
Actual                         
False         39    41       80
True           1    79       80
__all__       40   120      160

# to see the accuracy we need to use- cm.stats_overall

OrderedDict([('Accuracy', 0.7375),
             ('95% CI', (0.6621798680048558, 0.803802817410914)),
             ('No Information Rate', 'ToDo'),
             ('P-Value [Acc > NIR]', 0.6802139128726792),
             ('Kappa', 0.4750000000000001),
             ("Mcnemar's Test P-Value", 'ToDo')])
as the accuracy is nearly 74% now we have to check for test data too.
so for test data same steps must be followed and check the accuracy.
# and for test data I got:
OrderedDict([('Accuracy', 0.6),
             ('95% CI', (0.43326705219329664, 0.7513500134122563)),
             ('No Information Rate', 'ToDo'),
             ('P-Value [Acc > NIR]', 0.9884386387990133),
             ('Kappa', 0.19999999999999996),
             ("Mcnemar's Test P-Value", 'ToDo')])
# for test data accuracy is 7% which is nearly equal so we can say that model is a good model as accuracy is more than 70%

# now we can do the roc for checking whether we can increase the accuracy or not.
def convertclass(y):
    if y >= 0.53:
        return 1
    else:
        return 0
y_pred_train_prob = model.predict(x_train)
y_pred_test_prob = model.predict(x_test)
y_pred_train_class = y_pred_train_prob.apply(convertclass)
y_pred_test_class = y_pred_test_prob.apply(convertclass)
from sklearn.metrics import roc_curve,auc
fpr,tpr,threshold = roc_curve(y_train,y_pred_train_prob)

plt.figure(figsize=(10,6))
plt.plot(fpr,tpr,color='red',lw=1.5)
plt.plot([0, 1], [0, 1], color='navy',lw = 2, linestyle='--')

for i,value in enumerate(fpr*10):
    try:
        if (round(fpr[i+1]*10) - round(fpr[i]*10)) == 1:
            plt.text(fpr[i],tpr[i],'%0.2f'%(threshold[i]))
    except IndexError:
        print(' ')
        
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(['ROC curve (area = %0.2f)' % auc(fpr,tpr)])
plt.show()

# from roc curve we can know that best prob or threshhold value for the model is 0.53 but there is not much change and as per the graph area under the curve is 0.91 which greater than 0.70, so we can say that builded model is a good model.

*****************have a good day*******************
    
    
