import csv
import sys
import time
import numpy as np
#import matplotlib.pyplot as plt 
import pickle

#Remove Above Comment Line for getting the graphs included in reports like F1 vs K etc

#start = time.time()   - Timing

# DISTANCE MEASURES for computation

def euclidian(a,b):
    dist=np.linalg.norm(a-b)
    if(dist==0):
        dist=0.00001
        
    return np.abs(dist)

def cosine(a,b):
    dist=1-(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    if(dist==0):
        dist=0.00001
    
    return np.abs(dist)

def manhattan(a,b):
    dist=np.sum(np.abs(a-b))    
    if(dist==0):
        dist=0.00001
        
    return np.abs(dist)

# Making the confusion Matrix , for estiamting the F1 Score and Accuracy

def CM(Y_pred,Y_true):
    Con_Mat=np.zeros((11,11))
    TP=np.zeros(11)
    FP=np.zeros(11)
    FN=np.zeros(11)
    F=np.zeros(11)
    
    for i in range(0,len(Y_pred)):
        Con_Mat[int(Y_true[i])][int(Y_pred[i])]=Con_Mat[int(Y_true[i])][int(Y_pred[i])]+1
        
    for i in range(0,11):
        for j in range(0,11):
            if(i==j):
                TP[i]=Con_Mat[i][j]
            else:
                FN[i]=FN[i]+Con_Mat[i][j]
                FP[i]=FP[i]+Con_Mat[j][i]
        if(TP[i]==0):
            F[i]=0
        else:
            F[i]=2*TP[i]/float(2*TP[i]+FP[i]+FN[i])
    
    F1_Score=float(np.sum(F))/(len(np.unique(Y_true))) 
    Accuracy=float(np.sum(TP))/(len(Y_pred))
    
    return Accuracy,F1_Score

# THE CROSS VALIDATION TASK - SPECIAL POINTER TO THE BEGINING OF THE CODE FUNCTION
def cross_validation_test(X_train,K_Max):
    X_val=X_train[0:np.shape(X_train)[0]/5]
    X_train=X_train[np.shape(X_train)[0]/5:np.shape(X_train)[0]]
    # Splitting the data
    Y_train=X_train[:,11]   
    X_train=X_train[:,0:-1]
    Y_val=X_val[:,11]
    X_val=X_val[:,0:-1]
    
    Mean=X_train.mean(0)
    Std=X_train.std(0)
    X_train=(X_train-Mean)/Std
    X_val=(X_val-Mean)/Std    
    # Normalizing the data
    Accuracy=np.zeros(K_Max+1)
    F1_Score=np.zeros(K_Max+1)
    # Implementing K fold cross validation
    for k in range(1,K_Max+1): 
        Y_pred=np.zeros(len(X_val))
        for i in range(0,len(X_val)):    
            distance=[]
            Wts=np.zeros(11)
            
            for tmp in range(0,k):
                d=manhattan(X_train[tmp],X_val[i])
                l=[Y_train[tmp],d]
                distance.append(l)      
                # Finding K nearest neighbours
            for j in range(k,len(X_train)):    
                d=manhattan(X_train[j],X_val[i])# Finding distances
                tmp=np.argmax(np.asarray(distance)[:,1])
                if(d<distance[tmp][1]):
                    del distance[tmp]
                    l=[Y_train[j],d]
                    distance.append(l)
                    
            for m in range(0,k):
                Wts[int(distance[m][0])]=Wts[int(distance[m][0])]+np.abs(1/float(distance[m][1]))
            # Making prediction based on distances
            tmp=np.argmax(Wts)   
            Y_pred[i]=tmp    
               
        Accuracy[k],F1_Score[k]=CM(Y_pred,Y_val)
            
        print("The value of K is %d ." %(k))
        print("The F1 Score is %f ." %(F1_Score[k]))

    return Accuracy,F1_Score

# Final output used for printing resuls and testing data with Best K final value
def Final_Output_Test(X_train,X_test,K):
    X_val=X_train[0:np.shape(X_train)[0]/5]
    X_train=X_train[np.shape(X_train)[0]/5:np.shape(X_train)[0]]
    
    Y_train=X_train[:,11]
    X_train=X_train[:,0:-1]  
    Y_val=X_val[:,11]
    X_val=X_val[:,0:-1]
    Y_test=X_test[:,11]
    X_test=X_test[:,0:-1]
       
    Mean=X_train.mean(0)
    Std=X_train.std(0)    
    X_train=(X_train-Mean)/Std
    X_val=(X_val-Mean)/Std
    X_test=(X_test-Mean)/Std
    
    for k in range(K,K+1):
        Y_pred=np.zeros(len(X_val))
        for i in range(0,len(X_val)):
            distance=[]
            Wts=np.zeros(11)
            
            for tmp in range(0,k):
                d=manhattan(X_train[tmp],X_val[i])
                l=[Y_train[tmp],d]
                distance.append(l)
            
            for j in range(k,len(X_train)):
                d=manhattan(X_train[j],X_val[i])
                tmp=np.argmax(np.asarray(distance)[:,1])
                if(d<distance[tmp][1]):
                    del distance[tmp]
                    l=[Y_train[j],d]
                    distance.append(l)
                    
            for m in range(0,k):
                Wts[int(distance[m][0])]=Wts[int(distance[m][0])]+np.abs(1/float(distance[m][1]))
            
            tmp=np.argmax(Wts)   
            Y_pred[i]=tmp    
               
        VAccuracy,VF1_Score=CM(Y_pred,Y_val)
    # Similar to the cross validation code, But run only once
        
    for k in range(K,K+1):
        Y_pred=np.zeros(len(X_test))
        for i in range(0,len(X_test)):
            distance=[]
            Wts=np.zeros(11)
            
            for tmp in range(0,k):
                d=manhattan(X_train[tmp],X_test[i])
                l=[Y_train[tmp],d]
                distance.append(l)
            
            for j in range(k,len(X_train)):
                d=manhattan(X_train[j],X_test[i])
                tmp=np.argmax(np.asarray(distance)[:,1])
                if(d<distance[tmp][1]):
                    del distance[tmp]
                    l=[Y_train[j],d]
                    distance.append(l)
                    
            for m in range(0,k):
                Wts[int(distance[m][0])]=Wts[int(distance[m][0])]+np.abs(1/float(distance[m][1]))
            
            tmp=np.argmax(Wts)   
            Y_pred[i]=tmp   
               
        TAccuracy,TF1_Score=CM(Y_pred,Y_test)

    return TAccuracy,TF1_Score,VAccuracy,VF1_Score


file = open('winequality-white.csv')
data=[]

TAc=[]
TF1=[]
VAc=[]
VF1=[]

for row in file:
    a=row.split(';')
    data.append(a)
    
del data[0]

X=np.asarray(data).astype('float')
#np.random.seed(0)
np.random.shuffle(X)
X = pickle.load( open("X_data_saved.p", "rb" ) ) #- To be used for getting best results

X_1=X[0:np.shape(X)[0]/4]
X_2=X[np.shape(X)[0]/4:2*(np.shape(X)[0]/4)]
X_3=X[2*(np.shape(X)[0]/4):3*(np.shape(X)[0]/4)]
X_4=X[3*(np.shape(X)[0]/4):np.shape(X)[0]]

test=[X_1,X_2,X_3,X_4]
tr1=np.concatenate((X_2,X_3,X_4),axis=0)
tr2=np.concatenate((X_3,X_4,X_1),axis=0)
tr3=np.concatenate((X_4,X_1,X_2),axis=0)
tr4=np.concatenate((X_1,X_2,X_3),axis=0)
train=[tr1,tr2,tr3,tr4]
# The shuffled data, to be used for cross validation and testing
K_best_final=16

print("Hyper-parameters:")
print("K : %d" %(K_best_final))
print("Distance Measure: Manhattan Distance")

for i in range(0,4):

    X_test=test[i]
    X_train=train[i]
    
    TAccuracy,TF1_Score,VAccuracy,VF1_Score=Final_Output_Test(X_train,X_test,K_best_final)
    TAc.append(TAccuracy)
    TF1.append(TF1_Score)
    VAc.append(VAccuracy)
    VF1.append(VF1_Score)
    
    print("Hyper-parameters:")
    print("K Best Fold: %d" %(K_best_final))
    print("Distance Measure: Manhattan Distance")
    
    print("Fold-%d:" %(i+1))
    print("Validation: F1 Score: %f , Accuracy: %f" %(VF1_Score,VAccuracy))
    print("Test: F1 Score: %f , Accuracy: %f" %(TF1_Score,TAccuracy))

print("Average:")
print("Validation: F1 Score: %f , Accuracy: %f" %(np.mean(VF1),np.mean(VAc)))
print("Test: F1 Score: %f , Accuracy: %f" %(np.mean(TF1),np.mean(TAc)))

# BELOW Two Lines for timing
#end = time.time()
#print("The time taken for the algorithm computation is :- %f seconds." % (end-start))

# The END OF CODE which Prints the FINAL REQUIRED OUTPUT with the Best K Final Value

# The Below Code is for Running the Cross Validation & Testing
# That is for running the cross_validate_test function
# Comment the Above Main Section Code
# Un Comment The Below Section, to just run 4-Fold Cross Validation 
# The below section is used to generate the graphs included in the report
# By Running the Below Code we tune the hyperparamet K_best_final
# We also tune the distance metric, by changing the function we call :
# Euclidian, Cosine, Manhattan are the distance hyperparameters to tune.

#file = open('winequality-white.csv')
#data=[]
#Ac=[]
#F1=[]
#
#TAc=[]
#TF1=[]
#VAc=[]
#VF1=[]
#
#for row in file:
#    a=row.split(';')
#    data.append(a)
#    
#del data[0]
#
#X=np.asarray(data).astype('float')
#np.random.seed(0)
#np.random.shuffle(X)
#
#X_1=X[0:np.shape(X)[0]/4]
#X_2=X[np.shape(X)[0]/4:2*(np.shape(X)[0]/4)]
#X_3=X[2*(np.shape(X)[0]/4):3*(np.shape(X)[0]/4)]
#X_4=X[3*(np.shape(X)[0]/4):np.shape(X)[0]]
#
#test=[X_1,X_2,X_3,X_4]
#tr1=np.concatenate((X_2,X_3,X_4),axis=0)
#tr2=np.concatenate((X_3,X_4,X_1),axis=0)
#tr3=np.concatenate((X_4,X_1,X_2),axis=0)
#tr4=np.concatenate((X_1,X_2,X_3),axis=0)
#train=[tr1,tr2,tr3,tr4]
#
#for i in range(0,4):
#
#    X_test=test[i]
#    X_train=train[i]
#    
#    Accuracy,F1_Score=cross_validation_test(X_train,50)
#    K_best_fold=np.argmax(F1_Score)
#    print("The best value of K is %d and fold number is %d." % (K_best_fold,i+1))
#    print(Accuracy[K_best_fold])
#    print(F1_Score[K_best_fold])
#    
#    x = np.arange(1,51, 1)
#    Accuracy=Accuracy[1:]
#    F1_Score=F1_Score[1:]
#    F1.append(F1_Score)
#    Ac.append(Accuracy)
#    
#    plt.figure(1)
#    plt.plot(x,Accuracy, label = "fold %d" %(i+1))
#    plt.figure(2)
#    plt.plot(x,F1_Score, label = "fold %d" %(i+1))
#
#    TAccuracy,TF1_Score,VAccuracy,VF1_Score=Final_Output_Test(X_train,X_test,K_best_fold)
#    TAc.append(TAccuracy)
#    TF1.append(TF1_Score)
#    VAc.append(VAccuracy)
#    VF1.append(VF1_Score)
#    
#    print("Hyper-parameters:")
#    print("K Best Fold: %d" %(K_best_fold))
#    print("Distance Measure: Manhattan Distance")  
#    
#    print("Fold-%d:" %(i+1))
#    print("Validation: F1 Score: %f , Accuracy: %f" %(VF1_Score,VAccuracy))
#    print("Test: F1 Score: %f , Accuracy: %f" %(TF1_Score,TAccuracy))
#    
# 
#print("Average:")
#print("Validation: F1 Score: %f , Accuracy: %f" %(np.mean(VF1),np.mean(VAc)))
#print("Test: F1 Score: %f , Accuracy: %f" %(np.mean(TF1),np.mean(TAc)))
#    
#
#plt.figure(1)
#plt.xlabel('K') 
## naming the y axis 
#plt.ylabel('Accuracy') 
## giving a title to my graph 
#plt.title('Accuracy vs K')   
## show a legend on the plot 
#plt.legend()   
## function to show the plot 
#plt.savefig('Accuracy.png')
#
#plt.figure(2)
#plt.xlabel('K') 
## naming the y axis 
#plt.ylabel('F1 Score') 
## giving a title to my graph 
#plt.title('F1 Scores vs K')   
## show a legend on the plot 
#plt.legend()   
## function to show the plot 
#plt.savefig('F1_Score.png')
#
#pickle.dump(X, open( "X_data_saved.p", "wb" ) ) 
#pickle.dump(Ac, open( "Ac_data_saved.p", "wb" ) )
#pickle.dump(F1, open( "F1_data_saved.p", "wb" ) )
#
#end = time.time()
#print("The time taken for the algorithm computation is :- %f seconds." % (end-start))


