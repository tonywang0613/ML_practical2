
import os
import sys
import cPickle as pickle
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import grid_search
from sklearn import gaussian_process
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc

def main(argv):
    qgram=False
    final=False
    if not qgram:
        level=1
        
        if level==1:
            train_x=np.genfromtxt("train_x_level_1",delimiter=',')
            test_x=np.genfromtxt("test_x_level_1",delimiter=',')
            train_y=load_object("train_y_level_1_pickle")
            train_x_processed,test_x_processed=data_preprocess_simple_2(train_x,test_x)
        
        elif level==2:
            train_x=np.genfromtxt("train_x_level_2",delimiter=',')
            test_x=np.genfromtxt("test_x_level_2",delimiter=',')
            train_y=load_object("train_y_level_2_pickle")
            train_x_processed=data_preprocess_simple(train_x)
            test_x_processed=data_preprocess_simple(test_x) #train_x_processed,test_x_processed=data_preprocess_simple_2(train_x,test_x)
    
        if final:
            if level==1:
                
                #test_x_processed=data_preprocess_simple(test_x)
                test_id=load_object("test_id_level_1_pickle")
            elif level==2:
                #test_x_processed=data_preprocess(test_x)
                test_id=load_object("test_id_level_2_pickle")
            predictions=modeling(train_x_processed,train_y,test_x_processed)
            predictions=convert(predictions)
            write_predictions(predictions,test_id,"predict14")
            
    else:
        train_x=np.genfromtxt("train_x_qgram",delimiter=',')
        train_y=load_object("train_y_qgram_pickle")
        if final:
            test_x=np.genfromtxt("test_x_qgram",delimiter=',')
            test_id=load_object("test_id_qgram_pickle")
            predictions=tuneRand(train_x_processed,train_y,test_x_processed)
            predictions=convert(predictions)
            write_predictions(predictions,test_id,"predict15")
            
        #tuneRand(train_x,train_y)
        
    #train_x_level1=np.genfromtxt("train_x_level_1",delimiter=',')
    #test_x_level1=np.genfromtxt("test_x_level_1",delimiter=',')
    #train_x_level2=np.genfromtxt("train_x_level_2",delimiter=',')
    #print train_x_level2.shape
    #train_y_level2=load_object("train_y_level_2_pickle")
    #train_y_level2=load_object("train_y_level_2_pickle")
    #test_x_level1=load_object("test_x_level_1_pickle")
    #test_x_level2=np.genfromtxt("test_x_level_2",delimiter=',')
    #test_id=load_object("test_id_pickle")
    #print test_id
    #print test_x_level1.shape
    #train_x_level2_processed=data_preprocess(train_x_level2)
    #test_x_level2_processed=data_preprocess(test_x_level1)
    #print train_x_level1_processed.shape
    #print test_x_level1_processed.shape
    
    ##tuneGMP(train_x_processed,train_y)
    #tuneKNN(train_x_processed,train_y)
    #tuneSVM(train_x_processed,train_y)
    #predictions=modeling(train_x,train_y,test_x_level2_processed)
    #predictions=convert(predictions)
    
    evaluation(train_x,train_y,"all")
    

    
    #write_predictions(predictions,test_id,"predict3")
    
    ######plot########
    #evaluation(train_x,train_y,"all")

def combineTrainTest(x_train, y_train, x_test):
    predict=modeling(x_train,y_train,x_test)
    new_y=y_train+predict
    new_x=np.vstack((x_train,x_test))
    new_predict=modeling(new_x,new_y)
    
    
def confusion(y_test,y_predict):
    malware_classes = ["Agent","AutoRun","FraudLoad","FraudPack",
     "Hupigon","Krap","Lipler","Magania","None",
     "Poison","Swizzor","Tdss","VB","Virut","Zbot"]
    cm = confusion_matrix(y_test, y_predict,labels=malware_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plot_confusion(cm_normalized)
    plt.show()
    

    
def plot_confusion(cm,title='confusion matrix',cmap=plt.cm.Blues):
    malware_classes = ["Agent","AutoRun","FraudLoad","FraudPack",
     "Hupigon","Krap","Lipler","Magania","None",
     "Poison","Swizzor","Tdss","VB","Virut","Zbot"]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(15)
    plt.xticks(tick_marks, malware_classes, rotation=45)
    plt.yticks(tick_marks, malware_classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def roc(y_test,pred_prob):
    malware_classes = ["Agent","AutoRun","FraudLoad","FraudPack",
     "Hupigon","Krap","Lipler","Magania","None",
     "Poison","Swizzor","Tdss","VB","Virut","Zbot"]
    y_test=convert(y_test)
    y=label_binarize(y_test,classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    print y
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(15):
        fpr[i],tpr[i],_=roc_curve(y[:,i],pred_prob[:,i])
        roc_auc[i]=auc(fpr[i],tpr[i])
    plt.figure()
    for i in range(15):
        plt.plot(fpr[i],tpr[i],label='ROC curve of {0}(area={1:0.2f})'.format(malware_classes[i],roc_auc[i]))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC of malware')
    plt.legend(loc="lower right",fontsize=8)
    plt.show()
      

def evaluation(x,y,model):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=21)
    if model=="random" or model=="all":
        prediction,prediction_proba=modeling(x_train,y_train,x_test)
        #print prediction
        confusion(y_test,prediction)
        #print prediction_proba
        roc(y_test,prediction_proba)
        '''
        non_correct=[]
        for i in range(len(prediction)):
            if prediction[i]!=y_test[i]:
                non_correct.append([prediction[i],y_test[i]])
        print str(non_correct)
        out=open("non_correct_2","w")
        out.write(str(non_correct))
        out.close
        #print "random foreset score is {}".format(float(len(correct))/len(prediction))
        '''
        
#def cv(x,y,para,model):
    
    
def tuneGMP(x,y):
   #mean: 0.89566, std: 0.00330, params: {'min_samples_split': 50, 'n_estimators': 100, 'learning_rate': 0.1}
    parameters={'learning_rate':[0.05,0.15,0.25],'n_estimators':[40,100,150,200],'min_samples_split':[5,30,50]} 
    gmp=GradientBoostingClassifier(max_features='sqrt')
    clf=grid_search.GridSearchCV(gmp, parameters,cv=3,n_jobs=4)
    clf.fit(x,y)
    print clf.grid_scores_
    
    
def tuneKNN(x,y):
    parameters={'n_neighbors':[5,10,20,40,50],'algorithm':('auto','brute')}
    knn=KNeighborsClassifier(weights='distance')
    clf=grid_search.GridSearchCV(knn, parameters,cv=3,n_jobs=4)
    clf.fit(x,y)
    print clf.grid_scores_
    
    
def tuneRand(x,y,x_test):
    parameters = {'max_features':('auto', 'log2'),"n_estimators":[70,90,100,110,130,150]}
    rf = RandomForestClassifier(random_state=3)
    clf = grid_search.GridSearchCV(rf, parameters,cv=10,n_jobs=4)
    clf.fit(x,y)
    
    print clf.grid_scores_
    return clf.predict(x_test)
    
def tuneSVM(x,y):
    parameters = {"C":[1,10,20,30,50,70,90]}
    svm = SVC()
    clf = grid_search.GridSearchCV(svm, parameters,cv=5,n_jobs=4)
    clf.fit(x,y)
    print clf.grid_scores_
    
   
def write_predictions(predictions, ids, outfile):
    """
    assumes len(predictions) == len(ids), and that predictions[i] is the
    index of the predicted class with the malware_classes list above for 
    the executable corresponding to ids[i].
    outfile will be overwritten
    """
    with open(outfile,"w+") as f:
        # write header
        f.write("Id,Prediction\n")
        for i, history_id in enumerate(ids):
            f.write("%s,%s\n" % (history_id, predictions[i]))
            
def convert(predictions):
    malware_classes = ["Agent","AutoRun","FraudLoad","FraudPack",
     "Hupigon","Krap","Lipler","Magania","None",
     "Poison","Swizzor","Tdss","VB","Virut","Zbot"]
    prediction_id=[malware_classes.index(x) for x in predictions]
    return prediction_id


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
def load_object(filename):
    with open(filename, "rb") as input_file:
        e = pickle.load(input_file)
    return e



def l2_normalizer(vec):
    denom = np.sum([el**2 for el in vec])
    return [(el / math.sqrt(denom)) for el in vec]
    
# calculate the frequencey of the word, divide by sum

def data_preprocess_simple_2 (data1,data2):
    size=len(data1)
    data_total=np.vstack((data1,data2))
    data_total=scale(data_total)
    return data_total[:size,],data_total[size:,]
    
    
def data_preprocess_simple(data):
    return scale(data,axis=0)
    
    sum=data.sum(axis=1,dtype='float')
    print sum.shape
    return data/np.matrix(sum).T

def data_preprocess(data):
    
    data_tf=[]
    #l2_normalization

    
    #IDF frequency weighting
    #calculate the frequence of the word in all docu(the number of the doc contains this word divide by the total number of doc)
    idf=[]
    for colu in data.T:
        non_zeros=np.count_nonzero(colu)
        #print non_zeros
        all=len(colu)
        #print all
        idf.append(np.log(float(all)/(1+non_zeros)))
    #print idf
    
    def build_idf_matrix(idf_vector):
        idf_mat = np.zeros((len(idf_vector), len(idf_vector)))
        np.fill_diagonal(idf_mat, idf_vector)
        return idf_mat
    
    #create digonal matrix of idf 
    data_dif_matrix=build_idf_matrix(idf)
    
    #tf-idf matrix multiplication
    data_tfidf=[]
    for tf_vector in data:
        data_tfidf.append(np.dot(tf_vector,data_dif_matrix))
    
    #normalize
    data_tfidf_l2=[]
    for tfidf_vector in data_tfidf:
        data_tfidf_l2.append(l2_normalizer(tfidf_vector))
        
    return np.array(data_tfidf_l2)
    
def modeling(x_train,y_train,x_test):
    #min_samples_split': 50, 'n_estimators': 100, 'learning_rate': 0.1}
    #clf=GradientBoostingClassifier(max_features='sqrt',min_samples_split=50,n_estimators=100,learning_rate=0.1)
    clf = RandomForestClassifier(max_features="auto",n_estimators=90)
    clf.fit(x_train,y_train)
    print clf.classes_
    return clf.predict(x_test),clf.predict_proba(x_test)
   
    

def svm_model(x_train,y_train,x_test):
    clf = SVC()
    clf.fit(x_train, y_train) 
    return clf.predict(x_test)
    
if __name__=="__main__":
    main(sys.argv)
    


############# tune result ####################

###unprocess the data, v5 mist
#best  mean: 0.89371, std: 0.00251, params: {'max_features': 'auto', 'n_estimators': 150}

'''
[mean: 0.89274, std: 0.00315, params: {'max_features': 'auto', 'n_estimators': 100}, mean: 0.89307, std: 0.00259, params: {'max_features': 'auto', 'n_estimators': 110}, mean: 0.89209, std: 0.00301, params: {'max_features': 'auto', 'n_estimators': 120}, mean: 0.89307, std: 0.00227, params: {'max_features': 'auto', 'n_estimators': 130}, mean: 0.89371, std: 0.00251, params: {'max_features': 'auto', 'n_estimators': 150}, mean: 0.89047, std: 0.00165, params: {'max_features': 'auto', 'n_estimators': 170}, mean: 0.89242, std: 0.00350, params: {'max_features': 'auto', 'n_estimators': 200}, mean: 0.89469, std: 0.00230, params: {'max_features': 'auto', 'n_estimators': 250}, mean: 0.89177, std: 0.00256, params: {'max_features': 'auto', 'n_estimators': 270}, mean: 0.89080, std: 0.00363, params: {'max_features': 'auto', 'n_estimators': 300}, mean: 0.88950, std: 0.00018, params: {'max_features': 'log2', 'n_estimators': 100}, mean: 0.88950, std: 0.00403, params: {'max_features': 'log2', 'n_estimators': 110}, mean: 0.88756, std: 0.00481, params: {'max_features': 'log2', 'n_estimators': 120}, mean: 0.89274, std: 0.00269, params: {'max_features': 'log2', 'n_estimators': 130}, mean: 0.88983, std: 0.00105, params: {'max_features': 'log2', 'n_estimators': 150}, mean: 0.89145, std: 0.00204, params: {'max_features': 'log2', 'n_estimators': 170}, mean: 0.89242, std: 0.00154, params: {'max_features': 'log2', 'n_estimators': 200}, mean: 0.88983, std: 0.00147, params: {'max_features': 'log2', 'n_estimators': 250}, mean: 0.89177, std: 0.00113, params: {'max_features': 'log2', 'n_estimators': 270}, mean: 0.88983, std: 0.00124, params: {'max_features': 'log2', 'n_estimators': 300}]

'''
###simple process the data, divide bby the total count , tune with random forest
#best is the auto 50
'''
[mean: 0.89177, std: 0.00063, params: {'max_features': 'auto', 'n_estimators': 50}, mean: 0.89080, std: 0.00329, params: {'max_features': 'auto', 'n_estimators': 100}, mean: 0.89047, std: 0.00272, params: {'max_features': 'auto', 'n_estimators': 250}, mean: 0.89145, std: 0.00367, params: {'max_features': 'auto', 'n_estimators': 500}, mean: 0.88918, std: 0.00266, params: {'max_features': 'log2', 'n_estimators': 50}, mean: 0.89145, std: 0.00135, params: {'max_features': 'log2', 'n_estimators': 100}, mean: 0.88788, std: 0.00217, params: {'max_features': 'log2', 'n_estimators': 250}, mean: 0.89112, std: 0.00208, params: {'max_features': 'log2', 'n_estimators': 500}]
'''

### random forest on level 1 input ####

#best is the auto 100 estimator and auto
''''
[mean: 0.88658, std: 0.00496, params: {'max_features': 'auto', 'n_estimators': 50}, mean: 0.89209, std: 0.00541, params: {'max_features': 'auto', 'n_estimators': 100}, mean: 0.88885, std: 0.00363, params: {'max_features': 'auto', 'n_estimators': 250}, mean: 0.88918, std: 0.00586, params: {'max_features': 'auto', 'n_estimators': 500}, mean: 0.88464, std: 0.00371, params: {'max_features': 'log2', 'n_estimators': 50}, mean: 0.88950, std: 0.00562, params: {'max_features': 'log2', 'n_estimators': 100}, mean: 0.88658, std: 0.00524, params: {'max_features': 'log2', 'n_estimators': 250}, mean: 0.88723, std: 0.00443, params: {'max_features': 'log2', 'n_estimators': 500}]
Haos-MacBook-Pro:test haowang$ 
'''   



### random foreset on level 2 input ###
#with preproceed 
#best log2 50
''''
[mean: 0.88464, std: 0.00371, params: {'max_features': 'auto', 'n_estimators': 50}, mean: 0.88464, std: 0.00581, params: {'max_features': 'auto', 'n_estimators': 100}, mean: 0.88432, std: 0.00584, params: {'max_features': 'auto', 'n_estimators': 250}, mean: 0.88367, std: 0.00651, params: {'max_features': 'auto', 'n_estimators': 500}, mean: 0.88723, std: 0.00476, params: {'max_features': 'log2', 'n_estimators': 50}, mean: 0.88237, std: 0.00363, params: {'max_features': 'log2', 'n_estimators': 100}, mean: 0.88399, std: 0.00644, params: {'max_features': 'log2', 'n_estimators': 250}, mean: 0.88399, std: 0.00735, params: {'max_features': 'log2', 'n_estimators': 500}]
'''
#without processed
#best auto 500
'''
 [mean: 0.89209, std: 0.00103, params: {'max_features': 'auto', 'n_estimators': 50}, mean: 0.89501, std: 0.00268, params: {'max_features': 'auto', 'n_estimators': 100}, mean: 0.89469, std: 0.00389, params: {'max_features': 'auto', 'n_estimators': 250}, mean: 0.89566, std: 0.00257, params: {'max_features': 'auto', 'n_estimators': 500}, mean: 0.89339, std: 0.00175, params: {'max_features': 'log2', 'n_estimators': 50}, mean: 0.89501, std: 0.00526, params: {'max_features': 'log2', 'n_estimators': 100}, mean: 0.89469, std: 0.00257, params: {'max_features': 'log2', 'n_estimators': 250}, mean: 0.89469, std: 0.00325, params: {'max_features': 'log2', 'n_estimators': 500}]
### svm tune on level 1 input ###
'''

##################qgram tuning###########
## randome foreset best auto 250
'''
[mean: 0.89760, std: 0.00105, params: {'max_features': 'auto', 'n_estimators': 50}, mean: 0.89922, std: 0.00283, params: {'max_features': 'auto', 'n_estimators': 100}, mean: 0.90084, std: 0.00364, params: {'max_features': 'auto', 'n_estimators': 250}, mean: 0.89955, std: 0.00554, params: {'max_features': 'auto', 'n_estimators': 500}, mean: 0.89857, std: 0.00033, params: {'max_features': 'log2', 'n_estimators': 50}, mean: 0.89566, std: 0.00302, params: {'max_features': 'log2', 'n_estimators': 100}, mean: 0.89857, std: 0.00317, params: {'max_features': 'log2', 'n_estimators': 250}, mean: 0.89857, std: 0.00300, params: {'max_features': 'log2', 'n_estimators': 500}]
##for rbf c from 0.1 to 15
'''
''''
[mean: 0.52981, std: 0.01784, params: {'C': 0.1}, mean: 0.76410, std: 0.00783, params: {'C': 1}, mean: 0.77803, std: 0.00430, params: {'C': 2}, mean: 0.78257, std: 0.00306, params: {'C': 3}, mean: 0.79067, std: 0.00929, params: {'C': 4}, mean: 0.79909, std: 0.01322, params: {'C': 5}, mean: 0.80266, std: 0.01425, params: {'C': 6}, mean: 0.80298, std: 0.01427, params: {'C': 7}, mean: 0.80557, std: 0.01544, params: {'C': 8}, mean: 0.80655, std: 0.01567, params: {'C': 9}, mean: 0.80784, std: 0.01358, params: {'C': 10}, mean: 0.80752, std: 0.01355, params: {'C': 11}, mean: 0.81238, std: 0.01494, params: {'C': 12}, mean: 0.81789, std: 0.01193, params: {'C': 13}, mean: 0.81918, std: 0.01185, params: {'C': 14}, mean: 0.82113, std: 0.01216, params: {'C': 15}]

'''
##for rbf c from 15 to 23
''''
[mean: 0.82113, std: 0.01216, params: {'C': 15}, mean: 0.82113, std: 0.01216, params: {'C': 16}, mean: 0.82307, std: 0.01257, params: {'C': 17}, mean: 0.82340, std: 0.01284, params: {'C': 18}, mean: 0.82599, std: 0.01357, params: {'C': 19}, mean: 0.82761, std: 0.01299, params: {'C': 20}, mean: 0.82793, std: 0.01261, params: {'C': 21}, mean: 0.82858, std: 0.01224, params: {'C': 22}, mean: 0.82988, std: 0.01179, params: {'C': 23}]
'''
##for rbf c from 30 to 60
''''
[mean: 0.83603, std: 0.01161, params: {'C': 30}, mean: 0.83895, std: 0.01198, params: {'C': 35}, mean: 0.83895, std: 0.01102, params: {'C': 40}, mean: 0.83992, std: 0.01039, params: {'C': 45}, mean: 0.83960, std: 0.00992, params: {'C': 50}, mean: 0.84219, std: 0.00690, params: {'C': 55}, mean: 0.84413, std: 0.00738, params: {'C': 60}
'''

#for rbf c from 60 to 200
''''
[mean: 0.84413, std: 0.00738, params: {'C': 60}, mean: 0.84413, std: 0.00821, params: {'C': 70}, mean: 0.84770, std: 0.00674, params: {'C': 80}, mean: 0.84997, std: 0.00765, params: {'C': 90}, mean: 0.85126, std: 0.00809, params: {'C': 100}, mean: 0.85321, std: 0.00837, params: {'C': 110}, mean: 0.85450, std: 0.00735, params: {'C': 120}, mean: 0.85872, std: 0.00721, params: {'C': 130}, mean: 0.85969, std: 0.00717, params: {'C': 140}, mean: 0.86001, std: 0.00771, params: {'C': 150}, mean: 0.86099, std: 0.00644, params: {'C': 160}, mean: 0.86163, std: 0.00571, params: {'C': 170}, mean: 0.86099, std: 0.00788, params: {'C': 180}, mean: 0.86099, std: 0.00840, params: {'C': 190}, mean: 0.86196, std: 0.00715, params: {'C': 200}]
'''
''''
[mean: 0.86844, std: 0.00836, params: {'C': 250}, mean: 0.86941, std: 0.00890, params: {'C': 300}, mean: 0.87071, std: 0.00815, params: {'C': 350}, mean: 0.87071, std: 0.00893, params: {'C': 400}, mean: 0.87135, std: 0.01018, params: {'C': 450}, mean: 0.87200, std: 0.00922, params: {'C': 500}, mean: 0.87168, std: 0.00936, params: {'C': 550}, mean: 0.87330, std: 0.00842, params: {'C': 600}, mean: 0.87459, std: 0.00955, params: {'C': 650}, mean: 0.87427, std: 0.00981, params: {'C': 700}, mean: 0.87524, std: 0.00893, params: {'C': 800}, mean: 0.87524, std: 0.00807, params: {'C': 900}, mean: 0.87557, std: 0.00755, params: {'C': 1000}]
'''
#for rbf kernel :linear poly sigmoid, c from 250 to 1000
''''
[mean: 0.87751, std: 0.01020, params: {'kernel': 'linear', 'C': 250},
 mean: 0.52139, std: 0.00333, params: {'kernel': 'poly', 'C': 250}, 
 mean: 0.85483, std: 0.00722, params: {'kernel': 'sigmoid', 'C': 250}, 
 
 mean: 0.87816, std: 0.00957, params: {'kernel': 'linear', 'C': 300}, 
 mean: 0.52139, std: 0.00333, params: {'kernel': 'poly', 'C': 300}, 
 mean: 0.85710, std: 0.00757, params: {'kernel': 'sigmoid', 'C': 300},
 
  
 mean: 0.87881, std: 0.00997, params: {'kernel': 'linear', 'C': 350}, 
 mean: 0.52139, std: 0.00333, params: {'kernel': 'poly', 'C': 350}, 
 mean: 0.85839, std: 0.00678, params: {'kernel': 'sigmoid', 'C': 350}, 
 
 mean: 0.87881, std: 0.00977, params: {'kernel': 'linear', 'C': 400}, 
 mean: 0.52139, std: 0.00333, params: {'kernel': 'poly', 'C': 400}, 
 mean: 0.86001, std: 0.00719, params: {'kernel': 'sigmoid', 'C': 400}, 
 
 mean: 0.87751, std: 0.00988, params: {'kernel': 'linear', 'C': 450}, 
 mean: 0.52139, std: 0.00333, params: {'kernel': 'poly', 'C': 450}, 
 mean: 0.86066, std: 0.00770, params: {'kernel': 'sigmoid', 'C': 450}, 
 
 mean: 0.87654, std: 0.00991, params: {'kernel': 'linear', 'C': 500}, 
 mean: 0.52139, std: 0.00333, params: {'kernel': 'poly', 'C': 500}, 
 mean: 0.86163, std: 0.00845, params: {'kernel': 'sigmoid', 'C': 500}, 
 
 mean: 0.87686, std: 0.00950, params: {'kernel': 'linear', 'C': 550}, 
 mean: 0.52981, std: 0.01784, params: {'kernel': 'poly', 'C': 550}, 
 mean: 0.86196, std: 0.00831, params: {'kernel': 'sigmoid', 'C': 550}, 
 
 mean: 0.87654, std: 0.00864, params: {'kernel': 'linear', 'C': 600}, 
 mean: 0.57550, std: 0.00689, params: {'kernel': 'poly', 'C': 600}, 
 mean: 0.86358, std: 0.00747, params: {'kernel': 'sigmoid', 'C': 600}, 
 
 mean: 0.87589, std: 0.00969, params: {'kernel': 'linear', 'C': 650}, 
 mean: 0.63318, std: 0.02813, params: {'kernel': 'poly', 'C': 650}, 
 mean: 0.86455, std: 0.00768, params: {'kernel': 'sigmoid', 'C': 650}, 
 
 mean: 0.87622, std: 0.00974, params: {'kernel': 'linear', 'C': 700}, 
 mean: 0.65781, std: 0.01156, params: {'kernel': 'poly', 'C': 700}, 
 mean: 0.86585, std: 0.00790, params: {'kernel': 'sigmoid', 'C': 700}, 
 
 mean: 0.87524, std: 0.01051, params: {'kernel': 'linear', 'C': 800}, 
 mean: 0.73040, std: 0.03086, params: {'kernel': 'poly', 'C': 800}, 
 mean: 0.86617, std: 0.00765, params: {'kernel': 'sigmoid', 'C': 800}, 
 
 mean: 0.87492, std: 0.00873, params: {'kernel': 'linear', 'C': 900}, 
 mean: 0.75891, std: 0.01069, params: {'kernel': 'poly', 'C': 900}, 
 mean: 0.86811, std: 0.00844, params: {'kernel': 'sigmoid', 'C': 900}, 
 
 mean: 0.87589, std: 0.00922, params: {'kernel': 'linear', 'C': 1000}, 
 mean: 0.76086, std: 0.00813, params: {'kernel': 'poly', 'C': 1000}, 
 mean: 0.86714, std: 0.00933, params: {'kernel': 'sigmoid', 'C': 1000}]
 
'''
 
'''
 [mean: 0.77771, std: 0.00514, params: {'kernel': 'linear', 'C': 0.1}, mean: 0.52139, std: 0.00333, params: {'kernel': 'poly', 'C': 0.1}, mean: 0.52139, std: 0.00333, params: {'kernel': 'sigmoid', 'C': 0.1}, mean: 0.83085, std: 0.01032, params: {'kernel': 'linear', 'C': 1}, mean: 0.52139, std: 0.00333, params: {'kernel': 'poly', 'C': 1}, mean: 0.72748, std: 0.01888, params: {'kernel': 'sigmoid', 'C': 1}, mean: 0.86131, std: 0.00813, params: {'kernel': 'linear', 'C': 10}, mean: 0.52139, std: 0.00333, params: {'kernel': 'poly', 'C': 10}, mean: 0.79877, std: 0.01329, params: {'kernel': 'sigmoid', 'C': 10}, mean: 0.86811, std: 0.00931, params: {'kernel': 'linear', 'C': 30}, mean: 0.52139, std: 0.00333, params: {'kernel': 'poly', 'C': 30}, mean: 0.82048, std: 0.01167, params: {'kernel': 'sigmoid', 'C': 30}, mean: 0.86941, std: 0.01006, params: {'kernel': 'linear', 'C': 50}, mean: 0.52139, std: 0.00333, params: {'kernel': 'poly', 'C': 50}, mean: 0.83182, std: 0.01059, params: {'kernel': 'sigmoid', 'C': 50}, mean: 0.87233, std: 0.00972, params: {'kernel': 'linear', 'C': 70}, mean: 0.52139, std: 0.00333, params: {'kernel': 'poly', 'C': 70}, mean: 0.83830, std: 0.01123, params: {'kernel': 'sigmoid', 'C': 70}, mean: 0.87395, std: 0.00875, params: {'kernel': 'linear', 'C': 90}, mean: 0.52139, std: 0.00333, params: {'kernel': 'poly', 'C': 90}, mean: 0.83960, std: 0.01159, params: {'kernel': 'sigmoid', 'C': 90}, mean: 0.87459, std: 0.00869, params: {'kernel': 'linear', 'C': 120}, mean: 0.52139, std: 0.00333, params: {'kernel': 'poly', 'C': 120}, mean: 0.84219, std: 0.00725, params: {'kernel': 'sigmoid', 'C': 120}, mean: 0.87524, std: 0.00823, params: {'kernel': 'linear', 'C': 140}, mean: 0.52139, std: 0.00333, params: {'kernel': 'poly', 'C': 140}, mean: 0.84381, std: 0.00826, params: {'kernel': 'sigmoid', 'C': 140}, mean: 0.87557, std: 0.00872, params: {'kernel': 'linear', 'C': 160}, mean: 0.52139, std: 0.00333, params: {'kernel': 'poly', 'C': 160}, mean: 0.84738, std: 0.00692, params: {'kernel': 'sigmoid', 'C': 160}, mean: 0.87654, std: 0.01024, params: {'kernel': 'linear', 'C': 180}, mean: 0.52139, std: 0.00333, params: {'kernel': 'poly', 'C': 180}, mean: 0.84835, std: 0.00678, params: {'kernel': 'sigmoid', 'C': 180}, mean: 0.87557, std: 0.01018, params: {'kernel': 'linear', 'C': 200}, mean: 0.52139, std: 0.00333, params: {'kernel': 'poly', 'C': 200}, mean: 0.85094, std: 0.00837, params: {'kernel': 'sigmoid', 'C': 200}]
'''
##############svm with level 2#######

#for rbf c 0.1 to 900
'''
[mean: 0.52139, std: 0.00333, params: {'C': 0.1}, mean: 0.52139, std: 0.00333, params: {'C': 1}, mean: 0.52139, std: 0.00333, params: {'C': 10}, mean: 0.52139, std: 0.00333, params: {'C': 30}, mean: 0.62541, std: 0.01300, params: {'C': 50}]

[mean: 0.75081, std: 0.01289, params: {'C': 100}, mean: 0.76345, std: 0.01308, params: {'C': 200}, mean: 0.77738, std: 0.01339, params: {'C': 300}, mean: 0.78289, std: 0.00884, params: {'C': 400}, mean: 0.78743, std: 0.00928, params: {'C': 500}, mean: 0.79747, std: 0.00564, params: {'C': 700}, mean: 0.80525, std: 0.01033, params: {'C': 900}]
'''

##########knn with level 1
'''
[mean: 0.84867, std: 0.00708, params: {'n_neighbors': 1, 'algorithm': 'auto'}, mean: 0.82793, std: 0.00540, params: {'n_neighbors': 2, 'algorithm': 'auto'}, mean: 0.83798, std: 0.01001, params: {'n_neighbors': 3, 'algorithm': 'auto'}, mean: 0.84025, std: 0.00950, params: {'n_neighbors': 4, 'algorithm': 'auto'}, mean: 0.84122, std: 0.00855, params: {'n_neighbors': 5, 'algorithm': 'auto'}, mean: 0.84835, std: 0.00679, params: {'n_neighbors': 1, 'algorithm': 'ball_tree'}, mean: 0.82793, std: 0.00614, params: {'n_neighbors': 2, 'algorithm': 'ball_tree'}, mean: 0.83830, std: 0.00910, params: {'n_neighbors': 3, 'algorithm': 'ball_tree'}, mean: 0.84025, std: 0.00950, params: {'n_neighbors': 4, 'algorithm': 'ball_tree'}, mean: 0.84122, std: 0.00931, params: {'n_neighbors': 5, 'algorithm': 'ball_tree'}, mean: 0.84867, std: 0.00708, params: {'n_neighbors': 1, 'algorithm': 'kd_tree'}, mean: 0.82793, std: 0.00540, params: {'n_neighbors': 2, 'algorithm': 'kd_tree'}, mean: 0.83798, std: 0.01001, params: {'n_neighbors': 3, 'algorithm': 'kd_tree'}, mean: 0.84025, std: 0.00950, params: {'n_neighbors': 4, 'algorithm': 'kd_tree'}, mean: 0.84122, std: 0.00855, params: {'n_neighbors': 5, 'algorithm': 'kd_tree'}, mean: 0.84738, std: 0.00689, params: {'n_neighbors': 1, 'algorithm': 'brute'}, mean: 0.83085, std: 0.00948, params: {'n_neighbors': 2, 'algorithm': 'brute'}, mean: 0.83765, std: 0.01092, params: {'n_neighbors': 3, 'algorithm': 'brute'}, mean: 0.84122, std: 0.00993, params: {'n_neighbors': 4, 'algorithm': 'brute'}, mean: 0.84381, std: 0.00944, params: {'n_neighbors': 5, 'algorithm': 'brute'}]
'''