import xml.etree.ElementTree as ET
import os
from StringIO import StringIO
import sys
from collections import Counter
import cPickle as pickle
import numpy as np
import itertools

###############convert mist to vector space model###################
########calculate the freq of each string in the total features############
#input folder /Users/haowang/Documents/harvardex/machinelearning/practical2/train_mist    
#output folder
#/Users/haowang/Documents/harvardex/machinelearning/practical2/train_ndarray




#python feaEmbe.py /Users/haowang/Documents/harvardex/machinelearning/practical2/train_mist_v3 /Users/haowang/Documents/harvardex/machinelearning/practical2/test_mist_v3 1

#python feaEmbe.py /Users/haowang/Documents/harvardex/machinelearning/practical2/train_mist_v4 /Users/haowang/Documents/harvardex/machinelearning/practical2/test_mist_v4 1

#python feaEmbe.py /Users/haowang/Documents/harvardex/machinelearning/practical2/train_mist_v4 /Users/haowang/Documents/harvardex/machinelearning/practical2/test_mist_v4 2

#python feaEmbe.py /Users/haowang/Documents/harvardex/machinelearning/practical2/train_mist_v5 /Users/haowang/Documents/harvardex/machinelearning/practical2/test_mist_v5 1

#python feaEmbe.py /Users/haowang/Documents/harvardex/machinelearning/practical2/train_mist_v6 /Users/haowang/Documents/harvardex/machinelearning/practical2/test_mist_v6 2

#python feaEmbe.py /Users/haowang/Documents/harvardex/machinelearning/practical2/train_mist_v6 /Users/haowang/Documents/harvardex/machinelearning/practical2/test_mist_v6 1

def main(argv):
    train_in=argv[1]
    test_in=argv[2]
    level=argv[3]
    
    feaEmbed(train_in,test_in,level)
    #qgram(train_in,test_in)



def create_file_list(input_folder):
    files=[]
    if os.path.exists(input_folder):
    			for ffile in os.listdir(input_folder):
    				file = os.path.join(input_folder, ffile)
    				if os.path.isfile(file) and file.endswith(".txt"):
    					files.append(file)
    return files



def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
def load_object(filename):
    with open(filename,'rb') as input:
            obj=pickle.load(input)
    return obj

########### ################                 

def feaEmbed(train_input_folder,test_input_folder,level):
    
    level=level
    
    files_train=create_file_list(train_input_folder)
    files_test=create_file_list(test_input_folder)
    total_file=files_train+files_test
    
    total_level = set()
    train_y=[]
    total_count={}    
    for file in total_file:
    
        count=Counter()
        id=os.path.splitext(os.path.basename(file))[0]
        with open(file) as input:
            for line in input:
                data=[str(x) for x in line.replace("|"," ").split(" ") if x and x!='\n']
                data[1]=data[0]+data[1]
                
                if "#process" not in data:
                    #print data
                    if level=="1":
                        count[data[0]]+=1
                        count[data[1]]+=1
                        total_level.update(data[:2])
                    elif level=="2":
                        count[data[0]]+=1
                        count[data[1]]+=1
                        #print data
                        try:
                            data[2]=line
                            count[data[2]]+=1
                        except Exception:
                            pass
                        total_level.update(data[0:3])
    #save_object(total_level1,"total_level1.pickle")
        #create the np array and dp dataframe

    #generate the count of each feature in all the files
        total_count[id]=count
        #print count

    num_fea=len(total_level)
    num_samples_train=len(files_train)
    num_samples_test=len(files_test)
    
    train_x=[]
    
    test_x=[]
    
    
    

    test_ids=[]
    for id,count in total_count.iteritems():
        if not id.endswith("X"):
            clazz=id.split(".")[1]
            train_y.append(clazz)
            row=[]
            for item in total_level:
                itemcount=count[item]
                #if itemcount!=0:
                    #print itemcount
                row.append(itemcount)
            train_x.append(np.array(row))
        else:
            test_id=id.split(".")[0]
            test_ids.append(test_id)
            row=[]
            for item in total_level:
                itemcount=count[item]
                #if itemcount!=0:
                    #print itemcount
                row.append(itemcount)
            test_x.append(np.array(row))
        
    train_x=np.array(train_x)
    print train_x.shape
    test_x=np.array(test_x)
    print test_x.shape
        #print row
    
    
    train_y=np.array(train_y)

    np.savetxt("train_x_level_"+level,train_x,delimiter=',',newline='\n')
    np.savetxt("test_x_level_"+level,test_x,delimiter=',',newline='\n')

    save_object(train_y,"train_y_level_"+level+"_pickle")
    save_object(test_ids,"test_id_level_"+level+"_pickle")


def qgram(train_input_folder,test_input_folder):
    files_train=create_file_list(train_input_folder)
    files_test=create_file_list(test_input_folder)
    total_file=files_train+files_test
    
    
    sample_qgram={}
    total_level=set()
    for file in total_file:
        each_qgram=[]

        id=os.path.splitext(os.path.basename(file))[0]
        with open(file) as input:
            temp=""
            for line in input:
                data=[str(x) for x in line.replace("|"," ").split(" ") if x and x!='\n']
                data[1]=data[0]+data[1]
                if "#process" not in data:
                    if not temp:
                        temp=str(data[0])+","+str(data[1])
                    else:
                        qgram=(temp,str(data[0]+","+str(data[1])))
                        temp=str(data[0])+","+str(data[1])
                        each_qgram.append(qgram)
                        total_level.update([temp])
        sample_qgram[id]=each_qgram
            #print each_qgram
        
    total_qgram=list(itertools.product(*[total_level,total_level]))
    
    save_object(total_qgram,"total_qgram_pickle")
    save_object(sample_qgram,"sample_qgram_pickle")
    
    print "saved"
    
    total_qgram=load_object("total_qgram_pickle")
    sample_qgram=load_object("sample_qgram_pickle")
    
    print len(total_qgram)
    print total_qgram[0]
	
    train_x=[]
    train_y=[]
    test_x=[]
    test_ids=[]
	
    for id,each_qgram in sample_qgram.iteritems():
        if not id.endswith("X"):
            clazz=id.split(".")[1]
            train_y.append(clazz)
            row=[]
            for qgram in total_qgram:
                if qgram in each_qgram:
                    row.append(1)
                    #print 1
                else:
                    row.append(0)
            train_x.append(np.array(row))
        else:
            test_id=id.split(".")[0]
            test_ids.append(test_id)
            row=[]
            for qgram in total_qgram:
                if qgram in each_qgram:
                    row.append(1)
                else:
                    row.append(0)
            test_x.append(np.array(row))

    train_x=np.array(train_x)
    print train_x.shape
    test_x=np.array(test_x)
    print test_x.shape
    
    np.savetxt("train_x_qgram",train_x,delimiter=',',newline='\n')
    np.savetxt("test_x_qgram",test_x,delimiter=',',newline='\n')

    save_object(train_y,"train_y_qgram_pickle")
    save_object(test_ids,"test_id_qgram_pickle")
    

if __name__=="__main__":
    main(sys.argv)