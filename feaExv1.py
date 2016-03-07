import xml.etree.ElementTree as ET
import os
from StringIO import StringIO
import sys
from urlparse import urlparse
from collections import Counter
import cPickle as pickle


eleT=ET.parse("/Users/haowang/Documents/harvardex/machinelearning/practical2/test/conf/element2mistv6.xml")
eleRoot=eleT.getroot()

#tree = ET.parse('../train/0a6cbeee24cd37e42b5698319f9af6ebbed74395c.None.xml')
mist=StringIO()

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
def str2mist(value):
    cache={}
    val=value.lower()
    try:
    	return cache[val]
    except:
    	res = ELFHash(val)
    	result 	= int2hex(res, 8)
    	cache[val] = result
    	return result
        
def ELFHash(key):
		hash = 0
		x    = 0
		for i in range(len(key)):
			hash = (hash << 4) + ord(key[i])
			x = hash & 0xF0000000
			if x != 0:
				hash ^= (x >> 24)
			hash &= ~x
		return hash    

def int2hex(n, len):
		assert n   is not None
		assert len is not None
		try:
			hexval = ('0' * len) + "%x" % int(n)
		except ValueError:
			hexval = ('0' * len) + "%x" % int(n, 16)	
		return hexval[len * -1:]

def convertValue(tag,type,value,final_dic):
  
    #print type,value
    if tag=="srcfile":
        value=os.path.splitext(value)[1]
    if tag=="url":
        value=urlparse(value).netloc
    if tag=="filename" or "file":
        value=os.path.splitext(value)[1]
    if value=='':
        return ""
    final_dic[value]+=1
    if type == 'type_string':
    	result = str2mist(value)		
    elif type == 'type_hex':
    	result = value[2:10]
        while len(result) < 8:
    		result = "0" + result 			
    elif type == 'type_integer' or type =="type_binary":
        result = int2hex(value, 8)
    return result                
            
def write(mist,output):
	try:
		w_file = open(output, 'w')
		w_file.write(mist.getvalue())
		w_file.flush()
		w_file.close()
	except Exception, e:
		errormsg = e
		print errormsg
	return True
    
    
#input folder /Users/haowang/Documents/harvardex/machinelearning/practical2/train    
#output folder
#/Users/haowang/Documents/harvardex/machinelearning/practical2/train_mist

#python feaExv1.py /Users/haowang/Documents/harvardex/machinelearning/practical2/testdata   /Users/haowang/Documents/harvardex/machinelearning/practical2/test_mist

#python feaExv1.py /Users/haowang/Documents/harvardex/machinelearning/practical2/train   /Users/haowang/Documents/harvardex/machinelearning/practical2/train_mist_v4
#python feaExv1.py /Users/haowang/Documents/harvardex/machinelearning/practical2/testdata   /Users/haowang/Documents/harvardex/machinelearning/practical2/test_mist_v4

#python feaExv1.py /Users/haowang/Documents/harvardex/machinelearning/practical2/train   /Users/haowang/Documents/harvardex/machinelearning/practical2/train_mist_v5

#python feaExv1.py /Users/haowang/Documents/harvardex/machinelearning/practical2/testdata   /Users/haowang/Documents/harvardex/machinelearning/practical2/test_mist_v5

#python feaExv1.py /Users/haowang/Documents/harvardex/machinelearning/practical2/testdata   /Users/haowang/Documents/harvardex/machinelearning/practical2/test_mist_v6

#python feaExv1.py /Users/haowang/Documents/harvardex/machinelearning/practical2/train   /Users/haowang/Documents/harvardex/machinelearning/practical2/train_mist_v6

input_folder=sys.argv[1]
input_folder=sys.argv[1]
output_folder=sys.argv[2]

files=[]

if os.path.exists(input_folder):
			for ffile in os.listdir(input_folder):
				file = os.path.join(input_folder, ffile)
				if os.path.isfile(file) and file.endswith(".xml"):
					files.append(file)
final_dic=Counter()

for file in files:
    tree = ET.parse(file)
   
    mist=StringIO()
    processes=tree.findall("process")
    for process in processes:
        pid=process.attrib.get("pid")
        for thread in process:
            tid=thread.get("tid")
            mist.write("#process {} thread {}#\n".format(str(pid),str(tid)))
            for sections in thread.findall("all_section"):
                for api in sections:
                    apitag=api.tag
                    #print apitag
                    #api is the node in xml, we need to extract children info
                    apiattrib=api.attrib
                    if eleRoot.find(".//"+apitag) is not None:
                        maincategory=eleRoot.find(".//"+apitag+"/..")
                        #is the node like "load dll" in element2mist.xml
                        eleNode=eleRoot.find(".//"+apitag)
                        mist.write("{} {} |".format(maincategory.attrib["mist"],\
                        eleNode.attrib["mist"]))
                        #attri_node (vale, type pairs)
                        #print eleNode.tag
                        for attri_node in eleNode.getchildren():
                            #if the "value" or "key" is in the xml file input
                            if attri_node.tag in apiattrib:
                                 #print attri_node.tag
                                 value=apiattrib[attri_node.tag]
                                 value=                            convertValue(attri_node.tag,attri_node.attrib["type"],\
                                     value,final_dic)
                                 mist.write(" {} ".format(value))
                        mist.write("\n")
    basename=os.path.splitext(os.path.basename(file))[0]
    print output_folder+"/"+basename+".txt"
    if write(mist,output_folder+"/"+basename+".txt"):
        print "ok"
        
save_object(final_dic,"count_total_fea_pickle")

count=open("counter_total_fea","w")
for  a in final_dic:
    count.write("{}\t{}\n".format(repr(a),final_dic[a]))
                              
count.close()                  
                        

#print mist.getvalue()
                        
