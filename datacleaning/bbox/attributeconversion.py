import json
import csv
import urllib.parse
import boto3
import uuid
import os
import sys
import numpy as np
import pickle
import pandas as pd
import os
from sklearn.cluster import KMeans
import sklearn
from collections import Counter
from sklearn.metrics import pairwise_distances_argmin_min

# print('The scikit-learn version is {}.'.format(sklearn.__version__))

# Defining S3 client
s3 = boto3.client('s3')

#Env varibles
buc = os.environ['Bucket']
rating = os.environ['Rating']
result = os.environ['Result_matrix']
cluster_map_avg = os.environ['cluster_map_avg']
top_bottom_rating = os.environ['Top_bottom_rating']
upp = os.environ['upper']
low = os.environ['lower']
upper_model = os.environ['upper_model']
lower_model = os.environ['lower_model']

#Defining the lower and upper attribute list
lower_att=['Unnamed: 0', 't_floral', 't_stripe', 't_dot', 'f_denim', 'f_leather','f_cotton', 'f_knit', 'f_pleated', 's_fit', 's_pencil', 's_midi','s_mini', 's_maxi', 'p_zipper']
upper_att=['Unnamed: 0', 't_floral', 't_stripe', 't_dot', 'f_lace', 'f_denim','f_chiffon', 'f_cotton', 'f_leather', 'f_fur', 'p_sleeveless','p_long-sleeve', 'p_collar', 'p_pocket', 'p_v-neck', 'p_button','p_hooded', 'p_zipper']
    
citae_dic_lower={}
citae_dic_upper={}
    
def load_csv_files(fname):
    csvfile = s3.get_object(Bucket=buc, Key=fname)
    df=pd.read_csv(csvfile['Body'])
    return df

def load_model(modelname):
    s3 = boto3.resource('s3')
    model = pickle.loads(s3.Bucket(buc).Object(modelname).get()['Body'].read())
    return model
 
### helper function for collaborative filtering  
def helper(clusterIndex, topType):
    df_ratings_clean = load_csv_files(rating)
    # print("df_ratings_clean:", df_ratings_clean)
    col_idx = clusterIndex
    print("col_idx: ", col_idx)
    resultant_matrix=load_csv_files(result)
    print("resultant_matrix:", resultant_matrix)
    corr_mat = np.corrcoef(resultant_matrix)
    corr_specific = corr_mat[col_idx]
    print("corr_specific", corr_specific)
    bottomMatchingTops=pd.DataFrame({'corr_specific':corr_specific, 'TopBottom_corr_specific': df_ratings_clean.columns})\
    .sort_values('corr_specific', ascending=False)\
    .head(10)
    bottomMatchingTops=bottomMatchingTops.TopBottom_corr_specific.iloc[0:2].tolist()
    if(topType == "upper"):
        return identifyCluster("upper", bottomMatchingTops)
    else:
        return identifyCluster("lower", bottomMatchingTops)

### helper function to get the top or bottom index
def identifyCluster(topType, bottomMatchingTops):
    cluster_mapping_average=load_csv_files(cluster_map_avg)
    df_top_bottom_rating=load_csv_files(top_bottom_rating)

    if topType=="upper":
        clusterResult=findClusterRatings("upper", cluster_mapping_average, bottomMatchingTops, df_top_bottom_rating)
        print("clusterResult", clusterResult)
        topitems= findClusterItems("upper", clusterResult, df_top_bottom_rating)
        topLocationIndex=topitems.top.sample(frac=0.3)
        topLocationIndex=topLocationIndex.to_numpy()
        print("topLocationIndex", topLocationIndex)
        return topLocationIndex
        
    elif topType=="lower":
        clusterResult=findClusterRatings("lower", cluster_mapping_average, bottomMatchingTops, df_top_bottom_rating)
        print("clusterResult", clusterResult)
        bottomItems= findClusterItems("upper", clusterResult, df_top_bottom_rating)
        print("bottomItems", bottomItems)
        bottomLocationIndex=bottomItems.bottom.sample(frac=.3)
        bottomLocationIndex=bottomLocationIndex.to_numpy()
        print("bottomLocationIndex", bottomLocationIndex)
        return bottomLocationIndex
        
def findClusterRatings(itemType,cluster_mapping_average, bottomMatchingTops, df_top_bottom_rating):
    clusterResult=[]
    if(itemType=="upper"):
        for i in bottomMatchingTops:
            topCluster=cluster_mapping_average[cluster_mapping_average["topClusterRating"]==int(i)].topClusterRating
            clusterResult.append(topCluster.iloc[0])
        return clusterResult
    elif (itemType=="lower"):
        for i in bottomMatchingTops:
            bottomCluster=cluster_mapping_average[cluster_mapping_average["bottomClusterRating"]==int(i)].bottomClusterRating
            print("bottomCluster", bottomCluster)
            clusterResult.append(bottomCluster.iloc[0])
        return clusterResult


def findClusterItems(itemType, clusterResult, df_top_bottom_rating):
    itemClusterResult=[]
    if(itemType=="upper"):
        for i in clusterResult:
            topitems=df_top_bottom_rating[df_top_bottom_rating["topClusterRating"]==int(i)].sort_values('itemRating', ascending=False)
            itemClusterResult.append(topitems)
        df_topClusterResult = pd.concat(itemClusterResult)
        df_topClusterResult=df_topClusterResult.sort_values('itemRating', ascending=False)
        return df_topClusterResult
    elif(itemType=="lower"):
        for i in clusterResult:
            bottomItems=df_top_bottom_rating[df_top_bottom_rating["bottomClusterRating"]==int(i)].sort_values('itemRating', ascending=False)
            itemClusterResult.append(bottomItems)
        df_bottomClusterResult = pd.concat(itemClusterResult)
        df_bottomClusterResult=df_bottomClusterResult.sort_values('itemRating', ascending=False)
        return df_bottomClusterResult
        
## helper function to predict the attributes       
def getAttributes(itemType, indexNumber):
    if itemType=="upper":
        df_upper=load_csv_files(upp)
        df_lower=load_csv_files(low)
        attributesIndex=checkIndex(itemType, indexNumber, df_upper, df_lower)
        if(len(attributesIndex) == 0):
            return ("lower", df_upper.sample(n=5))
        return("lower", df_upper.iloc[attributesIndex])
    elif itemType=="lower":
        df_lower=load_csv_files(low)
        df_upper=load_csv_files(upp)
        attributesIndex=checkIndex(itemType, indexNumber, df_upper, df_lower)
        if(len(attributesIndex) == 0):
            return ("upper", df_upper.sample(n=5))
        return("upper", df_lower.iloc[attributesIndex])

def checkIndex(itemType, indexNumber, df_upper, df_lower):
    result=[]
    if itemType=="lower":
        for i in indexNumber:
            if i in df_lower.index:
                result.append(i)
    elif itemType=="upper":
        for i in indexNumber:
            if i in df_upper.index:
                result.append(i)
    return result

## helper function to predict cluster
def predictCluster(itemType, predValues):
    if itemType=="lower":
        model=load_model(lower_model)
        predCluster=helperPredictCluster(model, predValues)
        print("lowerpred", predCluster)
        return predCluster
    elif itemType=="upper":
        model=load_model(upper_model)
        predCluster=helperPredictCluster(model, predValues)
        print("upperpred", predCluster)
        return predCluster
        
def helperPredictCluster(model, predValues):
    pred=np.asarray(predValues)
    loaded_model_test=model.fit(pred.reshape(-1, 1))
    closest, _ = pairwise_distances_argmin_min(model.cluster_centers_, pred.reshape(-1, 1))
    centroidCounter = Counter(closest)
    cluster=centroidCounter.most_common(1)[0][0]
    return cluster

def checkIftheattributeexists(cloth_type, cloth_att, cloth_response):
    if not cloth_att and not cloth_response:
        return []
    # print("cloth_att", cloth_att)
    # print("cloth_response", cloth_response)
    cloth_att= () if not cloth_att else tuple(sorted(cloth_att))
    cloth_response= () if not cloth_response else tuple(sorted(cloth_response))
    # dic_lower={}
    # dic_upper={}
    if cloth_type=="upper":
        if cloth_att not in citae_dic_upper:
            citae_dic_upper[cloth_att]=cloth_response
        return list(citae_dic_upper[cloth_att])
    elif cloth_type=="lower":
        if cloth_att not in citae_dic_lower:
            citae_dic_lower[cloth_att]=cloth_response
        return list(citae_dic_lower[cloth_att])
            
def att_to_bin(event):
    print(event)
    cloth_type = (event["type"]).lower()
    print("cloth_type",cloth_type)
    cloth_att =event["attributes"]
    print(cloth_att, type(cloth_att))
    # att_list = cloth_att.split(",")
    bin_lis_u=[0 for _ in range(len(upper_att))]
    ind =0
    if cloth_type == "top":
        while ind <len(cloth_att):
            if cloth_att[ind] in upper_att:
                bin_lis_u[upper_att.index(cloth_att[ind])]=1
            ind +=1
    bin_lis = (bin_lis_u)
    print("bin lis u",bin_lis_u)
    bin_lis_b=[0 for _ in range(len(lower_att))]
    if cloth_type =="bottom":
        while ind <len(cloth_att):
            if cloth_att[ind] in lower_att:
                bin_lis_b[upper_att.index(cloth_att[ind])]=1
            ind +=1
    bin_lis = bin_lis_b
    print(bin_lis)
    return bin_lis
    
def rem_zeros(bin_lis):
    print("inside rem_zeros",bin_lis)
    only_attributes=[]
    for i in bin_lis:
        rem =(list(filter((0).__ne__, i)))
        only_attributes.append(rem)
    print("printing attributes after removing zero:", only_attributes)
    return only_attributes
    
    
def rem_int(final_resp):
    no_int=[]
    for i in final_resp:
        s = [x for x in i if not isinstance(x, int)]
        no_int.append(s)
    print("printing just attribute:",no_int)
    rem_empty_lis = [ele for ele in no_int if ele != []]
    print("removed empty list",rem_empty_lis)
    return rem_empty_lis
    
def type_mapping(event):
    cloth_type = (event["type"]).lower()
    if cloth_type == "top":
        return "upper"
    if cloth_type =="bottom":
        return "lower"

def ret_attributes(xy):
    print("in ret attribute function",xy)
    xy = xy.loc[:].replace(1, pd.Series(xy.columns, xy.columns))
    print("xy after replacing",xy)
    bin_output = xy.values.tolist()
    print(" List of dataframe",bin_output)
    return bin_output
    
def lambda_handler(event, context):
    # pred=[1,    1,    0,    1,    1,    1,    1,    1,    1,    0,    1,    0,    1,    1,  1, 1, 0]
    print(att_to_bin(event))
    pred = att_to_bin(event)
    print(pred)
    predCluster=predictCluster(type_mapping(event), pred)
    print("predCluster", predCluster)
    clusterIndex=predCluster
    topType=type_mapping(event)
    itemIndex=helper(clusterIndex, topType)
    print("itemIndex", itemIndex)
    if(topType=="lower"):
        y=getAttributes("upper", itemIndex)
        print("y[1]",y[1])
    elif(topType=="upper"):
        y=getAttributes("lower", itemIndex)
        print("y[1]",y[1])
    attributes_list = ret_attributes(y[1])
    final_resp = rem_zeros(attributes_list)
    print("final response:",final_resp)
    removed = rem_int(final_resp)
    print("removed", removed)
    print("eventattr",  event["attributes"])
    final_response=checkIftheattributeexists(topType, event["attributes"], removed)
    print("final_response", final_response)
    return json.dumps({"attributes": (final_response)})
    # return json.dumps({"attributes": attributes_list})
    # return 200; 
