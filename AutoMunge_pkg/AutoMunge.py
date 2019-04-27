#global imports
import numpy as np
import pandas as pd
from copy import deepcopy

#imports for process_numerical_class, postprocess_numerical_class
from pandas import Series
from sklearn import preprocessing

#imports for process_text_class, postprocess_text_class
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#imports for process_time_class, postprocess_time_class
import datetime as dt

#imports for process_bxcx_class
from scipy import stats

#imports for evalcategory
import collections
import datetime as dt

#imports for predictinfill, predictpostinfill, trainFSmodel
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

#imports for shuffleaccuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_log_error

#imports for automunge
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class AutoMunge:
  
  def __init__(self):
    pass
  
  
  

  def assembletransformdict(self, powertransform, binstransform):
    '''
    #assembles the range of transformations to be applied based on the evaluated \
    #category of data

    #the primitives are intented as follows:
    #_greatgrandparents_: supplemental column derived from source column, only applied
    #to first generation, with downstream transforms included
    #_grandparents_: supplemental column derived from source column, only applied
    #to first generation
    #_parents_: replace source column, with downstream trasnforms performed
    #_siblings_: supplemental column derived from source column, \
    #with downstream transforms performed
    #_auntsuncles_: replace source column, without downstream transforms performed
    #_cousins_: supplemental column derived from source column, \
    #without downstream transforms performed
    #downstream transform primitives are:
    #_children_: becomes downstream parent
    #_niecenephews_: treated like a downstream sibling
    #_coworkers_: becomes a downstream auntsuncles
    #_friends_: become downstream cousins
    
    
    #for example, if we set 'bxcx' entry to have both 'bxcx' as parents and \
    #'nmbr' as cousin, then the output would be column_nmbr, column_bxcx_nmbr, \
    #column_bxcx_nmbr_bins

    #however if we set 'bxcx' entry to have 'bxcx' as parent and 'nmbr' as sibling, then
    #the outpuyt would be column_nmbr, column_nmbr_bins, column_bxcx_nmbr, \
    #column_bxcx_nmbr_bins
    
    #note a future extension will allow automubnge class to run experiements
    #on different configurations of trasnform_dict to improve the feature selection
    '''

    transform_dict = {}

    #initialize bins based on what was passed through application of automunge(.)
    if binstransform == True:
      bins = 'bins'
      bint = 'bint'
    else:
      bins = None
      bint = None

    #initialize trasnform_dict. Note in a future extension the range of categories
    #is intended to be built out
    transform_dict.update({'nmbr' : {'greatgrandparents' : [], \
                                     'grandparents' : ['NArw'], \
                                     'parents' : ['nmbr'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : [bint]}})

    transform_dict.update({'bnry' : {'greatgrandparents' : [], \
                                     'grandparents' : ['NArw'], \
                                     'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bnry'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})

    transform_dict.update({'text' : {'greatgrandparents' : [], \
                                     'grandparents' : ['NArw'], \
                                     'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['text'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})

    transform_dict.update({'date' : {'greatgrandparents' : [], \
                                     'grandparents' : ['NArw'], \
                                     'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['date'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})

    transform_dict.update({'null' : {'greatgrandparents' : [], \
                                     'grandparents' : [], \
                                     'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['null'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'NArw' : {'greatgrandparents' : [], \
                                     'grandparents' : [], \
                                     'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['NArw'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'rgrl' : {'greatgrandparents' : [], \
                                     'grandparents' : [], \
                                     'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['nmbr'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'nbr2' : {'greatgrandparents' : [], \
                                     'grandparents' : ['NArw'], \
                                     'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['nmbr'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mnmx' : {'greatgrandparents' : [], \
                                     'grandparents' : ['NArw'], \
                                     'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mnmx'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mnm2' : {'greatgrandparents' : [], \
                                     'grandparents' : ['NArw'], \
                                     'parents' : ['nmbr'], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mnmx'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mnm3' : {'greatgrandparents' : [], \
                                     'grandparents' : ['NArw'], \
                                     'parents' : ['nmbr'], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mnm3'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    transform_dict.update({'mnm4' : {'greatgrandparents' : [], \
                                     'grandparents' : ['NArw'], \
                                     'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mnm3'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    transform_dict.update({'mnm5' : {'greatgrandparents' : [], \
                                     'grandparents' : ['NArw'], \
                                     'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mnmx'], \
                                     'cousins' : ['nmbr'], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'excl' : {'greatgrandparents' : [], \
                                     'grandparents' : [], \
                                     'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['excl'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
#     transform_dict.update({'exc2' : {'greatgrandparents' : [], \
#                                      'grandparents' : [], \
#                                      'parents' : [], \
#                                      'siblings': [], \
#                                      'auntsuncles' : [], \
#                                      'cousins' : ['exc2'], \
#                                      'children' : [], \
#                                      'niecesnephews' : [], \
#                                      'coworkers' : [], \
#                                      'friends' : []}})
    
    

    #initialize bxcx based on what was passed through application of automunge(.)
    if powertransform == True:

      transform_dict.update({'bxcx' : {'greatgrandparents' : [], \
                                       'grandparents' : ['NArw'], \
                                       'parents' : ['bxcx'], \
                                       'siblings': ['nmbr'], \
                                       'auntsuncles' : [], \
                                       'cousins' : [], \
                                       'children' : ['nmbr'], \
                                       'niecesnephews' : [], \
                                       'coworkers' : [], \
                                       'friends' : []}})

    else:

      transform_dict.update({'bxcx' : {'greatgrandparents' : [], \
                                       'grandparents' : ['NArw'], \
                                       'parents' : ['nmbr'], \
                                       'siblings': [], \
                                       'auntsuncles' : [], \
                                       'cousins' : [], \
                                       'children' : [], \
                                       'niecesnephews' : [bins], \
                                       'coworkers' : [], \
                                       'friends' : []}})

#     #currently bins will always be niecesnephews so not used as a key, but putting
#     #this here incase of future incorporation as a child
#     transform_dict.update({'bins' : {'greatgrandparents' : [], \
#                                      'grandparents' : ['NArw'], \
#                                      'parents' : [], \
#                                      'siblings': [], \
#                                      'auntsuncles' : ['bins'], \
#                                      'cousins' : [], \
#                                      'children' : [], \
#                                      'niecesnephews' : [], \
#                                      'coworkers' : [], \
#                                      'friends' : []}})


    return transform_dict
  
  
  
  
  def assembleprocessdict(self):
    '''
    #creates a dictionary storing all of the processing functions for each
    #category. Note that the convention is that every dualprocess entry 
    #(to process both train and text set in automunge) is meant
    #to have a coresponding postprocess entry (to process the test set in 
    #postmunge). If the dualprocess/postprocess pair aren't included a 
    #singleprocess funciton will be instead which processes a single column
    #at a time and is neutral to whether that set is from train or test data.
    
    #starting in version 1.79, this also stores entries for 'NArowtype' and
    #'MLinfilltype', which were added to facilitate user definition of 
    #custom processing functions
    
    #NArowtype entries are:
    # - 'numeric' for source columns with expected numeric entries
    # - 'justNaN' for source columns that may have expected entries other than numeric
    # - 'exclude' for source columns that aren't needing NArow columns derived
    
    #MLinfilltype entries are:
    # - 'numeric' for columns with numeric entries
    # - 'singlct' for single column sets with boolean entries
    # - 'multict' for multi column sets with boolean entrie
    # - 'exclude' for columns which will be excluded from ML infill
    '''
    
    process_dict = {}
    
    #categories are nmbr, bnry, text, date, bxcx, bins, bint, NArw, null
    #note a future extension will allow the definition of new categories 
    #to automunge

    #dual column functions
    process_dict.update({'nmbr' : {'dualprocess' : self.process_numerical_class, \
                                  'singleprocess' : None, \
                                  'postprocess' : self.postprocess_numerical_class, \
                                  'NArowtype' : 'numeric', \
                                  'MLinfilltype' : 'numeric'}})
    process_dict.update({'nbr2' : {'dualprocess' : self.process_numerical_class, \
                                  'singleprocess' : None, \
                                  'postprocess' : self.postprocess_numerical_class, \
                                  'NArowtype' : 'numeric', \
                                  'MLinfilltype' : 'numeric'}})
    process_dict.update({'mnmx' : {'dualprocess' : self.process_mnmx_class, \
                                  'singleprocess' : None, \
                                  'postprocess' : self.postprocess_mnmx_class, \
                                  'NArowtype' : 'numeric', \
                                  'MLinfilltype' : 'numeric'}})
    process_dict.update({'mnm2' : {'dualprocess' : self.process_mnmx_class, \
                                  'singleprocess' : None, \
                                  'postprocess' : self.postprocess_mnmx_class, \
                                  'NArowtype' : 'numeric', \
                                  'MLinfilltype' : 'numeric'}})
    process_dict.update({'mnm3' : {'dualprocess' : self.process_mnm3_class, \
                                  'singleprocess' : None, \
                                  'postprocess' : self.postprocess_mnm3_class, \
                                  'NArowtype' : 'numeric', \
                                  'MLinfilltype' : 'numeric'}})
    process_dict.update({'mnm4' : {'dualprocess' : self.process_mnm3_class, \
                                  'singleprocess' : None, \
                                  'postprocess' : self.postprocess_mnm3_class, \
                                  'NArowtype' : 'numeric', \
                                  'MLinfilltype' : 'numeric'}})
    process_dict.update({'mnm5' : {'dualprocess' : self.process_mnmx_class, \
                                  'singleprocess' : None, \
                                  'postprocess' : self.postprocess_mnmx_class, \
                                  'NArowtype' : 'numeric', \
                                  'MLinfilltype' : 'numeric'}})
    process_dict.update({'bnry' : {'dualprocess' : self.process_binary_class, \
                                  'singleprocess' : None, \
                                  'postprocess' : self.postprocess_binary_class, \
                                  'NArowtype' : 'justNaN', \
                                  'MLinfilltype' : 'singlct'}})
    process_dict.update({'text' : {'dualprocess' : self.process_text_class, \
                                  'singleprocess' : None, \
                                  'postprocess' : self.postprocess_text_class, \
                                  'NArowtype' : 'justNaN', \
                                  'MLinfilltype' : 'multirt'}})
    process_dict.update({'date' : {'dualprocess' : self.process_time_class, \
                                  'singleprocess' : None, \
                                  'postprocess' : self.postprocess_time_class, \
                                  'NArowtype' : 'justNaN', \
                                  'MLinfilltype' : 'exclude'}})
    process_dict.update({'bxcx' : {'dualprocess' : self.process_bxcx_class, \
                                  'singleprocess' : None, \
                                  'postprocess' : self.postprocess_bxcx_class, \
                                  'NArowtype' : 'numeric', \
                                  'MLinfilltype' : 'numeric'}})
    process_dict.update({'bins' : {'dualprocess' : self.process_bins_class, \
                                  'singleprocess' : None, \
                                  'postprocess' : self.postprocess_bins_class, \
                                  'NArowtype' : 'numeric', \
                                  'MLinfilltype' : 'multisp'}})
    process_dict.update({'bint' : {'dualprocess' : self.process_bint_class, \
                                  'singleprocess' : None, \
                                  'postprocess' : self.postprocess_bint_class, \
                                  'NArowtype' : 'numeric', \
                                  'MLinfilltype' : 'multisp'}})

    #single column functions
    process_dict.update({'NArw' : {'dualprocess' : None, \
                                  'singleprocess' : self.process_NArw_class, \
                                  'postprocess' : None, \
                                  'NArowtype' : 'exclude', \
                                  'MLinfilltype' : 'exclude'}})

    process_dict.update({'null' : {'dualprocess' : None, \
                                  'singleprocess' : self.process_null_class, \
                                  'postprocess' : None, \
                                  'NArowtype' : 'exclude', \
                                  'MLinfilltype' : 'exclude'}})
    process_dict.update({'excl' : {'dualprocess' : None, \
                                  'singleprocess' : self.process_excl_class, \
                                  'postprocess' : None, \
                                  'NArowtype' : 'exclude', \
                                  'MLinfilltype' : 'exclude'}})
#     process_dict.update({'exc2' : {'dualprocess' : None, \
#                                   'singleprocess' : self.process_exc2_class, \
#                                   'postprocess' : None, \
#                                   'NArowtype' : 'exclude', \
#                                   'MLinfilltype' : 'exclude'}})

    return process_dict

  
  
  def processancestors(self, df_train, df_test, column, category, origcategory, process_dict, \
                      transform_dict, postprocess_dict):
    '''
    #as automunge runs a for loop through each column in automunge, this is the  
    #processing function applied which runs through the grandparents family primitives
    #populated in the transform_dict by assembletransformdict, only applied to
    #first generation of transforms (others are recursive through the processfamily function)
    '''
    
    #process the grandparents (no downstream, supplemental, only applied ot first generation)
    for grandparent in transform_dict[category]['grandparents']:
      
      if grandparent != None:
      
        #note we use the processcousin function here
        df_train, df_test, postprocess_dict = \
        self.processcousin(df_train, df_test, column, grandparent, origcategory, \
                            process_dict, transform_dict, postprocess_dict)
      
    for greatgrandparent in transform_dict[category]['greatgrandparents']:
      
      
      if greatgrandparent != None:
        #note we use the processparent function here
        df_train, df_test, postprocess_dict = \
        self.processparent(df_train, df_test, column, greatgrandparent, origcategory, \
                          process_dict, transform_dict, postprocess_dict)
      
    
    return df_train, df_test, postprocess_dict
  
  

  def processfamily(self, df_train, df_test, column, category, origcategory, process_dict, \
                    transform_dict, postprocess_dict):
    '''
    #as automunge runs a for loop through each column in automunge, this is the master 
    #processing function applied which runs through the different family primitives
    #populated in the transform_dict by assembletransformdict
    '''
    
    #process the cousins (no downstream, supplemental)
    for cousin in transform_dict[category]['cousins']:
      
      if cousin != None:
      
        #note we use the processcousin function here
        df_train, df_test, postprocess_dict = \
        self.processcousin(df_train, df_test, column, cousin, origcategory, \
                            process_dict, transform_dict, postprocess_dict)


    #process the siblings (with downstream, supplemental)
    for sibling in transform_dict[category]['siblings']:
      
      if sibling != None:
        #note we use the processparent function here
        df_train, df_test, postprocess_dict = \
        self.processparent(df_train, df_test, column, sibling, origcategory, \
                          process_dict, transform_dict, postprocess_dict)

    #process the auntsuncles (no downstream, with replacement)
    for auntuncle in transform_dict[category]['auntsuncles']:

      if auntuncle != None:
      
        #note we use the processcousin function here
        df_train, df_test, postprocess_dict = \
        self.processcousin(df_train, df_test, column, auntuncle, origcategory, \
                            process_dict, transform_dict, postprocess_dict)

    #process the parents (with downstream, with replacement)
    for parent in transform_dict[category]['parents']:
      
      if parent != None:
      
        df_train, df_test, postprocess_dict = \
        self.processparent(df_train, df_test, column, parent, origcategory, \
                          process_dict, transform_dict, postprocess_dict)


    #if we had replacement transformations performed then mark column for deletion
    #(circle of life)
    if len(transform_dict[category]['auntsuncles']) \
    + len(transform_dict[category]['parents']) > 0:
      #here we'll only address downstream generaitons
      if column in postprocess_dict['column_dict']:
        postprocess_dict['column_dict'][column]['deletecolumn'] = True

    return df_train, df_test, postprocess_dict
  
  
  def circleoflife(self, df_train, df_test, column, category, origcategory, process_dict, \
                    transform_dict, postprocess_dict):
    '''
    #This function deletes source column for cases where family primitives 
    #included replacement.
    '''
    
    #if we had replacement transformations performed on first generation \
    #then delete the original column
    if len(transform_dict[category]['auntsuncles']) \
    + len(transform_dict[category]['parents']) > 0:
      del df_train[column]
      del df_test[column]
      
    #if we had replacement transformations performed on downstream generation \
    #then delete the associated parent column 
    for columndict_column in postprocess_dict['column_dict']:
      if postprocess_dict['column_dict'][columndict_column]['deletecolumn'] == True:
      
        #first we'll remove the column from columnslists 
        for columnslistcolumn in postprocess_dict['column_dict'][columndict_column]['columnslist']:

          if columndict_column in postprocess_dict['column_dict'][columnslistcolumn]['columnslist']:
          
            postprocess_dict['column_dict'][columnslistcolumn]['columnslist'].remove(columndict_column)

        #now we'll delete column
        #note this only worksa on single column  parents, need to incioroprate categorylist
        #for multicolumn parents (future extension)
        if columndict_column in list(df_train):
          del df_train[columndict_column]
          del df_test[columndict_column]
      
    return df_train, df_test, postprocess_dict
  

  def dictupdate(self, column, column_dict, postprocess_dict):
    '''
    #dictupdate function takes as input column_dict, postprocess_dict, then for cases
    #where origcolmn is the same fo rhte two combines the columnslist and the 
    #normalization_dict, then appends the column_dict onto the postprocess_dict
    #returns the column_dict and postprocess_dict. Note that the passed column name
    #"column" is the column name prior to the applicaiton of processing, and the
    #name of the column after the. last processing funciton is saved as a key
    #in the column_dict
    '''
    
    
    #(reason for "key2" instead of key1 is some shuffling during editing)
    for key2 in column_dict:

      #first address carry-though of origcolumn and origcategory from parent to child
      if column in postprocess_dict['column_dict']:

        #if column is not origcolumn in postprocess_dict
        if postprocess_dict['column_dict'][column]['origcolumn'] \
        != column:

          #assign origcolumn from postprocess_dict to column_dict
          column_dict[key2]['origcolumn'] = \
          postprocess_dict['column_dict'][column]['origcolumn']

          #assign origcategory from postprocess_dict to column_dict
          column_dict[key2]['origcategory'] = \
          postprocess_dict['column_dict'][column]['origcategory']

      for key1 in postprocess_dict['column_dict']:
        

        #if origcolumn is the same between column_dict saved in postprocess_dict and
        #the column_dict outputed from our processing, we'll combine a few values
        if postprocess_dict['column_dict'][key1]['origcolumn'] == column_dict[key2]['origcolumn']:
          #first we'll combine the columnslist capturing all columns 
          #originating from same origcolumn for these two sets
          postprocess_dict['column_dict'][key1]['columnslist'] = \
          list(set(postprocess_dict['column_dict'][key1]['columnslist'])|set(column_dict[key2]['columnslist']))
          #apply that value to the column_dict columnslist as well
          column_dict[key2]['columnslist'] = postprocess_dict['column_dict'][key1]['columnslist']


          #now we'll combine the normalization dictionary
          #remember that we are updating the strucutre to include the column+'_ctgy'
          #identifier as a key
          
          #first we'll append column_dict normalization_dict onto postprocess_dict, 
          postprocess_dict['column_dict'][key1]['normalization_dict'].update(column_dict[key2]['normalization_dict'])
          
          #then we'll copy postprocess_dict normalization_dict back onto column_dict
          column_dict[key2]['normalization_dict'] = postprocess_dict['column_dict'][key1]['normalization_dict']
          
          
            

          #now save the postprocess_dict normalization dicitonary to the column_dict
          #the idea is that every normalization parameter for every column is saved in 
          #every normalizaiton dict. Not extremely efficient but todo otherwise we 
          #would need to update our approach in postmunge in getting a column key
          column_dict[key2]['normalization_dict'] = \
          postprocess_dict['column_dict'][key1]['normalization_dict']
    

    #now append column_dict onto postprocess_dict
    postprocess_dict['column_dict'].update(column_dict)
    

    #return column_dict, postprocess_dict
    return postprocess_dict
  
  
  
  
  
  def processcousin(self, df_train, df_test, column, cousin, origcategory, \
                     process_dict, transform_dict, postprocess_dict):
    '''
    #cousin is one of the primitives for processfamily function, and it involves
    #transformations without downstream derivations without replacement of source column
    #although this same funciton can be used with the auntsuncles primitive
    #by following with a deletion of original column, also this funciton can be
    #used on the niecesnephews primitive downstream of parents or siblings since 
    #they don't have children (they're way to young for that)
    #note the processing funcitons are accessed through the process_dict
    '''
    
    
    #if this is a dual process function
    if process_dict[cousin]['dualprocess'] != None:
      df_train, df_test, column_dict_list = \
      process_dict[cousin]['dualprocess'](df_train, df_test, column, origcategory, \
                                          postprocess_dict)

    #else if this is a single process function process train and test seperately
    elif process_dict[cousin]['singleprocess'] != None:

      df_train, column_dict_list =  \
      process_dict[cousin]['singleprocess'](df_train, column, origcategory, \
                                            postprocess_dict)

      df_test, _1 = \
      process_dict[cousin]['singleprocess'](df_test, column, origcategory, \
                                            postprocess_dict)


    #update the columnslist and normalization_dict for both column_dict and postprocess_dict
    for column_dict in column_dict_list:
      postprocess_dict = self.dictupdate(column, column_dict, postprocess_dict)


    return df_train, df_test, postprocess_dict
  
  
  
  
  
  def processparent(self, df_train, df_test, column, parent, origcategory, \
                    process_dict, transform_dict, postprocess_dict):
    '''
    #parent is one of the primitives for processfamily function, and it involves
    #transformations with downstream derivations with replacement of source column
    #although this same funciton can be used with the siblinga primitive
    #by not following with a deletion of original column, also this funciton can be
    #used on the children primitive downstream of parents or siblings, allowing
    #the children to have children of their own, you know, grandchildren and stuff.
    #note the processing functions are accessed through the process_dict
    '''
    
    #if this is a dual process function
    if process_dict[parent]['dualprocess'] != None:

      df_train, df_test, column_dict_list = \
      process_dict[parent]['dualprocess'](df_train, df_test, column, origcategory, \
                                         postprocess_dict)

    #else if this is a single process function process train and test seperately
    elif process_dict[parent]['singleprocess'] != None:

      df_train, column_dict_list =  \
      process_dict[parent]['singleprocess'](df_train, column, origcategory, \
                                         postprocess_dict)

      df_test, _1 = \
      process_dict[parent]['singleprocess'](df_test, column, origcategory, \
                                         postprocess_dict)

    #update the columnslist and normalization_dict for both column_dict and postprocess_dict
    for column_dict in column_dict_list:
      postprocess_dict = self.dictupdate(column, column_dict, postprocess_dict)

      #note this only works for single column source, as currently implemented
      #multicolumn transforms (such as text or bins) cannot serve as parents
      #a future extension may check the categorylist from column_dict for 
      #purposes of transforms applied to multicolumn source
      parentcolumn = list(column_dict.keys())[0]



    #if transform_dict[parent] != None:

    #process any coworkers
    for coworker in transform_dict[parent]['coworkers']:

      if coworker != None:

        #process the coworker
        #note the function applied is processcousin
        df_train, df_test, postprocess_dict = \
        self.processcousin(df_train, df_test, parentcolumn, coworker, origcategory, \
                           process_dict, transform_dict, postprocess_dict)

    #process any friends
    for friend in transform_dict[parent]['friends']:

      if friend != None:

        #process the friend
        #note the function applied is processcousin
        df_train, df_test, postprocess_dict = \
        self.processcousin(df_train, df_test, parentcolumn, friend, origcategory, \
                           process_dict, transform_dict, postprocess_dict)


    #process any niecesnephews
    #note the function applied is comparable to processsibling, just a different
    #parent column
    for niecenephew in transform_dict[parent]['niecesnephews']:

      if niecenephew != None:

        #process the niecenephew
        #note the function applied is processfamily (using recursion)
        #parent column
        df_train, df_test, postprocess_dict = \
        self.processfamily(df_train, df_test, parentcolumn, niecenephew, origcategory, \
                           process_dict, transform_dict, postprocess_dict)

    #process any children
    for child in transform_dict[parent]['children']:

      if child != None:

        #process the child
        #note the function applied is processfamily (using recursion)
        #parent column
        df_train, df_test, postprocess_dict = \
        self.processfamily(df_train, df_test, parentcolumn, child, origcategory, \
                           process_dict, transform_dict, postprocess_dict)

#     #if we had replacement transformations performed then delete the original column 
#     #(circle of life)
#     if len(transform_dict[parent]['children']) \
#     + len(transform_dict[parent]['coworkers']) > 0:
#       del df_train[parentcolumn]
#       del df_test[parentcolumn]
# #       if column in postprocess_dict['column_dict']:
# #         postprocess_dict['column_dict'][parentcolumn]['deletecolumn'] = True
      
# #       else:
# #         postprocess_dict['column_dict'].update({parentcolumn : {'deletecolumn' : True}})

    #if we had replacement transformations performed then mark column for deletion
    #(circle of life)
    if len(transform_dict[parent]['children']) \
    + len(transform_dict[parent]['coworkers']) > 0:
      #here we'll only address downstream generaitons
      if parentcolumn in postprocess_dict['column_dict']:
        postprocess_dict['column_dict'][parentcolumn]['deletecolumn'] = True


    return df_train, df_test, postprocess_dict
  
  
  
  
  def process_NArw_class(self, df, column, category, postprocess_dict):
    '''
    #processing funciton that creates a boolean column indicating 1 for rows
    #corresponding to missing or improperly formated data in source column
    #note this uses the NArows function which has a category specific approach
    #returns same dataframe with new column of name column + '_NArw'
    #note this is a "singleprocess" function since is applied to single dataframe
    '''
    
    #add a second column with boolean expression indicating a missing cell
    #(using NArows(.) function defined below, column name will be column+'_NArows')
    NArows_nmbr = self.NArows(df, column, category, postprocess_dict)
    df[column + '_NArw'] = NArows_nmbr.copy()
    del NArows_nmbr

    #change NArows data type to 8-bit (1 byte) integers for memory savings
    df[column + '_NArw'] = df[column + '_NArw'].astype(np.int8)

    #create list of columns
    nmbrcolumns = [column + '_NArw']

    #create normalization dictionary
    NArwnormalization_dict = {column + '_NArw' : {}}

    #store some values in the nmbr_dict{} for use later in ML infill methods
    column_dict_list = []

    for nc in nmbrcolumns:

      if nc[-5:] == '_NArw':

        column_dict = { nc : {'category' : 'NArw', \
                             'origcategory' : category, \
                             'normalization_dict' : NArwnormalization_dict, \
                             'origcolumn' : column, \
                             'columnslist' : nmbrcolumns, \
                             'categorylist' : [nc], \
                             'infillmodel' : False, \
                             'infillcomplete' : False, \
                             'deletecolumn' : False}}

        column_dict_list.append(column_dict.copy())

    return df, column_dict_list

  
  
  
  def process_numerical_class(self, mdf_train, mdf_test, column, category, \
                              postprocess_dict):
    '''
    #process_numerical_class(mdf_train, mdf_test, column, category)
    #function to normalize data to mean of 0 and standard deviation of 1 \
    #z score normalization) 
    #takes as arguement pandas dataframe of training and test data (mdf_train), (mdf_test)\
    #and the name of the column string ('column') and parent category (category)
    #replaces missing or improperly formatted data with mean of remaining values
    #returns same dataframes with new column of name column + '_nmbr'
    #note this is a "dualprocess" function since is applied to both dataframes

    #expect this approach works better when the numerical distribution is thin tailed
    #if only have training but not test data handy, use same training data for both dataframe inputs
    '''
    
    #copy source column into new column
    mdf_train[column + '_nmbr'] = mdf_train[column].copy()
    mdf_test[column + '_nmbr'] = mdf_test[column].copy()

    #convert all values to either numeric or NaN
    mdf_train[column + '_nmbr'] = pd.to_numeric(mdf_train[column + '_nmbr'], errors='coerce')
    mdf_test[column + '_nmbr'] = pd.to_numeric(mdf_test[column + '_nmbr'], errors='coerce')

    #get mean of training data
    mean = mdf_train[column + '_nmbr'].mean()    

    #replace missing data with training set mean
    mdf_train[column + '_nmbr'] = mdf_train[column + '_nmbr'].fillna(mean)
    mdf_test[column + '_nmbr'] = mdf_test[column + '_nmbr'].fillna(mean)

    #subtract mean from column for both train and test
    mdf_train[column + '_nmbr'] = mdf_train[column + '_nmbr'] - mean
    mdf_test[column + '_nmbr'] = mdf_test[column + '_nmbr'] - mean

    #get standard deviation of training data
    std = mdf_train[column + '_nmbr'].std()

    #divide column values by std for both training and test data
    mdf_train[column + '_nmbr'] = mdf_train[column + '_nmbr'] / std
    mdf_test[column + '_nmbr'] = mdf_test[column + '_nmbr'] / std


    #create list of columns
    nmbrcolumns = [column + '_nmbr']


    nmbrnormalization_dict = {column + '_nmbr' : {'mean' : mean, 'std' : std}}

    #store some values in the nmbr_dict{} for use later in ML infill methods
    column_dict_list = []

    for nc in nmbrcolumns:

      if nc[-5:] == '_nmbr':

        column_dict = { nc : {'category' : 'nmbr', \
                             'origcategory' : category, \
                             'normalization_dict' : nmbrnormalization_dict, \
                             'origcolumn' : column, \
                             'columnslist' : nmbrcolumns, \
                             'categorylist' : [nc], \
                             'infillmodel' : False, \
                             'infillcomplete' : False, \
                             'deletecolumn' : False}}

        column_dict_list.append(column_dict.copy())
    

        
    return mdf_train, mdf_test, column_dict_list


  def process_mnmx_class(self, mdf_train, mdf_test, column, category, \
                         postprocess_dict):
    '''
    #process_mnmx_class(mdf_train, mdf_test, column, category)
    #function to scale data to minimum of 0 and maximum of 1 \
    #based on min/max values from training set for this column
    #takes as arguement pandas dataframe of training and test data (mdf_train), (mdf_test)\
    #and the name of the column string ('column') and parent category (category)
    #replaces missing or improperly formatted data with mean of remaining values
    #returns same dataframes with new column of name column + '_mnmx'
    #note this is a "dualprocess" function since is applied to both dataframes

    #expect this approach works better when the numerical distribution is thin tailed
    #if only have training but not test data handy, use same training data for both
    #dataframe inputs
    '''
    
    #copy source column into new column
    mdf_train[column + '_mnmx'] = mdf_train[column].copy()
    mdf_test[column + '_mnmx'] = mdf_test[column].copy()

    #convert all values to either numeric or NaN
    mdf_train[column + '_mnmx'] = pd.to_numeric(mdf_train[column + '_mnmx'], errors='coerce')
    mdf_test[column + '_mnmx'] = pd.to_numeric(mdf_test[column + '_mnmx'], errors='coerce')

    #get mean of training data
    mean = mdf_train[column + '_mnmx'].mean()    

    #replace missing data with training set mean
    mdf_train[column + '_mnmx'] = mdf_train[column + '_mnmx'].fillna(mean)
    mdf_test[column + '_mnmx'] = mdf_test[column + '_mnmx'].fillna(mean)
    
    #get maximum value of training column
    maximum = mdf_train[column + '_mnmx'].max()
    
    #get minimum value of training column
    minimum = mdf_train[column + '_mnmx'].min()
    
    #perform min-max scaling to train and test sets using values from train
    mdf_train[column + '_mnmx'] = (mdf_train[column + '_mnmx'] - minimum) / \
                                  (maximum - minimum)
    
    mdf_test[column + '_mnmx'] = (mdf_test[column + '_mnmx'] - minimum) / \
                                 (maximum - minimum)

    #create list of columns
    nmbrcolumns = [column + '_mnmx']


    nmbrnormalization_dict = {column + '_mnmx' : {'minimum' : minimum, \
                                                  'maximum' : maximum, \
                                                  'mean' : mean}}

    #store some values in the nmbr_dict{} for use later in ML infill methods
    column_dict_list = []

    for nc in nmbrcolumns:

      if nc[-5:] == '_mnmx':

        column_dict = { nc : {'category' : 'mnmx', \
                             'origcategory' : category, \
                             'normalization_dict' : nmbrnormalization_dict, \
                             'origcolumn' : column, \
                             'columnslist' : nmbrcolumns, \
                             'categorylist' : [nc], \
                             'infillmodel' : False, \
                             'infillcomplete' : False, \
                             'deletecolumn' : False}}

        column_dict_list.append(column_dict.copy())
    

        
    return mdf_train, mdf_test, column_dict_list



  def process_mnm3_class(self, mdf_train, mdf_test, column, category, \
                         postprocess_dict):
    '''
    #process_mnmx_class(mdf_train, mdf_test, column, category)
    #function to scale data to minimum of 0 and maximum of 1 \
    #after replacing extreme values above the 0.99 quantile with
    #the value of 0.99 quantile and extreme values below the 0.01
    #quantile with the value of 0.01 quantile
    #takes as arguement pandas dataframe of training and test data (mdf_train), (mdf_test)\
    #and the name of the column string ('column') and parent category (category)
    #replaces missing or improperly formatted data with mean of remaining values
    #returns same dataframes with new column of name column + '_mnmx'
    #note this is a "dualprocess" function since is applied to both dataframes
    '''

    #copy source column into new column
    mdf_train[column + '_mnm3'] = mdf_train[column].copy()
    mdf_test[column + '_mnm3'] = mdf_test[column].copy()

    #convert all values to either numeric or NaN
    mdf_train[column + '_mnm3'] = pd.to_numeric(mdf_train[column + '_mnm3'], errors='coerce')
    mdf_test[column + '_mnm3'] = pd.to_numeric(mdf_test[column + '_mnm3'], errors='coerce')

    #get mean of training data
    mean = mdf_train[column + '_mnm3'].mean()    

    #replace missing data with training set mean
    mdf_train[column + '_mnm3'] = mdf_train[column + '_mnm3'].fillna(mean)
    mdf_test[column + '_mnm3'] = mdf_test[column + '_mnm3'].fillna(mean)

    #get maximum value of training column
    quantilemax = mdf_train[column + '_mnm3'].quantile(.99)

    #get minimum value of training column
    quantilemin = mdf_train[column + '_mnm3'].quantile(.01)

    #replace values > quantilemax with quantilemax
    mdf_train.loc[mdf_train[column + '_mnm3'] > quantilemax, (column + '_mnm3')] \
    = quantilemax
    mdf_test.loc[mdf_train[column + '_mnm3'] > quantilemax, (column + '_mnm3')] \
    = quantilemax
    #replace values < quantile10 with quantile10
    mdf_train.loc[mdf_train[column + '_mnm3'] < quantilemin, (column + '_mnm3')] \
    = quantilemin
    mdf_test.loc[mdf_train[column + '_mnm3'] < quantilemin, (column + '_mnm3')] \
    = quantilemin


    #note this step is now performed after the quantile evaluation / replacement

    #get mean of training data
    mean = mdf_train[column + '_mnm3'].mean()    

    #replace missing data with training set mean
    mdf_train[column + '_mnm3'] = mdf_train[column + '_mnm3'].fillna(mean)
    mdf_test[column + '_mnm3'] = mdf_test[column + '_mnm3'].fillna(mean)


    #perform min-max scaling to train and test sets using values from train
    mdf_train[column + '_mnm3'] = (mdf_train[column + '_mnm3'] - quantilemin) / \
                                  (quantilemax - quantilemin)

    mdf_test[column + '_mnm2'] = (mdf_test[column + '_mnm3'] - quantilemin) / \
                                 (quantilemax - quantilemin)

    #create list of columns
    nmbrcolumns = [column + '_mnm3']


    nmbrnormalization_dict = {column + '_mnm3' : {'quantilemin' : quantilemin, \
                                                  'quantilemax' : quantilemax, \
                                                  'mean' : mean}}

    #store some values in the nmbr_dict{} for use later in ML infill methods
    column_dict_list = []

    for nc in nmbrcolumns:

      if nc[-5:] == '_mnm3':

        column_dict = { nc : {'category' : 'mnm3', \
                             'origcategory' : category, \
                             'normalization_dict' : nmbrnormalization_dict, \
                             'origcolumn' : column, \
                             'columnslist' : nmbrcolumns, \
                             'categorylist' : [nc], \
                             'infillmodel' : False, \
                             'infillcomplete' : False, \
                             'deletecolumn' : False}}

        column_dict_list.append(column_dict.copy())



    return mdf_train, mdf_test, column_dict_list



  def process_binary_class(self, mdf_train, mdf_test, column, category, \
                           postprocess_dict):
    '''
    #process_binary_class(mdf, column, missing)
    #converts binary classification values to 0 or 1
    #takes as arguement a pandas dataframe (mdf_train, mdf_test), \
    #the name of the column string ('column') \
    #and the category from parent columkn (category)
    #fills missing valules with most common value
    #returns same dataframes with new column of name column + '_bnry'
    #note this is a "dualprocess" function since is applied to both dataframes
    '''
    
    #copy column to column + '_bnry'
    mdf_train[column + '_bnry'] = mdf_train[column].copy()
    mdf_test[column + '_bnry'] = mdf_test[column].copy()

    #create plug value for missing cells as most common value
    valuecounts = mdf_train[column + '_bnry'].value_counts().index.tolist()
    binary_missing_plug = valuecounts[0]
    
    #note LabelBinarizer encodes alphabetically, with 1 assigned to first and 0 to second
    valuecounts.sort()
    #we'll save these in the normalization dictionary for future reference
    onevalue = valuecounts[0]
    zerovalue = valuecounts[1]


    #replace missing data with specified classification
    mdf_train[column + '_bnry'] = mdf_train[column + '_bnry'].fillna(binary_missing_plug)
    mdf_test[column + '_bnry'] = mdf_test[column + '_bnry'].fillna(binary_missing_plug)

    #if more than two remaining classifications, return error message    
    if len(mdf_train[column + '_bnry'].unique()) > 2 or len(mdf_test[column + '_bnry'].unique()) > 2:
        print('ERROR: number of categories in column for process_binary_class() call >2')
        return mdf_train

    #convert column to binary 0/1 classification
    lb = preprocessing.LabelBinarizer()
    mdf_train[column + '_bnry'] = lb.fit_transform(mdf_train[column + '_bnry'])
    mdf_test[column + '_bnry'] = lb.fit_transform(mdf_test[column + '_bnry'])

    #create list of columns
    bnrycolumns = [column + '_bnry']

    #change data types to 8-bit (1 byte) integers for memory savings
    mdf_train[column + '_bnry'] = mdf_train[column + '_bnry'].astype(np.int8)
    mdf_test[column + '_bnry'] = mdf_test[column + '_bnry'].astype(np.int8)

    #create list of columns associated with categorical transform (blank for now)
    categorylist = []

    bnrynormalization_dict = {column + '_bnry' : {'missing' : binary_missing_plug, \
                                                 'onevalue' : onevalue, \
                                                 'zerovalue' : zerovalue}}

    #store some values in the column_dict{} for use later in ML infill methods
    column_dict_list = []

    for bc in bnrycolumns:


      if bc[-5:] == '_bnry':

        column_dict = { bc : {'category' : 'bnry', \
                             'origcategory' : category, \
                             'normalization_dict' : bnrynormalization_dict, \
                             'origcolumn' : column, \
                             'columnslist' : bnrycolumns, \
                             'categorylist' : [bc], \
                             'infillmodel' : False, \
                             'infillcomplete' : False, \
                             'deletecolumn' : False}}

        column_dict_list.append(column_dict.copy())


    #return mdf, bnrycolumns, categorylist, column_dict_list
    return mdf_train, mdf_test, column_dict_list
  
  
  def process_text_class(self, mdf_train, mdf_test, column, category, \
                         postprocess_dict):
    '''
    #process_text_class(mdf_train, mdf_test, column, category)
    #preprocess column with text categories
    #takes as arguement two pandas dataframe containing training and test data respectively 
    #(mdf_train, mdf_test), and the name of the column string ('column')
    #and the name of the category from parent column (category)

    #note this trains both training and test data simultaneously due to unique treatment if any category
    #missing from training set but not from test set to ensure consistent formatting 

    #doesn't delete the original column from master dataframe but
    #creates onehot encodings
    #with columns named after column_ + text categories
    #any categories missing from the training set removed from test set
    #any category present in training but missing from test set given a column of zeros for consistent formatting
    #ensures order of all new columns consistent between both sets
    #returns two transformed dataframe (mdf_train, mdf_test) \
    #and a list of the new column names (textcolumns)
    
    #if only have training but not test data handy, use same training data for both dataframe inputs
    '''
    
    #store original column for later retrieval
    mdf_train[column + '_temp'] = mdf_train[column].copy()
    mdf_test[column + '_temp'] = mdf_test[column].copy()

    #convert column to category
    mdf_train[column] = mdf_train[column].astype('category')
    mdf_test[column] = mdf_test[column].astype('category')

    #if set is categorical we'll need the plug value for missing values included
    mdf_train[column] = mdf_train[column].cat.add_categories(['NAr2'])
    mdf_test[column] = mdf_test[column].cat.add_categories(['NAr2'])

    #replace NA with a dummy variable
    mdf_train[column] = mdf_train[column].fillna('NAr2')
    mdf_test[column] = mdf_test[column].fillna('NAr2')

    #replace numerical with string equivalent
    mdf_train[column] = mdf_train[column].astype(str)
    mdf_test[column] = mdf_test[column].astype(str)


    #extract categories for column labels
    #note that .unique() extracts the labels as a numpy array
    labels_train = mdf_train[column].unique()
    labels_train.sort(axis=0)
    labels_test = mdf_test[column].unique()
    labels_test.sort(axis=0)

    #transform text classifications to numerical id
    encoder = LabelEncoder()
    cat_train = mdf_train[column]
    cat_train_encoded = encoder.fit_transform(cat_train)

    cat_test = mdf_test[column]
    cat_test_encoded = encoder.fit_transform(cat_test)


    #apply onehotencoding
    onehotencoder = OneHotEncoder()
    cat_train_1hot = onehotencoder.fit_transform(cat_train_encoded.reshape(-1,1))
    cat_test_1hot = onehotencoder.fit_transform(cat_test_encoded.reshape(-1,1))

    #append column header name to each category listing
    #note the iteration is over a numpy array hence the [...] approach
    labels_train[...] = column + '_' + labels_train[...]
    labels_test[...] = column + '_' + labels_test[...]


    #convert sparse array to pandas dataframe with column labels
    df_train_cat = pd.DataFrame(cat_train_1hot.toarray(), columns=labels_train)
    df_test_cat = pd.DataFrame(cat_test_1hot.toarray(), columns=labels_test)

#     #add a missing column to train if it's not present
#     if column + '_NArw' not in df_train_cat.columns:
#       missingcolumn = pd.DataFrame(0, index=np.arange(df_train_cat.shape[0]), columns=[column+'_NArw'])
#       df_train_cat = pd.concat([missingcolumn, df_train_cat], axis=1)


    #Get missing columns in test set that are present in training set
    missing_cols = set( df_train_cat.columns ) - set( df_test_cat.columns )

    #Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        df_test_cat[c] = 0
    #Ensure the order of column in the test set is in the same order than in train set
    #Note this also removes categories in test set that aren't present in training set
    df_test_cat = df_test_cat[df_train_cat.columns]


    #concatinate the sparse set with the rest of our training data
    mdf_train = pd.concat([df_train_cat, mdf_train], axis=1)
    mdf_test = pd.concat([df_test_cat, mdf_test], axis=1)


    #replace original column from training data
    del mdf_train[column]    
    del mdf_test[column]

    mdf_train[column] = mdf_train[column + '_temp'].copy()
    mdf_test[column] = mdf_test[column + '_temp'].copy()

    del mdf_train[column + '_temp']    
    del mdf_test[column + '_temp']
    
    #delete _NArw column, this will be processed seperately in the processfamily function
    #delete support NArw2 column
    columnNArw = column + '_NArw'
    columnNAr2 = column + '_NAr2'
    if columnNAr2 in list(mdf_train):
      del mdf_train[columnNAr2]
    if columnNAr2 in list(mdf_test):
      del mdf_test[columnNAr2]

    
#     del mdf_train[column + '_NAr2']    
#     del mdf_test[column + '_NAr2']
    
    
    #create output of a list of the created column names
    NAcolumn = columnNAr2
    labels_train = list(df_train_cat)
    labels_train.remove(NAcolumn)
    textcolumns = labels_train
    
    #now we'll creaate a dicitonary of the columns : categories for later reference
    #reminder here is list of. unque values from original column
    #labels_train
    
    
    normalizationdictvalues = labels_train
    normalizationdictkeys = textcolumns
    
    normalizationdictkeys.sort()
    normalizationdictvalues.sort()
    
    textlabelsdict = dict(zip(normalizationdictkeys, normalizationdictvalues))
    
    
    
    #change data types to 8-bit (1 byte) integers for memory savings
    for textcolumn in textcolumns:
      mdf_train[textcolumn] = mdf_train[textcolumn].astype(np.int8)
      mdf_test[textcolumn] = mdf_test[textcolumn].astype(np.int8)


    #store some values in the text_dict{} for use later in ML infill methods
    column_dict_list = []

    categorylist = textcolumns.copy()
#     categorylist.remove(columnNArw)

    for tc in textcolumns:

      textnormalization_dict = {tc : {'textlabelsdict' : textlabelsdict}}
      
      if tc[-5:] != '_NArw':
      
        column_dict = {tc : {'category' : 'text', \
                             'origcategory' : category, \
                             'normalization_dict' : textnormalization_dict, \
                             'origcolumn' : column, \
                             'columnslist' : textcolumns, \
                             'categorylist' : categorylist, \
                             'infillmodel' : False, \
                             'infillcomplete' : False, \
                             'deletecolumn' : False}}

        column_dict_list.append(column_dict.copy())
      
      else:
        
        
        column_dict = {tc : {'category' : 'text', \
                             'origcategory' : category, \
                             'normalization_dict' : textnormalization_dict, \
                             'origcolumn' : column, \
                             'columnslist' : textcolumns, \
                             'categorylist' : categorylist, \
                             'infillmodel' : False, \
                             'infillcomplete' : False, \
                             'deletecolumn' : False}}
        
#         fixed what I think is an error, replaced 'categorylist' : [tc] w/ 'categorylist' : categorylist


        column_dict_list.append(column_dict.copy())

    
    #return mdf_train, mdf_test, textcolumns, categorylist
    return mdf_train, mdf_test, column_dict_list
  
  
  
  
  def process_time_class(self, mdf_train, mdf_test, column, category, \
                         postprocess_dict):
    '''
    #process_time_class(mdf_train, mdf_test, column, category)
    #preprocess column with time classifications
    #takes as arguement two pandas dataframe containing training and test data respectively 
    #(mdf_train, mdf_test), and the name of the column string ('column') and the
    #category fo the source column (category)

    #note this trains both training and test data simultaneously due to unique treatment if any category
    #missing from training set but not from test set to ensure consistent formatting 
    
    #creates distinct columns for year, month, day, hour, minute, second
    #each normalized to the mean and std, with missing values plugged with the mean
    #with columns named after column_ + time category
    #returns two transformed dataframe (mdf_train, mdf_test) and column_dict_list

    #if only have training but not test data handy, use same training data for both dataframe inputs
    '''
    
    #store original column for later retrieval
    mdf_train[column + '_temp'] = mdf_train[column].copy()
    mdf_test[column + '_temp'] = mdf_test[column].copy()


    #apply pd.to_datetime to column, note that the errors = 'coerce' needed for messy data
    mdf_train[column] = pd.to_datetime(mdf_train[column], errors = 'coerce')
    mdf_test[column] = pd.to_datetime(mdf_test[column], errors = 'coerce')

    #mdf_train[column].replace(-np.Inf, np.nan)
    #mdf_test[column].replace(-np.Inf, np.nan)

    #get mean of various categories of datetime objects to use to plug in missing cells
    meanyear = mdf_train[column].dt.year.mean()    
    meanmonth = mdf_train[column].dt.month.mean()
    meanday = mdf_train[column].dt.day.mean()
    meanhour = mdf_train[column].dt.hour.mean()
    meanminute = mdf_train[column].dt.minute.mean()
    meansecond = mdf_train[column].dt.second.mean()

    #get standard deviation of training data
    stdyear = mdf_train[column].dt.year.std()  
    stdmonth = mdf_train[column].dt.month.std()
    stdday = mdf_train[column].dt.day.std()
    stdhour = mdf_train[column].dt.hour.std()
    stdminute = mdf_train[column].dt.minute.std()
    stdsecond = mdf_train[column].dt.second.std()


    #create new columns for each category in train set
    mdf_train[column + '_year'] = mdf_train[column].dt.year
    mdf_train[column + '_month'] = mdf_train[column].dt.month
    mdf_train[column + '_day'] = mdf_train[column].dt.day
    mdf_train[column + '_hour'] = mdf_train[column].dt.hour
    mdf_train[column + '_minute'] = mdf_train[column].dt.minute
    mdf_train[column + '_second'] = mdf_train[column].dt.second

    #do same for test set
    mdf_test[column + '_year'] = mdf_test[column].dt.year
    mdf_test[column + '_month'] = mdf_test[column].dt.month
    mdf_test[column + '_day'] = mdf_test[column].dt.day
    mdf_test[column + '_hour'] = mdf_test[column].dt.hour
    mdf_test[column + '_minute'] = mdf_test[column].dt.minute 
    mdf_test[column + '_second'] = mdf_test[column].dt.second


    #replace missing data with training set mean
    mdf_train[column + '_year'] = mdf_train[column + '_year'].fillna(meanyear)
    mdf_train[column + '_month'] = mdf_train[column + '_month'].fillna(meanmonth)
    mdf_train[column + '_day'] = mdf_train[column + '_day'].fillna(meanday)
    mdf_train[column + '_hour'] = mdf_train[column + '_hour'].fillna(meanhour)
    mdf_train[column + '_minute'] = mdf_train[column + '_minute'].fillna(meanminute)
    mdf_train[column + '_second'] = mdf_train[column + '_second'].fillna(meansecond)

    #do same for test set
    mdf_test[column + '_year'] = mdf_test[column + '_year'].fillna(meanyear)
    mdf_test[column + '_month'] = mdf_test[column + '_month'].fillna(meanmonth)
    mdf_test[column + '_day'] = mdf_test[column + '_day'].fillna(meanday)
    mdf_test[column + '_hour'] = mdf_test[column + '_hour'].fillna(meanhour)
    mdf_test[column + '_minute'] = mdf_test[column + '_minute'].fillna(meanminute)
    mdf_test[column + '_second'] = mdf_test[column + '_second'].fillna(meansecond)

    #subtract mean from column for both train and test
    mdf_train[column + '_year'] = mdf_train[column + '_year'] - meanyear
    mdf_train[column + '_month'] = mdf_train[column + '_month'] - meanmonth
    mdf_train[column + '_day'] = mdf_train[column + '_day'] - meanday
    mdf_train[column + '_hour'] = mdf_train[column + '_hour'] - meanhour
    mdf_train[column + '_minute'] = mdf_train[column + '_minute'] - meanminute
    mdf_train[column + '_second'] = mdf_train[column + '_second'] - meansecond

    mdf_test[column + '_year'] = mdf_test[column + '_year'] - meanyear
    mdf_test[column + '_month'] = mdf_test[column + '_month'] - meanmonth
    mdf_test[column + '_day'] = mdf_test[column + '_day'] - meanday
    mdf_test[column + '_hour'] = mdf_test[column + '_hour'] - meanhour
    mdf_test[column + '_minute'] = mdf_test[column + '_minute'] - meanminute
    mdf_test[column + '_second'] = mdf_test[column + '_second'] - meansecond


    #divide column values by std for both training and test data
    mdf_train[column + '_year'] = mdf_train[column + '_year'] / stdyear
    mdf_train[column + '_month'] = mdf_train[column + '_month'] / stdmonth
    mdf_train[column + '_day'] = mdf_train[column + '_day'] / stdday
    mdf_train[column + '_hour'] = mdf_train[column + '_hour'] / stdhour
    mdf_train[column + '_minute'] = mdf_train[column + '_minute'] / stdminute
    mdf_train[column + '_second'] = mdf_train[column + '_second'] / stdsecond

    mdf_test[column + '_year'] = mdf_test[column + '_year'] / stdyear
    mdf_test[column + '_month'] = mdf_test[column + '_month'] / stdmonth
    mdf_test[column + '_day'] = mdf_test[column + '_day'] / stdday
    mdf_test[column + '_hour'] = mdf_test[column + '_hour'] / stdhour
    mdf_test[column + '_minute'] = mdf_test[column + '_minute'] / stdminute
    mdf_test[column + '_second'] = mdf_test[column + '_second'] / stdsecond


    #now replace NaN with 0
    mdf_train[column + '_year'] = mdf_train[column + '_year'].fillna(0)
    mdf_train[column + '_month'] = mdf_train[column + '_month'].fillna(0)
    mdf_train[column + '_day'] = mdf_train[column + '_day'].fillna(0)
    mdf_train[column + '_hour'] = mdf_train[column + '_hour'].fillna(0)
    mdf_train[column + '_minute'] = mdf_train[column + '_minute'].fillna(0)
    mdf_train[column + '_second'] = mdf_train[column + '_second'].fillna(0)

    #do same for test set
    mdf_test[column + '_year'] = mdf_test[column + '_year'].fillna(0)
    mdf_test[column + '_month'] = mdf_test[column + '_month'].fillna(0)
    mdf_test[column + '_day'] = mdf_test[column + '_day'].fillna(0)
    mdf_test[column + '_hour'] = mdf_test[column + '_hour'].fillna(0)
    mdf_test[column + '_minute'] = mdf_test[column + '_minute'].fillna(0)
    mdf_test[column + '_second'] = mdf_test[column + '_second'].fillna(0)

    #output of a list of the created column names
    datecolumns = [column + '_year', column + '_month', column + '_day', \
                  column + '_hour', column + '_minute', column + '_second']

    #this is to address an issue I found when parsing columns with only time no date
    #which returned -inf vlaues, so if an issue will just delete the associated 
    #column along with the entry in datecolumns
    checkyear = np.isinf(mdf_train.iloc[0][column + '_year'])
    if checkyear:
      del mdf_train[column + '_year']
      datecolumns.remove(column + '_year')
      if column + '_year' in mdf_test.columns:
        del mdf_test[column + '_year']

    checkmonth = np.isinf(mdf_train.iloc[0][column + '_month'])
    if checkmonth:
      del mdf_train[column + '_month']
      datecolumns.remove(column + '_month')
      if column + '_month' in mdf_test.columns:
        del mdf_test[column + '_month']

    checkday = np.isinf(mdf_train.iloc[0][column + '_day'])
    if checkday:
      del mdf_train[column + '_day']
      datecolumns.remove(column + '_day')
      if column + '_day' in mdf_test.columns:
        del mdf_test[column + '_day']


    #replace original column from training data
    del mdf_train[column]    
    del mdf_test[column]

    mdf_train[column] = mdf_train[column + '_temp'].copy()
    mdf_test[column] = mdf_test[column + '_temp'].copy()

    del mdf_train[column + '_temp']    
    del mdf_test[column + '_temp']

    #create list of columns associated with categorical transform (blank for now)
    categorylist = []


    #store some values in the date_dict{} for use later in ML infill methods

    column_dict_list = []

    categorylist = datecolumns.copy()



    for dc in categorylist:

      #save a dictionary of the associated column mean and std
      timenormalization_dict = \
      {dc : {'meanyear' : meanyear, 'meanmonth' : meanmonth, \
            'meanday' : meanday, 'meanhour' : meanhour, \
            'meanminute' : meanminute, 'meansecond' : meansecond,\
            'stdyear' : stdyear, 'stdmonth' : stdmonth, \
            'stdday' : stdday, 'stdhour' : stdhour, \
            'stdminute' : stdminute, 'stdsecond' : stdsecond}}

      column_dict = {dc : {'category' : 'date', \
                           'origcategory' : category, \
                           'normalization_dict' : timenormalization_dict, \
                           'origcolumn' : column, \
                           'columnslist' : datecolumns, \
                           'categorylist' : categorylist, \
                           'infillmodel' : False, \
                           'infillcomplete' : False, \
                           'deletecolumn' : False}}

      column_dict_list.append(column_dict.copy())
      
      
    return mdf_train, mdf_test, column_dict_list
  
  
  def process_bxcx_class(self, mdf_train, mdf_test, column, category, \
                         postprocess_dict):
    '''
    Applies Box-Cox transform to an all-positive numerical set.
    '''
    
    #df_train, nmbrcolumns, nmbrnormalization_dict, categorylist = \
    mdf_train, column_dict_list = \
    self.process_bxcx_support(mdf_train, column, category, 1, bxcx_lmbda = None, \
                              trnsfrm_mean = None)

    #grab the normalization_dict associated with the bxcx category
    columnkeybxcx = column + '_bxcx'
    for column_dict in column_dict_list:
      if columnkeybxcx in column_dict:
        bxcxnormalization_dict = column_dict[columnkeybxcx]['normalization_dict'][columnkeybxcx]

    #df_test, nmbrcolumns, _1, _2 = \
    mdf_test, _1 = \
    self.process_bxcx_support(mdf_test, column, category, 1, bxcx_lmbda = \
                             bxcxnormalization_dict['bxcx_lmbda'], \
                             trnsfrm_mean = bxcxnormalization_dict['trnsfrm_mean'])

    return mdf_train, mdf_test, column_dict_list
  
  
  
  
  def process_bxcx_support(self, df, column, category, bxcxerrorcorrect, \
                          bxcx_lmbda = None, trnsfrm_mean = None):
    '''                      
    #process_bxcx_class(df, column, bxcx_lmbda = None, trnsfrm_mean = None, trnsfrm_std = None)
    #function that takes as input a dataframe with numnerical column for purposes
    #of applying a box-cox transformation. If lmbda = None it will infer a suitable
    #lambda value by minimizing log likelihood using SciPy's stats boxcox call. If
    #we pass a mean or std value it will apply the mean for the initial infill and 
    #use the values to apply postprocess_numerical_class function. 
    #Returns transformed dataframe, a list nmbrcolumns of the associated columns,
    #and a normalization dictionary nmbrnormalization_dict which we'll use for our
    #postprocess_dict, and the parameter lmbda that was used

    #expect this approach works better than our prior numerical address when the 
    #distribution is less thin tailed
    '''
    
    #store original column for later reversion
    df[column + '_temp'] = df[column].copy()

    #convert all values to either numeric or NaN
    df[column] = pd.to_numeric(df[column], errors='coerce')

    #get the mean value to apply to infill
    if trnsfrm_mean == None:
      #get mean of training data
      mean = df[column].mean()  

    else:
      mean = trnsfrm_mean

    #replace missing data with training set mean
    df[column] = df[column].fillna(mean)

    #apply box-cox transformation to generate a new column
    #note the returns are different based on whether we passed a lmbda value

    if bxcx_lmbda == None:

      df[column + '_bxcx'], bxcx_lmbda = stats.boxcox(df[column])
      df[column + '_bxcx'] *= bxcxerrorcorrect

    else:

      df[column + '_bxcx'] = stats.boxcox(df[column], lmbda = bxcx_lmbda)
      df[column + '_bxcx'] *= bxcxerrorcorrect

    #this is to address an error when bxcx transofrm produces overflow
    #I'm not sure of cause, showed up in the housing set)
    bxcxerrorcorrect = 1
    if max(df[column + '_bxcx']) > (2 ** 31 - 1):
      bxcxerrorcorrect = 0
      df[column + '_bxcx'] = 0
      bxcxcolumn = column + '_bxcx'
      print("overflow condition found in boxcox transofrm, column set to 0: ", bxcxcolumn)



    #replace original column
    del df[column]

    df[column] = df[column + '_temp'].copy()

    del df[column + '_temp']


    #output of a list of the created column names
    #nmbrcolumns = [column + '_nmbr', column + '_bxcx', column + '_NArw']
    nmbrcolumns = [column + '_bxcx']

    #create list of columns associated with categorical transform (blank for now)
    categorylist = []


    #store some values in the nmbr_dict{} for use later in ML infill methods
    column_dict_list = []

    for nc in nmbrcolumns:


      #save a dictionary of the associated column mean and std

      normalization_dict = {nc : {'trnsfrm_mean' : mean, \
                                  'bxcx_lmbda' : bxcx_lmbda, \
                                  'bxcxerrorcorrect' : bxcxerrorcorrect, \
                                  'mean' : mean}}

      if nc[-5:] == '_bxcx':

        column_dict = { nc : {'category' : 'bxcx', \
                             'origcategory' : category, \
                             'normalization_dict' : normalization_dict, \
                             'origcolumn' : column, \
                             'columnslist' : nmbrcolumns, \
                             'categorylist' : [nc], \
                             'infillmodel' : False, \
                             'infillcomplete' : False, \
                             'deletecolumn' : False}}

        column_dict_list.append(column_dict.copy())




    #return df, nmbrcolumns, nmbrnormalization_dict, categorylist
    return df, column_dict_list
  
  
  
  
  
  
  def process_bins_class(self, mdf_train, mdf_test, column, category, \
                         postprocess_dict):
    '''
    #processes a numerical set by creating bins coresponding to post z score
    #normalization of <-2, -2-1, -10, 01, 12, >2 in one hot encoded columns
    
    #bins will be intended for a raw set that is not normalized
    #bint will be intended for a previously normalized set
    '''

    #store original column for later reversion
    mdf_train[column + '_temp'] = mdf_train[column].copy()
    mdf_test[column + '_temp'] = mdf_test[column].copy()

    #convert all values to either numeric or NaN
    mdf_train[column] = pd.to_numeric(mdf_train[column], errors='coerce')
    mdf_test[column] = pd.to_numeric(mdf_test[column], errors='coerce')

    #get mean of training data
    mean = mdf_train[column].mean()    

    #replace missing data with training set mean
    mdf_train[column] = mdf_train[column].fillna(mean)
    mdf_test[column] = mdf_test[column].fillna(mean)

    #subtract mean from column for both train and test
    mdf_train[column] = mdf_train[column] - mean
    mdf_test[column] = mdf_test[column] - mean

    #get standard deviation of training data
    std = mdf_train[column].std()

    #divide column values by std for both training and test data
    mdf_train[column] = mdf_train[column] / std
    mdf_test[column] = mdf_test[column] / std


    #create bins based on standard deviation increments
    binscolumn = column + '_bins'
    mdf_train[binscolumn] = \
    pd.cut( mdf_train[column], bins = [-float('inf'),-2,-1,0,1,2,float('inf')],  \
           labels = ['s<-2','s-21','s-10','s+01','s+12','s>+2'], precision=4)
    mdf_test[binscolumn] = \
    pd.cut( mdf_test[column], bins = [-float('inf'), -2, -1, 0, 1, 2, float('inf')],  \
           labels = ['s<-2','s-21','s-10','s+01','s+12','s>+2'], precision=4)



    textcolumns = \
    [binscolumn + '_s<-2', binscolumn + '_s-21', binscolumn + '_s-10', \
     binscolumn + '_s+01', binscolumn + '_s+12', binscolumn + '_s>+2']

    

    
    #we're going to use the postprocess_text_class function here since it 
    #allows us to force the columns even if no values present in the set
    #however to do so we're going to have to construct a fake postprocess_dict
    
    #a future extension should probnably build this capacity into a new distinct function
    
    #here are some data structures for reference to create the below
#     def postprocess_text_class(self, mdf_test, column, postprocess_dict, columnkey):
#     textcolumns = postprocess_dict['column_dict'][columnkey]['columnslist']
    
    tempkey = 'tempkey'
    temppostprocess_dict = {'column_dict' : {tempkey : {'columnslist' : textcolumns,\
                                                        'categorylist' : textcolumns}}}
    
    

    
    #process bins as a categorical set
    mdf_train = \
    self.postprocess_text_class(mdf_train, binscolumn, temppostprocess_dict, tempkey)
    mdf_test = \
    self.postprocess_text_class(mdf_test, binscolumn, temppostprocess_dict, tempkey)
    
    
    #delete the support column
    del mdf_train[binscolumn]
    del mdf_test[binscolumn]

    #replace original column
    del mdf_train[column]
    del mdf_test[column]
    mdf_train[column] = mdf_train[column + '_temp'].copy()
    mdf_test[column] = mdf_test[column + '_temp'].copy()
    del mdf_train[column + '_temp']
    del mdf_test[column + '_temp']



    #create list of columns
    nmbrcolumns = textcolumns



    #nmbrnormalization_dict = {'mean' : mean, 'std' : std}

    #store some values in the nmbr_dict{} for use later in ML infill methods
    column_dict_list = []

    for nc in nmbrcolumns:

      nmbrnormalization_dict = {nc : {'binsmean' : mean, 'binsstd' : std}}

      if nc in textcolumns:

        column_dict = { nc : {'category' : 'bins', \
                             'origcategory' : category, \
                             'normalization_dict' : nmbrnormalization_dict, \
                             'origcolumn' : column, \
                             'columnslist' : nmbrcolumns, \
                             'categorylist' : textcolumns, \
                             'infillmodel' : False, \
                             'infillcomplete' : False, \
                             'deletecolumn' : False}}

        column_dict_list.append(column_dict.copy())



    #return mdf_train, mdf_test, mean, std, nmbrcolumns, categorylist
    return mdf_train, mdf_test, column_dict_list
  
  
  
  
  def process_bint_class(self, mdf_train, mdf_test, column, category, \
                         postprocess_dict):
    '''
    #processes a numerical set by creating bins coresponding to post z score
    #normalization of <-2, -2-1, -10, 01, 12, >2 in one hot encoded columns
    
    #bins will be intended for a raw set that is not normalized
    #bint will be intended for a previously z-score normalized set
    #with mean 0 and std 1
    '''

    #store original column for later reversion
    mdf_train[column + '_temp'] = mdf_train[column].copy()
    mdf_test[column + '_temp'] = mdf_test[column].copy()

    #convert all values to either numeric or NaN
    mdf_train[column] = pd.to_numeric(mdf_train[column], errors='coerce')
    mdf_test[column] = pd.to_numeric(mdf_test[column], errors='coerce')

    #get mean of training data
    #mean = mdf_train[column].mean()
    mean = 0

    #replace missing data with training set mean
    mdf_train[column] = mdf_train[column].fillna(mean)
    mdf_test[column] = mdf_test[column].fillna(mean)

#     #subtract mean from column for both train and test
#     mdf_train[column] = mdf_train[column] - mean
#     mdf_test[column] = mdf_test[column] - mean

#     #get standard deviation of training data
#     std = mdf_train[column].std()

#     #divide column values by std for both training and test data
#     mdf_train[column] = mdf_train[column] / std
#     mdf_test[column] = mdf_test[column] / std


    #create bins based on standard deviation increments
    binscolumn = column + '_bint'
    mdf_train[binscolumn] = \
    pd.cut( mdf_train[column], bins = [-float('inf'),-2,-1,0,1,2,float('inf')],  \
           labels = ['t<-2','t-21','t-10','t+01','t+12','t>+2'], precision=4)
    mdf_test[binscolumn] = \
    pd.cut( mdf_test[column], bins = [-float('inf'), -2, -1, 0, 1, 2, float('inf')],  \
           labels = ['t<-2','t-21','t-10','t+01','t+12','t>+2'], precision=4)



    textcolumns = \
    [binscolumn + '_t<-2', binscolumn + '_t-21', binscolumn + '_t-10', \
     binscolumn + '_t+01', binscolumn + '_t+12', binscolumn + '_t>+2']

    

    
    #we're going to use the postprocess_text_class function here since it 
    #allows us to force the columns even if no values present in the set
    #however to do so we're going to have to construct a fake postprocess_dict
    
    #a future extension should probnably build this capacity into a new distinct function
    
    #here are some data structures for reference to create the below
#     def postprocess_text_class(self, mdf_test, column, postprocess_dict, columnkey):
#     textcolumns = postprocess_dict['column_dict'][columnkey]['columnslist']
    
    tempkey = 'tempkey'
    temppostprocess_dict = {'column_dict' : {tempkey : {'columnslist' : textcolumns, \
                                                        'categorylist' : textcolumns}}}
    
    #process bins as a categorical set
    mdf_train = \
    self.postprocess_text_class(mdf_train, binscolumn, temppostprocess_dict, tempkey)
    mdf_test = \
    self.postprocess_text_class(mdf_test, binscolumn, temppostprocess_dict, tempkey)
    
    
    #delete the support column
    del mdf_train[binscolumn]
    del mdf_test[binscolumn]

    #replace original column
    del mdf_train[column]
    del mdf_test[column]
    mdf_train[column] = mdf_train[column + '_temp'].copy()
    mdf_test[column] = mdf_test[column + '_temp'].copy()
    del mdf_train[column + '_temp']
    del mdf_test[column + '_temp']



    #create list of columns
    nmbrcolumns = textcolumns



    #nmbrnormalization_dict = {'mean' : mean, 'std' : std}

    #store some values in the nmbr_dict{} for use later in ML infill methods
    column_dict_list = []

    for nc in nmbrcolumns:

      nmbrnormalization_dict = {nc : {'bintmean' : 0, 'bintstd' : 1}}

      if nc in textcolumns:

        column_dict = { nc : {'category' : 'bint', \
                             'origcategory' : category, \
                             'normalization_dict' : nmbrnormalization_dict, \
                             'origcolumn' : column, \
                             'columnslist' : nmbrcolumns, \
                             'categorylist' : textcolumns, \
                             'infillmodel' : False, \
                             'infillcomplete' : False, \
                             'deletecolumn' : False}}

        column_dict_list.append(column_dict.copy())



    #return mdf_train, mdf_test, mean, std, nmbrcolumns, categorylist
    return mdf_train, mdf_test, column_dict_list
  
  
  
  
  def process_null_class(self, df, column, category, postprocess_dict):
    '''
    #here we'll delete any columns that returned a 'null' category
    #note this is a. singleprocess transform
    '''
    
    df = df.drop([column], axis=1)

    column_dict_list = []

    column_dict = {column + '_null' : {'category' : 'null', \
                                      'origcategory' : category, \
                                      'normalization_dict' : {column + '_null':{}}, \
                                      'origcolumn' : column, \
                                      'columnslist' : [column], \
                                      'categorylist' : [], \
                                      'infillmodel' : False, \
                                      'infillcomplete' : False, \
                                      'deletecolumn' : False}}
    
    #now append column_dict onto postprocess_dict
    column_dict_list.append(column_dict.copy())



    return df, column_dict_list


  def process_excl_class(self, df, column, category, postprocess_dict):
    '''
    #here we'll address any columns that returned a 'excl' category
    #note this is a. singleprocess transform
    #we'll simply maintain the same column but with a suffix to the header
    '''
    exclcolumn = column + '_excl'
    df[exclcolumn] = df[column].copy()
    #del df[column]
    
    column_dict_list = []

    column_dict = {exclcolumn : {'category' : 'excl', \
                                 'origcategory' : category, \
                                 'normalization_dict' : {exclcolumn:{}}, \
                                 'origcolumn' : column, \
                                 'columnslist' : [exclcolumn], \
                                 'categorylist' : [exclcolumn], \
                                 'infillmodel' : False, \
                                 'infillcomplete' : False, \
                                 'deletecolumn' : False}}
    
    #now append column_dict onto postprocess_dict
    column_dict_list.append(column_dict.copy())



    return df, column_dict_list  

    
#   #this method needs troubleshooting, for now just use excl
#   def process_exc2_class(self, df, column, category, postprocess_dict):
#     '''
#     #here we'll address any columns that returned a 'exc2' category
#     #note this is a. singleprocess transform
#     #we'll simply populate the column_dict, no new column
#     '''
#     #exclcolumn = column + '_excl'
#     exclcolumn = column
#     #df[exclcolumn] = df[column].copy()
#     #del df[column]
    
#     column_dict_list = []

#     column_dict = {exclcolumn : {'category' : 'excl', \
#                                  'origcategory' : category, \
#                                  'normalization_dict' : {exclcolumn:{}}, \
#                                  'origcolumn' : column, \
#                                  'columnslist' : [exclcolumn], \
#                                  'categorylist' : [exclcolumn], \
#                                  'infillmodel' : False, \
#                                  'infillcomplete' : False, \
#                                  'deletecolumn' : False}}
    
#     #now append column_dict onto postprocess_dict
#     column_dict_list.append(column_dict.copy())



#     return df, column_dict_list  


  def evalcategory(self, df, column, numbercategoryheuristic):
    '''
    #evalcategory(df, column)
    #Function that dakes as input a dataframe and associated column id \
    #evaluates the contents of cells and classifies the column into one of four categories
    #category 1, 'bnry', is for columns with only two categorys of text or integer
    #category 2, 'nmbr', is for columns with ndumerical integer or float values
    #category 3: 'bxcx', is for nmbr category with all positive values
    #category 4, 'text', is for columns with multiple categories appropriate for one-hot
    #category 5, 'date', is for columns with Timestamp data
    #category 6, 'null', is for columns with >85% null values (arbitrary figure)
    #returns category id as a string
    '''

    #I couldn't find a good pandas tool for evaluating data class, \
    #So will produce an array containing data types of each cell and \
    #evaluate for most common variable using the collections library

    type1_df = df[column].apply(lambda x: type(x)).values

    c = collections.Counter(type1_df)
    mc = c.most_common(1)
    mc2 = c.most_common(2)

    #free memory (dtypes are memory hogs)
    type1_df = None


    #additional array needed to check for time series

    #df['typecolumn2'] = df[column].apply(lambda x: type(pd.to_datetime(x, errors = 'coerce')))
    type2_df = df[column].apply(lambda x: type(pd.to_datetime(x, errors = 'coerce'))).values

    datec = collections.Counter(type2_df)
    datemc = datec.most_common(1)
    datemc2 = datec.most_common(2)

    #free memory (dtypes are memory hogs)
    type2_df = None


    #an extension of this approach could be for those columns that produce a text\
    #category to implement an additional text to determine the number of \
    #common groupings / or the amount of uniquity. For example if every row has\
    #a unique value then one-hot-encoding would not be appropriate. It would \
    #probably be apopropraite to either return an error message if this is found \
    #or alternatively find a furhter way to automate this processing such as \
    #look for contextual clues to groupings that can be inferred.

    #This is kind of hack to evaluate class by comparing these with output of mc
    checkint = 1
    checkfloat = 1.1
    checkstring = 'string'
    checkNAN = None

    #there's probably easier way to do this, here will create a check for date
    df_checkdate = pd.DataFrame([{'checkdate' : '7/4/2018'}])
    df_checkdate['checkdate'] = pd.to_datetime(df_checkdate['checkdate'], errors = 'coerce')


    #create dummy variable to store determined class (default is text class)
    category = 'text'


    #if most common in column is string and > two values, set category to text
    if isinstance(checkstring, mc[0][0]) and df[column].nunique() > 2:
      category = 'text'

    #if most common is date, set category to date
    if isinstance(df_checkdate['checkdate'][0], datemc[0][0]):
      category = 'date'

    #if most common in column is integer and > two values, set category to number of bxcx
    if isinstance(checkint, mc[0][0]) and df[column].nunique() > 2:

      #take account for numbercategoryheuristic
      if df[column].nunique() / df[column].shape[0] < numbercategoryheuristic:

        category = 'text'

      else:

        #if all postiive set category to bxcx

        #if (pd.to_numeric(df[column], errors = 'coerce').notnull() >= 0).all():

        #note we'll only allow bxcx category if all values greater than a clip value
        #>0 (currently set at 0.1)since there is an asymptote for box-cox at 0
        if (df[pd.to_numeric(df[column], errors='coerce').notnull()][column] >= 0.1).all():
          category = 'bxcx'
          #note a future extension may test for skewness before assigning bxcx category

        #note a future extension mayt test for skewness here and only assign category
        #of bxcx for skewness beyond a certain threshold

        else:
          category = 'nmbr'

    #if most common in column is float, set category to number or bxcx
    if isinstance(checkfloat, mc[0][0]):

      #take account for numbercategoryheuristic
      if df[column].nunique() / df[column].shape[0] < numbercategoryheuristic \
      or df[column].dtype.name == 'category':
        category = 'text'

      else:

        #if all postiive set category to bxcx

        #note we'll only allow bxcx category if all values greater than a clip value
        #>0 (currently set at 0.1) since there is an asymptote for box-cox at 0
        if (df[pd.to_numeric(df[column], errors='coerce').notnull()][column] >= 0.1).all():
          category = 'bxcx'

        else:
          category = 'nmbr'

    #if most common in column is integer and <= two values, set category to binary
    if isinstance(checkint, mc[0][0]) and df[column].nunique() <= 2:
      category = 'bnry'

    #if most common in column is string and <= two values, set category to binary
    if isinstance(checkstring, mc[0][0]) and df[column].nunique() <= 2:
      category = 'bnry'


    #if > 80% (ARBITRARY FIGURE) are NaN we'll just delete the column
    if df[column].isna().sum() >= df.shape[0] * 0.80:
      category = 'null'

    #else if most common in column is NaN, re-evaluate using the second most common type
    #(I suspect the below might have a bug somewhere but is working on my current 
    #tests so will leave be for now)
    elif df[column].isna().sum() >= df.shape[0] / 2:

      #if 2nd most common in column is string and > two values, set category to text
      if isinstance(checkstring, mc2[0][0]) and df[column].nunique() > 2:
        category = 'text'

      #if 2nd most common is date, set category to date   
      if isinstance(df_checkdate['checkdate'][0], datemc2[0][0]):
        category = 'date'

      #if 2nd most common in column is integer and > two values, set category to number
      if isinstance(checkint, mc2[0][0]) and df[column].nunique() > 2:


        #take account for numbercategoryheuristic
        if df[column].nunique() / df[column].shape[0] < numbercategoryheuristic:

          category = 'text'

        else:

          category = 'nmbr'

      #if 2nd most common in column is float, set category to number
      if isinstance(checkfloat, mc2[0][0]):

        #take account for numbercategoryheuristic
        if df[column].nunique() / df[column].shape[0] < numbercategoryheuristic:

          category = 'text'

        else:

          category = 'nmbr'

      #if 2nd most common in column is integer and <= two values, set category to binary
      if isinstance(checkint, mc2[0][0]) and df[column].nunique() <= 2:
        category = 'bnry'

      #if 2nd most common in column is string and <= two values, set category to binary
      if isinstance(checkstring, mc2[0][0]) and df[column].nunique() <= 2:
        category = 'bnry'


    return category





  def NArows(self, df, column, category, postprocess_dict):
    '''
    #NArows(df, column), function that when fed a dataframe, \
    #column id, and category label outputs a single column dataframe composed of \
    #True and False with the same number of rows as the input and the True's \
    #coresponding to those rows of the input that had missing or NaN data. This \
    #output can later be used to identify which rows for a column to infill with ML\
    # derived plug data
    '''
    
    NArowtype = postprocess_dict['process_dict'][category]['NArowtype']
    
    #if category == 'text':
    if NArowtype in ['justNaN']:

      #returns dataframe of True and False, where True coresponds to the NaN's
      #renames column name to column + '_NArows'
      NArows = pd.isna(df[column])
      NArows = pd.DataFrame(NArows)
      NArows = NArows.rename(columns = {column:column+'_NArows'})

#     if category == 'bnry':

#       #returns dataframe of True and False, where True coresponds to the NaN's
#       #renames column name to column + '_NArows'
#       NArows = pd.isna(df[column])
#       NArows = pd.DataFrame(NArows)
#       NArows = NArows.rename(columns = {column:column+'_NArows'})

    #if category == 'nmbr' or category == 'bxcx':
    #if category in ['nmbr', 'bxcx', 'nbr2']:
    if NArowtype in ['numeric']:

      #convert all values to either numeric or NaN
      df[column] = pd.to_numeric(df[column], errors='coerce')

      #returns dataframe of True and False, where True coresponds to the NaN's
      #renames column name to column + '_NArows'
      NArows = pd.isna(df[column])
      NArows = pd.DataFrame(NArows)
      NArows = NArows.rename(columns = {column:column+'_NArows'})


#     if category == 'date':

#       #returns dataframe column of all False
#       #renames column name to column + '_NArows'
#       NArows = pd.isna(df[column])
#       NArows = pd.DataFrame(NArows)
#       NArows = NArows.rename(columns = {column:column+'_NArows'})
      
#       NArows = pd.DataFrame(False, index=np.arange(df.shape[0]), columns=[column+'NA'])
#       NArows = pd.DataFrame(NArows)
#       NArows = NArows.rename(columns = {column:column+'_NArows'})

    #if category in ['excl']:
    if NArowtype in ['exclude']:
      
      NArows = pd.DataFrame(np.zeros((df.shape[0], 1)), columns=[column+'_NArows'])
      #NArows = NArows.rename(columns = {column:column+'_NArows'})
    

    return NArows




  def labelbinarizercorrect(self, npinput, columnslist):
    '''
    #labelbinarizercorrect(npinput, columnslist), function that takes as input the output\
    #array from scikit learn's LabelBinarizer() and ensures that the re-encoding is\
    #consistent with the original array prior to performing the argmax. This is \
    #needed because LabelBinarizer automatically takes two class sets to a binary\
    #setting and doesn't account for columns above index of active values based on\
    #my understanding. For a large enough dataset this probably won't be an issue \
    #but just trying to be thorough. Outputs a one-hot encoded array comparable to \
    #the format of our input to argmax.
    '''

    #if our array post application of LabelBinarizer has few coloumns than our \
    #column list then run through these loops
    if npinput.shape[1] < len(columnslist):

      #if only one column in our array means LabelEncoder must have binarized \
      #since we already established that there are more columns
      if npinput.shape[1] == 1:

        #this transfers from the binary encoding to two columns of one hot
        npinput = np.hstack((1 - npinput, npinput))

        np_corrected = npinput

      #if we still have fewer columns than the column list, means we'll need to \
      #pad out with columns containing zeros
      if npinput.shape[1] < len(columnslist):
        missingcols = len(columnslist) - npinput.shape[1]
        append = np.zeros((npinput.shape[0], missingcols))
        np_corrected = np.concatenate((npinput, append), axis=1)

    else:
      #otherwise just return the input array because it is in good shape
      np_corrected = npinput


    return np_corrected






  def predictinfill(self, category, df_train_filltrain, df_train_filllabel, \
                    df_train_fillfeatures, df_test_fillfeatures, randomseed, \
                    postprocess_dict, columnslist = []):
    '''
    #predictinfill(category, df_train_filltrain, df_train_filllabel, \
    #df_train_fillfeatures, df_test_fillfeatures, randomseed, columnslist), \
    #function that takes as input \
    #a category string, the output of createMLinfillsets(.), a seed for randomness \
    #and a list of columns produced by a text class preprocessor when applicable and 
    #returns predicted infills for the train and test feature sets as df_traininfill, \
    #df_testinfill based on derivations using scikit-learn, with the lenth of \
    #infill consistent with the number of True values from NArows, and the trained \
    #model

    #a reasonable extension of this funciton would be to allow ML inference with \
    #other ML architectures such a SVM or something SGD based for instance
    '''
    
    MLinfilltype = postprocess_dict['process_dict'][category]['MLinfilltype']
    
    #convert dataframes to numpy arrays
    np_train_filltrain = df_train_filltrain.values
    np_train_filllabel = df_train_filllabel.values
    np_train_fillfeatures = df_train_fillfeatures.values
    np_test_fillfeatures = df_test_fillfeatures.values

    #ony run the following if we have any rows needing infill
    if df_train_fillfeatures.shape[0] > 0:
      
      #if a numerical set
      #if category in ['nmbr', 'nbr2', 'bxcx']:
      if MLinfilltype in ['numeric']:

        #this is to address a weird error message suggesting I reshape the y with ravel()
        np_train_filllabel = np.ravel(np_train_filllabel)

        #train linear regression model using scikit-learn for numerical prediction
        #model = LinearRegression()
        #model = PassiveAggressiveRegressor(random_state = randomseed)
        #model = Ridge(random_state = randomseed)
        #model = RidgeCV()
        #note that SVR doesn't have an argument for random_state
        #model = SVR()
        model = RandomForestRegressor(n_estimators=100, random_state = randomseed, verbose=0)

        model.fit(np_train_filltrain, np_train_filllabel)    


        #predict infill values
        np_traininfill = model.predict(np_train_fillfeatures)

        #only run following if we have any test rows needing infill
        if df_test_fillfeatures.shape[0] > 0:
          np_testinfill = model.predict(np_test_fillfeatures)
        else:
          np_testinfill = np.array([0])

        #convert infill values to dataframe
        df_traininfill = pd.DataFrame(np_traininfill, columns = ['infill'])
        df_testinfill = pd.DataFrame(np_testinfill, columns = ['infill'])


#       if category == 'bxcx':

#         #this is to address a weird error message suggesting I reshape the y with ravel()
#         np_train_filllabel = np.ravel(np_train_filllabel)

#         #model = SVR()
#         model = RandomForestRegressor(random_state = randomseed)

#         model.fit(np_train_filltrain, np_train_filllabel)   

#         #predict infill values
#         np_traininfill = model.predict(np_train_fillfeatures)


#         #only run following if we have any test rows needing infill
#         if df_test_fillfeatures.shape[0] > 0:
#           np_testinfill = model.predict(np_test_fillfeatures)
#         else:
#           np_testinfill = np.array([0])

#         #convert infill values to dataframe
#         df_traininfill = pd.DataFrame(np_traininfill, columns = ['infill'])
#         df_testinfill = pd.DataFrame(np_testinfill, columns = ['infill'])     



      #if category == 'bnry':
      if MLinfilltype in ['singlct']:

        #this is to address a weird error message suggesting I reshape the y with ravel()
        np_train_filllabel = np.ravel(np_train_filllabel)

        #train logistic regression model using scikit-learn for binary classifier
        #model = LogisticRegression()
        #model = LogisticRegression(random_state = randomseed)
        #model = SGDClassifier(random_state = randomseed)
        #model = SVC(random_state = randomseed)
        model = RandomForestClassifier(n_estimators=100, random_state = randomseed, verbose=0)

        model.fit(np_train_filltrain, np_train_filllabel)

        #predict infill values
        np_traininfill = model.predict(np_train_fillfeatures)

        #only run following if we have any test rows needing infill
        if df_test_fillfeatures.shape[0] > 0:
          np_testinfill = model.predict(np_test_fillfeatures)
        else:
          np_testinfill = np.array([0])

        #convert infill values to dataframe
        df_traininfill = pd.DataFrame(np_traininfill, columns = ['infill'])
        df_testinfill = pd.DataFrame(np_testinfill, columns = ['infill'])

  #       print('category is bnry, df_traininfill is')
  #       print(df_traininfill)

      #if category in ['text', 'bins', 'bint']:
      if MLinfilltype in ['multirt', 'multisp']:

        #first convert the one-hot encoded set via argmax to a 1D array
        np_train_filllabel_argmax = np.argmax(np_train_filllabel, axis=1)

        #train logistic regression model using scikit-learn for binary classifier
        #with multi_class argument activated
        #model = LogisticRegression()
        #model = SGDClassifier(random_state = randomseed)
        #model = SVC(random_state = randomseed)
        model = RandomForestClassifier(n_estimators=100, random_state = randomseed, verbose=0)

        model.fit(np_train_filltrain, np_train_filllabel_argmax)

        #predict infill values
        np_traininfill = model.predict(np_train_fillfeatures)

        #only run following if we have any test rows needing infill
        if df_test_fillfeatures.shape[0] > 0:
          np_testinfill = model.predict(np_test_fillfeatures)
        else:
          #this needs to have same number of columns as text category
          np_testinfill = np.zeros(shape=(1,len(columnslist)))

        #convert the 1D arrary back to one hot encoding
        labelbinarizertrain = preprocessing.LabelBinarizer()
        labelbinarizertrain.fit(np_traininfill)
        np_traininfill = labelbinarizertrain.transform(np_traininfill)

        #only run following if we have any test rows needing infill
        if df_test_fillfeatures.shape[0] > 0:
          labelbinarizertest = preprocessing.LabelBinarizer()
          labelbinarizertest.fit(np_testinfill)
          np_testinfill = labelbinarizertest.transform(np_testinfill)



        #run function to ensure correct dimensions of re-encoded classifier array
        np_traininfill = self.labelbinarizercorrect(np_traininfill, columnslist)

        if df_test_fillfeatures.shape[0] > 0:
          np_testinfill = self.labelbinarizercorrect(np_testinfill, columnslist)


        #convert infill values to dataframe
        df_traininfill = pd.DataFrame(np_traininfill, columns = [columnslist])
        df_testinfill = pd.DataFrame(np_testinfill, columns = [columnslist]) 


  #       print('category is text, df_traininfill is')
  #       print(df_traininfill)

      #if category in ['date', 'NArw', 'null']:
      if MLinfilltype in ['exclude']:

        #create empty sets for now
        #an extension of this method would be to implement a comparable infill \
        #method for the time category, based on the columns output from the \
        #preprocessing
        df_traininfill = pd.DataFrame({'infill' : [0]}) 
        df_testinfill = pd.DataFrame({'infill' : [0]}) 

        model = False

  #       print('category is text, df_traininfill is')
  #       print(df_traininfill)


    #else if we didn't have any infill rows let's create some plug values
    else:

      if category == 'text':
        np_traininfill = np.zeros(shape=(1,len(columnslist)))
        np_testinfill = np.zeros(shape=(1,len(columnslist)))
        df_traininfill = pd.DataFrame(np_traininfill, columns = [columnslist])
        df_testinfill = pd.DataFrame(np_testinfill, columns = [columnslist]) 

      else :
        df_traininfill = pd.DataFrame({'infill' : [0]}) 
        df_testinfill = pd.DataFrame({'infill' : [0]}) 

      #set model to False, this will be be needed for this eventiality in 
      #test set post-processing
      model = False

    return df_traininfill, df_testinfill, model






  def createMLinfillsets(self, df_train, df_test, column, trainNArows, testNArows, \
                         category, randomseed, postprocess_dict, columnslist = [], \
                         categorylist = []):
    '''
    #update createMLinfillsets as follows:
    #instead of diferientiation by category, do a test for whether categorylist = []
    #if so do a single column transform excluding those other columns from columnslist
    #in the sets comparable to , otherwise do a transform comparable to text category

    #createMLinfillsets(df_train, df_test, column, trainNArows, testNArows, \
    #category, columnslist = []) function that when fed dataframes of train and\
    #test sets, column id, df of True/False corresponding to rows from original \
    #sets with missing values, a string category of 'text', 'date', 'nmbr', or \
    #'bnry', and a list of column id's for the text category if applicable. The \
    #function returns a seris of dataframes which can be applied to training a \
    #machine learning model to predict apppropriate infill values for those points \
    #that had missing values from the original sets, indlucing returns of \
    #df_train_filltrain, df_train_filllabel, df_train_fillfeatures, \
    #and df_test_fillfeatures
    '''
    
    MLinfilltype = postprocess_dict['process_dict'][category]['MLinfilltype']
    
    #create 3 new dataframes for each train column - the train and labels \
    #for rows not needing infill, and the features for rows needing infill \
    #also create a test features column 
    
    #categories are nmbr, bnry, text, date, bxcx, bins, bint, NArw, null
    #if category in ['nmbr', 'bxcx', 'bnry', 'text', 'bins', 'bint']:
    
    #if category in ['nmbr', 'nbr2', 'bxcx', 'bnry', 'text', 'bins', 'bint']:
    if MLinfilltype in ['numeric', 'singlct', 'multirt', 'multisp']:

      #if this is a single column set (not categorical)
      if len(categorylist) == 1:

        #first concatinate the NArows True/False designations to df_train & df_test
        df_train = pd.concat([df_train, trainNArows], axis=1)
        df_test = pd.concat([df_test, testNArows], axis=1)

        #create copy of df_train to serve as training set for fill
        df_train_filltrain = df_train.copy()
        #now delete rows coresponding to True
        df_train_filltrain = df_train_filltrain[df_train_filltrain[trainNArows.columns.get_values()[0]] == False]
        

        #now delete columns = columnslist and the NA labels (orig column+'_NArows') from this df
        df_train_filltrain = df_train_filltrain.drop(columnslist, axis=1)
        df_train_filltrain = df_train_filltrain.drop([trainNArows.columns.get_values()[0]], axis=1)



        #create a copy of df_train[column] for fill train labels
        df_train_filllabel = pd.DataFrame(df_train[column].copy())
        #concatinate with the NArows
        df_train_filllabel = pd.concat([df_train_filllabel, trainNArows], axis=1)
        #drop rows corresponding to True
        df_train_filllabel = df_train_filllabel[df_train_filllabel[trainNArows.columns.get_values()[0]] == False]

        #delete the NArows column
        df_train_filllabel = df_train_filllabel.drop([trainNArows.columns.get_values()[0]], axis=1)

        #create features df_train for rows needing infill
        #create copy of df_train (note it already has NArows included)
        df_train_fillfeatures = df_train.copy()
        #delete rows coresponding to False
        df_train_fillfeatures = df_train_fillfeatures[(df_train_fillfeatures[trainNArows.columns.get_values()[0]])]
        #delete columnslist and column+'_NArows'
        df_train_fillfeatures = df_train_fillfeatures.drop(columnslist, axis=1)
        df_train_fillfeatures = df_train_fillfeatures.drop([trainNArows.columns.get_values()[0]], axis=1)


        #create features df_test for rows needing infill
        #create copy of df_test (note it already has NArows included)
        df_test_fillfeatures = df_test.copy()
        #delete rows coresponding to False
        df_test_fillfeatures = df_test_fillfeatures[(df_test_fillfeatures[testNArows.columns.get_values()[0]])]
        #delete column and column+'_NArows'
        df_test_fillfeatures = df_test_fillfeatures.drop(columnslist, axis=1)
        df_test_fillfeatures = df_test_fillfeatures.drop([testNArows.columns.get_values()[0]], axis=1)

        #delete NArows from df_train, df_test
        df_train = df_train.drop([trainNArows.columns.get_values()[0]], axis=1)
        df_test = df_test.drop([testNArows.columns.get_values()[0]], axis=1)





      #else if categorylist wasn't single entry
      else:

        #create a list of columns representing columnslist exlucding elements from
        #categorylist
        noncategorylist = columnslist[:]
        #this removes categorylist elements from noncategorylist
        noncategorylist = list(set(noncategorylist).difference(set(categorylist)))


        #first concatinate the NArows True/False designations to df_train & df_test
        df_train = pd.concat([df_train, trainNArows], axis=1)
        df_test = pd.concat([df_test, testNArows], axis=1)

        #create copy of df_train to serve as training set for fill
        df_train_filltrain = df_train.copy()
        #now delete rows coresponding to True
        df_train_filltrain = df_train_filltrain[df_train_filltrain[trainNArows.columns.get_values()[0]] == False]

        #now delete columns = columnslist and the NA labels (orig column+'_NArows') from this df
        df_train_filltrain = df_train_filltrain.drop(columnslist, axis=1)
        df_train_filltrain = df_train_filltrain.drop([trainNArows.columns.get_values()[0]], axis=1)


        #create a copy of df_train[categorylist] for fill train labels
        df_train_filllabel = df_train[categorylist].copy()
        #concatinate with the NArows
        df_train_filllabel = pd.concat([df_train_filllabel, trainNArows], axis=1)
        #drop rows corresponding to True
        df_train_filllabel = df_train_filllabel[df_train_filllabel[trainNArows.columns.get_values()[0]] == False]



        #delete the NArows column
        df_train_filllabel = df_train_filllabel.drop([trainNArows.columns.get_values()[0]], axis=1)


        #create features df_train for rows needing infill
        #create copy of df_train (note it already has NArows included)
        df_train_fillfeatures = df_train.copy()
        #delete rows coresponding to False
        df_train_fillfeatures = df_train_fillfeatures[(df_train_fillfeatures[trainNArows.columns.get_values()[0]])]
        #delete columnslist and column+'_NArows'
        df_train_fillfeatures = df_train_fillfeatures.drop(columnslist, axis=1)
        df_train_fillfeatures = df_train_fillfeatures.drop([trainNArows.columns.get_values()[0]], axis=1)


        #create features df_test for rows needing infill
        #create copy of df_test (note it already has NArows included)
        df_test_fillfeatures = df_test.copy()
        #delete rows coresponding to False
        df_test_fillfeatures = df_test_fillfeatures[(df_test_fillfeatures[testNArows.columns.get_values()[0]])]
        #delete column and column+'_NArows'
        df_test_fillfeatures = df_test_fillfeatures.drop(columnslist, axis=1)
        df_test_fillfeatures = df_test_fillfeatures.drop([testNArows.columns.get_values()[0]], axis=1)

        #delete NArows from df_train, df_test
        df_train = df_train.drop([trainNArows.columns.get_values()[0]], axis=1)
        df_test = df_test.drop([testNArows.columns.get_values()[0]], axis=1)




    #if category == 'date':
    #if MLinfilltype in ['exclude']:
    else:

      #create empty sets for now
      #an extension of this method would be to implement a comparable method \
      #for the time category, based on the columns output from the preprocessing
      df_train_filltrain = pd.DataFrame({'foo' : []}) 
      df_train_filllabel = pd.DataFrame({'foo' : []})
      df_train_fillfeatures = pd.DataFrame({'foo' : []})
      df_test_fillfeatures = pd.DataFrame({'foo' : []})
    
    
    return df_train_filltrain, df_train_filllabel, df_train_fillfeatures, df_test_fillfeatures





  def insertinfill(self, df, column, infill, category, NArows, postprocess_dict, \
                   columnslist = [], categorylist = []):
    '''
    #insertinfill(df, column, infill, category, NArows, columnslist = [])
    #function that takes as input a dataframe, column id, category string of either\
    #'nmbr'/'text'/'bnry'/'date', a df column of True/False identifiying row id of\
    #rows that will recieve infill, and and a list of columns produced by a text \
    #class preprocessor when applicable. Replaces the column cells in rows \
    #coresponding to the NArows True values with the values from infill, returns\
    #the associated transformed dataframe.
    '''
    
    MLinfilltype = postprocess_dict['process_dict'][category]['MLinfilltype']
    
    #NArows column name uses original column name + _NArows as key
    #by convention, current column has original column name + '_ctgy' at end
    #so we'll drop final 5 characters from column string
    #origcolumnname = column[:-5]
    NArowcolumn = NArows.columns[0]

    #if category in ['nmbr', 'nbr2', 'bxcx', 'bnry', 'text']:
    if MLinfilltype in ['numeric', 'singlct', 'multirt']:

      #if this is a single column set (not categorical)
      if len(categorylist) == 1:

        #create new dataframe for infills wherein the infill values are placed in \
        #rows coresponding to NArows True values and rows coresponding to NArows \
        #False values are filled with a 0    


        #assign index values to a column
        df['tempindex1'] = df.index

        #concatinate our df with NArows
        df = pd.concat([df, NArows], axis=1)

        #create list of index numbers coresponding to the NArows True values
        infillindex = df.loc[df[NArowcolumn]]['tempindex1']

        #create a dictionary for use to insert infill using df's index as the key
        infill_dict = dict(zip(infillindex, infill.values))

        #replace 'tempindex1' column with infill in rows where NArows is True
        df['tempindex1'] = np.where(df[NArowcolumn], df['tempindex1'].replace(infill_dict), 'fill')

        #now carry that infill over to the target column for rows where NArows is True
        df[column] = np.where(df[NArowcolumn], df['tempindex1'], df[column])

        #remove the temporary columns from df
        df = df.drop(['tempindex1'], axis=1)
        df = df.drop([NArowcolumn], axis=1)




      #else if categorylist wasn't single value
      else:

        #create new dataframe for infills wherein the infill values are placed in \
        #rows coresponding to NArows True values and rows coresponding to NArows \
        #False values are filled with a 0

        #text infill contains multiple columns for each predicted calssification
        #which were derived from one-hot encoding the original column in preprocessing
        for textcolumnname in categorylist:

          #create newcolumn which will serve as the NArows specific to textcolumnname
          df['textNArows'] = NArows

          df['textNArows'] = df['textNArows'].replace(0, False)
          df['textNArows'] = df['textNArows'].replace(1, True)

          #assign index values to a column
          df['tempindex1'] = df.index


          #create list of index numbers coresponding to the NArows True values
          textinfillindex = pd.DataFrame(df.loc[df['textNArows']]['tempindex1'])
          #reset the index
          textinfillindex = textinfillindex.reset_index()

          #now before we create our infill dicitonaries, we're going to need to
          #create a seperate textinfillindex for each category

          infill['tempindex1'] = textinfillindex['tempindex1']

          #first let's create a copy of this textcolumn's infill column replacing 
          #0/1 with True False (this works because we are one hot encoding)
          infill[textcolumnname + '_bool'] = infill[textcolumnname].astype('bool')

          #we'll use the mask feature to create infillindex which only contains \
          #rows coresponding to the True value in the column we just created

          mask = (infill[[textcolumnname + '_bool']]==True).all(1)
          infillindex = infill[mask]['tempindex1']



          #we're only going to insert the infill to column textcolumnname if we \
          #have infill to insert

          if len(infillindex.values) > 0:

            df.loc[infillindex.values[0], textcolumnname] = 1


          #now we'll delete temporary support columns associated with textcolumnname
          
          #for some reason these infill drops are returning performance warnings
          #but since this function doesn't even return infill I'm just going to leave out
#           infill = infill.drop([textcolumnname + '_bool'], axis=1)
#           infill = infill.drop(['tempindex1'], axis=1)
          
          df = df.drop(['textNArows'], axis=1)
          df = df.drop(['tempindex1'], axis=1)


    #if category == 'date':
    if MLinfilltype in ['exclude', 'multisp']:
      #this spot reserved for future update to incorporate address of datetime\
      #category data
      df = df


    return df




  def MLinfillfunction (self, df_train, df_test, column, postprocess_dict, \
                        masterNArows_train, masterNArows_test, randomseed):
    '''
    #new function ML infill, generalizes the MLinfill application between categories

    #def MLinfill (df_train, df_test, column, postprocess_dict, \
    #masterNArows_train, masterNArows_test, randomseed)
    #function that applies series of functions of createMLinfillsets, 
    #predictinfill, and insertinfill to a categorical encoded set.

    #for the record I'm sure that the conversion of the single column
    #series to a dataframe is counter to the intent of pandas
    #it's probably less memory efficient but it's the current basis of
    #the functions so we're going to maintain that approach for now
    #the revision of these functions to accept pandas series is a
    #possible future extension
    '''

    if postprocess_dict['column_dict'][column]['infillcomplete'] == False \
    and postprocess_dict['column_dict'][column]['category'] not in ['date']:

      columnslist = postprocess_dict['column_dict'][column]['columnslist']
      categorylist = postprocess_dict['column_dict'][column]['categorylist']
      origcolumn = postprocess_dict['column_dict'][column]['origcolumn']
      category = postprocess_dict['column_dict'][column]['category']
      

      #createMLinfillsets
      df_train_filltrain, df_train_filllabel, df_train_fillfeatures, df_test_fillfeatures = \
      self.createMLinfillsets(df_train, \
                         df_test, column, \
                         pd.DataFrame(masterNArows_train[origcolumn+'_NArows']), \
                         pd.DataFrame(masterNArows_test[origcolumn+'_NArows']), \
                         category, randomseed, postprocess_dict, \
                         columnslist = columnslist, \
                         categorylist = categorylist)

      #predict infill values using defined function predictinfill(.)
      df_traininfill, df_testinfill, model = \
      self.predictinfill(category, df_train_filltrain, df_train_filllabel, \
                    df_train_fillfeatures, df_test_fillfeatures, \
                    randomseed, postprocess_dict, columnslist = columnslist)

      #now we'll add our trained model to the postprocess_dict
      postprocess_dict['column_dict'][column]['infillmodel'] \
      = model

      #troubleshooting note: it occurs to me that we're only saving our
      #trained model in the postprocess_dict for one of the text columns
      #not all, however since this will be the first column to be 
      #addressed here and also in the postmunge function (they're in 
      #same order) my expectation is that this will not be an issue and \
      #accidental bonus since we're only saving once results in reduced
      #file size

      #apply the function insertinfill(.) to insert missing value predicitons \
      #to df's associated column
      df_train = self.insertinfill(df_train, column, df_traininfill, category, \
                            pd.DataFrame(masterNArows_train[origcolumn+'_NArows']), \
                            postprocess_dict, columnslist = columnslist, \
                            categorylist = categorylist)

      #if we don't train the train set model on any features, that we won't be able 
      #to apply the model to predict the test set infill. 

      if any(x == True for x in masterNArows_train[origcolumn+'_NArows']):

        df_test = self.insertinfill(df_test, column, df_testinfill, category, \
                           pd.DataFrame(masterNArows_test[origcolumn+'_NArows']), \
                           postprocess_dict, columnslist = columnslist, \
                           categorylist = categorylist)

      #now change the infillcomplete marker in the text_dict for each \
      #associated text column
      for columnname in categorylist:
        postprocess_dict['column_dict'][columnname]['infillcomplete'] = True

        #now we'll add our trained text model to the postprocess_dict
        postprocess_dict['column_dict'][columnname]['infillmodel'] \
        = model

        #now change the infillcomplete marker in the dict for each associated column
        for columnname in categorylist:
          postprocess_dict['column_dict'][columnname]['infillcomplete'] = True

    return df_train, df_test, postprocess_dict




  def LabelSetGenerator(self, df, column, label):
    '''
    #LabelSetGenerator
    #takes as input dataframe for test set, label column name, and label
    #returns a dataframe set of all rows which included that label in the column
    '''

    df = df[df[column] == label]

    return df



  def LabelFrequencyLevelizer(self, train_df, labels_df, labelsencoding_dict, \
                              postprocess_dict):
    '''
    #LabelFrequencyLevelizer(.)
    #takes as input dataframes for train set, labels, and label category
    #combines them to single df, then creates sets for each label category
    #such as to add on multiples of each set to achieve near levelized
    #frequency of label occurence in training set (increases the size
    #of the training set by redundant inclusion of rows with lower frequency
    #labels.) Returns train_df, labels_df, trainID_df.
    '''

    columns_labels = list(labels_df)

    labelscategory = next(iter(labelsencoding_dict))
    
    MLinfilltype = postprocess_dict['process_dict'][labelscategory]['MLinfilltype']

    labels = list(labelsencoding_dict[labelscategory].keys())

    setnameslist = []
    setlengthlist = []
    multiplierlist = []

    #if labelscategory == 'bnry':
    if MLinfilltype in ['singlct']:


      for label in labels:

        #derive set of labels dataframe for counting length
        df = self.LabelSetGenerator(labels_df, columns_labels[0], label)


        #append length onto list
        setlength = df.shape[0]
        #setlengthlist = setlengthlist.append(setlength)
        setlengthlist.append(setlength)


      #length of biggest label set
      maxlength = max(setlengthlist)
      #set counter to 0
      i = 0
      for label in labels:
        #derive multiplier to levelize label frequency
        setlength = setlengthlist[i]
        if setlength > 0:
          labelmultiplier = int(round(maxlength / setlength))
        else:
          labelmultiplier = 0
        #append multiplier onto list
        #multiplierlist = multiplierlist.append(labelmultiplier)
        multiplierlist.append(labelmultiplier)
        #increment counter
        i+=1

      #concatinate labels onto train set
      train_df = pd.concat([train_df, labels_df], axis=1)

      #reset counter
      i=0
      #for loop through labels
      for label in labels:

        #create train subset corresponding to label
        df = self.LabelSetGenerator(train_df, columns_labels[0], label)

        #set j counter to 0
        j = 0
        #concatinate an additional copy of the label set multiplier times
        while j < multiplierlist[i]:
          train_df = pd.concat([train_df, df], axis=0)
          #train_df = train_df.reset_index()
          j+=1

      #now seperate the labels df from the train df
      labels_df = pd.DataFrame(train_df[columns_labels[0]].copy())
      #now delete the labels column from train set
      del train_df[columns_labels[0]]


    #if labelscategory in ['nmbr', 'bxcx']:
    if MLinfilltype in ['numeric']:

        columns_labels = []
        for label in list(labels_df):
          if label[-5:] in ['_t<-2', '_t-21', '_t-10', '_t+01', '_t+12', '_t>+2']:
            columns_labels.append(label)

    #if labelscategory in ['text', 'nmbr', 'bxcx']:
    if MLinfilltype in ['multirt', 'numeric']:


      i=0
      for label in labels:

        column = columns_labels[i]
        #derive set of labels dataframe for counting length
        df = self.LabelSetGenerator(labels_df, column, 1)

        #append length onto list
        setlength = df.shape[0]
        #setlengthlist = setlengthlist.append(setlength)
        setlengthlist.append(setlength)

        i+=1

      #length of biggest label set
      maxlength = max(setlengthlist)

      #set counter to 0
      i = 0
      for label in labels:

        #derive multiplier to levelize label frequency
        setlength = setlengthlist[i]
        if setlength > 0:
          labelmultiplier = int(round(maxlength / setlength))
        else:
          labelmultiplier = 0
        #append multiplier onto list
        #multiplierlist = multiplierlist.append(labelmultiplier)
        multiplierlist.append(labelmultiplier)
        #increment counter
        i+=1

      #concatinate labels onto train set
      train_df = pd.concat([train_df, labels_df], axis=1)

      #reset counter
      i=0
      #for loop through labels
      for label in labels:


        #create train subset corresponding to label
        column = columns_labels[i]
        df = self.LabelSetGenerator(train_df, column, 1)

        #set j counter to 0
        j = 0
        #concatinate an additional copy of the label set multiplier times
        while j < multiplierlist[i]:
          train_df = pd.concat([train_df, df], axis=0)
          #train_df = train_df.reset_index()
          j+=1

        i+=1

      #now seperate the labels df from the train df
      labels_df = train_df[columns_labels]
      #now delete the labels column from train set
      train_df = train_df.drop(columns_labels, axis=1)


    if labelscategory in ['date', 'bxcx', 'nmbr']:

      pass


    return train_df, labels_df

  
  def dictupdatetrim(self, column, postprocess_dict):
    '''
    dictupdatetrim addresses the maintenance of postprocess_dict for cases where
    a feature is struck due to the feature selection mechanism. Speciifcally, the
    function is intended to:
    - remove column entry from every case where it is included in a columnslist
    i.e. column in postprocess_dict['comlumn_dict'][key1]['columnslist'] for all key1
    - remove column entry from every case where it is included in a categorylist
    i.e. column in postprocess_dict['comlumn_dict'][key1]['categorylist'] for all key1
    - trim column's postprocess_dict['column_dict'][column]

    As a reminder, a columnslist is a list of every column that originated from the
    same source, such that we will need to edit the columnslist for every dervied
    column that originated from the same source.

    As a reminder, a categorylist is a list of every column derived as part of the 
    same single or multi-column transformation, such that we will need to edit the
    categorylist for every derived column that originated from the same transformation
    as the column we are trimming

    Trimming the postprocess_dict['column_dict'][column] is fairly strainghtforward

    Note that since we cant' edit a dictionary as we are cycling through it, we
    will use some helper objects to store details of the edits.

    For some reason creating this function was harder than it should have been.
    Sometimes it helps to just sketch it out again from scratch.
    '''

    #initialize helper objects
    helper1_dict = {}
    helper2_dict = {}
    helper3_list = []

    #this if probably isn't neccesary but just going to throw it in
    if column in postprocess_dict['column_dict']:

      #for every column_dict
      for key in postprocess_dict['column_dict']:

        #if key originates from the same source column as our column argument
        if postprocess_dict['column_dict'][key]['origcolumn'] == \
        postprocess_dict['column_dict'][key]['origcolumn']:

          #then we'll be editting the columnslist for key, first we'll store in helper1_dict
          helper1_dict.update({key : column})

          #now we'll check if key shares the same categorylist as column
          if column in postprocess_dict['column_dict'][key]['categorylist']:

            #if so we'll be removing column from key's categorylist entry, first
            #we'll store in helper2_dict
            helper2_dict.update({key : column})

      #now we'll strike the column's column_dict, firtst we'll store in helper3_list
      helper3_list = helper3_list + [column]

    #ok here we'll do the trimming associated with helper1_dict for columnslists
    for key1 in helper1_dict:
      if helper1_dict[key1] in postprocess_dict['column_dict'][key1]['columnslist']:
        postprocess_dict['column_dict'][key1]['columnslist'].remove(helper1_dict[key1])

    #ok here we'll do the trimming associated with helper2_dict for categorylists
    for key2 in helper2_dict:
      if helper2_dict[key2] in postprocess_dict['column_dict'][key2]['categorylist']:
        postprocess_dict['column_dict'][key2]['categorylist'].remove(helper2_dict[key2])

    #and finally we'll trim the column_dict for the column
    for column3 in helper3_list:

      del postprocess_dict['column_dict'][column3]
    
#     #here we'll address the postprocess_dict['origcolumn'] entry for columnkey
#     #basically if we trim the column associated with the columnkey, we'll need
#     #to assign a new columnkey for use in postmunge which has not been previously trimmed
    
#     origcolumn = postprocess_dict['column_dict'][column]['origcolumn']
#     newcolumnkey = ''
#     if column = postprocess_dict['origcolumn'][origcolumn]['columnkey']:
#       for potentialcolumnkey in postprocess_dict['origcolumn'][origcolumn]['columnkey']:
#         if potentialcolumnkey in list(postprocess_dict['column_dict'][column]['columnlist']):
#             newcolumnkey = potentialcolumnkey
#             postprocess_dict['origcolumn'][origcolumn]['columnkey'] = newcolumnkey
#             break

    return postprocess_dict
  

  
  def secondcircle(self, df_train, df_test, column, postprocess_dict):
  	
    '''
    quite simply, delete the columns, call dictupdatetrim to address postprocess_dict 
    '''
    
    origcolumn = postprocess_dict['column_dict'][column]['origcolumn']

    postprocess_dict = self.dictupdatetrim(column, postprocess_dict)
    
    del df_train[column]
    del df_test[column]
    
    
    #here we'll address the postprocess_dict['origcolumn'] entry for columnkey
    #basically if we trim the column associated with the columnkey, we'll need
    #to assign a new columnkey for use in postmunge which has not been previously trimmed
    
    
    #origcolumn = postprocess_dict['column_dict'][column]['origcolumn']
    #newcolumnkey = ''
    
    columnkeybefore = postprocess_dict['origcolumn'][origcolumn]['columnkey']
    columnkeylistbefore = postprocess_dict['origcolumn'][origcolumn]['columnkeylist']
    column_dict_list = list(postprocess_dict['column_dict'])
    
    if column == columnkeybefore:
      for potentialcolumnkey in columnkeylistbefore:
        if potentialcolumnkey in column_dict_list:
            if potentialcolumnkey[-5:] != '_NArw':
              newcolumnkey = potentialcolumnkey
              postprocess_dict['origcolumn'][origcolumn]['columnkey'] = newcolumnkey
              break
    
    columnkeyafter = postprocess_dict['origcolumn'][origcolumn]['columnkey']
        
    return df_train, df_test, postprocess_dict
  

  
  def trainFSmodel(self, am_subset, am_labels, randomseed, labelsencoding_dict, \
                   process_dict):
    
    '''
    trains model for purpose of evaluating features
    '''
    
    
    #convert dataframes to numpy arrays
    np_subset = am_subset.values
    np_labels = am_labels.values
    
    #get category of labels from labelsencoding_dict
    labelscategory = next(iter(labelsencoding_dict))
    
    MLinfilltype = process_dict[labelscategory]['MLinfilltype']
    
    #if labelscategory in ['nmbr']:
    if MLinfilltype in ['numeric']:
      
      #this is specific to the current means of address for numeric label sets
      #as we build out our label engineering methods this will need to. be updated
      for labelcolumn in list(am_labels):
        if labelcolumn[-5:] == '_nmbr':
          np_labels = am_labels[labelcolumn].values
          break
      
      #this is to address a weird error message suggesting I reshape the y with ravel()
      np_labels = np.ravel(np_labels)

      FSmodel = RandomForestRegressor(n_estimators=100, random_state = randomseed, verbose=0)

      FSmodel.fit(np_subset, np_labels)
      
      baseaccuracy = self.shuffleaccuracy(am_subset, am_labels, FSmodel, randomseed, \
                                          labelsencoding_dict, process_dict)
        
    #if labelscategory in ['bnry']:
    if MLinfilltype in ['singlct']:
      
      #this is to address a weird error message suggesting I reshape the y with ravel()
      np_labels = np.ravel(np_labels)

      #train logistic regression model using scikit-learn for binary classifier
      FSmodel = RandomForestClassifier(n_estimators=100, random_state = randomseed, verbose=0)

      FSmodel.fit(np_subset, np_labels)
      
      baseaccuracy = self.shuffleaccuracy(am_subset, am_labels, FSmodel, randomseed, \
                                          labelsencoding_dict, process_dict)
      
    #if labelscategory in ['text']:
    if MLinfilltype in ['multirt']:
      
      #first convert the one-hot encoded set via argmax to a 1D array
      np_labels_argmax = np.argmax(np_labels, axis=1)

      #train logistic regression model using scikit-learn for binary classifier
      #with multi_class argument activated
      FSmodel = RandomForestClassifier(n_estimators=100, random_state = randomseed, verbose=0)

      FSmodel.fit(np_train_filltrain, np_train_filllabel_argmax)
      
      baseaccuracy = self.shuffleaccuracy(am_subset, am_labels, FSmodel, randomseed, \
                                          labelsencoding_dict, process_dict)
      
      del np_labels_argmax
        
    #I think this will clear some memory
    del np_labels, np_subset
    
    return FSmodel, baseaccuracy
      
  
  def createFSsets(self, am_subset, column, columnslist, randomseed):
    '''
    very simply shuffles rows of columns from columnslist with randomseed
    then returns the resulting dataframe
    
    hat tip for permutation method from "Beware Default Random Forest Importances"
    by Terence Parr, Kerem Turgutlu, Christopher Csiszar, and Jeremy Howard
    '''
    
    shuffleset = am_subset.copy()
    
    for clcolumn in columnslist:
      
      #shuffleset[column] = shuffle(shuffleset[column], random_state = randomseed)
      shuffleset[clcolumn] = shuffle(shuffleset[clcolumn].values, random_state = randomseed)
      
    return shuffleset

  def createFSsets2(self, am_subset, column, columnslist, randomseed):
    '''
    similar to createFSsets except performed such as to only leave one column from
    the columnslist untouched and shuffle the rest 
    '''
    shuffleset2 = am_subset.copy()
    
    for clcolumn in columnslist:
        
      if clcolumn != column:
            
        shuffleset2[clcolumn] = shuffle(shuffleset2[clcolumn].values, random_state = randomseed)
        
    
    return shuffleset2
    
  
  def shuffleaccuracy(self, shuffleset, am_labels, FSmodel, randomseed, \
                      labelsencoding_dict, process_dict):
    '''
    measures accuracy of predictions of shuffleset (which had permutation method)
    against the model trained on the unshuffled set
    '''
    
    #convert dataframes to numpy arrays
    np_shuffleset = shuffleset.values
    np_labels = am_labels.values
    
    #get category of labels from labelsencoding_dict
    labelscategory = next(iter(labelsencoding_dict))
    
    MLinfilltype = process_dict[labelscategory]['MLinfilltype']
    
    #if labelscategory in ['nmbr']:
    if MLinfilltype in ['numeric']:
      
      #this is specific to the current means of address for numeric label sets
      #as we build out our label engineering methods this will need to. be updated
      for labelcolumn in list(am_labels):
        if labelcolumn[-5:] == '_nmbr':
          np_labels = am_labels[labelcolumn].values
          break
      
      #this is to address a weird error message suggesting I reshape the y with ravel()
      np_labels = np.ravel(np_labels)
      
      #generate predictions
      np_predictions = FSmodel.predict(np_shuffleset)
      
      #just in case this returned any negative predictions
      np_predictions = np.absolute(np_predictions)
      #and we're trying to generalize here so will go ahead and apply to labels
      np_labels = np.absolute(np_labels)
      
      #evaluate accuracy metric
      #columnaccuracy = accuracy_score(np_labels, np_predictions)
      #columnaccuracy = mean_squared_log_error(np_labels, np_predictions)
      columnaccuracy = 1 - mean_squared_log_error(np_labels, np_predictions)
      
    #if labelscategory in ['bnry']:
    if MLinfilltype in ['singlct']:
      
      #this is to address a weird error message suggesting I reshape the y with ravel()
      np_labels = np.ravel(np_labels)
      
      #generate predictions
      np_predictions = FSmodel.predict(np_shuffleset)
      
      #evaluate accuracy metric
      columnaccuracy = accuracy_score(np_labels, np_predictions)
      
    #if labelscategory in ['text']:
    if MLinfilltype in ['multirt']:
      
      #first convert the one-hot encoded set via argmax to a 1D array
      np_labels_argmax = np.argmax(np_labels, axis=1)
      
      #generate predictions
      np_predictions = FSmodel.predict(np_shuffleset)
      
      #evaluate accuracy metric
      columnaccuracy = accuracy_score(np_labels, np_predictions)

      del np_labels_argmax
        
    #I think this will clear some memory
    del np_labels, np_shuffleset
    
    return columnaccuracy
  
  
  #def assemblemadethecut(self, FScolumn_dict, featurepct, am_subset_columns):
  def assemblemadethecut(self, FScolumn_dict, featurepct, featuremetric, featuremethod, \
                         am_subset_columns):
    '''
    takes as input the FScolumn_dict and the passed automunge argument featurepct
    and a list of the columns from automunge application in featureselect
    and uses to assemble a list of columns that made it through the feature
    selection process
    
    returns list madethecut
    '''
    
    #create empty dataframe for sorting purposes
    FSsupport_df = pd.DataFrame(columns=['FS_column', 'metric', 'category'])
    
    #Future extension:
    #FSsupport_df = pd.DataFrame(columns=['FS_column', 'metric', 'metric2', 'category'])
    
    #add rows to the dataframe for each column
    for key in FScolumn_dict:
      
      column_df = pd.DataFrame([[key, FScolumn_dict[key]['metric'], FScolumn_dict[key]['category']]], \
                               columns=['FS_column', 'metric', 'category'])
  
      FSsupport_df = pd.concat([FSsupport_df, column_df], axis=0)
    
    #sort the rows by metric (from large to small, not that higher metric implies
    #more predictive power associated with that column's feature)
    #(note that NaN rows will have NaN values at bottom of list)
    FSsupport_df = FSsupport_df.sort_values(['metric'], ascending=False)
    
    #create list of candidate entries for madethecut
    candidates = list(FSsupport_df['FS_column'])
    
    #count the number of NaN values originating form NArw cells
    NaNcount = FSsupport_df['metric'].isna().sum()
    #count the total number of rows
    totalrowcount =  FSsupport_df.shape[0]
    #count ranked rows
    metriccount = totalrowcount - NaNcount
    
    #create list of NArws
    candidateNArws = candidates[-NaNcount:]
    #create list of feature rows
    candidatefeaturerows = candidates[:-NaNcount]
    
#     #calculate the number of features we'll keep using the ratio passed from automunge
#     numbermakingcut = int(metriccount * featurepct)
    
    if featuremethod not in ['pct', 'metric']:
      print("error featuremethod object must be one of ['pct', 'metric']")
    
    if featuremethod == 'pct':

      #calculate the number of features we'll keep using the ratio passed from automunge
      numbermakingcut = int(metriccount * featurepct)
      
    if featuremethod == 'metric':
      
      #calculate the number of features we'll keep using the ratio passed from automunge
      numbermakingcut = len(FSsupport_df[FSsupport_df['metric'] >= featuremetric])
      
  
    #generate list of rows making the cut
    madethecut = candidatefeaturerows[:numbermakingcut]
    #add on the NArws
    madethecut = madethecut + candidateNArws
    
    return madethecut


    
  def featureselect(self, df_train, labels_column, trainID_column, \
                    powertransform, binstransform, randomseed, \
                    numbercategoryheuristic, assigncat, transformdict, \
                    process_dict, featurepct, featuremetric, featuremethod):
    '''
    featureselect is a function called within automunge() that applies methods
    to evaluate predictive power of derived features towards a downstream model
    such as to trim the branches of the transform tree.
    
    The function returns a list of column names that "made the cut" so that
    automunge() can then remove extraneous branches.
    '''
    
    
    #now we'll use automunge() to prepare the subset for feature evaluation
    #note the passed arguments, these are all intentional (no MLinfill applied,
    #primary goal here is to produce a processed dataframe for df_subset
    #with corresponding labels)
    am_train, _1, am_labels, am_validation1, _3, \
    am_validationlabels1, _5, _6, _7, \
    _8, _9, labelsencoding_dict, finalcolumns_train, _10,  \
    _11, FSpostprocess_dict = \
    self.automunge(df_train, df_test = False, labels_column = labels_column, trainID_column = trainID_column, \
                  testID_column = False, valpercent1 = 0.33, valpercent2 = 0.0, \
                  shuffletrain = False, TrainLabelFreqLevel = False, powertransform = powertransform, \
                  binstransform = binstransform, MLinfill = False, infilliterate=1, randomseed = randomseed, \
                  numbercategoryheuristic = numbercategoryheuristic, pandasoutput = True, \
                  featureselection = False, featurepct = 1.00, featuremetric = featuremetric, \
                  featuremethod = 'pct', assigncat = assigncat, \
                  assigninfill = {'stdrdinfill':[], 'MLinfill':[], 'zeroinfill':[], 'adjinfill':[]}, \
                  transformdict = transformdict, processdict = process_dict)
    
    
    #if am_labels is not an empty set
    if am_labels.empty == False:
      
      #apply function trainFSmodel
      FSmodel, baseaccuracy = \
      self.trainFSmodel(am_train, am_labels, randomseed, labelsencoding_dict, \
                        process_dict)
      
      #get list of columns
      am_train_columns = list(am_train)
      
      #initialize dictionary FScolumn_dict = {}
      FScolumn_dict = {}
      
      #assemble FScolumn_dict to support the feature evaluation
      for column in am_train_columns:
        
        #pull categorylist, category, columnslist
        categorylist = FSpostprocess_dict['column_dict'][column]['categorylist']
        category = FSpostprocess_dict['column_dict'][column]['category']
        columnslist = FSpostprocess_dict['column_dict'][column]['columnslist']
        
        #create entry to FScolumn_dict
        FScolumn_dict.update({column : {'categorylist' : categorylist, \
                                        'category' : category, \
                                        'columnslist' : columnslist, \
                                        'FScomplete' : False, \
                                        'shuffleaccuracy' : None, \
                                        'shuffleaccuracy2' : None, \
                                        'baseaccuracy' : baseaccuracy, \
                                        'metric' : None, \
                                        'metric2' : None}})
        
      #perform feature evaluation on each column
      for column in am_train_columns:
        
        if column[-5:] != '_NArw' \
        and FScolumn_dict[column]['FScomplete'] == False:
            
          #categorylist = FScolumn_dict[column]['categorylist']
          #update version 1.80, let's perform FS on columnslist instead of categorylist
          columnslist = FScolumn_dict[column]['columnslist']
          
          #create set with columns shuffle from columnslist
          #shuffleset = self.createFSsets(am_train, column, categorylist, randomseed)
          #shuffleset = self.createFSsets(am_train, column, columnslist, randomseed)
          shuffleset = self.createFSsets(am_validation1, column, columnslist, randomseed)
          
          #determine resulting accuracy after shuffle
#           columnaccuracy = self.shuffleaccuracy(shuffleset, am_labels, FSmodel, \
#                                                 randomseed, labelsencoding_dict, \
#                                                 process_dict)
          columnaccuracy = self.shuffleaccuracy(shuffleset, am_validationlabels1, \
                                                FSmodel, randomseed, labelsencoding_dict, \
                                                process_dict)

          
          #I think this will clear some memory
          del shuffleset
          
          #category accuracy penalty metric
          metric = baseaccuracy - columnaccuracy
          #metric2 = baseaccuracy - columnaccuracy2
        
          
          
          #save accuracy to FScolumn_dict and set FScomplete to True
          #(for each column in the categorylist)
          #for categorycolumn in FSpostprocess_dict['column_dict'][column]['categorylist']:
          for categorycolumn in FSpostprocess_dict['column_dict'][column]['columnslist']:
            
            FScolumn_dict[categorycolumn]['FScomplete'] = True
            FScolumn_dict[categorycolumn]['shuffleaccuracy'] = columnaccuracy
            FScolumn_dict[categorycolumn]['metric'] = metric
            #FScolumn_dict[categorycolumn]['shuffleaccuracy2'] = columnaccuracy2
            #FScolumn_dict[categorycolumn]['metric2'] = metric2
          
                    

        #if column[-5:] != '_NArw':
        if True == True:
          
          columnslist = FScolumn_dict[column]['columnslist']
            
          #create second set with all but one columns shuffled from columnslist
          #this will allow us to compare the relative importance between columns
          #derived from the same parent
          #shuffleset2 = self.createFSsets2(am_train, column, columnslist, randomseed)
          shuffleset2 = self.createFSsets2(am_validation1, column, columnslist, randomseed)
          
          #determine resulting accuracy after shuffle
#           columnaccuracy2 = self.shuffleaccuracy(shuffleset2, am_labels, FSmodel, \
#                                                 randomseed, labelsencoding_dict, \
#                                                 process_dict)
          columnaccuracy2 = self.shuffleaccuracy(shuffleset2, am_validationlabels1, \
                                                FSmodel, randomseed, labelsencoding_dict, \
                                                process_dict)
          
          metric2 = baseaccuracy - columnaccuracy2
          
          FScolumn_dict[column]['shuffleaccuracy2'] = columnaccuracy2
          FScolumn_dict[column]['metric2'] = metric2
        
        
#         if column[-5:] == '_NArw':
          
#           #we'll simply introduce a convention that NArw columns are not ranked
#           #for feature importance by default
#           #...
#           pass
          
          
    #madethecut = self.assemblemadethecut(FScolumn_dict, featurepct, am_subset_columns)
    madethecut = self.assemblemadethecut(FScolumn_dict, featurepct, featuremetric, \
                                         featuremethod, am_train_columns)
    
    
    #if the only column left in madethecut from origin column is a NArw, delete from the set
    #(this is going to lean on the column ID string naming conventions)
    #couldn't get this to work, this functionality a future extension
#     trimfrommtc = []
#     for traincolumn in list(df_train):
#       if (traincolumn + '_') not in [checkmtc[:(len(traincolumn)+1)] for checkmtc in madethecut]:
#         for mtc in madethecut:
#           #if mtc originated from traincolumn
#           if mtc[:(len(traincolumn)+1)] == traincolumn + '_':
#             #count the number of same instance in madethecut set
#             madethecut_trim = [mdc_trim[:(len(traincolumn)+1)] for mdc_trim in madethecut]
#             if madethecut_trim.count(mtc[:(len(traincolumn)+1)]) == 1 \
#             and mtc[-5:] == '_NArw':
#               trimfrommtc = trimfrommtc + [mtc]
#     madethecut = list(set(madethecut).difference(set(trimfrommtc)))
          
       
    #apply function madethecut(FScolumn_dict, featurepct)
    #return madethecut
    #where featurepct is the percent of features that we intend to keep
    #(might want to make this a passed argument from automunge)
    
    #I think this will clear some memory
    del am_train, _1, am_labels, am_validation1, _3, \
    am_validationlabels1, _5, _6, _7, \
    _8, _9, labelsencoding_dict, finalcolumns_train, _10,  \
    FSpostprocess_dict
    
    
    return madethecut, FSmodel, FScolumn_dict



  def assemblepostprocess_assigninfill(self, assigninfill, infillcolumns_list, 
                                       columns_train, postprocess_dict):
    #so the convention we'll follow is a column is not explicitly included in 
    #any of the infill methods we'll add it to stdrdinfill
    
    #where as a reminder assigninfill is the dictionary passed to automunge(.)
    #which allows user to assign infill method to pre-rpocessed columns
    #and infillcolumns_list is a list of columns from df_train after processing
    #and columns_train is the list of original columns preceding processing
    
#     and postprocess_dict is our datas strucutre for passing around info
#     abotu the various columns to the functions
    
    #note that the assigned infill methods in the assigninfill are pre-rpocessed
    #and the columns listed in infillcolumns_list are post process
    #so we'll need to do some convertions here to assemble the final returned
    #set finalassignedinfill which will represent the list of postprocessed
    #columns abd their corresponding infill method
    
    #we'll return a dicitonary comparable to assigninfill but containing
    #posprocess columns

    #create list of all specificied infill columns
    allspecdinfill_list = []
    #for each of the pre-processed columns
    for key in assigninfill:
      if key != 'stdrdinfill':
        for infillcolumn in assigninfill[key]:
          if infillcolumn in allspecdinfill_list:
            print("___________________")
            print("error: column entered for more than one infill method in assigninfill dicitonary.")
            print("___________________")
      allspecdinfill_list = allspecdinfill_list + assigninfill[key]
    
    #so no we have a list of all infill pre-processed columns which will use 
    #stdrdinfill which we'll call allspecdinfill_list, again these are
    #pre-processed columns
    
    addthesecolumns = []

    #for infillcolumn in infillcolumns_list:
    for infillcolumn in columns_train:
      
      if infillcolumn not in allspecdinfill_list:
        addthesecolumns = addthesecolumns + [infillcolumn]
    
    
    allstdrdinfill_list = addthesecolumns + assigninfill['stdrdinfill']
    
    
    #ok all of that was mostly to assemble our list of pre-processed columns
    #for standardinfill
    #now the next step is to assemble a dicitonary comparable to assigninfill
    #but containing postprocess columns, to do so we'll use the info stored in
    #postprocess_dict to support
    
    #first initialize the dictionary we'll return from the function
    #which when complete will be comparable to the assigninfill passed to
    #automunge but containing postprocessed columns
    postprocess_assigninfill_dict = {'stdrdinfill':[]}
    
    #first let's do the standard infill methods, this will assemble a list
    #of corresponding postprocess columns
    for stndrdcolumn in allstdrdinfill_list:
      
      columnkey = postprocess_dict['origcolumn'][stndrdcolumn]['columnkey']
      
      postprocess_assigninfill_dict['stdrdinfill'] = \
      postprocess_assigninfill_dict['stdrdinfill'] + \
      postprocess_dict['column_dict'][columnkey]['columnslist']
      
      
    #ok great now let's do the other infill methods  
    for infillcatkey in assigninfill:
      
      if infillcatkey != 'stdrdinfill':
        
        postprocess_assigninfill_dict.update({infillcatkey: []})
        
        for infillcolumn in assigninfill[infillcatkey]:
          
          columnkey = postprocess_dict['origcolumn'][infillcolumn]['columnkey']
          
          postprocess_assigninfill_dict[infillcatkey] = \
          postprocess_assigninfill_dict[infillcatkey] + \
          postprocess_dict['column_dict'][columnkey]['columnslist']
        
    
    return postprocess_assigninfill_dict


  def zeroinfillfunction(self, df, column, postprocess_dict, \
                        masterNArows):


    #create infill dataframe of all zeros with number of rows corepsonding to the
    #number of 1's found in masterNArows
    NArw_columnname = \
    postprocess_dict['column_dict'][column]['origcolumn'] + '_NArows'

    NAcount = len(masterNArows[masterNArows[NArw_columnname] == 1])

    infill = pd.DataFrame(np.zeros((NAcount, 1)))

    category = postprocess_dict['column_dict'][column]['category']
    columnslist = postprocess_dict['column_dict'][column]['columnslist']
    categorylist = postprocess_dict['column_dict'][column]['categorylist']

    #insert infill
    df = self.insertinfill(df, column, infill, category, \
                           pd.DataFrame(masterNArows[NArw_columnname]), \
                           postprocess_dict, columnslist = columnslist, \
                           categorylist = categorylist)

    return df

  def adjinfillfunction(self, df, column, postprocess_dict, \
                        masterNArows):

    #create infill dataframe of all nan with number of rows corepsonding to the
    #number of 1's found in masterNArows
    NArw_columnname = \
    postprocess_dict['column_dict'][column]['origcolumn'] + '_NArows'

    NAcount = len(masterNArows[masterNArows[NArw_columnname] == 1])
    

    infill = pd.DataFrame(np.zeros((NAcount, 1)))
    infill = infill.replace(0, np.nan)
    
    category = postprocess_dict['column_dict'][column]['category']
    columnslist = postprocess_dict['column_dict'][column]['columnslist']
    categorylist = postprocess_dict['column_dict'][column]['categorylist']

    #insert infill
    df = self.insertinfill(df, column, infill, category, \
                           pd.DataFrame(masterNArows[NArw_columnname]), \
                           postprocess_dict, columnslist = columnslist, \
                           categorylist = categorylist)
    
    
    #this is hack
    df[column] = df[column].replace('nan', np.nan)
    
    #apply ffill to replace NArows with value from adjacent cell in pre4ceding row
    df[column] = df[column].fillna(method='ffill')
    
    #we'll follow with a bfill just in case first row had a nan
    df[column] = df[column].fillna(method='bfill')
    
    #(still a potential bug if both first and last row had a nan, we'll leave 
    #that to chance for now)
    
    #df[[column]] = df[[column]].fillna(method='bfill')
    

    return df



  def automunge(self, df_train, df_test = False, labels_column = False, trainID_column = False, \
                testID_column = False, valpercent1=0.20, valpercent2 = 0.10, \
                shuffletrain = True, TrainLabelFreqLevel = False, powertransform = True, \
                binstransform = True, MLinfill = True, infilliterate=1, randomseed = 42, \
                numbercategoryheuristic = 0.000, pandasoutput = False, \
                featureselection = True, featurepct = 1.0, featuremetric = 0.0, \
                featuremethod = 'pct', \
                assigncat = {'nmbr':[], 'nbr2':[], 'bxcx':[], 'bnry':[], 'text':[], \
                             'date':[], 'excl':[]}, \
                assigninfill = {'stdrdinfill':[], 'MLinfill':[], 'zeroinfill':[], 'adjinfill':[]}, \
                transformdict = {}, processdict = {}):

    '''
    #automunge(df_train, df_test, labels_column, valpercent=0.20, powertransform = True, \
    #MLinfill = True, infilliterate=1, randomseed = 42, \
    pandasoutput = False, featureselection = True) \
    #Function that when fed a train and test data set automates the process \
    #of evaluating each column for determination and applicaiton of appropriate \
    #preprocessing. Takes as arguement pandas dataframes of training and test data \
    #(mdf_train), (mdf_test), the name of the column from train set containing \
    #labels, a string identifying the ID column for train and test, a value for \
    #percent of training data to be applied to a validation set, a True'False selector \
    #to determine if a power law transoffrmation will be applied to numerical sets, a \
    #True/False selector to determine if MLinfill methods will be applied to any missing \
    #points, an integer indication how many iterations of infill predfictions to \
    #run, a random seed integer, and a list of any stroing column names that are to\
    #be excluded from processing. (If MLinfill = False, missing points are addressed \
    #with mean for numerical, most common value for binary, new column for one-hot \
    #encoding, and mean for datetime). Note that the ML method for datetime data is \
    #future extension. Based on an evaluation of columns selectively applies one of \
    #four preprocessing functions to each. Shuffles the data and splits the training \
    #set into train and validation sets. 
    #returns train, trainID, labels, validation, validationID, validationlabels, \
    #test, testID, labelsencoding_dict, finalcolumns_train, finalcolumns_test,  \
    #postprocess_dict

    #Note that this approach assumes that the test data is available at time of training
    #For subsequent processing of test data the postmung function can be applied
    #with as input the postprocess_dict returned by automunge's address of the train set

    #The thinking with the infilliterate approach is that for particularly messy \
    #sets the predictinfill method will be influenced by the initial plug value \
    #for missing cells, and so multiple iterations of the predictinfill should \
    #trend towards better predictions. Initial tests of this iteration did not \
    #demonstrate much effect so this probably is not neccesary for common use.

    #a word of caution: if you are excluding any columns from processing via \
    #excludetransformscolumns list make sure they are already in a suitable state \
    #for application of ML (e.g. numerical) otherwise the MLinfill technique will \
    #return errors - update vs 1.77, replaced with new assignable category 'excl'

    #An extension could be to test the input data here for non-dataframe format \
    #(such as csv) to convert it to pandas within the function. 
    
    #update here with version 1.77 is to allow the passing of a custom category assignment
    #as assigncat = {'category1':['column1', etc], 'category2':['column2', etc]}
    
    #also update with version 1.77 is to allow passing of a custom infill assignment
    #as assigninfill = {'stdrdinfill':['column1', etc], 'MLinfill':[], 'zeroinfill':[], 'adjinfill':[]}
    
    #also update with version 1.77 is to. allow passing of a custom transform_dict
    #such as to allow a user to program the steps of feature engineering address
    #along with a custom process_dict such as to define and import new categories with
    #corresponding processing functions
    '''
    
    
#     #initialize processing dicitonaries
#     if bool(transformdict) == False:
#       transform_dict = \
#       self.assembletransformdict(powertransform, binstransform)
#     else:
#       transform_dict = transformdict
    
#     if bool(processdict) == False:
#       process_dict = \
#       self.assembleprocessdict()
#     else:
#       process_dict = processdict
    
    #initialize processing dicitonaries
    transform_dict = self.assembletransformdict(powertransform, binstransform)
    
    if bool(transformdict) != False:
      
#       #first print a notification if we are overwriting anything
#       for keytd in list(transformdict.keys()):
#         #keytd = key
#         if keytd in list(transform_dict.keys()):
#           print("Note that a key in the user passed transformdict already exists in library")
#           print("Overwriting entry for trasnformdict key ", keytd)
      
      #now update the trasnformdict
      transform_dict.update(transformdict)
    
    #initialize process_dict
    process_dict = self.assembleprocessdict()
    
    if bool(processdict) != False:
      
#       #first print a notification if we are overwriting anything
#       for keypd in list(processdict.keys()):
#         #keypd = key
#         if keypd in list(processdict.keys()):
#           print("Note that a key in the user passed processdict already exists in library")
#           print("Overwriting entry for processdict key ", keypd)
      
      #now update the processdict
      process_dict.update(processdict)
    
    
    
    #feature selection analysis performed here if elected
    if featureselection == True:
      
      madethecut, FSmodel, FScolumn_dict = \
      self.featureselect(df_train, labels_column, trainID_column, \
                        powertransform, binstransform, randomseed, \
                        numbercategoryheuristic, assigncat, transformdict, \
                        process_dict, featurepct, featuremetric, featuremethod)
                                     
    else:
    
      madethecut = []
      FSmodel = None
      FScolumn_dict = {}
      

    #we'll introduce convention that if df_test provided as False then we'll create
    #a dummy set derived from df_train's first 10 rows
    if not isinstance(df_test, pd.DataFrame):
      df_test = df_train[0:10].copy()
      testID_column = trainID_column
      if labels_column != False:
        del df_test[labels_column] 

    #copy input dataframes to internal state so as not to edit exterior objects
    df_train = df_train.copy()
    df_test = df_test.copy()        
        
    #my understanding is it is good practice to convert any None values into NaN \
    #so I'll just get that out of the way
    df_train.fillna(value=float('nan'), inplace=True)
    df_test.fillna(value=float('nan'), inplace=True)

    #we'll delete any rows from training set missing values in the labels column
    if labels_column != False:
      df_train = df_train.dropna(subset=[labels_column])

    #extract the ID columns from train and test set
    if trainID_column != False:
      df_trainID = pd.DataFrame(df_train[trainID_column])
      del df_train[trainID_column]
    else:
      df_trainID = pd.DataFrame()

    if testID_column != False:
      df_testID = pd.DataFrame(df_test[testID_column])
      del df_test[testID_column]
    else:
      df_testID = pd.DataFrame()

    #extract labels from train set
    #an extension to this function could be to delete the training set rows\
    #where the labels are missing or improperly formatted prior to performing\
    #this step
    if labels_column != False:
      df_labels = pd.DataFrame(df_train[labels_column])

#       #create copy of labels to support the translation dictionary for use after \
#       #prediction to convert encoded predictions back to the original label
#       df_labels2 = pd.DataFrame(df_labels.copy())

      del df_train[labels_column]
    
    else:
      df_labels = pd.DataFrame()


    #confirm consistency of train an test sets

    #check number of columns is consistent
    if df_train.shape[1] != df_test.shape[1]:
      print("error, different number of columns in train and test sets")
      print("(This assesment excludes labels and ID columns.)")
      return

    #check column headers are consistent (this works independent of order)
    columns_train = set(list(df_train))
    columns_test = set(list(df_test))
    if columns_train != columns_test:
      print("error, different column labels in the train and test set")
      print("(This assesment excludes labels and ID columns.)")
      return

    columns_train = list(df_train)
    columns_test = list(df_test)
    if columns_train != columns_test:
      print("error, different order of column labels in the train and test set")
      print("(This assesment excludes labels and ID columns.)")
      return

    #extract column lists again but this time as a list
    columns_train = list(df_train)
    columns_test = list(df_test)
    
    
    #carve out the validation rows
    
    #set randomness seed number
    answer = randomseed

    #first shuffle if that was selected
    
    if shuffletrain == True:
      #shuffle training set and labels
      df_train = shuffle(df_train, random_state = answer)
      df_labels = shuffle(df_labels, random_state = answer)

      if trainID_column != False:
        df_trainID = shuffle(df_trainID, random_state = answer)

    
    #ok now carve out the validation rows. We'll process these later
    #(we're processing train data from validation data seperately to
    #ensure no leakage)

    totalvalidationratio = valpercent1 + valpercent2

    if totalvalidationratio > 0.0:
      
      val2ratio = valpercent2 / totalvalidationratio

      if labels_column != False:
#         #split validation1 sets from training and labels
#         df_train, df_validation1, df_labels, df_validationlabels1 = \
#         train_test_split(df_train, df_labels, test_size=totalvalidationratio, \
#                          shuffle = False)
        #we'll wait to split out the validation labels
        df_train, df_validation1 = \
        train_test_split(df_train, test_size=totalvalidationratio, shuffle = False)


      else:
        df_train, df_validation1 = \
        train_test_split(df_train, test_size=totalvalidationratio, shuffle = False)
        df_labels = pd.DataFrame()
        df_validationlabels1 = pd.DataFrame()



      if trainID_column != False:
        df_trainID, df_validationID1 = \
        train_test_split(df_trainID, test_size=totalvalidationratio, shuffle = False)
  #       df_trainID, df_validationID1 = \
  #       train_test_split(df_trainID, test_size=valpercent1, shuffle = False)


      else:
        df_trainID = pd.DataFrame()
        df_validationID1 = pd.DataFrame()


      df_train = df_train.reset_index(drop=True)
      df_validation1 = df_validation1.reset_index(drop=True)
#       df_labels = df_labels.reset_index(drop=True)
#       df_validationlabels1 = df_validationlabels1.reset_index(drop=True)
      df_trainID = df_trainID.reset_index(drop=True)
      df_validationID1 = df_validationID1.reset_index(drop=True)
      
    #else if total validation was <= 0.0
    else:
      df_validation1 = pd.DataFrame()
      df_validationID1 = pd.DataFrame()
        
        
    #create an empty dataframe to serve as a store for each column's NArows
    #the column id's for this df will follow convention from NArows of 
    #column+'_NArows' for each column in columns_train
    #these are used in the ML infill methods
    masterNArows_train = pd.DataFrame()
    masterNArows_test = pd.DataFrame()


    #create an empty dictionary to serve as store for categorical transforms lists
    #of associated columns
    multicolumntransform_dict = {}

    #create an empty dictionary to serve as a store of processing variables from \
    #processing that were specific to the train dataset. These can be used for \
    #future processing of a later test set without the need to reprocess the \
    #original train. The dictionary will be populated with an entry for each \
    #column post processing, and will contain a column specific and category \
    #specific (i.e. nmbr, bnry, text, date) set of variable.
    postprocess_dict = {'column_dict' : {}, 'origcolumn' : {}, \
                        'process_dict' : process_dict}
    
    

    #For each column, determine appropriate processing function
    #processing function will be based on evaluation of train set
    for column in columns_train:

      #re-initialize the column specific dictionary for later insertion into
      #our postprocess_dict
      column_dict = {}

      #we're only going to process columns that weren't in our excluded set
      #if column not in excludetransformscolumns:
      if True == True:
        
        categorycomplete = False
        
        if bool(assigncat) == True:

          for key in assigncat:
            if column in assigncat[key]:
              category = key
              category_test = key
              categorycomplete = True
            
        if categorycomplete == False:
          
          category = self.evalcategory(df_train, column, numbercategoryheuristic)

          #special case for categorical
          if df_train[column].dtype.name == 'category':
            category = 'text'

          #let's make sure the category is consistent between train and test sets
          category_test = self.evalcategory(df_test, column, numbercategoryheuristic)

          #special case for categorical
          if df_test[column].dtype.name == 'category':
            category_test = 'text'


          #for the special case of train category = bxcx and test category = nmbr
          #(meaning there were no negative values in train but there were in test)
          #we'll resolve by reseting the train category to nmbr
          if category == 'bxcx' and category_test == 'nmbr':
            category = 'nmbr'

          #one more bxcx special case: if user elects not to apply boxcox transform
          #default to 'nmbr' category instead of 'bxcx'
          if category == 'bxcx' and powertransform == False:
            category = 'nmbr'
            category_test = 'nmbr'

          #one more special case, if train was a numerical set to categorical based
          #on heuristic, let's force test to as well
          if category == 'text' and category_test == 'nmbr':
            category_test = 'text'

        #otherwise if train category != test category return error
        if category != category_test:
          print('error - different category between train and test sets for column ',\
               column)

#         #here we'll delete any columns that returned a 'null' category
#         if category == 'null':
#           df_train = df_train.drop([column], axis=1)
#           df_test = df_test.drop([column], axis=1)

#           column_dict = { column + '_null' : {'category' : 'null', \
#                                               'origcategory' : 'null', \
#                                               'normalization_dict' : {}, \
#                                               'origcolumn' : column, \
#                                               'columnslist' : [column], \
#                                               'categorylist' : [], \
#                                               'infillmodel' : False, \
#                                               'infillcomplete' : False }}

#           #now append column_dict onto postprocess_dict
#           postprocess_dict['column_dict'].update(column_dict)


        #so if we didn't delete the column let's proceed
        else:
          
          #to support the postprocess_dict entry below, let's first create a temp
          #list of columns
          templist1 = list(df_train)
          
          #create NArows (column of True/False where True coresponds to missing data)
          trainNArows = self.NArows(df_train, column, category, postprocess_dict)
          testNArows = self.NArows(df_test, column, category, postprocess_dict)

          #now append that NArows onto a master NA rows df
          masterNArows_train = pd.concat([masterNArows_train, trainNArows], axis=1)
          masterNArows_test = pd.concat([masterNArows_test, testNArows], axis=1)

          #troubleshoot
          #print("processing column ", column)
          
          #now process ancestors
          df_train, df_test, postprocess_dict = \
          self.processancestors(df_train, df_test, column, category, category, process_dict, \
                                transform_dict, postprocess_dict)
          
          #now process family
          df_train, df_test, postprocess_dict = \
          self.processfamily(df_train, df_test, column, category, category, process_dict, \
                            transform_dict, postprocess_dict)
          
          #now delete columns that were subject to replacement
          df_train, df_test, postprocess_dict = \
          self.circleoflife(df_train, df_test, column, category, category, process_dict, \
                            transform_dict, postprocess_dict)
          
          #here's another templist to support the postprocess_dict entry below
          templist2 = list(df_train)
          
          #ok now we're going to pick one of the new entries in templist2 to serve 
          #as a "columnkey" for pulling datas from the postprocess_dict down the road
          #columnkeylist = list(set(templist2) - set(templist1))[0]
          columnkeylist = list(set(templist2) - set(templist1))
            
          #so last line I believe returns string if only one entry, so let's run a test
          if isinstance(columnkeylist, str):
            columnkey = columnkeylist
          else:
            #if list is empty
            if len(columnkeylist) == 0:
              columnkey = column
            else:
              columnkey = columnkeylist[0]
              if len(columnkey) >= 5:
                if columnkey[-5:] == '_NArw':
                  columnkey = columnkeylist[1]
              
          
          
          #ok this is sort of a hack, originating in version 1.77,
          #we're going to create an entry to postprocess_dict to
          #store a columnkey for each of the original columns
          postprocess_dict['origcolumn'].update({column : {'category' : category, \
                                                           'columnkeylist' : columnkeylist, \
                                                           'columnkey' : columnkey}})
          
          
    

    
    
    #now that we've pre-processed all of the columns, let's run through them again\
    #using infill to derive plug values for the previously missing cells
    
    infillcolumns_list = list(df_train)
    
#     #Here is the list of columns for the stdrdinfill approach
#     #(bassically using MLinfill if MLinfill elected for default, otherwise
#     #using mean for numerical, most common for binary, and unique column for categorical)
#     allstdrdinfill_list = self.stdrdinfilllist(assigninfill, infillcolumns_list)
    
    #Here is the application of assemblepostprocess_assigninfill
    postprocess_assigninfill_dict = \
    self.assemblepostprocess_assigninfill(assigninfill, infillcolumns_list, \
                                          columns_train, postprocess_dict)
    
    
    columns_train_zero = postprocess_assigninfill_dict['zeroinfill']
    
    for column in columns_train_zero:
      
      categorylistlength = len(postprocess_dict['column_dict'][column]['categorylist'])
      
      #if (column not in excludetransformscolumns) \
      if (column not in postprocess_assigninfill_dict['stdrdinfill']) \
      and (column[-5:] != '_NArw') \
      and (categorylistlength == 1):
        #noting that currently we're only going to infill 0 for single column categorylists
        #some comparable address for multi-column categories is a future extension
        
        df_train = \
        self.zeroinfillfunction(df_train, column, postprocess_dict, \
                                masterNArows_train)
        
        df_test = \
        self.zeroinfillfunction(df_test, column, postprocess_dict, \
                                masterNArows_test)
        

    columns_train_adj = postprocess_assigninfill_dict['adjinfill']
    for column in columns_train_adj:

      
      #if column not in excludetransformscolumns \
      if column not in postprocess_assigninfill_dict['stdrdinfill'] \
      and column[-5:] != '_NArw':
        
        df_train = \
        self.adjinfillfunction(df_train, column, postprocess_dict, \
                               masterNArows_train)
        
        df_test = \
        self.adjinfillfunction(df_test, column, postprocess_dict, \
                               masterNArows_test)
    
    
    if MLinfill == True:
      
      columns_train_ML = list(set().union(postprocess_assigninfill_dict['stdrdinfill'], \
                                          postprocess_assigninfill_dict['MLinfill']))
      
      #columns_test_ML = list(df_test)

    else:
      
      columns_train_ML = postprocess_assigninfill_dict['MLinfill']
      
    
    iteration = 0
    while iteration < infilliterate:


      #for key in postprocess_dict['column_dict']:
      for key in columns_train_ML:
        postprocess_dict['column_dict'][key]['infillcomplete'] = False


      for column in columns_train_ML:

        #we're only going to process columns that weren't in our excluded set
        #or aren't identifiers for NA rows
        #if column not in excludetransformscolumns \
        if column[-5:] != '_NArw':


          df_train, df_test, postprocess_dict = \
          self.MLinfillfunction(df_train, df_test, column, postprocess_dict, \
                  masterNArows_train, masterNArows_test, randomseed)


      iteration += 1    
    
    
    #Here's where we'll trim the columns that were stricken as part of featureselection method
    
    #copy postprocess_dict in current state (prior to feature selection updates)
    #for use in postmunge postprocessfamily functions
    #preFSpostprocess_dict = postprocess_dict.copy()
    preFSpostprocess_dict = deepcopy(postprocess_dict)
    
    #trim branches here associated with featureselect

    if featureselection == True:

      #get list of columns currently included
      currentcolumns = list(df_train)

      #get list of columns to trim
      madethecutset = set(madethecut)
      trimcolumns = [b for b in currentcolumns if b not in madethecutset]

      #trim columns using circle of life function
      for trimmee in trimcolumns:
        
        df_train, df_test, postprocess_dict = \
        self.secondcircle(df_train, df_test, trimmee, postprocess_dict)
    
    



    if labels_column != False:
      
      #for now we'll just assume consistent processing approach for labels as for data
      #a future extension may segregate this approach
      
      #initialize processing dicitonaries (we'll use same as for train set)
      #a future extension may allow custom address for labels
      labelstransform_dict = transform_dict
      
      labelsprocess_dict = process_dict
      
      #we'll allow user to assign category to labels as well via assigncat call
      categorycomplete = False
        
      if bool(assigncat) == True:

        for key in assigncat:
          if labels_column in assigncat[key]:
            labelscategory = key
            categorycomplete = True
            
      if categorycomplete == False:
        
        #determine labels category and apply appropriate function
        labelscategory = self.evalcategory(df_labels, labels_column, numbercategoryheuristic)


      #copy dummy labels "test" df for our preprocessing functions
      #labelsdummy = pd.DataFrame()
      labelsdummy = df_labels.copy()

      #initialize a dictionary to serve as the store between labels and their \
      #associated encoding
      labelsencoding_dict = {labelscategory:{}}

      #apply appropriate processing function to this column based on the result
      if labelscategory == 'bnry':
        
        
#             bnrynormalization_dict = {column + '_bnry' : {'missing' : binary_missing_plug, \
#                                                  'onevalue' : onevalue, \
#                                                  'zerovalue' : zerovalue}}
        
        
    
        #now process ancestors
        df_labels, _1, postprocess_dict = \
        self.processancestors(df_labels, labelsdummy, labels_column, labelscategory, labelscategory, \
                              labelsprocess_dict, labelstransform_dict, postprocess_dict)


        #now process family
        df_labels, _1, postprocess_dict = \
        self.processfamily(df_labels, labelsdummy, labels_column, labelscategory, labelscategory, \
                          labelsprocess_dict, labelstransform_dict, postprocess_dict)
        
        #now delete columns subject to replacement
        df_labels, _1, postprocess_dict = \
        self.circleoflife(df_labels, labelsdummy, labels_column, labelscategory, labelscategory, \
                          labelsprocess_dict, labelstransform_dict, postprocess_dict)
        
#         labels_binary_missing_plug = df_labels[labels_column].value_counts().index.tolist()[0]
#         df_labels, _1 = self.process_binary_class(df_labels, labels_column, labels_binary_missing_plug)

        del df_labels[labels_column + '_NArw']

        finalcolumns_labels = list(df_labels)
    


        #here we'll populate the dictionery pairing values from the encoded labels \
        #column with the original value for transformation post prediciton
        
        labelsnormalization_dict = postprocess_dict['column_dict'][labels_column + '_' + labelscategory]['normalization_dict'][labels_column + '_' + labelscategory]
        
        labelsencoding_dict[labelscategory] = dict(zip([1,0], [labelsnormalization_dict['onevalue'], labelsnormalization_dict['zerovalue']]))
        
#         i = 0

#         for row in df_labels.iterrows():
#           if row[1][0] in labelsencoding_dict[labelscategory].keys():
#             i += 1
#           else:
#             labelsencoding_dict[labelscategory].update({row[1][0] : df_labels2.iloc[i][0]})
#             i += 1


      if labelscategory == 'nmbr' or labelscategory == 'bxcx':
        #(a future extension will address label processing for bxcx category seperately)

  #       #if labels category is 'nmbr' we won't apply any further processing to the \
  #       #column as my experience with linear regression methods is that this is not\
  #       #required. Further processing of numerical labels would need to be addressed\
  #       #by returning mean and std from the process_numerical_class method so as to\
  #       #potentially store in our labelsencoding_dict
  #       #a future expansino could be to facilitate supplemental numerical trasnformations\
  #       #such as we implemented with the boxcox transform
  #       pass
        
        
#         #made an executive decision not to perform full range of feature engineering
#         #methods on numerical labels, as is not common practice and some frameowrks
#         #onkly allow one or either of classification or regression (eg sklearn)
#         #so we'll just do a copy of original column and a z-score normalizaiton
#         #using new category 'rgrl' (stands for 'regression label')
#         labelscategory = 'rgrl'
        
        #made a further executive decision, I dont' think it's common to apply zscore
        #normalization to the labels of. a linear regression. So let's just
        #leave numerical labels untouched. I think there's some room for debate.
        labelscategory = 'excl'
        
#         #for numerical we'll want the original column unaltered for predictions
#         df_labels[labels_column+'_orig'] = df_labels[labels_column].copy()

        #however it may also benefit to parallel train model to predict transformations
        #plus we'll use the std bins for leveling the frequency of labels for oversampling
        
        #now process ancestors
        df_labels, _1, postprocess_dict = \
        self.processancestors(df_labels, labelsdummy, labels_column, labelscategory, labelscategory, \
                              labelsprocess_dict, labelstransform_dict, postprocess_dict)


        #now process family
        df_labels, _1, postprocess_dict = \
        self.processfamily(df_labels, labelsdummy, labels_column, labelscategory, labelscategory, \
                          labelsprocess_dict, labelstransform_dict, postprocess_dict)
        
        #now delete columns subject to replacement
        df_labels, _1, postprocess_dict = \
        self.circleoflife(df_labels, labelsdummy, labels_column, labelscategory, labelscategory, \
                          labelsprocess_dict, labelstransform_dict, postprocess_dict)
        
        
        
#         df_labels, labelsdummy, labels_column_dict_list = \
#         self.process_numerical_class(df_labels, labelsdummy, labels_column)

#         del df_labels[labels_column + '_NArw']

        finalcolumns_labels = list(df_labels)

        #for the labelsencoding_dict we'll save the bin labels and asscoiated columns
        labelsencoding_dict = {'nmbr':{}}
        columns_labels = []
        for label in list(df_labels):
          if label[-5:] in ['_t<-2', '_t-21', '_t-10', '_t+01', '_t+12', '_t>+2']:
            labelsencoding_dict['nmbr'].update({label[-4:]:label})




      #it occurs to me there might be an argument for preferring a single numerical \
      #classifier for labels to keep this to a single column, if so scikitlearn's \
      #LabelEcncoder could be used here, will assume that onehot encoding is acceptable
      if labelscategory == 'text':
        
        #now process ancestors
        df_labels, _1, postprocess_dict = \
        self.processancestors(df_labels, labelsdummy, labels_column, labelscategory, labelscategory, \
                              labelsprocess_dict, labelstransform_dict, postprocess_dict)


        #now process family
        df_labels, _1, postprocess_dict = \
        self.processfamily(df_labels, labelsdummy, labels_column, labelscategory, labelscategory, \
                          labelsprocess_dict, labelstransform_dict, postprocess_dict)

        #now delete columns subject to replacement
        df_labels, _1, postprocess_dict = \
        self.circleoflife(df_labels, labelsdummy, labels_column, labelscategory, labelscategory, \
                          labelsprocess_dict, labelstransform_dict, postprocess_dict)
        
        
#         df_labels, labelsdummy, _1 = \
#         self.process_text_class(df_labels, labelsdummy, labels_column)

#         del df_labels[labels_column + '_NArw']

        finalcolumns_labels = list(df_labels)

  
        labelsnormalization_dict = postprocess_dict['column_dict'][finalcolumns_labels[0]]['normalization_dict'][finalcolumns_labels[0]]

    
    
        #labelsencoding_dict[labelscategory] = dict(zip([1,0], [labelsnormalization_dict['onevalue'], labelsnormalization_dict['zerovalue']]))
        
        labelsencoding_dict[labelscategory] = labelsnormalization_dict['textlabelsdict']
  

  
  
#         i = 0

#         for row in df_labels2.iterrows():
#           if row[1][0] in labelsencoding_dict[labelscategory].keys():
#             i += 1
#           else:
#             labelsencoding_dict[labelscategory].\
#             update({row[1][0] : labels_column+'_'+row[1][0]})
#             i += 1

    else:
      df_labels = pd.DataFrame([])
      labelsencoding_dict = {}

    #great the data is processed now let's do a few moore global training preps

    #here's a list of final column names saving here since the translation to \
    #numpy arrays scrubs the column names
    finalcolumns_train = list(df_train)
    finalcolumns_test = list(df_test)


    #ok here's where we'll sploit out the validation1 labels from df_labels
    #(after processing labels but before trainlabelfreqlevel)
    if totalvalidationratio > 0.0:
      df_labels, df_validationlabels1 = \
      train_test_split(df_labels, test_size=totalvalidationratio, \
                       shuffle = False)
      
      df_labels = df_labels.reset_index(drop=True)
      df_validationlabels1 = df_validationlabels1.reset_index(drop=True)
      
    else:
      df_validationlabels1 = pd.DataFrame()
    


    #here is the process to levelize the frequency of label rows in train data
    #currently only label categories of 'bnry' or 'text' are considered
    #a future extension will include numerical labels by adding supplemental 
    #label columns to designate inclusion in some fractional bucket of the distribution
    #e.g. such as quintiles for instance
    if TrainLabelFreqLevel == True \
    and labels_column != False:


#       train_df = pd.DataFrame(np_train, columns = finalcolumns_train)
#       labels_df = pd.DataFrame(np_labels, columns = finalcolumns_labels)
      if trainID_column != False:
#         trainID_df = pd.DataFrame(np_trainID, columns = [trainID_column])
        #add trainID set to train set for consistent processing
#         train_df = pd.concat([train_df, trainID_df], axis=1)                        
        df_train = pd.concat([df_train, df_trainID], axis=1)                        
      
      
      if postprocess_dict['process_dict'][labelscategory]['MLinfilltype'] \
      in ['numeric', 'singlct', 'multirt']:
        
        #apply LabelFrequencyLevelizer defined function
        df_train, df_labels = \
        self.LabelFrequencyLevelizer(df_train, df_labels, labelsencoding_dict, \
                                     postprocess_dict)
      
#       if labelscategory in ['bnry', 'text']:

#         #apply LabelFrequencyLevelizer defined function
# #         train_df, labels_df = \
# #         self.LabelFrequencyLevelizer(train_df, labels_df, labelsencoding_dict)
#         df_train, df_labels = \
#         self.LabelFrequencyLevelizer(df_train, df_labels, labelsencoding_dict, \
#                                      postprocess_dict)

#       elif labelscategory in ['nmbr', 'bxcx']:

#         #apply LabelFrequencyLevelizer defined function
# #         train_df, labels_df = \
# #         self.LabelFrequencyLevelizer(train_df, labels_df, labelsencoding_dict)
#         df_train, df_labels = \
#         self.LabelFrequencyLevelizer(df_train, df_labels, labelsencoding_dict, \
#                                      postprocess_dict)




      #extract trainID
      if trainID_column != False:

#         trainID_df = pd.DataFrame(train_df[trainID_column])
#         del train_df[trainID_column]
        df_trainID = pd.DataFrame(df_train[trainID_column])
        del df_train[trainID_column]
      
        
      #shuffle one more time as part of levelized label frequency
      if shuffletrain == True:
        #shuffle training set and labels
        df_train = shuffle(df_train, random_state = answer)
        df_labels = shuffle(df_labels, random_state = answer)

        if trainID_column != False:
          df_trainID = shuffle(df_trainID, random_state = answer)


    #here we'll populate the postprocess_dci8t that is returned from automunge
    #as it. will be. used in the postmunge call beow to process validation sets
    postprocess_dict.update({'origtraincolumns' : columns_train, \
                             'finalcolumns_train' : finalcolumns_train, \
                             'testID_column' : testID_column, \
                             'MLinfill' : MLinfill, \
                             'infilliterate' : infilliterate, \
                             'randomseed' : randomseed, \
                             'powertransform' : powertransform, \
                             'binstransform' : binstransform, \
                             'numbercategoryheuristic' : numbercategoryheuristic, \
                             'pandasoutput' : pandasoutput, \
                             'labelsencoding_dict' : labelsencoding_dict, \
                             'preFSpostprocess_dict' : preFSpostprocess_dict, \
                             'featureselection' : featureselection, \
                             'featurepct' : featurepct, \
                             'featuremetric' : featuremetric, \
                             'featuremethod' : featuremethod, \
                             'FSmodel' : FSmodel, \
                             'FScolumn_dict' : FScolumn_dict, \
                             'madethecut' : madethecut, \
                             'assigncat' : assigncat, \
                             'assigninfill' : assigninfill, \
                             'transformdict' : transformdict, \
                             'processdict' : processdict, \
                             'automungeversion' : '1.799' })

    
    
    if totalvalidationratio > 0.0:
    
      #process validation set consistent to train set with postmunge here
      df_validation1, _2, _3, _4 = \
      self.postmunge(postprocess_dict, df_validation1, testID_column = False, \
                    pandasoutput = True)
    
    
    
#     #process validation labels set consistent to train labels with postmunge
#     df_validationlabels1, _2, _3, _4 = \
#     self.postmunge(postprocess_dict, df_validationlabels1, testID_column = False, \
#                   pandasoutput = True)
    #(instead of using postmunge we just waited to. split out labels)
    
    
    
    #ok now that validation is processed, carve out second validation set if one was elected
    
    if totalvalidationratio > 0.0:

      if val2ratio > 0.0:


        if labels_column != False:
          #split validation2 sets from training and labels
          df_validation1, df_validation2, df_validationlabels1, df_validationlabels2 = \
          train_test_split(df_validation1, df_validationlabels1, test_size=val2ratio, \
                           random_state = answer)

        else:

          df_validation1, df_validation2 = \
          train_test_split(df_validation1, test_size=val2ratio, \
                           random_state = answer)

          df_validationlabels2 = pd.DataFrame()

        if trainID_column != False:
          df_validationID1, df_validationID2 = \
          train_test_split(df_validationID1, test_size=val2ratio, random_state = answer)
        else:
          df_trainID = pd.DataFrame()
          df_validationID2 = pd.DataFrame()

      else:
        df_validation2 = pd.DataFrame()
        df_validationlabels2 = pd.DataFrame()
        df_validationID2 = pd.DataFrame()

    #else if totalvalidationratio <= 0.0
    else:
      df_validation1 = pd.DataFrame()
      df_validationlabels1 = pd.DataFrame()
      df_validationID1 = pd.DataFrame()
      df_validation2 = pd.DataFrame()
      df_validationlabels2 = pd.DataFrame()
      df_validationID2 = pd.DataFrame()


    if testID_column != False:
      df_testID = df_testID
    else:
      df_testID = pd.DataFrame()


    df_test = df_test

    
    #for reference
#     return np_train, np_trainID, np_labels, np_validation1, np_validationID1, \
#     np_validationlabels1, np_validation2, np_validationID2, np_validationlabels2, \
#     np_test, np_testID, labelsencoding_dict, finalcolumns_train, finalcolumns_test,  \
#     postprocess_dict

    
    #set output format based on pandasoutput argument
    if pandasoutput == True:
#       np_train, np_trainID, np_labels, np_validation1, np_validationID1, \
#       np_validationlabels1, np_validation2, np_validationID2, np_validationlabels2, \
#       np_test, np_testID = \
#       df_train, df_trainID, df_labels, df_validation1, df_validationID1, \
#       df_validationlabels1, df_validation2, df_validationID2, df_validationlabels2, \
#       df_test, df_testID
      
      np_train = df_train
      np_trainID = df_trainID
      np_labels = df_labels
      np_validation1 = df_validation1
      np_validationID1 = df_validationID1
      np_validationlabels1 = df_validationlabels1
      np_validation2 = df_validation2
      np_validationID2 = df_validationID2
      np_validationlabels2 = df_validationlabels2
      np_test = df_test
      np_testID = df_testID
    
    #else set output to numpy arrays
    else:
#       np_train, np_trainID, np_labels, np_validation1, np_validationID1, \
#       np_validationlabels1, np_validation2, np_validationID2, np_validationlabels2, \
#       np_test, np_testID = \
#       df_train.values, df_trainID.values, df_labels.values, df_validation1.values, \
#       df_validationID1.values, \
#       df_validationlabels1.values, df_validation2.values, df_validationID2.values, \
#       df_validationlabels2.values, \
#       df_test.values, df_testID.values
      
      np_train = df_train.values
      np_trainID = df_trainID.values
      np_labels = df_labels.values
      np_validation1 = df_validation1.values
      np_validationID1 = df_validationID1.values
      np_validationlabels1 = df_validationlabels1.values
      np_validation2 = df_validation2.values
      np_validationID2 = df_validationID2.values
      np_validationlabels2 = df_validationlabels2.values
      np_test = df_test.values
      np_testID = df_testID.values
      

    #a reasonable extension would be to perform some validation functions on the\
    #sets here (or also prior to transofrm to numpuy arrays) and confirm things \
    #like consistency between format of columns and data between our train and \
    #test sets and if any issues return a coresponding error message to alert user


    return np_train, np_trainID, np_labels, np_validation1, np_validationID1, \
    np_validationlabels1, np_validation2, np_validationID2, np_validationlabels2, \
    np_test, np_testID, labelsencoding_dict, finalcolumns_train, finalcolumns_test,  \
    FScolumn_dict, postprocess_dict



  # #Here is a summary of the postprocess_dict structure from automunge:



#         postprocess_dict.update({'origtraincolumns' : columns_train, \
#                              'finalcolumns_train' : finalcolumns_train, \
#                              'testID_column' : testID_column, \
#                              'MLinfill' : MLinfill, \
#                              'infilliterate' : infilliterate, \
#                              'randomseed' : randomseed, \
#                              'powertransform' : powertransform, \
#                              'binstransform' : binstransform, \
#                              'numbercategoryheuristic' : numbercategoryheuristic, \
#                              'pandasoutput' : pandasoutput, \
#                              'labelsencoding_dict' : labelsencoding_dict, \
#                              'preFSpostprocess_dict' : preFSpostprocess_dict, \
#                              'featureselection' : featureselection, \
#                              'featurepct' : featurepct, \
#                              'featuremetric' : featuremetric, \
#                              'featuremethod' : featuremethod, \
#                              'FSmodel' : FSmodel, \
#                              'FScolumn_dict' : FScolumn_dict, \
#                              'madethecut' : madethecut, \
#                              'assigncat' : assigncat, \
#                              'assigninfill' : assigninfill, \
#                              'transformdict' : transformdict, \
#                              'processdict' : processdict, \
#                              'automungeversion' : '1.77' })
# #also in postprocess_dict
# postprocess_dict['column_dict'][one entry per processed column]
# postprocess_dict['origcolumn']
# *note that postprocess_dict['preFSpostprocess_dict'] is a copy of the 
# postprocess_dict['column_dict'] taken before trimming of columns associated with feature evaluation

  # (example of bnry)
  # column_dict = { bc : {'category' : 'bnry', \
  #                      'origcategory' : 'bnry', \
  #                      'normalization_dict' : bnrynormalization_dict, \
  #                      'origcolumn' : column, \
  #                      'columnslist' : bnrycolumns, \
  #                      'categorylist' : categorylist, \
  #                      'infillmodel' : False, \
  #                      'infillcomplete' : False}}

  # nmbr
  # nmbrnormalization_dict = {'mean' : mean, 'std' : std}

  # bxcx
  # nmbrnormalization_dict = {'trnsfrm_mean' : mean, 'trnsfrm_std' : std, \
  #                           'bxcx_lmbda' : bxcx_lmbda}



  # bnrynormalization_dict = {'missing' : binary_missing_plug}



  # timenormalization_dict = {'meanyear' : meanyear, 'meanmonth' : meanmonth, \
  #                           'meanday' : meanday, 'meanhour' : meanhour, \
  #                           'meanminute' : meanminute, 'meansecond' : meansecond,\
  #                           'stdyear' : stdyear, 'stdmonth' : stdmonth, \
  #                           'stdday' : stdday, 'stdhour' : stdhour, \
  #                           'stdminute' : stdminute, 'stdsecond' : stdsecond}



  def postprocessancestors(self, df_test, column, category, origcategory, process_dict, \
                          transform_dict, postprocess_dict, columnkey):
    '''
    #as automunge runs a for loop through each column in automunge, this is the  
    #processing function applied which runs through the grandparents family primitives
    #populated in the transform_dict by assembletransformdict, only applied to
    #first generation of transforms (others are recursive through the processfamily function)
    '''
    
    
    #process the grandparents (no downstream, supplemental, only applied ot first generation)
    for grandparent in transform_dict[category]['grandparents']:
      
#       print("grandparent =. ", grandparent)
      
      if grandparent != None:
        #note we use the processsibling function here
        df_test = \
        self.postprocesscousin(df_test, column, grandparent, category, process_dict, \
                                transform_dict, postprocess_dict, columnkey)
      
    for greatgrandparent in transform_dict[category]['greatgrandparents']:
      
#       print("greatgrandparent = ", greatgrandparent)
      
      if greatgrandparent != None:
        #note we use the processparent function here
        df_test = \
        self.postprocessparent(df_test, column, greatgrandparent, category, process_dict, \
                              transform_dict, postprocess_dict, columnkey)
      
    
    return df_test
  
  

  def postprocessfamily(self, df_test, column, category, origcategory, process_dict, \
                        transform_dict, postprocess_dict, columnkey):
    '''
    #as automunge runs a for loop through each column in automunge, this is the  
    #processing function applied which runs through the family primitives
    #populated in the transform_dict by assembletransformdict.
    '''
  
#     print("postprocessfamily")
#     print("column = ", column)
#     print("category = ", category)
    
    #process the cousins (no downstream, supplemental)
    for cousin in transform_dict[category]['cousins']:
      
#       print("cousin = ", cousin)

      if cousin != None:
        #note we use the processsibling function here
        df_test = \
        self.postprocesscousin(df_test, column, cousin, origcategory, process_dict, \
                                transform_dict, postprocess_dict, columnkey)

    #process the siblings (with downstream, supplemental)
    for sibling in transform_dict[category]['siblings']:
      
#       print("sibling = ", sibling)

      if sibling != None:
        #note we use the processparent function here
        df_test = \
        self.postprocessparent(df_test, column, sibling, origcategory, process_dict, \
                              transform_dict, postprocess_dict, columnkey)

    #process the auntsuncles (no downstream, with replacement)
    for auntuncle in transform_dict[category]['auntsuncles']:
      
#       print("auntuncle = ", auntuncle)

      if auntuncle != None:
        df_test = \
        self.postprocesscousin(df_test, column, auntuncle, origcategory, process_dict, \
                                transform_dict, postprocess_dict, columnkey)

    #process the parents (with downstream, with replacement)
    for parent in transform_dict[category]['parents']:

#       print("parent = ", parent)

      if parent != None:
        df_test = \
        self.postprocessparent(df_test, column, parent, origcategory, process_dict, \
                              transform_dict, postprocess_dict, columnkey)


#     #if we had replacement transformations performed then delete the original column 
#     #(circle of life)
#     if len(transform_dict[category]['auntsuncles']) + len(transform_dict[category]['parents']) > 0:
#       del df_test[column]

    return df_test
  
  
  def postcircleoflife(self, df_test, column, category, origcategory, process_dict, \
                        transform_dict, postprocess_dict, columnkey):
    '''
    This functino deletes source columns for family primitives that included replacement.
    '''
    
    #if we had replacement transformations performed on first generation \
    #then delete the original column
    if len(transform_dict[category]['auntsuncles']) \
    + len(transform_dict[category]['parents']) > 0:
      del df_test[column]
      
    #if we had replacement transformations performed on downstream generation \
    #then delete the associated parent column 
    for columndict_column in postprocess_dict['column_dict']:
      if postprocess_dict['column_dict'][columndict_column]['deletecolumn'] == True:
      
        #first we'll remove the column from columnslists 
        for columnslistcolumn in postprocess_dict['column_dict'][columndict_column]['columnslist']:

          if columndict_column in postprocess_dict['column_dict'][columnslistcolumn]['columnslist']:
          
            postprocess_dict['column_dict'][columnslistcolumn]['columnslist'].remove(columndict_column)

        

        #now we'll delete column
        #note this only worksa on single column  parents, need to incioroprate categorylist
        #for multicolumn parents (future extension)
        if columndict_column in list(df_test):
          del df_test[columndict_column]
      
    return df_test
  
  
  
  def postprocesscousin(self, df_test, column, cousin, origcategory, process_dict, \
                       transform_dict, postprocess_dict, columnkey):
    
    #if this is a dual process function
    if process_dict[cousin]['postprocess'] != None:
      df_test = \
      process_dict[cousin]['postprocess'](df_test, column, postprocess_dict, \
                                           columnkey)

    #else if this is a single process function
    elif process_dict[cousin]['singleprocess'] != None:

      df_test, _1 = \
      process_dict[cousin]['singleprocess'](df_test, column, origcategory, \
                                            postprocess_dict)

    return df_test
  
  
  
  
  def postprocessparent(self, df_test, column, parent, origcategory, process_dict, \
                      transform_dict, postprocess_dict, columnkey):
    
    #this is used to derive the new columns from the trasform
    origcolumnsset = set(list(df_test))
    
    #if this is a dual process function
    if process_dict[parent]['postprocess'] != None:

      df_test = \
      process_dict[parent]['postprocess'](df_test, column, postprocess_dict, \
                                           columnkey)

    #else if this is a single process function process train and test seperately
    elif process_dict[parent]['singleprocess'] != None:

      df_test, _1 = \
      process_dict[parent]['singleprocess'](df_test, column, origcategory, \
                                            postprocess_dict)
    
    #this is used to derive the new columns from the trasform
    newcolumnsset = set(list(df_test))
    
    #derive the new columns from the trasform
    categorylist = list(origcolumnsset^newcolumnsset)
    
    if len(categorylist) > 1:
      #future extension
      pass

    else:
      parentcolumn = categorylist[0]
      

    #process any coworkers
    for coworker in transform_dict[parent]['coworkers']:

      if coworker != None:

        #process the coworker
        #note the function applied is processcousin
        df_test = \
        self.postprocesscousin(df_test, parentcolumn, coworker, origcategory, \
                               process_dict, transform_dict, postprocess_dict, columnkey)

    #process any friends
    for friend in transform_dict[parent]['friends']:

      if friend != None:

        #process the friend
        #note the function applied is processcousin
        df_test = \
        self.postprocesscousin(df_test, parentcolumn, friend, origcategory, \
                               process_dict, transform_dict, postprocess_dict, columnkey)


    #process any niecesnephews
    #note the function applied is comparable to processsibling, just a different
    #parent column
    for niecenephew in transform_dict[parent]['niecesnephews']:

      if niecenephew != None:

        #process the niecenephew
        df_test = \
        self.postprocessfamily(df_test, parentcolumn, niecenephew, origcategory, \
                               process_dict, transform_dict, postprocess_dict, columnkey)

    #process any children
    for child in transform_dict[parent]['children']:

      if child != None:

        #process the child
        #note the function applied is processfamily (using recursion)
        #parent column
        df_test = \
        self.postprocessfamily(df_test, parentcolumn, child, origcategory, process_dict, \
                              transform_dict, postprocess_dict, columnkey)


#     #if we had replacement transformations performed then delete the original column 
#     #(circle of life)
#     if len(transform_dict[parent]['children']) \
#     + len(transform_dict[parent]['coworkers']) > 0:
#       del df_test[parentcolumn]
      

    return df_test
  
  
  
  def postprocess_numerical_class(self, mdf_test, column, postprocess_dict, columnkey):
    '''
    #postprocess_numerical_class(mdf_test, column, postprocess_dict, columnkey)
    #function to normalize data to mean of 0 and standard deviation of 1 from training distribution
    #takes as arguement pandas dataframe of training and test data (mdf_train), (mdf_test)\
    #and the name of the column string ('column'), and the mean and std from the train set \
    #stored in postprocess_dict
    #replaces missing or improperly formatted data with mean of remaining values
    #leaves original specified column in dataframe
    #returns transformed dataframe

    #expect this approach works better when the numerical distribution is thin tailed
    #if only have training but not test data handy, use same training data for both dataframe inputs
    '''
    
    
    #retrieve normalizastion parameters from postprocess_dict
    normkey = column + '_nmbr'
    
    mean = \
    postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['mean']
    std = \
    postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['std']

    #copy original column for implementation
    mdf_test[column + '_nmbr'] = mdf_test[column].copy()


    #convert all values to either numeric or NaN
    mdf_test[column + '_nmbr'] = pd.to_numeric(mdf_test[column + '_nmbr'], errors='coerce')

    #get mean of training data
    mean = mean  

    #replace missing data with training set mean
    mdf_test[column + '_nmbr'] = mdf_test[column + '_nmbr'].fillna(mean)

    #subtract mean from column
    mdf_test[column + '_nmbr'] = mdf_test[column + '_nmbr'] - mean

    #get standard deviation of training data
    std = std

    #divide column values by std
    mdf_test[column + '_nmbr'] = mdf_test[column + '_nmbr'] / std


    return mdf_test
  

  def postprocess_mnmx_class(self, mdf_test, column, postprocess_dict, columnkey):
    '''
    #postprocess_mnmx_class(mdf_test, column, postprocess_dict, columnkey)
    #function to scale data to minimum of 0 and maximum of 1 based on training distribution
    #takes as arguement pandas dataframe of training and test data (mdf_train), (mdf_test)\
    #and the name of the column string ('column'), and the normalization parameters \
    #stored in postprocess_dict
    #replaces missing or improperly formatted data with mean of training values
    #leaves original specified column in dataframe
    #returns transformed dataframe

    #expect this approach works better when the numerical distribution is thin tailed
    #if only have training but not test data handy, use same training data for both dataframe inputs
    '''
    
    
    #retrieve normalizastion parameters from postprocess_dict
    normkey = column + '_mnmx'
    
    mean = \
    postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['mean']
    
    minimum = \
    postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['minimum']
    
    maximum = \
    postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['maximum']

    #copy original column for implementation
    mdf_test[column + '_mnmx'] = mdf_test[column].copy()


    #convert all values to either numeric or NaN
    mdf_test[column + '_mnmx'] = pd.to_numeric(mdf_test[column + '_mnmx'], errors='coerce')

    #get mean of training data
    mean = mean  

    #replace missing data with training set mean
    mdf_test[column + '_mnmx'] = mdf_test[column + '_mnmx'].fillna(mean)

    #perform min-max scaling to test set using values from train
    mdf_test[column + '_mnmx'] = (mdf_test[column + '_mnmx'] - minimum) / \
                                 (maximum - minimum)


    return mdf_test


  def postprocess_mnm3_class(self, mdf_test, column, postprocess_dict, columnkey):
    '''
    #postprocess_mnmx_class(mdf_test, column, postprocess_dict, columnkey)
    #function to scale data to minimum of 0 and maximum of 1 based on training distribution
    #quantiles with values exceeding quantiles capped
    #takes as arguement pandas dataframe of training and test data (mdf_train), (mdf_test)\
    #and the name of the column string ('column'), and the normalization parameters \
    #stored in postprocess_dict
    #replaces missing or improperly formatted data with mean of training values
    #leaves original specified column in dataframe
    #returns transformed dataframe

    #expect this approach works better when the numerical distribution is thin tailed
    #if only have training but not test data handy, use same training data for both dataframe inputs
    '''


    #retrieve normalizastion parameters from postprocess_dict
    normkey = column + '_mnm3'

    mean = \
    postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['mean']

    quantilemin = \
    postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['quantilemin']

    quantilemax = \
    postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['quantilemax']

    #copy original column for implementation
    mdf_test[column + '_mnm3'] = mdf_test[column].copy()


    #convert all values to either numeric or NaN
    mdf_test[column + '_mnm3'] = pd.to_numeric(mdf_test[column + '_mnm3'], errors='coerce')

    #get mean of training data
    mean = mean  

    #replace missing data with training set mean
    mdf_test[column + '_mnm3'] = mdf_test[column + '_mnm3'].fillna(mean)

    #perform min-max scaling to test set using values from train
    mdf_test[column + '_mnm3'] = (mdf_test[column + '_mnm3'] - quantilemin) / \
                                 (quantilemax - quantilemin)


    return mdf_test


  
  def postprocess_binary_class(self, mdf_test, column, postprocess_dict, columnkey):
    '''
    #postprocess_binary_class(mdf, column, postprocess_dict, columnkey)
    #converts binary classification values to 0 or 1
    #takes as arguement a pandas dataframe (mdf_test), \
    #the name of the column string ('column') \
    #and the string classification to assign to missing data ('missing')
    #saved in the postprocess_dict
    #replaces original specified column in dataframe
    #returns transformed dataframe

    #missing category must be identical to one of the two existing categories
    #returns error message if more than two categories remain
    '''
    
    #retrieve normalization parameters
    normkey = column + '_bnry'
    binary_missing_plug = \
    postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['missing']

    #change column name to column + '_bnry'
    mdf_test[column + '_bnry'] = mdf_test[column].copy()


    #replace missing data with specified classification
    mdf_test[column + '_bnry'] = mdf_test[column + '_bnry'].fillna(binary_missing_plug)

    #if more than two remaining classifications, return error message    
    #if len(mdf_train[column + '_bnry'].unique()) > 2 or len(mdf_test[column + '_bnry'].unique()) > 2:
    if len(mdf_test[column + '_bnry'].unique()) > 2:
        print('ERROR: number of categories in column for process_binary_class() call >2')
        return mdf_train

    #convert column to binary 0/1 classification
    lb = preprocessing.LabelBinarizer()
    mdf_test[column + '_bnry'] = lb.fit_transform(mdf_test[column + '_bnry'])

    #create list of columns
    bnrycolumns = [column + '_bnry']

    #change data types to 8-bit (1 byte) integers for memory savings
    mdf_test[column + '_bnry'] = mdf_test[column + '_bnry'].astype(np.int8)


    return mdf_test
  
  
  
  def postprocess_text_class(self, mdf_test, column, postprocess_dict, columnkey):
    


    '''
    #postprocess_text_class(mdf_test, column, postprocess_dict, columnkey)
    #process column with text classifications
    #takes as arguement pandas dataframe containing test data  
    #(mdf_test), and the name of the column string ('column'), and an array of
    #the associated transformed column s from the train set (textcolumns)
    #which is saved in the postprocess_dict

    #note this aligns formatting of transformed columns to the original train set
    #fromt he original treatment with automunge

    #retains the original column from master dataframe and
    #adds onehot encodings
    #with columns named after column_ + text classifications
    #missing data replaced with category label 'missing'+column
    #any categories missing from the training set removed from test set
    #any category present in training but missing from test set given a column of zeros for consistent formatting
    #ensures order of all new columns consistent between both sets
    #returns two transformed dataframe (mdf_train, mdf_test) \
    #and a list of the new column names (textcolumns)
    '''
    
    #note it is kind of a hack here to create a column for missing values with \
    #two underscores (__) in the column name to ensure appropriate order for cases\
    #where NaN present in test data but not train data, if a category starts with|
    #an underscore such that it preceeds '__missing' alphabetically in this scenario\
    #this might create error due to different order of columns, address of this \
    #potential issue will be a future extension

#     #add _NArw to textcolumns to ensure a column gets populated even if no missing
#     textcolumns = [column + '_NArw'] + textcolumns

    

    #note this will need to be revised in a future extension where 
    #downstream transforms are performed on multicolumn parents 
    #by pulling the categorylist instead of columnslist (noting that will require
    #a more exact evaluation for columnkey somehow)
    
    
    #textcolumns = postprocess_dict['column_dict'][columnkey]['columnslist']
    textcolumns = postprocess_dict['column_dict'][columnkey]['categorylist']
    
    #create copy of original column for later retrieval
    mdf_test[column + '_temp'] = mdf_test[column].copy()

    #convert column to category
    mdf_test[column] = mdf_test[column].astype('category')

#     #if set is categorical we'll need the plug value for missing values included
#     mdf_test[column] = mdf_test[column].cat.add_categories(['NArw'])

#     #replace NA with a dummy variable
#     mdf_test[column] = mdf_test[column].fillna('NArw')
    
    #if set is categorical we'll need the plug value for missing values included
    mdf_test[column] = mdf_test[column].cat.add_categories(['NAr2'])

    #replace NA with a dummy variable
    mdf_test[column] = mdf_test[column].fillna('NAr2')

    #replace numerical with string equivalent
    #mdf_train[column] = mdf_train[column].astype(str)
    mdf_test[column] = mdf_test[column].astype(str)


    #extract categories for column labels
    #note that .unique() extracts the labels as a numpy array

    #we'll get the category names from the textcolumns array by stripping the \
    #prefixes of column name + '_'
    prefixlength = len(column)+1
    labels_train = textcolumns[:]
    for textcolumn in labels_train:
      textcolumn = textcolumn[prefixlength :]
    #labels_train.sort(axis=0)
    labels_train.sort()
    labels_test = mdf_test[column].unique()
    labels_test.sort(axis=0)

    #transform text classifications to numerical id
    encoder = LabelEncoder()

    cat_test = mdf_test[column]
    cat_test_encoded = encoder.fit_transform(cat_test)


    #apply onehotencoding
    onehotencoder = OneHotEncoder()
    cat_test_1hot = onehotencoder.fit_transform(cat_test_encoded.reshape(-1,1))

    #append column header name to each category listing
    #note the iteration is over a numpy array hence the [...] approach
    labels_test[...] = column + '_' + labels_test[...]


    #convert sparse array to pandas dataframe with column labels
    df_test_cat = pd.DataFrame(cat_test_1hot.toarray(), columns=labels_test)


    #Get missing columns in test set that are present in training set
    missing_cols = set( textcolumns ) - set( df_test_cat.columns )

    #Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        df_test_cat[c] = 0

    #Ensure the order of column in the test set is in the same order than in train set
    #Note this also removes categories in test set that aren't present in training set
    df_test_cat = df_test_cat[textcolumns]


    #concatinate the sparse set with the rest of our training data
    mdf_test = pd.concat([df_test_cat, mdf_test], axis=1)
    

    #replace original column
    del mdf_test[column]
    mdf_test[column] = mdf_test[column + '_temp'].copy()
    del mdf_test[column + '_temp']
    
    #delete support NArw2 column
    columnNAr2 = column + '_NAr2'
    if columnNAr2 in list(mdf_test):
      del mdf_test[columnNAr2]
    
#     #troubleshooting version 1.77
#     columnNArw = column + '_NArw'
#     if columnNArw in list(mdf_test):
#       del mdf_test[columnNArw]
    
    

    
    #change data types to 8-bit (1 byte) integers for memory savings
    for textcolumn in textcolumns:
      
      
      
      mdf_test[textcolumn] = mdf_test[textcolumn].astype(np.int8)

    
    return mdf_test
  
  
  
  
  
  def postprocess_time_class(self, mdf_test, column, postprocess_dict, columnkey):

    '''
    #postprocess_time_class(mdf_test, column, postprocess_dict, columnkey)
    #postprocess test column with of date category
    #takes as arguement pandas dataframe containing test data 
    #(mdf_test), the name of the column string ('column'), and the timenormalization_dict 
    #from the original application of automunge to the associated date column from train set
    #(saved in the postprocess_dict)

    #retains the original column from master dataframe and
    #adds distinct columns for year, month, day, hour, minute, second
    #each normalized to the mean and std from original train set, 
    #with missing values plugged with the mean
    #with columns named after column_ + time category
    #returns two transformed dataframe (mdf_train, mdf_test)
    '''
    
    #retrieve normalization parameters from postprocess_dict
    normkey1 = column+'_year'
    normkey2 = column+'_month'
    normkey3 = column+'_day'
    normkey4 = column+'hour'
    normkey5 = column+'minute'
    normkey6 = column+'_second'
    normkeylist = [normkey1, normkey2, normkey3, normkey4, normkey5, normkey6]
    datecolumns = postprocess_dict['column_dict'][columnkey]['categorylist']
    for normkeyiterate in normkeylist:
      if normkeyiterate in datecolumns:
        normkey = normkeyiterate
        break
    
    timenormalization_dict = postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]

    #create copy of original column for later retrieval
    mdf_test[column + '_temp'] = mdf_test[column].copy()

    #apply pd.to_datetime to column, note that the errors = 'coerce' needed for messy data
    mdf_test[column] = pd.to_datetime(mdf_test[column], errors = 'coerce')

    #mdf_train[column].replace(-np.Inf, np.nan)
    #mdf_test[column].replace(-np.Inf, np.nan)

    #get mean of various categories of datetime objects to use to plug in missing cells
  #   meanyear = mdf_train[column].dt.year.mean()    
  #   meanmonth = mdf_train[column].dt.month.mean()
  #   meanday = mdf_train[column].dt.day.mean()
  #   meanhour = mdf_train[column].dt.hour.mean()
  #   meanminute = mdf_train[column].dt.minute.mean()
  #   meansecond = mdf_train[column].dt.second.mean()

    meanyear = timenormalization_dict['meanyear']
    meanmonth = timenormalization_dict['meanmonth']
    meanday = timenormalization_dict['meanday']
    meanhour = timenormalization_dict['meanhour']
    meanminute = timenormalization_dict['meanminute']
    meansecond = timenormalization_dict['meansecond']


    #get standard deviation of training data
  #   stdyear = mdf_train[column].dt.year.std()  
  #   stdmonth = mdf_train[column].dt.month.std()
  #   stdday = mdf_train[column].dt.day.std()
  #   stdhour = mdf_train[column].dt.hour.std()
  #   stdminute = mdf_train[column].dt.minute.std()
  #   stdsecond = mdf_train[column].dt.second.std()

    stdyear = timenormalization_dict['stdyear']
    stdmonth = timenormalization_dict['stdmonth']
    stdday = timenormalization_dict['stdday']
    stdhour = timenormalization_dict['stdhour']
    stdminute = timenormalization_dict['stdminute']
    stdsecond = timenormalization_dict['stdsecond']


  #   #create new columns for each category in train set
  #   mdf_train[column + '_year'] = mdf_train[column].dt.year
  #   mdf_train[column + '_month'] = mdf_train[column].dt.month
  #   mdf_train[column + '_day'] = mdf_train[column].dt.day
  #   mdf_train[column + '_hour'] = mdf_train[column].dt.hour
  #   mdf_train[column + '_minute'] = mdf_train[column].dt.minute
  #   mdf_train[column + '_second'] = mdf_train[column].dt.second

    #create new columns for each category in test set
    mdf_test[column + '_year'] = mdf_test[column].dt.year
    mdf_test[column + '_month'] = mdf_test[column].dt.month
    mdf_test[column + '_day'] = mdf_test[column].dt.day
    mdf_test[column + '_hour'] = mdf_test[column].dt.hour
    mdf_test[column + '_minute'] = mdf_test[column].dt.minute 
    mdf_test[column + '_second'] = mdf_test[column].dt.second


  #   #replace missing data with training set mean
  #   mdf_train[column + '_year'] = mdf_train[column + '_year'].fillna(meanyear)
  #   mdf_train[column + '_month'] = mdf_train[column + '_month'].fillna(meanmonth)
  #   mdf_train[column + '_day'] = mdf_train[column + '_day'].fillna(meanday)
  #   mdf_train[column + '_hour'] = mdf_train[column + '_hour'].fillna(meanhour)
  #   mdf_train[column + '_minute'] = mdf_train[column + '_minute'].fillna(meanminute)
  #   mdf_train[column + '_second'] = mdf_train[column + '_second'].fillna(meansecond)

    #do same for test set (replace missing data with training set mean)
    mdf_test[column + '_year'] = mdf_test[column + '_year'].fillna(meanyear)
    mdf_test[column + '_month'] = mdf_test[column + '_month'].fillna(meanmonth)
    mdf_test[column + '_day'] = mdf_test[column + '_day'].fillna(meanday)
    mdf_test[column + '_hour'] = mdf_test[column + '_hour'].fillna(meanhour)
    mdf_test[column + '_minute'] = mdf_test[column + '_minute'].fillna(meanminute)
    mdf_test[column + '_second'] = mdf_test[column + '_second'].fillna(meansecond)

    #subtract mean from column for both train and test
  #   mdf_train[column + '_year'] = mdf_train[column + '_year'] - meanyear
  #   mdf_train[column + '_month'] = mdf_train[column + '_month'] - meanmonth
  #   mdf_train[column + '_day'] = mdf_train[column + '_day'] - meanday
  #   mdf_train[column + '_hour'] = mdf_train[column + '_hour'] - meanhour
  #   mdf_train[column + '_minute'] = mdf_train[column + '_minute'] - meanminute
  #   mdf_train[column + '_second'] = mdf_train[column + '_second'] - meansecond

    mdf_test[column + '_year'] = mdf_test[column + '_year'] - meanyear
    mdf_test[column + '_month'] = mdf_test[column + '_month'] - meanmonth
    mdf_test[column + '_day'] = mdf_test[column + '_day'] - meanday
    mdf_test[column + '_hour'] = mdf_test[column + '_hour'] - meanhour
    mdf_test[column + '_minute'] = mdf_test[column + '_minute'] - meanminute
    mdf_test[column + '_second'] = mdf_test[column + '_second'] - meansecond


    #divide column values by std for both training and test data
  #   mdf_train[column + '_year'] = mdf_train[column + '_year'] / stdyear
  #   mdf_train[column + '_month'] = mdf_train[column + '_month'] / stdmonth
  #   mdf_train[column + '_day'] = mdf_train[column + '_day'] / stdday
  #   mdf_train[column + '_hour'] = mdf_train[column + '_hour'] / stdhour
  #   mdf_train[column + '_minute'] = mdf_train[column + '_minute'] / stdminute
  #   mdf_train[column + '_second'] = mdf_train[column + '_second'] / stdsecond

    mdf_test[column + '_year'] = mdf_test[column + '_year'] / stdyear
    mdf_test[column + '_month'] = mdf_test[column + '_month'] / stdmonth
    mdf_test[column + '_day'] = mdf_test[column + '_day'] / stdday
    mdf_test[column + '_hour'] = mdf_test[column + '_hour'] / stdhour
    mdf_test[column + '_minute'] = mdf_test[column + '_minute'] / stdminute
    mdf_test[column + '_second'] = mdf_test[column + '_second'] / stdsecond


    #now replace NaN with 0
  #   mdf_train[column + '_year'] = mdf_train[column + '_year'].fillna(0)
  #   mdf_train[column + '_month'] = mdf_train[column + '_month'].fillna(0)
  #   mdf_train[column + '_day'] = mdf_train[column + '_day'].fillna(0)
  #   mdf_train[column + '_hour'] = mdf_train[column + '_hour'].fillna(0)
  #   mdf_train[column + '_minute'] = mdf_train[column + '_minute'].fillna(0)
  #   mdf_train[column + '_second'] = mdf_train[column + '_second'].fillna(0)

    #do same for test set
    mdf_test[column + '_year'] = mdf_test[column + '_year'].fillna(0)
    mdf_test[column + '_month'] = mdf_test[column + '_month'].fillna(0)
    mdf_test[column + '_day'] = mdf_test[column + '_day'].fillna(0)
    mdf_test[column + '_hour'] = mdf_test[column + '_hour'].fillna(0)
    mdf_test[column + '_minute'] = mdf_test[column + '_minute'].fillna(0)
    mdf_test[column + '_second'] = mdf_test[column + '_second'].fillna(0)


  #   #output of a list of the created column names
  #   datecolumns = [column + '_year', column + '_month', column + '_day', \
  #                 column + '_hour', column + '_minute', column + '_second']

    #this is to address an issue I found when parsing columns with only time no date
    #which returned -inf vlaues, so if an issue will just delete the associated 
    #column along with the entry in datecolumns
  #   checkyear = np.isinf(mdf_train.iloc[0][column + '_year'])
  #   if checkyear:
  #     del mdf_train[column + '_year']
  #     if column + '_year' in mdf_test.columns:
  #       del mdf_test[column + '_year']

  #   checkmonth = np.isinf(mdf_train.iloc[0][column + '_month'])
  #   if checkmonth:
  #     del mdf_train[column + '_month']
  #     if column + '_month' in mdf_test.columns:
  #       del mdf_test[column + '_month']

  #   checkday = np.isinf(mdf_train.iloc[0][column + '_day'])
  #   if checkmonth:
  #     del mdf_train[column + '_day']
  #     if column + '_day' in mdf_test.columns:
  #       del mdf_test[column + '_day']

    #instead we'll just delete a column from test set if not found in train set
    if column + '_year' not in datecolumns:
      del mdf_test[column + '_year']
  #     datecolumns.remove(column + '_year')
    if column + '_month' not in datecolumns:
      del mdf_test[column + '_month'] 
  #     datecolumns.remove(column + '_month')
    if column + '_day' not in datecolumns:
      del mdf_test[column + '_day']  
  #     datecolumns.remove(column + '_day')
    if column + '_hour' not in datecolumns:
      del mdf_test[column + '_hour']
  #     datecolumns.remove(column + '_hour')
    if column + '_minute' not in datecolumns:
      del mdf_test[column + '_minute'] 
  #     datecolumns.remove(column + '_minute')
    if column + '_second' not in datecolumns:
      del mdf_test[column + '_second'] 
  #     datecolumns.remove(column + '_second')


  #   #delete original column from training data
  #   if column in mdf_test.columns:
  #     del mdf_test[column]  

    #replace original column
    del mdf_test[column]
    mdf_test[column] = mdf_test[column + '_temp'].copy()
    del mdf_test[column + '_temp']


  #   #output a dictionary of the associated column mean and std

  #   timenormalization_dict = {'meanyear' : meanyear, 'meanmonth' : meanmonth, \
  #                             'meanday' : meanday, 'meanhour' : meanhour, \
  #                             'meanminute' : meanminute, 'meansecond' : meansecond,\
  #                             'stdyear' : stdyear, 'stdmonth' : stdmonth, \
  #                             'stdday' : stdday, 'stdhour' : stdhour, \
  #                             'stdminute' : stdminute, 'stdsecond' : stdsecond}


    return mdf_test
  
  
  
  
  
  def postprocess_bxcx_class(self, mdf_test, column, postprocess_dict, columnkey):
    '''
    Applies box-cox method within postmunge function.
    '''
  #   #df_train, nmbrcolumns, nmbrnormalization_dict, categorylist = \
  #   mdf_train, column_dict_list = \
  #   self.process_bxcx_support(mdf_train, column, category, 1, bxcx_lmbda = None, \
  #                             trnsfrm_mean = None, trnsfrm_std = None)

  #   #grab the normalization_dict associated with the bxcx category
  #   columnkeybxcx = column + '_bxcx'
  #   for column_dict in column_dict_list:
  #     if columnkeybxcx in column_dict:
  #       bxcxnormalization_dict = column_dict[columnkeybxcx]['normalization_dict'][columnkey]


    bxcxkey = columnkey[:-5] + '_bxcx'

    #grab the normalization_dict associated with the bxcx category
    normkey = column+'_bxcx'
    bxcxnormalization_dict = \
    postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]
    #postprocess_dict['column_dict'][bxcxkey]['normalization_dict'][bxcxkey]
    
    temporigcategoryplug = 'bxcx'

    #df_test, nmbrcolumns, _1, _2 = \
    mdf_test, _1 = \
    self.process_bxcx_support(mdf_test, column, temporigcategoryplug, 1, bxcx_lmbda = \
                             bxcxnormalization_dict['bxcx_lmbda'], \
                             trnsfrm_mean = bxcxnormalization_dict['trnsfrm_mean'])

    return mdf_test
  
  
  
  
  def postprocess_bins_class(self, mdf_test, column, postprocess_dict, columnkey):
    '''
    #note that bins is intended for raw data that has not yet been nromalized
    #bint is intended for values that have already recieved z-score normalization
    
    
    #process_numerical_class(mdf_train, mdf_test, column)
    #function to normalize data to mean of 0 and standard deviation of 1 \
    #z score normalization) and also create set of onehot encoded bins based \
    #on standaqrds deviation increments from training distribution \
    #takes as arguement pandas dataframe of training and test data (mdf_train), (mdf_test)\
    #and the name of the column string ('column') 
    #replaces missing or improperly formatted data with mean of remaining values
    #replaces original specified column in dataframe
    #returns transformed dataframe

    #expect this approach works better when the numerical distribution is thin tailed
    #if only have training but not test data handy, use same training data for both dataframe inputs
    '''
    
    #retrieve normalization parameters from postprocess_dict
    normkey = column +'_bins_s<-2'
    mean = postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['binsmean']
    std = postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['binsstd']

    #store original column for later reversion
    mdf_test[column + '_temp'] = mdf_test[column].copy()

    #convert all values to either numeric or NaN
    mdf_test[column] = pd.to_numeric(mdf_test[column], errors='coerce')

    #replace missing data with training set mean
    mdf_test[column] = mdf_test[column].fillna(mean)

    #subtract mean from column for test
    mdf_test[column] = mdf_test[column] - mean

    #divide column values by std for both training and test data
    mdf_test[column] = mdf_test[column] / std


    #create bins based on standard deviation increments
    binscolumn = column + '_bins'
    mdf_test[binscolumn] = \
    pd.cut( mdf_test[column], bins = [-float('inf'), -2, -1, 0, 1, 2, float('inf')],  \
           labels = ['s<-2','s-21','s-10','s+01','s+12','s>+2'], precision=4)



    textcolumns = \
    [binscolumn + '_s<-2', binscolumn + '_s-21', binscolumn + '_s-10', \
     binscolumn + '_s+01', binscolumn + '_s+12', binscolumn + '_s>+2']

#     #process bins as a categorical set
#     mdf_test = \
#     self.postprocess_text_class(mdf_test, binscolumn, textcolumns)

    #we're going to use the postprocess_text_class function here since it 
    #allows us to force the columns even if no values present in the set
    #however to do so we're going to have to construct a fake postprocess_dict
    
    #a future extension should probnably build this capacity into a new distinct function
    
    #here are some data structures for reference to create the below
#     def postprocess_text_class(self, mdf_test, column, postprocess_dict, columnkey):
#     textcolumns = postprocess_dict['column_dict'][columnkey]['columnslist']
    
    tempkey = 'tempkey'
    temppostprocess_dict = {'column_dict' : {tempkey : {'columnslist' : textcolumns, \
                                                       'categorylist' : textcolumns}}}
    
    #process bins as a categorical set
    mdf_test = \
    self.postprocess_text_class(mdf_test, binscolumn, temppostprocess_dict, tempkey)
    
    
    #delete the support column
    del mdf_test[binscolumn]
    
    #replace original column
    del mdf_test[column]
    mdf_test[column] = mdf_test[column + '_temp'].copy()
    del mdf_test[column + '_temp']



    #create list of columns
    #nmbrcolumns = [column + '_nmbr', column + '_NArw'] + textcolumns
    nmbrcolumns = textcolumns



    #nmbrnormalization_dict = {'mean' : mean, 'std' : std}

    #store some values in the nmbr_dict{} for use later in ML infill methods
    column_dict_list = []
    
    return mdf_test
  
  
  def postprocess_bint_class(self, mdf_test, column, postprocess_dict, columnkey):
    '''
    #note that bins is intended for raw data that has not yet been nromalized
    #bint is intended for values that have already recieved z-score normalization
    
    
    #process_numerical_class(mdf_train, mdf_test, column)
    #function to normalize data to mean of 0 and standard deviation of 1 \
    #z score normalization) and also create set of onehot encoded bins based \
    #on standaqrds deviation increments from training distribution \
    #takes as arguement pandas dataframe of training and test data (mdf_train), (mdf_test)\
    #and the name of the column string ('column') 
    #replaces missing or improperly formatted data with mean of remaining values
    #replaces original specified column in dataframe
    #returns transformed dataframe

    #expect this approach works better when the numerical distribution is thin tailed
    #if only have training but not test data handy, use same training data for both dataframe inputs
    '''
    
    #retrieve normalization parameters from postprocess_dict
    normkey = column + '_bint_t<-2'
    mean = postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['bintmean']
    std = postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['bintstd']

    #store original column for later reversion
    mdf_test[column + '_temp'] = mdf_test[column].copy()

    #convert all values to either numeric or NaN
    mdf_test[column] = pd.to_numeric(mdf_test[column], errors='coerce')

    #replace missing data with training set mean
    mdf_test[column] = mdf_test[column].fillna(mean)

#     #subtract mean from column for test
#     mdf_test[column] = mdf_test[column] - mean

#     #divide column values by std for both training and test data
#     mdf_test[column] = mdf_test[column] / std


    #create bins based on standard deviation increments
    binscolumn = column + '_bint'
    mdf_test[binscolumn] = \
    pd.cut( mdf_test[column], bins = [-float('inf'), -2, -1, 0, 1, 2, float('inf')],  \
           labels = ['t<-2','t-21','t-10','t+01','t+12','t>+2'], precision=4)



    textcolumns = \
    [binscolumn + '_t<-2', binscolumn + '_t-21', binscolumn + '_t-10', \
     binscolumn + '_t+01', binscolumn + '_t+12', binscolumn + '_t>+2']

#     #process bins as a categorical set
#     mdf_test = \
#     self.postprocess_text_class(mdf_test, binscolumn, textcolumns)

    #we're going to use the postprocess_text_class function here since it 
    #allows us to force the columns even if no values present in the set
    #however to do so we're going to have to construct a fake postprocess_dict
    
    #a future extension should probnably build this capacity into a new distinct function
    
    #here are some data structures for reference to create the below
#     def postprocess_text_class(self, mdf_test, column, postprocess_dict, columnkey):
#     textcolumns = postprocess_dict['column_dict'][columnkey]['columnslist']
    
    tempkey = 'tempkey'
    temppostprocess_dict = {'column_dict' : {tempkey : {'columnslist' : textcolumns, \
                                                        'categorylist' : textcolumns}}}
    
    #process bins as a categorical set
    mdf_test = \
    self.postprocess_text_class(mdf_test, binscolumn, temppostprocess_dict, tempkey)
    
    
    #delete the support column
    del mdf_test[binscolumn]
    

    #replace original column
    del mdf_test[column]
    mdf_test[column] = mdf_test[column + '_temp'].copy()
    del mdf_test[column + '_temp']



    #create list of columns
    #nmbrcolumns = [column + '_nmbr', column + '_NArw'] + textcolumns
    nmbrcolumns = textcolumns



    #nmbrnormalization_dict = {'mean' : mean, 'std' : std}

    #store some values in the nmbr_dict{} for use later in ML infill methods
    column_dict_list = []
    
    return mdf_test
  




  def createpostMLinfillsets(self, df_test, column, testNArows, category, \
                             postprocess_dict, columnslist = [], categorylist = []):
    '''
    #createpostMLinfillsets(df_test, column, testNArows, category, \
    #columnslist = []) function that when fed dataframe of
    #test set, column id, df of True/False corresponding to rows from original \
    #sets with missing values, a string category of 'text', 'date', 'nmbr', or \
    #'bnry', and a list of column id's for the text category if applicable. The \
    #function returns a series of dataframes which can be applied to apply a \
    #machine learning model previously trained on our train set as part of the 
    #original automunge application to predict apppropriate infill values for those\
    #points that had missing values from the original sets, returning the dataframe\
    #df_test_fillfeatures
    '''
    
    MLinfilltype = postprocess_dict['process_dict'][category]['MLinfilltype']

    #if category in ['nmbr', 'nbr2', 'bxcx', 'bnry', 'text', 'bins', 'bint']:
    if MLinfilltype in ['numeric', 'singlct', 'multirt', 'multisp']:

      #if this is a single column set (not categorical)
      #if categorylist == []:
      if len(categorylist) == 1:

        #first concatinate the NArows True/False designations to df_train & df_test
  #       df_train = pd.concat([df_train, trainNArows], axis=1)
        df_test = pd.concat([df_test, testNArows], axis=1)

  #       #create copy of df_train to serve as training set for fill
  #       df_train_filltrain = df_train.copy()
  #       #now delete rows coresponding to True
  #       df_train_filltrain = df_train_filltrain[df_train_filltrain[trainNArows.columns.get_values()[0]] == False]

  #       #now delete columns = columnslist and the NA labels (orig column+'_NArows') from this df
  #       df_train_filltrain = df_train_filltrain.drop(columnslist, axis=1)
  #       df_train_filltrain = df_train_filltrain.drop([trainNArows.columns.get_values()[0]], axis=1)


  #       #create a copy of df_train[column] for fill train labels
  #       df_train_filllabel = pd.DataFrame(df_train[column].copy())
  #       #concatinate with the NArows
  #       df_train_filllabel = pd.concat([df_train_filllabel, trainNArows], axis=1)
  #       #drop rows corresponding to True
  #       df_train_filllabel = df_train_filllabel[df_train_filllabel[trainNArows.columns.get_values()[0]] == False]

  #       #delete the NArows column
  #       df_train_filllabel = df_train_filllabel.drop([trainNArows.columns.get_values()[0]], axis=1)

  #       #create features df_train for rows needing infill
  #       #create copy of df_train (note it already has NArows included)
  #       df_train_fillfeatures = df_train.copy()
  #       #delete rows coresponding to False
  #       df_train_fillfeatures = df_train_fillfeatures[(df_train_fillfeatures[trainNArows.columns.get_values()[0]])]
  #       #delete columnslist and column+'_NArows'
  #       df_train_fillfeatures = df_train_fillfeatures.drop(columnslist, axis=1)
  #       df_train_fillfeatures = df_train_fillfeatures.drop([trainNArows.columns.get_values()[0]], axis=1)


        #create features df_test for rows needing infill
        #create copy of df_test (note it already has NArows included)
        df_test_fillfeatures = df_test.copy()
        #delete rows coresponding to False
        df_test_fillfeatures = df_test_fillfeatures[(df_test_fillfeatures[testNArows.columns.get_values()[0]])]
        #delete column and column+'_NArows'
        df_test_fillfeatures = df_test_fillfeatures.drop(columnslist, axis=1)
        df_test_fillfeatures = df_test_fillfeatures.drop([testNArows.columns.get_values()[0]], axis=1)

        #delete NArows from df_train, df_test
  #       df_train = df_train.drop([trainNArows.columns.get_values()[0]], axis=1)
        df_test = df_test.drop([testNArows.columns.get_values()[0]], axis=1)

      #else if categorylist wasn't empty
      else:

        #create a list of columns representing columnslist exlucding elements from
        #categorylist
        noncategorylist = columnslist[:]
        #this removes categorylist elements from noncategorylist
        noncategorylist = list(set(noncategorylist).difference(set(categorylist)))


        #first concatinate the NArows True/False designations to df_train & df_test
  #       df_train = pd.concat([df_train, trainNArows], axis=1)
        df_test = pd.concat([df_test, testNArows], axis=1)

  #       #create copy of df_train to serve as training set for fill
  #       df_train_filltrain = df_train.copy()
  #       #now delete rows coresponding to True
  #       df_train_filltrain = df_train_filltrain[df_train_filltrain[trainNArows.columns.get_values()[0]] == False]

  #       #now delete columns = columnslist and the NA labels (orig column+'_NArows') from this df
  #       df_train_filltrain = df_train_filltrain.drop(columnslist, axis=1)
  #       df_train_filltrain = df_train_filltrain.drop([trainNArows.columns.get_values()[0]], axis=1)


  #       #create a copy of df_train[columnslist] for fill train labels
  #       df_train_filllabel = df_train[columnslist].copy()
  #       #concatinate with the NArows
  #       df_train_filllabel = pd.concat([df_train_filllabel, trainNArows], axis=1)
  #       #drop rows corresponding to True
  #       df_train_filllabel = df_train_filllabel[df_train_filllabel[trainNArows.columns.get_values()[0]] == False]

  #       #now delete columns = noncategorylist from this df
  #       df_train_filltrain = df_train_filltrain.drop(noncategorylist, axis=1)

  #       #delete the NArows column
  #       df_train_filllabel = df_train_filllabel.drop([trainNArows.columns.get_values()[0]], axis=1)


  #       #create features df_train for rows needing infill
  #       #create copy of df_train (note it already has NArows included)
  #       df_train_fillfeatures = df_train.copy()
  #       #delete rows coresponding to False
  #       df_train_fillfeatures = df_train_fillfeatures[(df_train_fillfeatures[trainNArows.columns.get_values()[0]])]
  #       #delete columnslist and column+'_NArows'
  #       df_train_fillfeatures = df_train_fillfeatures.drop(columnslist, axis=1)
  #       df_train_fillfeatures = df_train_fillfeatures.drop([trainNArows.columns.get_values()[0]], axis=1)


        #create features df_test for rows needing infill
        #create copy of df_test (note it already has NArows included)
        df_test_fillfeatures = df_test.copy()
        #delete rows coresponding to False
        df_test_fillfeatures = df_test_fillfeatures[(df_test_fillfeatures[testNArows.columns.get_values()[0]])]
        #delete column and column+'_NArows'
        df_test_fillfeatures = df_test_fillfeatures.drop(columnslist, axis=1)
        df_test_fillfeatures = df_test_fillfeatures.drop([testNArows.columns.get_values()[0]], axis=1)
        

        #delete NArows from df_train, df_test
  #       df_train = df_train.drop([trainNArows.columns.get_values()[0]], axis=1)
        
        df_test = df_test.drop([testNArows.columns.get_values()[0]], axis=1)
        
        
        

    #if category == 'date':
    #if MLinfilltype in ['exclude']:
    else:

      #create empty sets for now
      #an extension of this method would be to implement a comparable method \
      #for the time category, based on the columns output from the preprocessing
  #     df_train_filltrain = pd.DataFrame({'foo' : []}) 
  #     df_train_filllabel = pd.DataFrame({'foo' : []})
  #     df_train_fillfeatures = pd.DataFrame({'foo' : []})
      df_test_fillfeatures = pd.DataFrame({'foo' : []})
    
    return df_test_fillfeatures





  def predictpostinfill(self, category, model, df_test_fillfeatures, \
                        postprocess_dict, columnslist = []):
    '''
    #predictpostinfill(category, model, df_test_fillfeatures, \
    #columnslist = []), function that takes as input \
    #a category string, a model trained as part of automunge on the coresponding \
    #column from the train set, the output of createpostMLinfillsets(.), a seed \
    #for randomness, and a list of columns \
    #produced by a text class preprocessor when applicable and returns \
    #predicted infills for the test feature sets as df_testinfill based on \
    #derivations using scikit-learn, with the lenth of \
    #infill consistent with the number of True values from NArows

    #a reasonable extension of this funciton would be to allow ML inference with \
    #other ML architectures such a SVM or something SGD based for instance
    '''
    
    MLinfilltype = postprocess_dict['process_dict'][category]['MLinfilltype']
    
    #convert dataframes to numpy arrays
  #   np_train_filltrain = df_train_filltrain.values
  #   np_train_filllabel = df_train_filllabel.values
  #   np_train_fillfeatures = df_train_fillfeatures.values
    np_test_fillfeatures = df_test_fillfeatures.values

    #ony run the following if we have any rows needing infill
  #   if df_train_fillfeatures.shape[0] > 0:
    #since we don't have df_train_fillfeatures to work with we'll look at the 
    #model which will be set to False if there was no infill model trained
    #if model[0] != False:
    if model != False:

      #if category in ['nmbr', 'bxcx', 'nbr2']:
      if MLinfilltype in ['numeric']:

  #       #train linear regression model using scikit-learn for numerical prediction
  #       #model = LinearRegression()
  #       #model = PassiveAggressiveRegressor(random_state = randomseed)
  #       #model = Ridge(random_state = randomseed)
  #       #model = RidgeCV()
  #       #note that SVR doesn't have an argument for random_state
  #       model = SVR()
  #       model.fit(np_train_filltrain, np_train_filllabel)    


  #       #predict infill values
  #       np_traininfill = model.predict(np_train_fillfeatures)

        #only run following if we have any test rows needing infill
        if df_test_fillfeatures.shape[0] > 0:
          np_testinfill = model.predict(np_test_fillfeatures)
        else:
          np_testinfill = np.array([0])

        #convert infill values to dataframe
  #       df_traininfill = pd.DataFrame(np_traininfill, columns = ['infill'])
        df_testinfill = pd.DataFrame(np_testinfill, columns = ['infill'])

  #       print('category is nmbr, df_traininfill is')
  #       print(df_traininfill)



#       if category == 'bxcx':


#   #       model = SVR()
#   #       model.fit(np_train_filltrain, np_train_filllabel)   

#   #       #predict infill values
#   #       np_traininfill = model.predict(np_train_fillfeatures)



#         #only run following if we have any test rows needing infill
#         if df_test_fillfeatures.shape[0] > 0:
#           np_testinfill = model.predict(np_test_fillfeatures)
#         else:
#           np_testinfill = np.array([0])

#         #convert infill values to dataframe
#   #       df_traininfill = pd.DataFrame(np_traininfill, columns = ['infill'])
#         df_testinfill = pd.DataFrame(np_testinfill, columns = ['infill'])     


      #if category == 'bnry':
      if MLinfilltype in ['singlct']:

  #       #train logistic regression model using scikit-learn for binary classifier
  #       #model = LogisticRegression()
  #       #model = LogisticRegression(random_state = randomseed)
  #       #model = SGDClassifier(random_state = randomseed)
  #       model = SVC(random_state = randomseed)

  #       model.fit(np_train_filltrain, np_train_filllabel)

  #       #predict infill values
  #       np_traininfill = model.predict(np_train_fillfeatures)

        #only run following if we have any test rows needing infill
        if df_test_fillfeatures.shape[0] > 0:
          np_testinfill = model.predict(np_test_fillfeatures)
        else:
          np_testinfill = np.array([0])

        #convert infill values to dataframe
  #       df_traininfill = pd.DataFrame(np_traininfill, columns = ['infill'])
        df_testinfill = pd.DataFrame(np_testinfill, columns = ['infill'])


      #if category in ['text', 'bins', 'bint']:
      if MLinfilltype in ['multirt', 'multisp']:

  #       #first convert the one-hot encoded set via argmax to a 1D array
  #       np_train_filllabel_argmax = np.argmax(np_train_filllabel, axis=1)

  #       #train logistic regression model using scikit-learn for binary classifier
  #       #with multi_class argument activated
  #       #model = LogisticRegression()
  #       #model = SGDClassifier(random_state = randomseed)
  #       model = SVC(random_state = randomseed)

  #       model.fit(np_train_filltrain, np_train_filllabel_argmax)

  #       #predict infill values
  #       np_traininfill = model.predict(np_train_fillfeatures)

        #only run following if we have any test rows needing infill
        if df_test_fillfeatures.shape[0] > 0:
          np_testinfill = model.predict(np_test_fillfeatures)
        else:
          #this needs to have same number of columns as text category
          np_testinfill = np.zeros(shape=(1,len(columnslist)))

        #convert the 1D arrary back to one hot encoding
  #       labelbinarizertrain = preprocessing.LabelBinarizer()
  #       labelbinarizertrain.fit(np_traininfill)
  #       np_traininfill = labelbinarizertrain.transform(np_traininfill)

        #only run following if we have any test rows needing infill
        if df_test_fillfeatures.shape[0] > 0:
          labelbinarizertest = preprocessing.LabelBinarizer()
          labelbinarizertest.fit(np_testinfill)
          np_testinfill = labelbinarizertest.transform(np_testinfill)



        #run function to ensure correct dimensions of re-encoded classifier array
  #       np_traininfill = labelbinarizercorrect(np_traininfill, columnslist)

        if df_test_fillfeatures.shape[0] > 0:
          np_testinfill = self.labelbinarizercorrect(np_testinfill, columnslist)


        #convert infill values to dataframe
  #       df_traininfill = pd.DataFrame(np_traininfill, columns = [columnslist])
        df_testinfill = pd.DataFrame(np_testinfill, columns = [columnslist]) 


  #       print('category is text, df_traininfill is')
  #       print(df_traininfill)

      #if category == 'date':
      if MLinfilltype in ['exclude']:

        #create empty sets for now
        #an extension of this method would be to implement a comparable infill \
        #method for the time category, based on the columns output from the \
        #preprocessing
  #       df_traininfill = pd.DataFrame({'infill' : [0]}) 
        df_testinfill = pd.DataFrame({'infill' : [0]}) 

  #       model = False

  #       print('category is text, df_traininfill is')
  #       print(df_traininfill)


    #else if we didn't have any infill rows let's create some plug values
    else:

      if category == 'text':
  #       np_traininfill = np.zeros(shape=(1,len(columnslist)))
        np_testinfill = np.zeros(shape=(1,len(columnslist)))
  #       df_traininfill = pd.DataFrame(np_traininfill, columns = [columnslist])
        df_testinfill = pd.DataFrame(np_testinfill, columns = [columnslist]) 

      else :
  #       df_traininfill = pd.DataFrame({'infill' : [0]}) 
        df_testinfill = pd.DataFrame({'infill' : [0]}) 

  #     model = False

    return df_testinfill



  def postMLinfillfunction(self, df_test, column, postprocess_dict, \
                            masterNArows_test):

    '''
    #new function ML infill, generalizes the MLinfill application

    #def MLinfill (df_train, df_test, column, postprocess_dict, \
    #masterNArows_train, masterNArows_test, randomseed)
    #function that applies series of functions of createMLinfillsets, 
    #predictinfill, and insertinfill to a categorical encoded set.

    #for the record I'm sure that the conversion of the single column
    #series to a dataframe is counter to the intent of pandas
    #it's probably less memory efficient but it's the current basis of
    #the functions so we're going to maintain that approach for now
    #the revision of these functions to accept pandas series is a
    #possible future extension
    '''
    
    if postprocess_dict['column_dict'][column]['infillcomplete'] == False \
    and postprocess_dict['column_dict'][column]['category'] not in ['date']:

      columnslist = postprocess_dict['column_dict'][column]['columnslist']
      categorylist = postprocess_dict['column_dict'][column]['categorylist']
      origcolumn = postprocess_dict['column_dict'][column]['origcolumn']
      category = postprocess_dict['column_dict'][column]['category']
      model = postprocess_dict['column_dict'][column]['infillmodel']
      
      #createMLinfillsets
      df_test_fillfeatures = \
      self.createpostMLinfillsets(df_test, column, \
                         pd.DataFrame(masterNArows_test[origcolumn+'_NArows']), \
                         category, postprocess_dict, \
                         columnslist = columnslist, \
                         categorylist = categorylist)

      #predict infill values using defined function predictinfill(.)
      df_testinfill = \
      self.predictpostinfill(category, model, df_test_fillfeatures, \
                             postprocess_dict, columnslist = columnslist)


      #if model != False:
      if postprocess_dict['column_dict'][column]['infillmodel'] != False:

        df_test = self.insertinfill(df_test, column, df_testinfill, category, \
                               pd.DataFrame(masterNArows_test[origcolumn+'_NArows']), \
                               postprocess_dict, columnslist = columnslist, \
                               categorylist = categorylist)

      #now change the infillcomplete marker in the text_dict for each \
      #associated text column
      for columnname in categorylist:
        postprocess_dict['column_dict'][columnname]['infillcomplete'] = True


        #now change the infillcomplete marker in the dict for each associated column
        for columnname in categorylist:
          postprocess_dict['column_dict'][columnname]['infillcomplete'] = True

    return df_test, postprocess_dict






  def postmunge(self, postprocess_dict, df_test, testID_column = False, \
                pandasoutput = False):
    '''
    #postmunge(df_test, testID_column, postprocess_dict) Function that when fed a \
    #test data set coresponding to a previously processed train data set which was \
    #processed using the automunge function automates the process \
    #of evaluating each column for determination and applicaiton of appropriate \
    #preprocessing. Takes as arguement pandas dataframes of test data \
    #(mdf_test), a string identifying the ID column for test (testID_column), a \
    #dictionary containing keys for the processing which had been generated by the \
    #original processing of the coresponding train set using automunge function. \
    #Returns following sets as numpy arrays: 
    #test, testID, labelsencoding_dict, finalcolumns_test

    #Requires consistent column naming and order as original train set pre \
    #application of automunge. Requires postprocess_dict from original applicaiton. \
    #Currently assumes coinbsistent columns carved out from application of munging \
    #from original automunge, a potential future extension is to allow for additional \
    #columns to be excluded from processing.
    '''
    
#     #initialize processing dicitonaries
#     powertransform = postprocess_dict['powertransform']
#     binstransform = postprocess_dict['binstransform']
    
#     if bool(postprocess_dict['transformdict']) == False:
#       transform_dict = \
#       self.assembletransformdict(powertransform, binstransform)
#     else:
#       transform_dict = postprocess_dict['transformdict']
    
#     if bool(postprocess_dict['processdict']) == False:
#       process_dict = \
#       self.assembleprocessdict()
#     else:
#       process_dict = postprocess_dict['processdict']
    
    #initialize processing dicitonaries
    
    powertransform = postprocess_dict['powertransform']
    binstransform = postprocess_dict['binstransform']
    
    transform_dict = self.assembletransformdict(powertransform, binstransform)
    
    if bool(postprocess_dict['transformdict']) != False:
      transform_dict.update(postprocess_dict['transformdict'])
    
    process_dict = self.assembleprocessdict()
    
    if bool(postprocess_dict['processdict']) != False:
      process_dict.update(postprocess_dict['processdict'])
      
    #initialize the preFS postprocess_dict for use here
    preFSpostprocess_dict = deepcopy(postprocess_dict['preFSpostprocess_dict'])
    
    #postprocess_dict.update({'preFSpostprocess_dict' : preFSpostprocess_dict})
    
    #copy input dataframes to internal state so as not to edit exterior objects
    df_test = df_test.copy()

    #my understanding is it is good practice to convert any None values into NaN \
    #so I'll just get that out of the way
    df_test.fillna(value=float('nan'), inplace=True)

    #extract the ID columns from test set
    if testID_column != False:
      df_testID = pd.DataFrame(df_test[testID_column])
      del df_test[testID_column]

    #confirm consistency of train an test sets

    #check number of columns is consistent
    if len(postprocess_dict['origtraincolumns'])!= df_test.shape[1]:
      print("error, different number of original columns in train and test sets")
      return

    #check column headers are consistent (this works independent of order)
    columns_train_set = set(postprocess_dict['origtraincolumns'])
    columns_test_set = set(list(df_test))
    if columns_train_set != columns_test_set:
      print("error, different column labels in the train and test set")
      return

    #check order of column headers are consistent
    columns_train = postprocess_dict['origtraincolumns']
    columns_test = list(df_test)
    if columns_train != columns_test:
      print("error, different order of column labels in the train and test set")
      return

    #create an empty dataframe to serve as a store for each column's NArows
    #the column id's for this df will follow convention from NArows of 
    #column+'_NArows' for each column in columns_train
    #these are used in the ML infill methods
    #masterNArows_train = pd.DataFrame()
    masterNArows_test = pd.DataFrame()


    #For each column, determine appropriate processing function
    #processing function will be based on evaluation of train set
    for column in columns_train:


      #we're only going to process columns that weren't in our excluded set
      #note a foreseeable workflow would be for there to be additional\
      #columns desired for exclusion in post processing, consider adding\
      #additional excluded columns as future extensionl
      #if column not in postprocess_dict['excludetransformscolumns']:
      if True == True:
        
        #ok this replaces some methods from 1.76 and earlier for finding a column key
        columnkey = postprocess_dict['origcolumn'][column]['columnkey']        
        #traincategory = postprocess_dict['column_dict'][columnkey]['origcategory']
        traincategory = postprocess_dict['origcolumn'][column]['category']
        
        #originally I seperately used evalcategory to check the actual category of
        #the test set, but now that we are allowing assigned categories that could
        #get too complex, this type of functionality could be a future extension
        #for now let's just make explicit assumption that test set has same 
        #properties as train set
        
        category = traincategory
        
#         #ok postprocess_dict stores column data by the key of column names after\
#         #they have gone through our pre-processing functions, which means the act \
#         #of processing will have \
#         #created new columns and deleted the original column - so since we are \
#         #currently walking through the original column names we'll need to \
#         #pull a post-process column name for the associated columns to serve as \
#         #a key for our postprocess_dict which we'll call columnkey. Also the  \
#         #original category from train set (traincategory) will be accessed to \
#         #serve as a check for consistency between train and test sets.
#         traincategory = False

#         #for postprocesscolumn in postprocess_dict['finalcolumns_train']:
#         for postprocesscolumn in list(preFSpostprocess_dict['column_dict']):
        
#           #if postprocess_dict['column_dict'][postprocesscolumn]['category'] == 'text':
#           if preFSpostprocess_dict['column_dict'][postprocesscolumn]['origcategory'] == 'text':
#             if column == preFSpostprocess_dict['column_dict'][postprocesscolumn]['origcolumn'] \
#             and postprocesscolumn[-5:] != '_NArw':
#             #and postprocesscolumn in postprocess_dict['column_dict'][postprocesscolumn]['categorylist']:
#               traincategory = 'text'
#               columnkey = postprocesscolumn
#               #break

#           #elif postprocess_dict['column_dict'][postprocesscolumn]['category'] == 'date':
#           elif preFSpostprocess_dict['column_dict'][postprocesscolumn]['origcategory'] == 'date':
#             if column == preFSpostprocess_dict['column_dict'][postprocesscolumn]['origcolumn'] \
#             and postprocesscolumn[-5:] != '_NArw':
#               traincategory = 'date'
#               columnkey = postprocesscolumn
#               #break

# #           #elif postprocess_dict['column_dict'][postprocesscolumn]['category'] == 'bxcx':
# #           elif postprocess_dict['column_dict'][postprocesscolumn]['origcategory'] == 'bxcx':
# #             if postprocess_dict['column_dict'][postprocesscolumn]['category'] == 'bxcx':
# #               if column == postprocess_dict['column_dict'][postprocesscolumn]['origcolumn'] \
# #               and postprocesscolumn[-5:] != '_NArw':
# #                 traincategory = 'bxcx'
# #                 columnkey = postprocesscolumn
# #                 #break
# #             #this is kind of a hack, will have to put some thought into if there is a \
# #             #better way to generalize this, as long as we maintain the column naming\
# #             #convention this works
# #             if postprocess_dict['column_dict'][postprocesscolumn]['category'] == 'nmbr':
# #               if column == postprocess_dict['column_dict'][postprocesscolumn]['origcolumn'] \
# #               and postprocesscolumn[-5:] != '_NArw':
# #                 traincategory = 'bxcx'
# #                 columnkey = postprocesscolumn[:-5]+'_bxcx'
# #                 #break
                
#           elif preFSpostprocess_dict['column_dict'][postprocesscolumn]['origcategory'] == 'bxcx':
#             if column == preFSpostprocess_dict['column_dict'][postprocesscolumn]['origcolumn']:
#               traincategory = 'bxcx'
#               columnkey = column + '_bxcx'


#           #elif postprocess_dict['column_dict'][postprocesscolumn]['category'] == 'bnry':
#           elif preFSpostprocess_dict['column_dict'][postprocesscolumn]['origcategory'] == 'bnry':
#             if column == preFSpostprocess_dict['column_dict'][postprocesscolumn]['origcolumn'] \
#             and postprocesscolumn[-5:] != '_NArw':
#               traincategory = 'bnry'
#               columnkey = postprocesscolumn
#               #break

#           #elif postprocess_dict['column_dict'][postprocesscolumn]['category'] == 'nmbr':
#           elif preFSpostprocess_dict['column_dict'][postprocesscolumn]['origcategory'] == 'nmbr':
#             if column == preFSpostprocess_dict['column_dict'][postprocesscolumn]['origcolumn'] \
#             and postprocesscolumn[-5:] != '_NArw':
#               traincategory = 'nmbr'
#               columnkey = postprocesscolumn
#               #break

#           elif traincategory == False:
#             traincategory = 'null'
#             #break



 
        

#         #for the special case of train category = bxcx and test category = nmbr
#         #(meaning there were no negative values in train but there were in test)
#         #we'll resolve by clipping all test values that were <0.1 and setting to 
#         #NaN then resetting the test category to bxcx to be consistent with train
#         if traincategory == 'bxcx' and category == 'nmbr':

#           #convert all values to either numeric or NaN
#           df_test[column] = pd.to_numeric(df_test[column], errors='coerce')


#           df_test[column] = df_test[column].mask(df_test[column] < 0.1)
#           category = 'bxcx'
#           print('Note that values < 0.1 found in test set were reset to NaN')
#           print('to allow consistent box-cox transform as train set.')

#         #another special case, if train category is nmbr and test category is bxcx
#         #default test category to nmbr
#         if traincategory == 'nmbr' and category == 'bxcx':
#           category = 'nmbr'

#         #one more special case, if train was a numerical set to categorical based
#         #on heuristic, let's force test to as well
#         if traincategory == 'text' and category == 'nmbr':
#           category = 'text'
        
#         #one more one more special case, this is certainly an edge case for
#         #very small sets, let's say:
#         if traincategory == 'text' and category == 'bnry':
#           category = 'text'

#         #let's make sure the category is consistent between train and test sets
#         if category != traincategory:
#           print('error - different category between train and test sets for column ',\
#                 column)


        #here we'll delete any columns that returned a 'null' category
        if category == 'null':
          df_test = df_test.drop([column], axis=1)

        #so if we didn't delete the column let's proceed
        else:

          #create NArows (column of True/False where True coresponds to missing data)
          testNArows = self.NArows(df_test, column, category, postprocess_dict)

          #now append that NArows onto a master NA rows df
          masterNArows_test = pd.concat([masterNArows_test, testNArows], axis=1)

          #now process using postprocessfamily functions
          
          #process ancestors
          df_test = \
          self.postprocessancestors(df_test, column, category, category, process_dict, \
                                    transform_dict, preFSpostprocess_dict, columnkey)
          
          #process family
          df_test = \
          self.postprocessfamily(df_test, column, category, category, process_dict, \
                                transform_dict, preFSpostprocess_dict, columnkey)
         
          
          #delete columns subject to replacement
          df_test = \
          self.postcircleoflife(df_test, column, category, category, process_dict, \
                                transform_dict, preFSpostprocess_dict, columnkey)
    
          
    #now that we've pre-processed all of the columns, let's run through them again\
    #using infill to derive plug values for the previously missing cells
    
    infillcolumns_list = list(df_test)
    
    #excludetransformscolumns = postprocess_dict['excludetransformscolumns']
    
    #Here is the list of columns for the stdrdinfill approach
    #(bassically using MLinfill if MLinfill elected for default, otherwise
    #using mean for numerical, most common for binary, and unique column for categorical)
    assigninfill = postprocess_dict['assigninfill']
    
    
    #allstdrdinfill_list = self.stdrdinfilllist(assigninfill, infillcolumns_list)
    
    #Here is the application of assemblepostprocess_assigninfill
    postprocess_assigninfill_dict = \
    self.assemblepostprocess_assigninfill(assigninfill, infillcolumns_list, \
                                          columns_test, preFSpostprocess_dict)
    
    
    columns_train_zero = postprocess_assigninfill_dict['zeroinfill']
    
    for column in columns_train_zero:
      
      categorylistlength = len(preFSpostprocess_dict['column_dict'][column]['categorylist'])
      
      #if (column not in excludetransformscolumns) \
      if (column not in postprocess_assigninfill_dict['stdrdinfill']) \
      and (column[-5:] != '_NArw') \
      and (categorylistlength == 1):
        #noting that currently we're only going to infill 0 for single column categorylists
        #some comparable address for multi-column categories is a future extension
        
        df_test = \
        self.zeroinfillfunction(df_test, column, preFSpostprocess_dict, \
                                masterNArows_test)    
    
    
    columns_train_adj = postprocess_assigninfill_dict['adjinfill']
    for column in columns_train_adj:

      
      #if column not in excludetransformscolumns \
      if column not in postprocess_assigninfill_dict['stdrdinfill'] \
      and column[-5:] != '_NArw':
        
        df_test = \
        self.adjinfillfunction(df_test, column, preFSpostprocess_dict, \
                               masterNArows_test)    
    

    
    
    #if MLinfill == True:
    if postprocess_dict['MLinfill'] == True:

      #columns_train_ML = list(df_train)
#       columns_test_ML = postprocess_assigninfill_dict['stdrdinfill'] \
#                          + postprocess_assigninfill_dict['MLinfill']
      columns_test_ML = list(set().union(postprocess_assigninfill_dict['stdrdinfill'], \
                                          postprocess_assigninfill_dict['MLinfill']))
        

    else:
      
      columns_test_ML = postprocess_assigninfill_dict['MLinfill']
    
    iteration = 0
    #while iteration < infilliterate:
    while iteration < postprocess_dict['infilliterate']:


      #since we're reusing the text_dict and date_dict from our original automunge
      #we're going to need to re-initialize the infillcomplete markers
      #actually come to this of it we need to go back to automunge and do this
      #for the MLinfill iterations as well

      #re-initialize the infillcomplete marker in column _dict's
      #for key in postprocess_dict['column_dict']:
      for key in columns_test_ML:
        preFSpostprocess_dict['column_dict'][key]['infillcomplete'] = False



      for column in columns_test_ML:

#           #troubleshoot
#           print("for column in columns_test_ML:, column = ", column)

        #we're only going to process columns that weren't in our excluded set
        #if column not in excludetransformscolumns:
        #troublshoot
        #print("what the heck yo, orint list(psotprocess_dict)")
        #print(list(postprocess_dict))
        #if column not in postprocess_dict['excludetransformscolumns'] \
        #if column not in excludetransformscolumns \
        if column[-5:] != '_NArw':

          df_test, preFSpostprocess_dict = \
          self.postMLinfillfunction (df_test, column, preFSpostprocess_dict, \
                                masterNArows_test)


      iteration += 1                     



    #trim branches associated with feature selection
    if postprocess_dict['featureselection'] == True:
      
      
      #get list of columns currently includedt
      currentcolumns = list(df_test)
      
      #get list of columns to trim
      madethecutset = set(postprocess_dict['madethecut'])
      trimcolumns = [b for b in currentcolumns if b not in madethecutset]
      
      #trim columns manually
      for trimmee in trimcolumns:
        del df_test[trimmee]

    #here's a list of final column names saving here since the translation to \
    #numpy arrays scrubs the column names
    finalcolumns_test = list(df_test)
    
    #determine output type based on pandasoutput argument
    if pandasoutput == True:
      #global processing to test set including conversion to numpy array
      test = df_test

      if testID_column != False:
        testID = df_testID
      else:
        testID = pd.DataFrame()
    
    #else output numpy arrays
    else:
      #global processing to test set including conversion to numpy array
      test = df_test.values

      if testID_column != False:
        testID = df_testID.values
      else:
        testID = []

    labelsencoding_dict = postprocess_dict['labelsencoding_dict']
    

    

    return test, testID, labelsencoding_dict, finalcolumns_test

    
