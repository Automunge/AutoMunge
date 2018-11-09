#global imports
import numpy as np
import pandas as pd

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

#imports for predictinfill, predictpostinfill
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

#imports for automunge
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class AutoMunge:
  
  def __init__(self):
    pass
  
  
  def process_numerical_class(self, mdf_train, mdf_test, column):

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

    #add a second column with boolean expression indicating a missing cell
    #(using NArows(.) function defined below, column name will be column+'_NArows')
    NArows_nmbr_train = self.NArows(mdf_train, column, 'nmbr')
    NArows_nmbr_test = self.NArows(mdf_test, column, 'nmbr')
    mdf_train[column + '_NArw'] = NArows_nmbr_train.copy()
    mdf_test[column + '_NArw'] = NArows_nmbr_test.copy()
    del NArows_nmbr_train
    del NArows_nmbr_test

    #change NArows data type to 8-bit (1 byte) integers for memory savings
    mdf_train[column + '_NArw'] = mdf_train[column + '_NArw'].astype(np.int8)
    mdf_test[column + '_NArw'] = mdf_test[column + '_NArw'].astype(np.int8)

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


    #change column name to column + '_nmbr'
    mdf_train[column + '_nmbr'] = mdf_train[column].copy()
    mdf_test[column + '_nmbr'] = mdf_test[column].copy()
    del mdf_train[column]
    del mdf_test[column]


    #create bins based on standard deviation increments
    binscolumn = column + '_bins'
    nmbrcolumn = column + '_nmbr'
    mdf_train[binscolumn] = \
    pd.cut( mdf_train[nmbrcolumn], bins = [-float('inf'),-2,-1,0,1,2,float('inf')],  \
           labels = ['_<-2','_-21','_-10','_+01','_+12','_>+2'], precision=4)
    mdf_test[binscolumn] = \
    pd.cut( mdf_test[nmbrcolumn], bins = [-float('inf'), -2, -1, 0, 1, 2, float('inf')],  \
           labels = ['_<-2','_-21','_-10','_+01','_+12','_>+2'], precision=4)



    textcolumns = \
    [binscolumn + '__<-2', binscolumn + '__-21', binscolumn + '__-10', \
     binscolumn + '__+01', binscolumn + '__+12', binscolumn + '__>+2']

    #process bins as a categorical set
    mdf_train = \
    self.postprocess_text_class(mdf_train, binscolumn, textcolumns)
    mdf_test = \
    self.postprocess_text_class(mdf_test, binscolumn, textcolumns)



    #create list of columns
    nmbrcolumns = [column + '_nmbr', column + '_NArw'] + textcolumns



    nmbrnormalization_dict = {'mean' : mean, 'std' : std}

    #store some values in the nmbr_dict{} for use later in ML infill methods
    column_dict_list = []

    for nc in nmbrcolumns:

      if nc[-5:] == '_NArw':

        column_dict = { nc : {'category' : 'NArw', \
                             'origcategory' : 'nmbr', \
                             'normalization_dict' : nmbrnormalization_dict, \
                             'origcolumn' : column, \
                             'columnslist' : nmbrcolumns, \
                             'categorylist' : [nc], \
                             'infillmodel' : False, \
                             'infillcomplete' : False}}

        column_dict_list.append(column_dict.copy())

      if nc[-5:] == '_nmbr':

        column_dict = { nc : {'category' : 'nmbr', \
                             'origcategory' : 'nmbr', \
                             'normalization_dict' : nmbrnormalization_dict, \
                             'origcolumn' : column, \
                             'columnslist' : nmbrcolumns, \
                             'categorylist' : [nc], \
                             'infillmodel' : False, \
                             'infillcomplete' : False}}

        column_dict_list.append(column_dict.copy())

      if nc in textcolumns:

        column_dict = { nc : {'category' : 'text', \
                             'origcategory' : 'nmbr', \
                             'normalization_dict' : nmbrnormalization_dict, \
                             'origcolumn' : column, \
                             'columnslist' : nmbrcolumns, \
                             'categorylist' : textcolumns, \
                             'infillmodel' : False, \
                             'infillcomplete' : False}}

        column_dict_list.append(column_dict.copy())



    #return mdf_train, mdf_test, mean, std, nmbrcolumns, categorylist
    return mdf_train, mdf_test, column_dict_list




  def process_binary_class(self, mdf, column, missing):

    #process_binary_class(mdf, column, missing)
    #converts binary classification values to 0 or 1
    #takes as arguement a pandas dataframe (mdf), \
    #the name of the column string ('column') \
    #and the string classification to assign to missing data ('missing')
    #replaces original specified column in dataframe
    #returns transformed dataframe

    #missing category must be identical to one of the two existing categories
    #returns error message if more than two categories remain

    #add a second column with boolean expression indicating a missing cell
    #(using NArows(.) function defined below, column name will be column+'_NArows')
    NArows_bnry = self.NArows(mdf, column, 'bnry')
    mdf[column + '_NArw'] = NArows_bnry.copy()
    del NArows_bnry

    #replace missing data with specified classification
    mdf[column] = mdf[column].fillna(missing)

    #if more than two remaining classifications, return error message    
    if len(mdf[column].unique()) > 2:
        print('ERROR: number of categories in column for process_binary_class() call >2')
        return mdf

    #convert column to binary 0/1 classification
    lb = preprocessing.LabelBinarizer()
    mdf[column] = lb.fit_transform(mdf[column])

    #change column name to column + '_bnry'
    mdf[column + '_bnry'] = mdf[column].copy()
    del mdf[column]

    #create list of columns
    bnrycolumns = [column + '_bnry', column + '_NArw']

    #change data types to 8-bit (1 byte) integers for memory savings
    for bnrycolumn in bnrycolumns:
      mdf[bnrycolumn] = mdf[bnrycolumn].astype(np.int8)

    #create list of columns associated with categorical transform (blank for now)
    categorylist = []

    bnrynormalization_dict = {'missing' : missing}

    #store some values in the column_dict{} for use later in ML infill methods
    column_dict_list = []

    for bc in bnrycolumns:

      if bc[-5:] == '_NArw':

        column_dict = { bc : {'category' : 'NArw', \
                             'origcategory' : 'bnry', \
                             'normalization_dict' : bnrynormalization_dict, \
                             'origcolumn' : column, \
                             'columnslist' : bnrycolumns, \
                             'categorylist' : [bc], \
                             'infillmodel' : False, \
                             'infillcomplete' : False}}

        column_dict_list.append(column_dict.copy())


      if bc[-5:] == '_bnry':

        column_dict = { bc : {'category' : 'bnry', \
                             'origcategory' : 'bnry', \
                             'normalization_dict' : bnrynormalization_dict, \
                             'origcolumn' : column, \
                             'columnslist' : bnrycolumns, \
                             'categorylist' : [bc], \
                             'infillmodel' : False, \
                             'infillcomplete' : False}}

        column_dict_list.append(column_dict.copy())


    #return mdf, bnrycolumns, categorylist, column_dict_list
    return mdf, column_dict_list






  def process_text_class(self, mdf_train, mdf_test, column):

    #process_text_class(mdf_train, mdf_test, column)
    #preprocess column with text classifications
    #takes as arguement two pandas dataframe containing training and test data respectively 
    #(mdf_train, mdf_test), and the name of the column string ('column')

    #note this trains both training and test data simultaneously due to unique treatment if any category
    #missing from training set but not from test set to ensure consistent formatting 

    #deletes the original column from master dataframe and
    #replaces with onehot encodings
    #with columns named after column_ + text classifications
    #missing data replaced with category label 'missing'+column
    #any categories missing from the training set removed from test set
    #any category present in training but missing from test set given a column of zeros for consistent formatting
    #ensures order of all new columns consistent between both sets
    #returns two transformed dataframe (mdf_train, mdf_test) \
    #and a list of the new column names (textcolumns)

    #if only have training but not test data handy, use same training data for both dataframe inputs

    #note it is kind of a hack here to create a column for missing values with \
    #two underscores (__) in the column name to ensure appropriate order for cases\
    #where NaN present in test data but not train data, if a category starts with|
    #an underscore such that it preceeds '__missing' alphabetically in this scenario\
    #this might create error due to different order of columns, address of this \
    #potential issue will be a future extension

    #convert column to category
    mdf_train[column] = mdf_train[column].astype('category')
    mdf_test[column] = mdf_test[column].astype('category')

    #if set is categorical we'll need the plug value for missing values included
    mdf_train[column] = mdf_train[column].cat.add_categories(['NArw'])
    mdf_test[column] = mdf_test[column].cat.add_categories(['NArw'])

    #replace NA with a dummy variable
    mdf_train[column] = mdf_train[column].fillna('NArw')
    mdf_test[column] = mdf_test[column].fillna('NArw')

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

    #add a missing column to train if it's not present
    if column + '_NArw' not in df_train_cat.columns:
      missingcolumn = pd.DataFrame(0, index=np.arange(df_train_cat.shape[0]), columns=[column+'_NArw'])
      df_train_cat = pd.concat([missingcolumn, df_train_cat], axis=1)


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


    #delete original column from training data
    del mdf_train[column]    
    del mdf_test[column]

    #create output of a list of the created column names
    labels_train = list(df_train_cat)
    textcolumns = labels_train

    #change data types to 8-bit (1 byte) integers for memory savings
    for textcolumn in textcolumns:
      mdf_train[textcolumn] = mdf_train[textcolumn].astype(np.int8)
      mdf_test[textcolumn] = mdf_test[textcolumn].astype(np.int8)


    #store some values in the text_dict{} for use later in ML infill methods
    column_dict_list = []

    categorylist = textcolumns.copy()

    categorylist.remove(column + '_NArw')

    for tc in textcolumns:

      if tc[-5:] == '_NArw':

        column_dict = { tc : {'category' : 'NArw', \
                             'origcategory' : 'text', \
                             'normalization_dict' : {}, \
                             'origcolumn' : column, \
                             'columnslist' : textcolumns, \
                             'categorylist' : [tc], \
                             'infillmodel' : False, \
                             'infillcomplete' : False}}

        column_dict_list.append(column_dict.copy())

    for tc in categorylist:

      column_dict = {tc : {'category' : 'text', \
                           'origcategory' : 'text', \
                           'normalization_dict' : {}, \
                           'origcolumn' : column, \
                           'columnslist' : textcolumns, \
                           'categorylist' : categorylist, \
                           'infillmodel' : False, \
                           'infillcomplete' : False }}

      column_dict_list.append(column_dict.copy())


    #return mdf_train, mdf_test, textcolumns, categorylist
    return mdf_train, mdf_test, column_dict_list




  def process_time_class(self, mdf_train, mdf_test, column):

    #process_time_class(mdf_train, mdf_test, column)
    #preprocess column with time classifications
    #takes as arguement two pandas dataframe containing training and test data respectively 
    #(mdf_train, mdf_test), and the name of the column string ('column')

    #note this trains both training and test data simultaneously due to unique treatment if any category
    #missing from training set but not from test set to ensure consistent formatting 

    #deletes the original column from master dataframe and
    #replaces with distinct columns for year, month, day, hour, minute, second
    #each normalized to the mean and std, with missing values plugged with the mean
    #with columns named after column_ + time category
    #returns two transformed dataframe (mdf_train, mdf_test)

    #if only have training but not test data handy, use same training data for both dataframe inputs

    #add a second column with boolean expression indicating a missing cell
    #(using NArows(.) function defined below, column name will be column+'_NArows')
    NArows_nmbr_train = self.NArows(mdf_train, column, 'nmbr')
    NArows_nmbr_test = self.NArows(mdf_test, column, 'nmbr')
    mdf_train[column + '_NArw'] = NArows_nmbr_train.copy()
    mdf_test[column + '_NArw'] = NArows_nmbr_test.copy()
    del NArows_nmbr_train
    del NArows_nmbr_test

    #change NArows data type to 8-bit (1 byte) integers for memory savings
    mdf_train[column + '_NArw'] = mdf_train[column + '_NArw'].astype(np.int8)
    mdf_test[column + '_NArw'] = mdf_test[column + '_NArw'].astype(np.int8)

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
                  column + '_hour', column + '_minute', column + '_second', \
                  column + '_NArw']

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


    #delete original column from training data
    del mdf_train[column]    
    if column in mdf_test.columns:
      del mdf_test[column]  


    #output a dictionary of the associated column mean and std

    timenormalization_dict = {'meanyear' : meanyear, 'meanmonth' : meanmonth, \
                              'meanday' : meanday, 'meanhour' : meanhour, \
                              'meanminute' : meanminute, 'meansecond' : meansecond,\
                              'stdyear' : stdyear, 'stdmonth' : stdmonth, \
                              'stdday' : stdday, 'stdhour' : stdhour, \
                              'stdminute' : stdminute, 'stdsecond' : stdsecond}

    #create list of columns associated with categorical transform (blank for now)
    categorylist = []


    #store some values in the date_dict{} for use later in ML infill methods

    column_dict_list = []

    categorylist = datecolumns.copy()

    categorylist.remove(column + '_NArw')

    for dc in datecolumns:

      if dc[-5:] == '_NArw':

        column_dict = { dc : {'category' : 'NArw', \
                             'origcategory' : 'date', \
                             'normalization_dict' : {}, \
                             'origcolumn' : column, \
                             'columnslist' : datecolumns, \
                             'categorylist' : [dc], \
                             'infillmodel' : False, \
                             'infillcomplete' : False}}

        column_dict_list.append(column_dict.copy())



    for dc in categorylist:

      column_dict = {dc : {'category' : 'date', \
                           'origcategory' : 'date', \
                           'normalization_dict' : timenormalization_dict, \
                           'origcolumn' : column, \
                           'columnslist' : datecolumns, \
                           'categorylist' : categorylist, \
                           'infillmodel' : False, \
                           'infillcomplete' : False }}

      column_dict_list.append(column_dict.copy())




    #return mdf_train, mdf_test, datecolumns, timenormalization_dict, categorylist
    return mdf_train, mdf_test, column_dict_list





  def process_bxcx_class(self, df, column, bxcxerrorcorrect, bxcx_lmbda = None, \
                         trnsfrm_mean = None, trnsfrm_std = None):

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

    #add a second column with boolean expression indicating a missing cell
    #(using NArows(.) function defined below, column name will be column+'_NArows')
    NArows_bxcx = self.NArows(df, column, 'nmbr')
    df[column + '_NArw'] = NArows_bxcx.copy()
    del NArows_bxcx


    #change NArows data type to 8-bit (1 byte) integers for memory savings
    df[column + '_NArw'] = df[column + '_NArw'].astype(np.int8)

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


    #apply process_numerical_class to other column and change name
    #although if we were given a mean and std value then apply postprocess function

    #if std is none
    if trnsfrm_std == None:

      #get std from train set
      std = df[column].std()

      #apply numerical transforms
      df[column + '_nmbr'] = df[column] - mean
      df[column + '_nmbr'] = df[column + '_nmbr'] / std

    else:

      std = trnsfrm_std

      #apply numerical transforms
      df[column + '_nmbr'] = df[column] - mean
      df[column + '_nmbr'] = df[column + '_nmbr'] / std 

    #delete the original column
    del df[column]  


    #output of a list of the created column names
    nmbrcolumns = [column + '_nmbr', column + '_bxcx', column + '_NArw']

    #output a dictionary of the associated column mean and std

    normalization_dict = {'trnsfrm_mean' : mean, 'trnsfrm_std' : std, \
                          'bxcx_lmbda' : bxcx_lmbda, \
                          'bxcxerrorcorrect' : bxcxerrorcorrect, \
                          'mean' : mean, 'std' : std}

    #create list of columns associated with categorical transform (blank for now)
    categorylist = []


    #store some values in the nmbr_dict{} for use later in ML infill methods
    column_dict_list = []

    for nc in nmbrcolumns:

      if nc[-5:] == '_NArw':

        column_dict = { nc : {'category' : 'NArw', \
                             'origcategory' : 'bxcx', \
                             'normalization_dict' : normalization_dict, \
                             'origcolumn' : column, \
                             'columnslist' : nmbrcolumns, \
                             'categorylist' : [nc], \
                             'infillmodel' : False, \
                             'infillcomplete' : False}}

        column_dict_list.append(column_dict.copy())


      if nc[-5:] == '_nmbr':

        column_dict = { nc : {'category' : 'nmbr', \
                             'origcategory' : 'bxcx', \
                             'normalization_dict' : normalization_dict, \
                             'origcolumn' : column, \
                             'columnslist' : nmbrcolumns, \
                             'categorylist' : [nc], \
                             'infillmodel' : False, \
                             'infillcomplete' : False}}

        column_dict_list.append(column_dict.copy())


      if nc[-5:] == '_bxcx':

        column_dict = { nc : {'category' : 'bxcx', \
                             'origcategory' : 'bxcx', \
                             'normalization_dict' : normalization_dict, \
                             'origcolumn' : column, \
                             'columnslist' : nmbrcolumns, \
                             'categorylist' : [nc], \
                             'infillmodel' : False, \
                             'infillcomplete' : False}}

        column_dict_list.append(column_dict.copy())



    #return df, nmbrcolumns, nmbrnormalization_dict, categorylist
    return df, column_dict_list




  def evalcategory(self, df, column, numbercategoryheuristic):

    #evalcategory(df, column)
    #Function that dakes as input a dataframe and associated column id \
    #evaluates the contents of cells and classifies the column into one of four categories
    #category 1, 'bnry', is for columns with only two categorys of text or integer
    #category 2, 'nmbr', is for columns with numerical integer or float values
    #category 3: 'bxcx', is for nmbr category with all positive values
    #category 4, 'text', is for columns with multiple categories appropriate for one-hot
    #category 5, 'date', is for columns with Timestamp data
    #category 6, 'null', is for columns with >85% null values (arbitrary figure)
    #returns category id as a string


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
      if df[column].nunique() / df[column].shape[0] < numbercategoryheuristic:

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
      if isinstance(checkstring, mc2[1][0]) and df[column].nunique() > 2:
        category = 'text'

      #if 2nd most common is date, set category to date   
      if isinstance(df_checkdate['checkdate'][0], datemc2[0][0]):
        category = 'date'

      #if 2nd most common in column is integer and > two values, set category to number
      if isinstance(checkint, mc2[1][0]) and df[column].nunique() > 2:


        #take account for numbercategoryheuristic
        if df[column].nunique() / df[column].shape[0] < numbercategoryheuristic:

          category = 'text'

        else:

          category = 'nmbr'

      #if 2nd most common in column is float, set category to number
      if isinstance(checkfloat, mc2[1][0]):

        #take account for numbercategoryheuristic
        if df[column].nunique() / df[column].shape[0] < numbercategoryheuristic:

          category = 'text'

        else:

          category = 'nmbr'

      #if 2nd most common in column is integer and <= two values, set category to binary
      if isinstance(checkint, mc2[1][0]) and df[column].nunique() <= 2:
        category = 'bnry'

      #if 2nd most common in column is string and <= two values, set category to binary
      if isinstance(checkstring, mc2[1][0]) and df[column].nunique() <= 2:
        category = 'bnry'


    return category





  def NArows(self, df, column, category):

    #NArows(df, column), function that when fed a dataframe, \
    #column id, and category label outputs a single column dataframe composed of \
    #True and False with the same number of rows as the input and the True's \
    #coresponding to those rows of the input that had missing or NaN data. This \
    #output can later be used to identify which rows for a column to infill with ML\
    # derived plug data



    if category == 'text':

      #returns dataframe of True and False, where True coresponds to the NaN's
      #renames column name to column + '_NArows'
      NArows = pd.isna(df[column])
      NArows = pd.DataFrame(NArows)
      NArows = NArows.rename(columns = {column:column+'_NArows'})

    if category == 'bnry':

      #returns dataframe of True and False, where True coresponds to the NaN's
      #renames column name to column + '_NArows'
      NArows = pd.isna(df[column])
      NArows = pd.DataFrame(NArows)
      NArows = NArows.rename(columns = {column:column+'_NArows'})

    if category == 'nmbr' or category == 'bxcx':

      #convert all values to either numeric or NaN
      df[column] = pd.to_numeric(df[column], errors='coerce')

      #returns dataframe of True and False, where True coresponds to the NaN's
      #renames column name to column + '_NArows'
      NArows = pd.isna(df[column])
      NArows = pd.DataFrame(NArows)
      NArows = NArows.rename(columns = {column:column+'_NArows'})


    if category == 'date':

      #returns dataframe column of all False
      #renames column name to column + '_NArows'
      NArows = pd.DataFrame(False, index=np.arange(df.shape[0]), columns=[column+'NA'])
      NArows = pd.DataFrame(NArows)
      NArows = NArows.rename(columns = {column:column+'_NArows'})


    return NArows




  def labelbinarizercorrect(self, npinput, columnslist):

    #labelbinarizercorrect(npinput, columnslist), function that takes as input the output\
    #array from scikit learn's LabelBinarizer() and ensures that the re-encoding is\
    #consistent with the original array prior to performing the argmax. This is \
    #needed because LabelBinarizer automatically takes two class sets to a binary\
    #setting and doesn't account for columns above index of active values based on\
    #my understanding. For a large enough dataset this probably won't be an issue \
    #but just trying to be thorough. Outputs a one-hot encoded array comparable to \
    #the format of our input to argmax.


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
                    columnslist = []):

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

    #convert dataframes to numpy arrays
    np_train_filltrain = df_train_filltrain.values
    np_train_filllabel = df_train_filllabel.values
    np_train_fillfeatures = df_train_fillfeatures.values
    np_test_fillfeatures = df_test_fillfeatures.values

    #ony run the following if we have any rows needing infill
    if df_train_fillfeatures.shape[0] > 0:

      if category == 'nmbr':

        #this is to address a weird error message suggesting I reshape the y with ravel()
        np_train_filllabel = np.ravel(np_train_filllabel)

        #train linear regression model using scikit-learn for numerical prediction
        #model = LinearRegression()
        #model = PassiveAggressiveRegressor(random_state = randomseed)
        #model = Ridge(random_state = randomseed)
        #model = RidgeCV()
        #note that SVR doesn't have an argument for random_state
        #model = SVR()
        model = RandomForestRegressor(random_state = randomseed)

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


      if category == 'bxcx':

        #this is to address a weird error message suggesting I reshape the y with ravel()
        np_train_filllabel = np.ravel(np_train_filllabel)

        #model = SVR()
        model = RandomForestRegressor(random_state = randomseed)

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



      if category == 'bnry':

        #this is to address a weird error message suggesting I reshape the y with ravel()
        np_train_filllabel = np.ravel(np_train_filllabel)

        #train logistic regression model using scikit-learn for binary classifier
        #model = LogisticRegression()
        #model = LogisticRegression(random_state = randomseed)
        #model = SGDClassifier(random_state = randomseed)
        #model = SVC(random_state = randomseed)
        model = RandomForestClassifier(random_state = randomseed)

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

      if category == 'text':

        #first convert the one-hot encoded set via argmax to a 1D array
        np_train_filllabel_argmax = np.argmax(np_train_filllabel, axis=1)

        #train logistic regression model using scikit-learn for binary classifier
        #with multi_class argument activated
        #model = LogisticRegression()
        #model = SGDClassifier(random_state = randomseed)
        #model = SVC(random_state = randomseed)
        model = RandomForestClassifier(random_state = randomseed)

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

      if category == 'date':

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
                         category, columnslist = [], categorylist = []):

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



    #create 3 new dataframes for each train column - the train and labels \
    #for rows not needing infill, and the features for rows needing infill \
    #also create a test features column 

    if category in ['nmbr', 'bxcx', 'bnry', 'text']:

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




    if category == 'date':

      #create empty sets for now
      #an extension of this method would be to implement a comparable method \
      #for the time category, based on the columns output from the preprocessing
      df_train_filltrain = pd.DataFrame({'foo' : []}) 
      df_train_filllabel = pd.DataFrame({'foo' : []})
      df_train_fillfeatures = pd.DataFrame({'foo' : []})
      df_test_fillfeatures = pd.DataFrame({'foo' : []})


    return df_train_filltrain, df_train_filllabel, df_train_fillfeatures, df_test_fillfeatures





  def insertinfill(self, df, column, infill, category, NArows, columnslist = [], \
                   categorylist = []):

    #insertinfill(df, column, infill, category, NArows, columnslist = [])
    #function that takes as input a dataframe, column id, category string of either\
    #'nmbr'/'text'/'bnry'/'date', a df column of True/False identifiying row id of\
    #rows that will recieve infill, and and a list of columns produced by a text \
    #class preprocessor when applicable. Replaces the column cells in rows \
    #coresponding to the NArows True values with the values from infill, returns\
    #the associated transformed dataframe.

    #NArows column name uses original column name + _NArows as key
    #by convention, current column has original column name + '_ctgy' at end
    #so we'll drop final 5 characters from column string
    origcolumnname = column[:-5]

    if category in ['nmbr', 'bxcx', 'bnry', 'text']:

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
        infillindex = df.loc[df[origcolumnname+'_NArows']]['tempindex1']

        #create a dictionary for use to insert infill using df's index as the key
        infill_dict = dict(zip(infillindex, infill.values))

        #replace 'tempindex1' column with infill in rows where NArows is True
        df['tempindex1'] = np.where(df[origcolumnname+'_NArows'], df['tempindex1'].replace(infill_dict), 'fill')

        #now carry that infill over to the target column for rows where NArows is True
        df[column] = np.where(df[origcolumnname+'_NArows'], df['tempindex1'], df[column])

        #remove the temporary columns from df
        df = df.drop(['tempindex1'], axis=1)
        df = df.drop([origcolumnname+'_NArows'], axis=1)




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
          infill = infill.drop([textcolumnname + '_bool'], axis=1)
          infill = infill.drop(['tempindex1'], axis=1)
          df = df.drop(['textNArows'], axis=1)
          df = df.drop(['tempindex1'], axis=1)




    if category == 'date':
      #this spot reserved for future update to incorporate address of datetime\
      #category data
      df = df


    return df




  def MLinfillfunction (self, df_train, df_test, column, postprocess_dict, \
                        masterNArows_train, masterNArows_test, randomseed):

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
                         category, \
                         columnslist = columnslist, \
                         categorylist = categorylist)

      #predict infill values using defined function predictinfill(.)
      df_traininfill, df_testinfill, model = \
      self.predictinfill(category, df_train_filltrain, df_train_filllabel, \
                    df_train_fillfeatures, df_test_fillfeatures, \
                    randomseed, columnslist = columnslist)

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
                            columnslist = columnslist, categorylist = categorylist)

      #if we don't train the train set model on any features, that we won't be able 
      #to apply the model to predict the test set infill. 

      if any(x == True for x in masterNArows_train[origcolumn+'_NArows']):

        df_test = self.insertinfill(df_test, column, df_testinfill, category, \
                           pd.DataFrame(masterNArows_test[origcolumn+'_NArows']), \
                           columnslist = columnslist, categorylist = categorylist)

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

    #LabelSetGenerator
    #takes as input dataframe for test set, label column name, and label
    #returns a dataframe set of all rows which included that label in the column


    df = df[df[column] == label]

    return df



  def LabelFrequencyLevelizer(self, train_df, labels_df, labelsencoding_dict):

    #LabelFrequencyLevelizer(.)
    #takes as input dataframes for train set, labels, and label category
    #combines them to single df, then creates sets for each label category
    #such as to add on multiples of each set to achieve near levelized
    #frequency of label occurence in training set (increases the size
    #of the training set by redundant inclusion of rows with lower frequency
    #labels.) Returns train_df, labels_df, trainID_df.


    columns_labels = list(labels_df)

    labelscategory = next(iter(labelsencoding_dict))

    labels = list(labelsencoding_dict[labelscategory].keys())

    setnameslist = []
    setlengthlist = []
    multiplierlist = []

    if labelscategory == 'bnry':


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


    if labelscategory in ['nmbr', 'bxcx']:

        columns_labels = []
        for label in list(labels_df):
          if label[-5:] in ['__<-2', '__-21', '__-10', '__+01', '__+12', '__>+2']:
            columns_labels.append(label)

    if labelscategory in ['text', 'nmbr', 'bxcx']:


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





  def automunge(self, df_train, df_test = False, labels_column = False, trainID_column = False, \
                testID_column = False, valpercent1=0.20, valpercent2 = 0.10, \
                shuffletrain = True, TrainLabelFreqLevel = True, powertransform = True, \
                MLinfill = True, infilliterate=1, randomseed = 42, \
                forcetocategoricalcolumns = [], numbercategoryheuristic = 0.000, \
                excludetransformscolumns = []):


    #automunge(df_train, df_test, labels_column, valpercent=0.20, powertransform = True, \
    #MLinfill = True, infilliterate=1, randomseed = 42, excludetransformscolumns = []) \
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
    #return errors

    #An extension could be to test the input data here for non-dataframe format \
    #(such as csv) to convert it to pandas within the function. 

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

    if testID_column != False:
      df_testID = pd.DataFrame(df_test[testID_column])
      del df_test[testID_column]

    #extract labels from train set
    #an extension to this function could be to delete the training set rows\
    #where the labels are missing or improperly formatted prior to performing\
    #this step
    if labels_column != False:
      df_labels = pd.DataFrame(df_train[labels_column])

      #create copy of labels to support the translation dictionary for use after \
      #prediction to convert encoded predictions back to the original label
      df_labels2 = pd.DataFrame(df_labels.copy())

      del df_train[labels_column]


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
    postprocess_dict = {'column_dict' : {}}


    #For each column, determine appropriate processing function
    #processing function will be based on evaluation of train set
    for column in columns_train:

      #re-initialize the column specific dictionary for later insertion into
      #our postprocess_dict
      column_dict = {}

      #we're only going to process columns that weren't in our excluded set
      if column not in excludetransformscolumns:

        category = self.evalcategory(df_train, column, numbercategoryheuristic)

        #special case for force to categorical
        if column in forcetocategoricalcolumns:
          category = 'text'

        #let's make sure the category is consistent between train and test sets
        category_test = self.evalcategory(df_test, column, numbercategoryheuristic)

        #special case for force to categorical
        if column in forcetocategoricalcolumns \
        or df_train[column].dtype.name == 'category':
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

        #here we'll delete any columns that returned a 'null' category
        if category == 'null':
          df_train = df_train.drop([column], axis=1)
          df_test = df_test.drop([column], axis=1)

          column_dict = { column + '_null' : {'category' : 'null', \
                                              'origcategory' : 'null', \
                                              'normalization_dict' : {}, \
                                              'origcolumn' : column, \
                                              'columnslist' : [column], \
                                              'categorylist' : [], \
                                              'infillmodel' : False, \
                                              'infillcomplete' : False }}

          #now append column_dict onto postprocess_dict
          postprocess_dict['column_dict'].update(column_dict)


        #so if we didn't delete the column let's proceed
        else:




          #create NArows (column of True/False where True coresponds to missing data)
          trainNArows = self.NArows(df_train, column, category)
          testNArows = self.NArows(df_test, column, category)

          #now append that NArows onto a master NA rows df
          masterNArows_train = pd.concat([masterNArows_train, trainNArows], axis=1)
          masterNArows_test = pd.concat([masterNArows_test, testNArows], axis=1)


          #(now normalize as would normally)



          #for binary class use the majority field for missing plug value
          if category == 'bnry':
            binary_missing_plug = df_train[column].value_counts().index.tolist()[0]


            #apply appropriate processing function to this column based on the result
            df_train, column_dict_list = self.process_binary_class(df_train, column, binary_missing_plug)
            df_test, _1 = self.process_binary_class(df_test, column, binary_missing_plug)


            for column_dict in column_dict_list:

              #now append column_dict onto postprocess_dict
              postprocess_dict['column_dict'].update(column_dict)


          if category == 'nmbr':
            #df_train, df_test, mean, std, nmbrcolumns, categorylist = \
            df_train, df_test, column_dict_list = \
            self.process_numerical_class(df_train, df_test, column)


            for column_dict in column_dict_list:

              #now append column_dict onto postprocess_dict
              postprocess_dict['column_dict'].update(column_dict)


          if category == 'bxcx':

            #df_train, nmbrcolumns, nmbrnormalization_dict, categorylist = \
            df_train, column_dict_list = \
            self.process_bxcx_class(df_train, column, 1, bxcx_lmbda = None, \
                               trnsfrm_mean = None, trnsfrm_std = None)

            #grab the normalization_dict associated with the bxcx category
            columnkeybxcx = column + '_bxcx'
            for column_dict in column_dict_list:
              if columnkeybxcx in column_dict:
                bxcxnormalization_dict = column_dict[columnkeybxcx]['normalization_dict']

            #df_test, nmbrcolumns, _1, _2 = \
            df_test, _1 = \
            self.process_bxcx_class(df_test, column, 1, bxcx_lmbda = \
                               bxcxnormalization_dict['bxcx_lmbda'], \
                               trnsfrm_mean = bxcxnormalization_dict['trnsfrm_mean'], \
                               trnsfrm_std = bxcxnormalization_dict['trnsfrm_std'])



            for column_dict in column_dict_list:

              #now append column_dict onto postprocess_dict
              postprocess_dict['column_dict'].update(column_dict)

          if category == 'text':
            #df_train, df_test, textcolumns, categorylist = \
            df_train, df_test, column_dict_list = \
            self.process_text_class(df_train, df_test, column)


            for column_dict in column_dict_list:

              #now append column_dict onto postprocess_dict
              postprocess_dict['column_dict'].update(column_dict)


          if category == 'date':

            df_train, df_test, column_dict_list = \
            self.process_time_class(df_train, df_test, column)


            for column_dict in column_dict_list:

              #now append column_dict onto postprocess_dict
              postprocess_dict['column_dict'].update(column_dict)


    #now that we've pre-processed all of the columns, let's run through them again\
    #using ML to derive plug values for the previously missing cells


    if MLinfill == True:


      columns_train_ML = list(df_train)
      columns_test_ML = list(df_test)


      iteration = 0

      while iteration < infilliterate:


        for key in postprocess_dict['column_dict']:
          postprocess_dict['column_dict'][key]['infillcomplete'] = False


        for column in columns_train_ML:


          #we're only going to process columns that weren't in our excluded set
          #or aren't identifiers for NA rows
          if column not in excludetransformscolumns \
          and column[-5:] != '_NArw':


            df_train, df_test, postprocess_dict = \
            self.MLinfillfunction(df_train, df_test, column, postprocess_dict, \
                    masterNArows_train, masterNArows_test, randomseed)


        iteration += 1



    if labels_column != False:

      #determine labels category and apply appropriate function
      labelscategory = self.evalcategory(df_labels, labels_column, numbercategoryheuristic)

      #if labels in forcetocategory then revise category
      if labels_column in forcetocategoricalcolumns:
        labelscategory = 'text'


      #copy dummy labels "test" df for our preprocessing functions
      #labelsdummy = pd.DataFrame()
      labelsdummy = df_labels.copy()

      #initialize a dictionary to serve as the store between labels and their \
      #associated encoding
      labelsencoding_dict = {labelscategory:{}}

      #apply appropriate processing function to this column based on the result
      if labelscategory == 'bnry':
        labels_binary_missing_plug = df_labels[labels_column].value_counts().index.tolist()[0]
        df_labels, _1 = self.process_binary_class(df_labels, labels_column, labels_binary_missing_plug)

        del df_labels[labels_column + '_NArw']

        finalcolumns_labels = list(df_labels)

        #here we'll populate the dictionery pairing values from the encoded labels \
        #column with the original value for transformation post prediciton


        i = 0

        for row in df_labels.iterrows():
          if row[1][0] in labelsencoding_dict[labelscategory].keys():
            i += 1
          else:
            labelsencoding_dict[labelscategory].update({row[1][0] : df_labels2.iloc[i][0]})
            i += 1


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

        #for numerical we'll want the original column unaltered for predictions
        df_labels[labels_column+'_orig'] = df_labels[labels_column].copy()

        #however it may also benefit to parallel train model to predict transformations
        #plus we'll use the std bins for leveling the frequency of labels for oversampling
        df_labels, labelsdummy, labels_column_dict_list = \
        self.process_numerical_class(df_labels, labelsdummy, labels_column)

        del df_labels[labels_column + '_NArw']

        finalcolumns_labels = list(df_labels)

        #for the labelsencoding_dict we'll save the bin labels and asscoiated columns
        labelsencoding_dict = {'nmbr':{}}
        columns_labels = []
        for label in list(df_labels):
          if label[-5:] in ['__<-2', '__-21', '__-10', '__+01', '__+12', '__>+2']:
            labelsencoding_dict['nmbr'].update({label[-4:]:label})





      #it occurs to me there might be an argument for preferring a single numerical \
      #classifier for labels to keep this to a single column, if so scikitlearn's \
      #LabelEcncoder could be used here, will assume that onehot encoding is acceptable
      if labelscategory == 'text':

        df_labels, labelsdummy, _1 = \
        self.process_text_class(df_labels, labelsdummy, labels_column)

        del df_labels[labels_column + '_NArw']

        finalcolumns_labels = list(df_labels)

        i = 0

        for row in df_labels2.iterrows():
          if row[1][0] in labelsencoding_dict[labelscategory].keys():
            i += 1
          else:
            labelsencoding_dict[labelscategory].\
            update({row[1][0] : labels_column+'_'+row[1][0]})
            i += 1

    else:
      df_labels = pd.DataFrame([])
      labelsencoding_dict = {}

    #great the data is processed now let's do a few moore global training preps


    #here's a list of final column names saving here since the translation to \
    #numpy arrays scrubs the column names
    finalcolumns_train = list(df_train)
    finalcolumns_test = list(df_test)


    #convert all of our dataframes to numpy arrays (train, test, labels, and ID)
    #    df_trainID, df_testID
    np_train = df_train.values
    np_test = df_test.values
    np_labels = df_labels.values

    if trainID_column != False:
      np_trainID = df_trainID.values
    if testID_column != False:
      np_testID = df_testID.values


    #set randomness seed number
    answer = randomseed
    #a reasonable extension would be to tie this in with randomness seed for \
    #ML infill methods calls to scikit learn


    if shuffletrain == True:
      #shuffle training set and labels
      np_train = shuffle(np_train, random_state = answer)
      np_labels = shuffle(np_labels, random_state = answer)
  #     df_train = shuffle(df_train, random_state = answer)
  #     df_labels = shuffle(df_labels, random_state = answer)

      if trainID_column != False:
        np_trainID = shuffle(np_trainID, random_state = answer)
  #       df_trainID = shuffle(df_trainID, random_state = answer)






    totalvalidationratio = valpercent1 + valpercent2
    val2ratio = valpercent2 / totalvalidationratio

    if totalvalidationratio > 0.0:

      if labels_column != False:
        #split validation1 sets from training and labels
        np_train, np_validation1, np_labels, np_validationlabels1 = \
        train_test_split(np_train, np_labels, test_size=totalvalidationratio, \
                         shuffle = False)
      else:
        np_train, np_validation1 = \
        train_test_split(np_train, test_size=totalvalidationratio, shuffle = False)
        np_validationlabels1 = np_labels



      if trainID_column != False:
        np_trainID, np_validationID1 = \
        train_test_split(np_trainID, test_size=totalvalidationratio, shuffle = False)
  #       df_trainID, df_validationID1 = \
  #       train_test_split(df_trainID, test_size=valpercent1, shuffle = False)


      else:
        np_trainID = []
        np_validationID1 = []



      if val2ratio > 0.0:


        if labels_column != False:
          #split validation2 sets from training and labels
          np_validation1, np_validation2, np_validationlabels1, np_validationlabels2 = \
          train_test_split(np_validation1, np_validationlabels1, test_size=val2ratio, \
                           random_state = answer)

        else:

          np_validation1, np_validation2 = \
          train_test_split(np_validation1, test_size=val2ratio, \
                           random_state = answer)

          np_validationlabels2 = []

        if trainID_column != False:
          np_validationID1, np_validationID2 = \
          train_test_split(np_trainID, test_size=val2ratio, random_state = answer)
        else:
          np_trainID = []
          np_validationID2 = []

      else:
        np_validation2 = []
        np_validationlabels2 = []
        np_validationID2 = []

    else:
      np_validation1 = []
      np_validationlabels1 = []
      np_validationID1 = [] 


    if testID_column != False:
      np_testID = np_testID
    else:
      np_testID = []


    np_test = np_test






    #here is the process to levelize the frequency of label rows in train data
    #currently only label categories of 'bnry' or 'text' are considered
    #a future extension will include numerical labels by adding supplemental 
    #label columns to designate inclusion in some fractional bucket of the distribution
    #e.g. such as quintiles for instance
    if TrainLabelFreqLevel == True \
    and labels_column != False:


      train_df = pd.DataFrame(np_train, columns = finalcolumns_train)
      labels_df = pd.DataFrame(np_labels, columns = finalcolumns_labels)
      if trainID_column != False:
        trainID_df = pd.DataFrame(np_trainID, columns = [trainID_column])
        #add trainID set to train set for consistent processing
        train_df = pd.concat([train_df, trainID_df], axis=1)                        

      if labelscategory in ['bnry', 'text']:



        #apply LabelFrequencyLevelizer defined function
        train_df, labels_df = \
        self.LabelFrequencyLevelizer(train_df, labels_df, labelsencoding_dict)

      elif labelscategory in ['nmbr', 'bxcx']:

        #apply LabelFrequencyLevelizer defined function
        train_df, labels_df = \
        self.LabelFrequencyLevelizer(train_df, labels_df, labelsencoding_dict)




      #extract trainID
      if trainID_column != False:

        trainID_df = pd.DataFrame(train_df[trainID_column])
        del train_df[trainID_column]

      #convert levelized train sets to Numpy arrays
      np_train = train_df.values
      np_labels = labels_df.values

      if trainID_column != False:
        np_trainID = trainID_df.values

      #shuffle one more time as part of levelized label frequency
      if shuffletrain == True:
        #shuffle training set and labels
        np_train = shuffle(np_train, random_state = answer)
        np_labels = shuffle(np_labels, random_state = answer)

        if trainID_column != False:
          np_trainID = shuffle(np_trainID, random_state = answer)





    postprocess_dict.update({'origtraincolumns' : columns_train, \
                             'finalcolumns_train' : finalcolumns_train, \
                             'testID_column' : testID_column, \
                             'MLinfill' : MLinfill, \
                             'infilliterate' : infilliterate, \
                             'randomseed' : randomseed, \
                             'forcetocategoricalcolumns' : forcetocategoricalcolumns, \
                             'numbercategoryheuristic' : numbercategoryheuristic, \
                             'excludetransformscolumns' : excludetransformscolumns,\
                             'labelsencoding_dict' : labelsencoding_dict, \
                             'automungeversion' : '1.1' })





    #a reasonable extension would be to perform some validation functions on the\
    #sets here (or also prior to transofrm to numpuy arrays) and confirm things \
    #like consistency between format of columns and data between our train and \
    #test sets and if any issues return a coresponding error message to alert user


    return np_train, np_trainID, np_labels, np_validation1, np_validationID1, \
    np_validationlabels1, np_validation2, np_validationID2, np_validationlabels2, \
    np_test, np_testID, labelsencoding_dict, finalcolumns_train, finalcolumns_test,  \
    postprocess_dict



  # #Here is a summary of the postprocess_dict structure from automunge:



  # postprocess_dict.update({'origtraincolumns' : columns_train, \
  #                          'finalcolumns_train' : finalcolumns_train, \
  #                          'testID_column' : testID_column, \
  #                          'MLinfill' : MLinfill, \
  #                          'infilliterate' : infilliterate, \
  #                          'randomseed' : randomseed, \
  #                          'excludetransformscolumns' : excludetransformscolumns,\
  #                          'labelsencoding_dict' : labelsencoding_dict, \
  #                          'automungeversion' : '1.1', 
  #                          'column_dict' : {}})

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




  def postprocess_numerical_class(self, mdf_test, column, mean, std):

    #postprocess_numerical_class(mdf_test, column, mean, std)
    #function to normalize data to mean of 0 and standard deviation of 1 from training distribution
    #takes as arguement pandas dataframe of training and test data (mdf_train), (mdf_test)\
    #and the name of the column string ('column'), and the mean and std from the train set \
    #replaces missing or improperly formatted data with mean of remaining values
    #replaces original specified column in dataframe
    #returns transformed dataframe

    #expect this approach works better when the numerical distribution is thin tailed
    #if only have training but not test data handy, use same training data for both dataframe inputs


    #add a second column with boolean expression indicating a missing cell
    #(using NArows(.) function defined below, column name will be column+'_NArows')
    NArows_nmbr_test = self.NArows(mdf_test, column, 'nmbr')
    mdf_test[column + '_NArw'] = NArows_nmbr_test.copy()
    del NArows_nmbr_test

    #change NArows data type to 8-bit (1 byte) integers for memory savings
    mdf_test[column + '_NArw'] = mdf_test[column + '_NArw'].astype(np.int8)

    #convert all values to either numeric or NaN
    mdf_test[column] = pd.to_numeric(mdf_test[column], errors='coerce')

    #get mean of training data
    mean = mean  

    #replace missing data with training set mean
    mdf_test[column] = mdf_test[column].fillna(mean)

    #subtract mean from column
    mdf_test[column] = mdf_test[column] - mean

    #get standard deviation of training data
    std = std

    #divide column values by std
    mdf_test[column] = mdf_test[column] / std

    #change column name to column + '_nmbr'
    mdf_test[column + '_nmbr'] = mdf_test[column].copy()
    del mdf_test[column]




    #create bins based on standard deviation increments
    binscolumn = column + '_bins'
    nmbrcolumn = column + '_nmbr'

    mdf_test[binscolumn] = \
    pd.cut( mdf_test[nmbrcolumn], bins = [-float('inf'), -2, -1, 0, 1, 2, float('inf')],  \
           labels = ['_<-2','_-21','_-10','_+01','_+12','_>+2'], precision=4)

    textcolumns = \
    [binscolumn + '__<-2', binscolumn + '__-21', binscolumn + '__-10', \
     binscolumn + '__+01', binscolumn + '__+12', binscolumn + '__>+2']

    #process bins as a categorical set
    mdf_test = \
    self.postprocess_text_class(mdf_test, binscolumn, textcolumns) 

  #   #delete the _NArw column from this categorical bins derivation
  #   del mdf_test[binscolumn + '_NArw' ]


    #create list of columns
    nmbrcolumns = [column + '_nmbr', column + '_NArw'] + textcolumns


    return mdf_test




  def postprocess_text_class(self, mdf_test, column, textcolumns):


    #postprocess_text_class(mdf_test, column, textcolumns)
    #process column with text classifications
    #takes as arguement pandas dataframe containing test data  
    #()mdf_test), and the name of the column string ('column'), and an array of
    #the associated transformed column s from the train set (textcolumns)

    #note this aligns formatting of transformed columns to the original train set
    #fromt he original treatment with automunge

    #deletes the original column from master dataframe and
    #replaces with onehot encodings
    #with columns named after column_ + text classifications
    #missing data replaced with category label 'missing'+column
    #any categories missing from the training set removed from test set
    #any category present in training but missing from test set given a column of zeros for consistent formatting
    #ensures order of all new columns consistent between both sets
    #returns two transformed dataframe (mdf_train, mdf_test) \
    #and a list of the new column names (textcolumns)

    #note it is kind of a hack here to create a column for missing values with \
    #two underscores (__) in the column name to ensure appropriate order for cases\
    #where NaN present in test data but not train data, if a category starts with|
    #an underscore such that it preceeds '__missing' alphabetically in this scenario\
    #this might create error due to different order of columns, address of this \
    #potential issue will be a future extension

  #   #add _NArw to textcolumns to ensure a column gets populated even if no missing
  #   textcolumns = [column + '_NArw'] + textcolumns

    #convert column to category
    mdf_test[column] = mdf_test[column].astype('category')

    #if set is categorical we'll need the plug value for missing values included
    mdf_test[column] = mdf_test[column].cat.add_categories(['NArw'])

    #replace NA with a dummy variable
    mdf_test[column] = mdf_test[column].fillna('NArw')

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
  #   cat_train = mdf_train[column]
  #   cat_train_encoded = encoder.fit_transform(cat_train)

    cat_test = mdf_test[column]
    cat_test_encoded = encoder.fit_transform(cat_test)


    #apply onehotencoding
    onehotencoder = OneHotEncoder()
  #   cat_train_1hot = onehotencoder.fit_transform(cat_train_encoded.reshape(-1,1))
    cat_test_1hot = onehotencoder.fit_transform(cat_test_encoded.reshape(-1,1))

    #append column header name to each category listing
    #note the iteration is over a numpy array hence the [...] approach  
  #   labels_train[...] = column + '_' + labels_train[...]
    labels_test[...] = column + '_' + labels_test[...]


    #convert sparse array to pandas dataframe with column labels
  #   df_train_cat = pd.DataFrame(cat_train_1hot.toarray(), columns=labels_train)
    df_test_cat = pd.DataFrame(cat_test_1hot.toarray(), columns=labels_test)

  #   #add a missing column to train if it's not present
  #   if column + '__missing' not in df_train_cat.columns:
  #     missingcolumn = pd.DataFrame(0, index=np.arange(df_train_cat.shape[0]), columns=[column+'__missing'])
  #     df_train_cat = pd.concat([missingcolumn, df_train_cat], axis=1)


    #Get missing columns in test set that are present in training set
    missing_cols = set( textcolumns ) - set( df_test_cat.columns )

    #Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        df_test_cat[c] = 0

    #Ensure the order of column in the test set is in the same order than in train set
    #Note this also removes categories in test set that aren't present in training set
    df_test_cat = df_test_cat[textcolumns]


    #concatinate the sparse set with the rest of our training data
  #   mdf_train = pd.concat([df_train_cat, mdf_train], axis=1)
    mdf_test = pd.concat([df_test_cat, mdf_test], axis=1)


    #delete original column from training data
  #   del mdf_train[column]    
    del mdf_test[column]

  #   #create output of a list of the created column names
  #   labels_train = list(df_train_cat)
  #   textcolumns = labels_train

    #change data types to 8-bit (1 byte) integers for memory savings
    for textcolumn in textcolumns:
      mdf_test[textcolumn] = mdf_test[textcolumn].astype(np.int8)


    return mdf_test





  def postprocess_time_class(self, mdf_test, column, datecolumns, timenormalization_dict):

    #postprocess_time_class(mdf_test, column, datecolumns, timenormalization_dict)
    #postprocess test column with of date category
    #takes as arguement pandas dataframe containing test data 
    #(mdf_test), the name of the column string ('column'), and the timenormalization_dict 
    #from the original application of automunge to the associated date column from train set

    #deletes the original column from master dataframe and
    #replaces with distinct columns for year, month, day, hour, minute, second
    #each normalized to the mean and std from original train set, 
    #with missing values plugged with the mean
    #with columns named after column_ + time category
    #returns two transformed dataframe (mdf_train, mdf_test)

    #add a second column with boolean expression indicating a missing cell
    #(using NArows(.) function defined below, column name will be column+'_NArows')
    NArows_nmbr_test = self.NArows(mdf_test, column, 'nmbr')
    mdf_test[column + '_NArw'] = NArows_nmbr_test.copy()
    del NArows_nmbr_test

    #change NArows data type to 8-bit (1 byte) integers for memory savings
    mdf_test[column + '_NArw'] = mdf_test[column + '_NArw'].astype(np.int8)


    #apply pd.to_datetime to column, note that the errors = 'coerce' needed for messy data
  #   mdf_train[column] = pd.to_datetime(mdf_train[column], errors = 'coerce')
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


    #delete original column from training data
    if column in mdf_test.columns:
      del mdf_test[column]  


  #   #output a dictionary of the associated column mean and std

  #   timenormalization_dict = {'meanyear' : meanyear, 'meanmonth' : meanmonth, \
  #                             'meanday' : meanday, 'meanhour' : meanhour, \
  #                             'meanminute' : meanminute, 'meansecond' : meansecond,\
  #                             'stdyear' : stdyear, 'stdmonth' : stdmonth, \
  #                             'stdday' : stdday, 'stdhour' : stdhour, \
  #                             'stdminute' : stdminute, 'stdsecond' : stdsecond}


    return mdf_test




  def createpostMLinfillsets(self, df_test, column, testNArows, category, \
                             columnslist = [], categorylist = []):


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

    if category in ['nmbr', 'bxcx', 'bnry', 'text']:

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

    if category == 'date':

      #create empty sets for now
      #an extension of this method would be to implement a comparable method \
      #for the time category, based on the columns output from the preprocessing
  #     df_train_filltrain = pd.DataFrame({'foo' : []}) 
  #     df_train_filllabel = pd.DataFrame({'foo' : []})
  #     df_train_fillfeatures = pd.DataFrame({'foo' : []})
      df_test_fillfeatures = pd.DataFrame({'foo' : []})


    return df_test_fillfeatures





  def predictpostinfill(self, category, model, df_test_fillfeatures, \
                        columnslist = []):

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

      if category == 'nmbr':

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



      if category == 'bxcx':


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


      if category == 'bnry':

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


      if category == 'text':

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

      if category == 'date':

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



  def postMLinfillfunction (self, df_test, column, postprocess_dict, \
                            masterNArows_test):


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
                         category, \
                         columnslist = columnslist, \
                         categorylist = categorylist)

      #predict infill values using defined function predictinfill(.)
      df_testinfill = \
      self.predictpostinfill(category, model, df_test_fillfeatures, \
                    columnslist = columnslist)


      #if model != False:
      if postprocess_dict['column_dict'][column]['infillmodel'] != False:

        df_test = self.insertinfill(df_test, column, df_testinfill, category, \
                               pd.DataFrame(masterNArows_test[origcolumn+'_NArows']), \
                               columnslist = columnslist, categorylist = categorylist)

      #now change the infillcomplete marker in the text_dict for each \
      #associated text column
      for columnname in categorylist:
        postprocess_dict['column_dict'][columnname]['infillcomplete'] = True


        #now change the infillcomplete marker in the dict for each associated column
        for columnname in categorylist:
          postprocess_dict['column_dict'][columnname]['infillcomplete'] = True

    return df_test, postprocess_dict






  def postmunge(self, postprocess_dict, df_test, testID_column = False):

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
      if column not in postprocess_dict['excludetransformscolumns']:


        category = self.evalcategory(df_test, column, \
                                postprocess_dict['numbercategoryheuristic'])

        if column in postprocess_dict['forcetocategoricalcolumns']:
          category = 'text'



        #ok postprocess_dict stores column data by the key of column names after\
        #they have gone through our pre-processing functions, which means the act \
        #of processing will have \
        #created new columns and deleted the original column - so since we are \
        #currently walking through the original column names we'll need to \
        #pull a post-process column name for the associated columns to serve as \
        #a key for our postprocess_dict which we'll call columnkey. Also the  \
        #original category from train set (traincategory) will be accessed to \
        #serve as a check for consistency between train and test sets.
        traincategory = False

        for postprocesscolumn in postprocess_dict['finalcolumns_train']:



          #if postprocess_dict['column_dict'][postprocesscolumn]['category'] == 'text':
          if postprocess_dict['column_dict'][postprocesscolumn]['origcategory'] == 'text':
            if column == postprocess_dict['column_dict'][postprocesscolumn]['origcolumn'] \
            and postprocesscolumn[-5:] != '_NArw':
            #and postprocesscolumn in postprocess_dict['column_dict'][postprocesscolumn]['categorylist']:
              traincategory = 'text'
              columnkey = postprocesscolumn
              #break

          #elif postprocess_dict['column_dict'][postprocesscolumn]['category'] == 'date':
          elif postprocess_dict['column_dict'][postprocesscolumn]['origcategory'] == 'date':
            if column == postprocess_dict['column_dict'][postprocesscolumn]['origcolumn'] \
            and postprocesscolumn[-5:] != '_NArw':
              traincategory = 'date'
              columnkey = postprocesscolumn
              #break

          #elif postprocess_dict['column_dict'][postprocesscolumn]['category'] == 'bxcx':
          elif postprocess_dict['column_dict'][postprocesscolumn]['origcategory'] == 'bxcx':
            if postprocess_dict['column_dict'][postprocesscolumn]['category'] == 'bxcx':
              if column == postprocess_dict['column_dict'][postprocesscolumn]['origcolumn'] \
              and postprocesscolumn[-5:] != '_NArw':
                traincategory = 'bxcx'
                columnkey = postprocesscolumn
                #break
            #this is kind of a hack, will have to put some thought into if there is a \
            #better way to generalize this, as long as we maintain the column naming\
            #convention this works
            if postprocess_dict['column_dict'][postprocesscolumn]['category'] == 'nmbr':
              if column == postprocess_dict['column_dict'][postprocesscolumn]['origcolumn'] \
              and postprocesscolumn[-5:] != '_NArw':
                traincategory = 'bxcx'
                columnkey = postprocesscolumn[:-5]+'_bxcx'
                #break


          #elif postprocess_dict['column_dict'][postprocesscolumn]['category'] == 'bnry':
          elif postprocess_dict['column_dict'][postprocesscolumn]['origcategory'] == 'bnry':
            if column == postprocess_dict['column_dict'][postprocesscolumn]['origcolumn'] \
            and postprocesscolumn[-5:] != '_NArw':
              traincategory = 'bnry'
              columnkey = postprocesscolumn
              #break

          #elif postprocess_dict['column_dict'][postprocesscolumn]['category'] == 'nmbr':
          elif postprocess_dict['column_dict'][postprocesscolumn]['origcategory'] == 'nmbr':
            if column == postprocess_dict['column_dict'][postprocesscolumn]['origcolumn'] \
            and postprocesscolumn[-5:] != '_NArw':
              traincategory = 'nmbr'
              columnkey = postprocesscolumn
              #break

          elif traincategory == False:
            traincategory = 'null'
            #break




        #for the special case of train category = bxcx and test category = nmbr
        #(meaning there were no negative values in train but there were in test)
        #we'll resolve by clipping all test values that were <0.1 and setting to 
        #NaN then resetting the test category to bxcx to be consistent with train
        if traincategory == 'bxcx' and category == 'nmbr':

          #convert all values to either numeric or NaN
          df_test[column] = pd.to_numeric(df_test[column], errors='coerce')


          df_test[column] = df_test[column].mask(df_test[column] < 0.1)
          category = 'bxcx'
          print('Note that values < 0.1 found in test set were reset to NaN')
          print('to allow consistent box-cox transform as train set.')

        #another special case, if train category is nmbr and test category is bxcx
        #default test category to nmbr
        if traincategory == 'nmbr' and category == 'bxcx':
          category = 'nmbr'

        #one more special case, if train was a numerical set to categorical based
        #on heuristic, let's force test to as well
        if traincategory == 'text' and category == 'nmbr':
          category = 'text'


        #let's make sure the category is consistent between train and test sets
        if category != traincategory:
          print('error - different category between train and test sets for column ',\
                column)


        #here we'll delete any columns that returned a 'null' category
        if category == 'null':
          df_test = df_test.drop([column], axis=1)

        #so if we didn't delete the column let's proceed
        else:

          #create NArows (column of True/False where True coresponds to missing data)
          testNArows = self.NArows(df_test, column, category)

          #now append that NArows onto a master NA rows df
          masterNArows_test = pd.concat([masterNArows_test, testNArows], axis=1)

          #(now normalize as would normally)

          #for binary class use the train majority field for missing plug value
          if category == 'bnry':
            binary_missing_plug = postprocess_dict['column_dict'][columnkey]['normalization_dict']['missing']

          #apply appropriate processing function to this column based on the result
          #original bnry processing function still works since only had one df input
          if category == 'bnry':
            #df_test, _1, _2 = process_binary_class(df_test, column, binary_missing_plug)
            df_test, _1 = self.process_binary_class(df_test, column, binary_missing_plug)

          #for nmbr category process test set with function postprocess_numerical_class
          if category == 'nmbr':

            df_test = self.postprocess_numerical_class(df_test, column, \
                                                  postprocess_dict['column_dict'][columnkey]['normalization_dict']['mean'], \
                                                  postprocess_dict['column_dict'][columnkey]['normalization_dict']['std'])

          #for bxcx category processing
          if category == 'bxcx':          


            #df_test, nmbrcolumns, nmbrnormalization_dict, _ = \
            df_test, _1 = \
            self.process_bxcx_class(df_test, column, \
                               postprocess_dict['column_dict'][columnkey]['normalization_dict']['bxcxerrorcorrect'], \
                               bxcx_lmbda = \
                               postprocess_dict['column_dict'][columnkey]['normalization_dict']['bxcx_lmbda'], \
                               trnsfrm_mean = postprocess_dict['column_dict'][columnkey]['normalization_dict']['trnsfrm_mean'], \
                               trnsfrm_std = postprocess_dict['column_dict'][columnkey]['normalization_dict']['trnsfrm_std'])


          #for text category process test set with function postprocess_text_class
          if category == 'text':


            df_test = self.postprocess_text_class(df_test, column, postprocess_dict['column_dict'][columnkey]['columnslist'])

          #for date category process test set with function postprocess_date_class
          if category == 'date':
            df_test = self.postprocess_time_class(df_test, column, \
                                             postprocess_dict['column_dict'][columnkey]['columnslist'], \
                                             postprocess_dict['column_dict'][columnkey]['normalization_dict'])


    #now that we've pre-processed all of the columns, let's run through them again\
    #using ML to derive plug values for the previously missing cells

  #   #troubleshoot
  #   print("list(df_test) = ", list(df_test))


    #if MLinfill == True
    if postprocess_dict['MLinfill'] == True:


      #now let's create a list of columns just like we did in automunge
      columns_test_ML = list(df_test)

      iteration = 0
      #while iteration < infilliterate:
      while iteration < postprocess_dict['infilliterate']:


        #since we're reusing the text_dict and date_dict from our original automunge
        #we're going to need to re-initialize the infillcomplete markers
        #actually come to this of it we need to go back to automunge and do this
        #for the MLinfill iterations as well

        #re-initialize the infillcomplete marker in column _dict's
        for key in postprocess_dict['column_dict']:
          postprocess_dict['column_dict'][key]['infillcomplete'] = False



        for column in columns_test_ML:



          #we're only going to process columns that weren't in our excluded set
          #if column not in excludetransformscolumns:
          if column not in postprocess_dict['excludetransformscolumns'] \
          and column[-5:] != '_NArw':

            df_test, postprocess_dict = \
            self.postMLinfillfunction (df_test, column, postprocess_dict, \
                                  masterNArows_test)


        iteration += 1                          



    #here's a list of final column names saving here since the translation to \
    #numpy arrays scrubs the column names
    finalcolumns_test = list(df_test)

    #global processing to test set including conversion to numpy array
    np_test = df_test.values

    if testID_column != False:
      np_testID = df_testID.values
      testID = np_testID
    else:
      testID = []

    test = np_test

    labelsencoding_dict = postprocess_dict['labelsencoding_dict']


    return test, testID, labelsencoding_dict, finalcolumns_test
