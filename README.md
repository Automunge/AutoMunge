# Automunge

![image](https://user-images.githubusercontent.com/44011748/76485571-4855b280-63f3-11ea-9574-c39a66e45e4e.png)

#  

## Table of Contents
* [Introduction](https://github.com/Automunge/AutoMunge#introduction)
* [Install, Initialize, and Basics](https://github.com/Automunge/AutoMunge#install-initialize-and-basics)
 ___ 
* [automunge(.)](https://github.com/Automunge/AutoMunge#automunge-1)
* [automunge(.) returned sets](https://github.com/Automunge/AutoMunge#automunge-returned-sets)
* [automunge(.) passed parameters](https://github.com/Automunge/AutoMunge#automunge-passed-parameters)
 ___ 
* [postmunge(.)](https://github.com/Automunge/AutoMunge#postmunge)
* [postmunge(.) returned sets](https://github.com/Automunge/AutoMunge#postmunge-returned-sets)
* [postmunge(.) passed parameters](https://github.com/Automunge/AutoMunge#postmunge-passed-parameters)
 ___ 
* [Default Transformations](https://github.com/Automunge/AutoMunge#default-transformations)
* [Library of Transformations](https://github.com/Automunge/AutoMunge#library-of-transformations)
* [Custom Transformation Functions](https://github.com/Automunge/AutoMunge#custom-transformation-functions)
 ___ 
* [Conclusion](https://github.com/Automunge/AutoMunge#conclusion)
 ___ 

## Introduction
[Automunge](https://automunge.com) is a python library platform for preparing 
tabular data for machine learning. A user has options between automated inference 
of column properties for application of appropriate simple feature engineering
methods, or may also assign to distinct columns custom feature engineering 
transformations, custom sets (e.g. "family trees") of feature engineering 
transformations, and select from options for missing data infill. The feature 
engineering transformations may be accessed from the internal library (aka a 
"feature transformation store"), or may also be externally user defined with minimal 
requirements of simple data structures. The tool includes options for automated feature 
importance evaluation, automated "ML infill" for derivation of missing data inputation 
using machine learning models trained on the set, automated 
preparation for oversampling for labels class imbalance, automated dimensionality 
reductions such as based on feature importance, principal component analysis (PCA), 
or binary encoding, automated evaluation of data property drift between training 
data and subsequent data, and perhaps most importantly the simplest means for 
consistent processing of additional data with a single function call. 

> In other words, put simply:<br/>
>  - **automunge(.)** prepares tabular data for machine learning.<br/>
>  - **postmunge(.)** consistently prepares additional data very efficiently.<br/>
>  
> We make machine learning easy.

The automunge(.) function takes as input tabular training data intended to
train a machine learning model with any corresponding labels if available 
included in the set, and also if available consistently formatted test data 
that can then be used to generate predictions from that trained model. When 
fed pandas dataframes or numpy arrays for these sets the function returns a 
series of transformed pandas dataframes (or numpy arrays per selection) which 
are numerically encoded and suitable for the direct application of machine 
learning algorithms. A user has an option between default feature engineering 
based on inferred properties of the data with feature transformations such as 
z-score normalization, binary encoding for categorical sets, hashing for
high cardinality cateogric sets, time series 
agregation to sin and cos transforms (with bins for business hours, weekdays, 
and holidays), and more (full documentation below); assigning distinct column 
feature engineering methods using a built-in library of feature engineering 
transformations; or alternatively the passing of user-defined custom 
transformation functions incorporating simple data structures such as to allow 
custom methods to each column while still making use of all of the built-in 
features of the tool (such as ML infill, feature importance, dimensionality 
reduction, and most importantly the simplest way for the consistent preparation 
of subsequently available data using just a single function call of the 
postmunge(.) function). Missing data points in the sets are also available to be 
addressed by either the default automated "ML infill" method which predicts 
infill using machine learning models trained on the rest of the set in a fully 
generalized and automated fashion or alternatively by assigning methods from an 
imputation libary. automunge(.) returns a populated python dictionary which can 
be used as input along with a subsequent data set to the postmunge(.) function 
for consistent preparations of additional data such as for inference. Be sure to
check out our [Tutorial Notebooks](https://github.com/Automunge/AutoMunge/tree/master/Tutorials).

## Install, Initialize, and Basics

Automunge is now available for pip install:

```
pip install Automunge
```
Or to upgrade (we currently roll out upgrades pretty frequently):
```
pip install Automunge --upgrade
```

Once installed, run this in a local session to initialize:

```
from Automunge import *
am = AutoMunge()
```

Where e.g. for train set processing  with default parameters run:

```
train, train_ID, labels, \
val, val_ID, val_labels, \
test, test_ID, test_labels, \
postprocess_dict \
= am.automunge(df_train)
```
Or for subsequent consistent processing of train or test data, using the
dictionary returned from original application of automunge(.), run:

```
test, test_ID, test_labels, \
postreports_dict \
= am.postmunge(postprocess_dict, df_test)
```
Importantly, if the df_train set passed to automunge(.) includes a column
intended for use as labels, it should be designated with the labels_column
parameter.

I find it helpful to pass these functions with the full range of arguments
included for reference, thus a user may simply copy and past this form.

```
#for automunge(.) function on original train and test data

train, train_ID, labels, \
val, val_ID, val_labels, \
test, test_ID, test_labels, \
postprocess_dict = \
am.automunge(df_train, df_test = False, \
             labels_column = False, trainID_column = False, testID_column = False, \
             valpercent=0.0, floatprecision = 32, shuffletrain = True, \
             dupl_rows = False, TrainLabelFreqLevel = False, powertransform = False, binstransform = False, \
             MLinfill = True, infilliterate=1, randomseed = False, eval_ratio = .5, \
             numbercategoryheuristic = 255, pandasoutput = True, NArw_marker = True, \
             featureselection = False, featurethreshold = 0., inplace = False, \
             Binary = False, PCAn_components = False, PCAexcl = [], excl_suffix = False, \
             ML_cmnd = {'autoML_type':'randomforest', \
                        'MLinfill_cmnd':{'RandomForestClassifier':{}, 'RandomForestRegressor':{}}, \
                        'PCA_type':'default', \
                        'PCA_cmnd':{}}, \
             assigncat = {'nmbr':[], 'retn':[], 'mnmx':[], 'mean':[], 'MAD3':[], 'lgnm':[], \
                          'bins':[], 'bsor':[], 'pwrs':[], 'pwr2':[], 'por2':[], 'bxcx':[], \
                          'addd':[], 'sbtr':[], 'mltp':[], 'divd':[], \
                          'log0':[], 'log1':[], 'logn':[], 'sqrt':[], 'rais':[], 'absl':[], \
                          'bnwd':[], 'bnwK':[], 'bnwM':[], 'bnwo':[], 'bnKo':[], 'bnMo':[], \
                          'bnep':[], 'bne7':[], 'bne9':[], 'bneo':[], 'bn7o':[], 'bn9o':[], \
                          'bkt1':[], 'bkt2':[], 'bkt3':[], 'bkt4':[], \
                          'nbr2':[], 'nbr3':[], 'MADn':[], 'MAD2':[], 'tlbn':[], \
                          'mnm2':[], 'mnm3':[], 'mnm4':[], 'mnm5':[], 'mnm6':[], \
                          'ntgr':[], 'ntg2':[], 'ntg3':[], 'mea2':[], 'mea3':[], 'bxc2':[], \
                          'dxdt':[], 'd2dt':[], 'd3dt':[], 'dxd2':[], 'd2d2':[], 'd3d2':[], \
                          'nmdx':[], 'nmd2':[], 'nmd3':[], 'mmdx':[], 'mmd2':[], 'mmd3':[], \
                          'shft':[], 'shf2':[], 'shf3':[], 'shf4':[], 'shf7':[], 'shf8':[], \
                          'bnry':[], 'onht':[], 'text':[], 'txt2':[], '1010':[], 'smth':[], \
                          'ordl':[], 'ord3':[], 'hash':[], 'hsh2':[], 'hs10':[], \
                          'Unht':[], 'Utxt':[], 'Utx2':[], 'Uor3':[], 'Uor6':[], 'U101':[], \
                          'splt':[], 'spl2':[], 'spl5':[], 'sp15':[], 'sp19':[], 'sbst':[], \
                          'spl8':[], 'spl9':[], 'sp10':[], 'sp16':[], 'sp20':[], 'sbs2':[], \
                          'srch':[], 'src2':[], 'src4':[], 'strn':[], 'lngt':[], 'aggt':[], \
                          'nmrc':[], 'nmr2':[], 'nmcm':[], 'nmc2':[], 'nmEU':[], 'nmE2':[], \
                          'nmr7':[], 'nmr8':[], 'nmc7':[], 'nmc8':[], 'nmE7':[], 'nmE8':[], \
                          'ors2':[], 'ors5':[], 'ors6':[], 'ors7':[], 'ucct':[], 'Ucct':[], \
                          'or15':[], 'or17':[], 'or19':[], 'or20':[], 'or21':[], 'or22':[], \
                          'date':[], 'dat2':[], 'dat6':[], 'wkdy':[], 'bshr':[], 'hldy':[], \
                          'wkds':[], 'wkdo':[], 'mnts':[], 'mnto':[], \
                          'yea2':[], 'mnt2':[], 'mnt6':[], 'day2':[], 'day5':[], \
                          'hrs2':[], 'hrs4':[], 'min2':[], 'min4':[], 'scn2':[], 'DPrt':[], \
                          'DPnb':[], 'DPmm':[], 'DPbn':[], 'DPod':[], 'DP10':[], 'DPoh':[], \
                          'qbt1':[], 'qbt2':[], 'qbt3':[], 'qbt4':[], 'nmqb':[], 'mmqb':[], \
                          'excl':[], 'exc2':[], 'exc3':[], 'exc4':[], 'exc5':[], \
                          'null':[], 'copy':[], 'shfl':[], 'eval':[], 'ptfm':[]}, \
             assignparam = {'default_assignparam' : {'(category)' : {'(parameter)' : 42}}, \
                                     '(category)' : {'(column)'   : {'(parameter)' : 42}}}, \
             assigninfill = {'stdrdinfill':[], 'MLinfill':[], 'zeroinfill':[], 'oneinfill':[], \
                             'adjinfill':[], 'meaninfill':[], 'medianinfill':[], \
                             'modeinfill':[], 'lcinfill':[], 'naninfill':[]}, \
             assignnan = {'categories':{}, 'columns':{}, 'global':[]}, \
             transformdict = {}, processdict = {}, evalcat = False, \
             privacy_encode = False, printstatus = True)
```

Please remember to save the automunge(.) returned object postprocess_dict 
such as using pickle library, which can then be later passed to the postmunge(.) 
function to consistently prepare subsequently available data.

```
#Sample pickle code:

#sample code to download postprocess_dict dictionary returned from automunge(.)
import pickle
with open('filename.pickle', 'wb') as handle:
  pickle.dump(postprocess_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

#to upload for later use in postmunge(.) in another notebook
import pickle
with open('filename.pickle', 'rb') as handle:
  postprocess_dict = pickle.load(handle)

```
We can then apply the postprocess_dict saved from a prior application of automunge
for consistent processing of additional data.
```
#for postmunge(.) function on additional available train or test data
#using the postprocess_dict object returned from original automunge(.) application

test, test_ID, test_labels, \
postreports_dict = \
am.postmunge(postprocess_dict, df_test, \
             testID_column = False, labelscolumn = False, \
             pandasoutput = True, printstatus = True, \
             dupl_rows = False, TrainLabelFreqLevel = False, \
	     featureeval = False, traindata = False, \
             driftreport = False, inversion = False, \
             returnedsets = True, shuffletrain = False)
```

The functions depend on pandas dataframe formatted train and test data
or numpy arrays with consistent order of columns between train and test data. 
(For numpy arrays any label column should be positioned as final column in set.)
The functions return pandas dataframes (or numpy arrays per selection) numerically encoded 
and normalized such as to make them suitable for direct application to a 
machine learning model in the framework of a user's choice, including sets for 
the various activities of a generic machine learning project such as training, 
hyperparameter tuning validation (val), or data intended for use in generation 
of predictions from the trained model (test set). The functions also return a 
few other sets such as labels, column headers, ID sets, and etc if elected - a 
full list of returned dataframes is below.

When left to automation, the function works by inferring a category of 
data based on properties of each column to select the type of processing 
function to apply, for example whether a column is a numerical, categorical,
binary, or time-series set. Alternately, a user can pass column header IDs to 
assign specific processing functions to distinct columns - which processing functions
may be pulled from the internal library of transformations or alternately user
defined. Normalization parameters from the initial automunge application are
saved to a returned dictionary for subsequent consistent processing of test data
that wasn't available at initial address with the postmunge(.) function. 

The feature engineering transformations are recorded with a series of suffixes 
appended to the column header title in the returned sets, for one example the 
application of z-score normalization returns a column with header origname + '\_nmbr'. 
As another example, for binary encoded sets the set of columns are returned with
header origname + '\_1010_#' where # is integer to distinguish columns in same set.
Each transformation category has a unique suffix appender.

In automation, for numerical data, the functions generate a series of derived
transformations which may result in one or more child columns. For numerical data, if the
powertransform option is selected distribution properties are evaluated for 
potential application of z-score normalization, min-max scaling, power law transform 
via box-cox method, or mean absolute deviation scaling. Otherwise numerical data 
defaults to z-score, with the binstransform parameter for supplmenting normalized data 
with standard deviation bins for values in range <-2, -2-1, -10, 01, 12, >2 from the
mean. Further transformation options are detailed below. For time-series
data the model segregates the data by time-scale (year, month, day, hour, minute, 
second) and returns year z-score normalized, a pair of sets for combined month/day 
and combined hour / minute / second with sin and cos transformations at period of 
time-scale, and also returns binned sets identifying business hours, weekdays, and 
US holidays. For binary categorical data the functions return a single column with 
1/0 designation. For multimodal categorical data the functions return binary
encoded sets where categoric entries may be distinguished by zero, one, or more
simultaneous column activations. Alternatives for one-hot encoding, ordinal encoding, 
and etc are also available. String parsing methods are available to extract and encode 
grammatic structure shared between cateogric entries. For all cases the functions may 
generate a supplemental column (NArw) with a boolean integer identifier for cells that were 
subject to infill due to missing or improperly formatted data when the default NArw_marker 
parameter is left activated.

The functions also defaults to a method we call 'ML infill' which if elected
predicts infill for missing values in both the train and test sets using
machine learning models trained on the rest of the set in a fully
generalized and automated fashion. The ML infill works by initially
applying infill using traditional methods such as mean for a numerical
set, most common value for a binary set, and a boolean identifier for
categorical. The functions then generate a column specific set of
training data, labels, and feature sets for the derivation of infill.
The column's trained model is included in the outputted dictionary for
application of the same model in the postmunge function. Alternately, a
user can pass column headers to assign different infill methods to distinct 
columns. The method currently makes use of Scikit Random Forest models by 
default. A user may defer to default hyperparameters or alternatively pass
hyperparameters via the "ML_cmnd" object, and may also make use of grid 
or randomized CV hyperparameter tuning by passing the hyperparameters as
lists, ranges, or distributions of candidate parameters instead of distinct 
values.

The automunge(.) function also includes a method for feature importance 
evaluation, in which metrics are derived to measure the impact to predictive 
accuracy of original source columns as well as relative importance of 
derived columns using a permutation importance method. Permutation importance 
method was inspired by a fast.ai lecture and more information can be found in 
the paper "Beware Default Random Forest Importances" by Terrence Parr, Kerem 
Turgutlu, Christopher Csiszar, and Jeremy Howard. This method currently makes 
use of Scikit-Learn's Random Forest predictors. I believe the metric we refer to
as metric2 which evaluates relative importance between features derived from the 
same source column is a unique approach.

The function also includes a method we call 'TrainLabelFreqLevel' which
if elected applies multiples of the feature sets associated with each
label category in the returned training data so as to enable
oversampling of those labels which may be underrepresented in the
training data. This method is available for categorical labels or also
for numerical labels when the label processing includes binned aggregations
such as standard deviation bins or powers of ten bins. This method is 
expected to improve downstream model accuracy for training data with uneven 
distribution of labels. For more on the class imbalance problem see "A 
systematic study of the class imbalance problem in convolutional neural 
networks" - Buda, Maki, Mazurowski.

The function also can perform dimensionality reduction of the sets via 
principal component analysis (PCA), where the user can pass a desired number 
of features and their preference of type and parameters between linear PCA, 
Sparse PCA, or Kernel PCA - all currently implemented in Scikit-Learn. (We
recommend cuation in applying PCA with fat tailed distributions.)

The function also can perform dimensionality reduction of the sets via
the Binary option which takes the set of columns with boolean {1/0} encodings
and collectively applies a binary transform to consolidate these columns to
a set where distinct activation sets may be represented by a distinct activation set
in a reduced number of columns with zero, one, or more simultaneous activations.

## automunge(.)

The application of the automunge and postmunge functions requires the
assignment of the function to a series of named sets. We suggest using
consistent naming convention as follows:

```
#first you'll need to initialize
from Automunge import *
am = AutoMunge()

#then to run with default parameters
train, train_ID, labels, \
val, val_ID, val_labels, \
test, test_ID, test_labels, \
postprocess_dict \
= am.automunge(df_train)
```

The full set of parameters available to be passed are given here, with
explanations provided below: 

```
#first you'll need to initialize
from Automunge import *
am = AutoMunge()

#then if you want you can copy paste following to view all of parameter options
#where df_train is the target training data set to be prepared

train, train_ID, labels, \
val, val_ID, val_labels, \
test, test_ID, test_labels, \
postprocess_dict = \
am.automunge(df_train, df_test = False, \
             labels_column = False, trainID_column = False, testID_column = False, \
             valpercent=0.0, floatprecision = 32, shuffletrain = True, \
             dupl_rows = False, TrainLabelFreqLevel = False, powertransform = False, binstransform = False, \
             MLinfill = True, infilliterate=1, randomseed = False, eval_ratio = .5, \
             numbercategoryheuristic = 255, pandasoutput = True, NArw_marker = True, \
             featureselection = False, featurethreshold = 0., inplace = False, \
             Binary = False, PCAn_components = False, PCAexcl = [], excl_suffix = False, \
             ML_cmnd = {'autoML_type':'randomforest', \
                        'MLinfill_cmnd':{'RandomForestClassifier':{}, 'RandomForestRegressor':{}}, \
                        'PCA_type':'default', \
                        'PCA_cmnd':{}}, \
             assigncat = {'nmbr':[], 'retn':[], 'mnmx':[], 'mean':[], 'MAD3':[], 'lgnm':[], \
                          'bins':[], 'bsor':[], 'pwrs':[], 'pwr2':[], 'por2':[], 'bxcx':[], \
                          'addd':[], 'sbtr':[], 'mltp':[], 'divd':[], \
                          'log0':[], 'log1':[], 'logn':[], 'sqrt':[], 'rais':[], 'absl':[], \
                          'bnwd':[], 'bnwK':[], 'bnwM':[], 'bnwo':[], 'bnKo':[], 'bnMo':[], \
                          'bnep':[], 'bne7':[], 'bne9':[], 'bneo':[], 'bn7o':[], 'bn9o':[], \
                          'bkt1':[], 'bkt2':[], 'bkt3':[], 'bkt4':[], \
                          'nbr2':[], 'nbr3':[], 'MADn':[], 'MAD2':[], 'tlbn':[], \
                          'mnm2':[], 'mnm3':[], 'mnm4':[], 'mnm5':[], 'mnm6':[], \
                          'ntgr':[], 'ntg2':[], 'ntg3':[], 'mea2':[], 'mea3':[], 'bxc2':[], \
                          'dxdt':[], 'd2dt':[], 'd3dt':[], 'dxd2':[], 'd2d2':[], 'd3d2':[], \
                          'nmdx':[], 'nmd2':[], 'nmd3':[], 'mmdx':[], 'mmd2':[], 'mmd3':[], \
                          'shft':[], 'shf2':[], 'shf3':[], 'shf4':[], 'shf7':[], 'shf8':[], \
                          'bnry':[], 'onht':[], 'text':[], 'txt2':[], '1010':[], 'smth':[], \
                          'ordl':[], 'ord3':[], 'hash':[], 'hsh2':[], 'hs10':[], \
                          'Unht':[], 'Utxt':[], 'Utx2':[], 'Uor3':[], 'Uor6':[], 'U101':[], \
                          'splt':[], 'spl2':[], 'spl5':[], 'sp15':[], 'sp19':[], 'sbst':[], \
                          'spl8':[], 'spl9':[], 'sp10':[], 'sp16':[], 'sp20':[], 'sbs2':[], \
                          'srch':[], 'src2':[], 'src4':[], 'strn':[], 'lngt':[], 'aggt':[], \
                          'nmrc':[], 'nmr2':[], 'nmcm':[], 'nmc2':[], 'nmEU':[], 'nmE2':[], \
                          'nmr7':[], 'nmr8':[], 'nmc7':[], 'nmc8':[], 'nmE7':[], 'nmE8':[], \
                          'ors2':[], 'ors5':[], 'ors6':[], 'ors7':[], 'ucct':[], 'Ucct':[], \
                          'or15':[], 'or17':[], 'or19':[], 'or20':[], 'or21':[], 'or22':[], \
                          'date':[], 'dat2':[], 'dat6':[], 'wkdy':[], 'bshr':[], 'hldy':[], \
                          'wkds':[], 'wkdo':[], 'mnts':[], 'mnto':[], \
                          'yea2':[], 'mnt2':[], 'mnt6':[], 'day2':[], 'day5':[], \
                          'hrs2':[], 'hrs4':[], 'min2':[], 'min4':[], 'scn2':[], 'DPrt':[], \
                          'DPnb':[], 'DPmm':[], 'DPbn':[], 'DPod':[], 'DP10':[], 'DPoh':[], \
                          'qbt1':[], 'qbt2':[], 'qbt3':[], 'qbt4':[], 'nmqb':[], 'mmqb':[], \
                          'excl':[], 'exc2':[], 'exc3':[], 'exc4':[], 'exc5':[], \
                          'null':[], 'copy':[], 'shfl':[], 'eval':[], 'ptfm':[]}, \
             assignparam = {'default_assignparam' : {'(category)' : {'(parameter)' : 42}}, \
                                     '(category)' : {'(column)'   : {'(parameter)' : 42}}}, \
             assigninfill = {'stdrdinfill':[], 'MLinfill':[], 'zeroinfill':[], 'oneinfill':[], \
                             'adjinfill':[], 'meaninfill':[], 'medianinfill':[], \
                             'modeinfill':[], 'lcinfill':[], 'naninfill':[]}, \
             assignnan = {'categories':{}, 'columns':{}, 'global':[]}, \
             transformdict = {}, processdict = {}, evalcat = False, \
             privacy_encode = False, printstatus = True)
```

Or for the postmunge function:

```
#for postmunge(.) function on additional or subsequently available test (or train) data
#using the postprocess_dict object returned from original automunge(.) application

#first you'll need to initialize
from Automunge import *
am = AutoMunge()

#then to run with default parameters
test, test_ID, test_labels, \
postreports_dict = \
am.postmunge(postprocess_dict, df_test)
```

With the full set of arguments available to be passed as:

```
#first you'll need to initialize
from Automunge import *
am = AutoMunge()

#then if you want you can copy paste following to view all of parameter options
#here postprocess_dict was returned from corresponding automunge(.) call
#and df_test is the target data set to be prepared

test, test_ID, test_labels, \
postreports_dict = \
am.postmunge(postprocess_dict, df_test, \
             testID_column = False, labelscolumn = False, \
             pandasoutput = True, printstatus = True, inplace = False, \
             dupl_rows = False, TrainLabelFreqLevel = False, \
	     featureeval = False, traindata = False, \
             driftreport = False, inversion = False, \
             returnedsets = True, shuffletrain = False)
```

Note that the only required argument to the automunge function is the
train set dataframe, the other arguments all have default values if
nothing is passed. The postmunge function requires as minimum the
postprocess_dict object (a python dictionary returned from the application of
automunge) and a dataframe test set consistently formatted as those sets
that were originally applied to automunge. 

Note that there is a potential source of error if the returned column header 
title strings, which will include suffix appenders based on transformations applied, 
match any of the original column header titles passed to automunge. This is an edge 
case not expected to occur in common practice and will return error message at 
conclusion of printouts.


...

Here now are descriptions for the returned sets from automunge, which
will be followed by descriptions of the parameters which can be passed to
the function, followed by similar treatment for postmunge returned sets
and arguments. Further below is documentation for the library of transformations.

...

## automunge returned sets:

Automunge defaults to returning data sets as pandas dataframes, or for
single column sets as pandas series.

* train: a numerically encoded set of data intended to be used to train a
downstream machine learning model in the framework of a user's choice

* train_ID: the set of ID values corresponding to the train set if a ID
column(s) was passed to the function. This set may be useful if the shuffle
option was applied. Note that an ID column may serve multiple purposes such
as row identifiers or for pairing tabular data rows with a corresponding
image file for instance. Also included in this set is a derived column
titled 'Automunge_index', this column serves as an index identifier for order
of rows as they were received in passed data, such as may be beneficial
when data is shuffled.

* labels: a set of numerically encoded labels corresponding to the
train set if a label column was passed. Note that the function
assumes the label column is originally included in the train set. Note
that if the labels set is a single column a returned numpy array is 
flattened (e.g. [[1,2,3]] converted to [1,2,3] )

* val: a set of training data carved out from the train set
that is intended for use in hyperparameter tuning of a downstream model.

* val_ID: the set of ID values corresponding to the val
set. Comparable to columns returned in train_ID.

* val_labels: the set of labels corresponding to the val
set

* test: the set of features, consistently encoded and normalized as the
training data, that can be used to generate predictions from a
downstream model trained with train. Note that if no test data is
available during initial address this processing will take place in the
postmunge(.) function. 

* test_ID: the set of ID values corresponding to the test set. Comparable 
to columns returned in train_ID.

* test_labels: a set of numerically encoded labels corresponding to the
test set if a label column was passed.

* postprocess_dict: a returned python dictionary that includes
normalization parameters and trained ML infill models used to
generate consistent processing of additional train or test data such as 
may not have been available at initial application of automunge. It is 
recommended that this dictionary be externally saved on each application 
used to train a downstream model so that it may be passed to postmunge(.) 
to consistently process subsequently available test data, such as 
demonstrated with the pickle library above.

A few useful entries in the postprocess_dict include:
- postprocess_dict['finalcolumns_train']: list of returned column headers for train set including suffix appenders
- postprocess_dict['columntype_report']: a report classifying the returned column types
- postprocess_dict['column_map']:  a report mapping the input columns to their associated returned columns (excluding those consolidated as part of a dimensionality reduction)
- postprocess_dict['FS_sorted]: sorted results of feature importance evaluation if elected

...

## automunge(.) passed parameters

```
train, train_ID, labels, \
val, val_ID, val_labels, \
test, test_ID, test_labels, \
postprocess_dict = \
am.automunge(df_train, df_test = False, \
             labels_column = False, trainID_column = False, testID_column = False, \
             valpercent=0.0, floatprecision = 32, shuffletrain = True, \
             dupl_rows = False, TrainLabelFreqLevel = False, powertransform = False, binstransform = False, \
             MLinfill = True, infilliterate=1, randomseed = False, eval_ratio = .5, \
             numbercategoryheuristic = 255, pandasoutput = True, NArw_marker = True, \
             featureselection = False, featurethreshold = 0., inplace = False, \
             Binary = False, PCAn_components = False, PCAexcl = [], excl_suffix = False, \
             ML_cmnd = {'autoML_type':'randomforest', \
                        'MLinfill_cmnd':{'RandomForestClassifier':{}, 'RandomForestRegressor':{}}, \
                        'PCA_type':'default', \
                        'PCA_cmnd':{}}, \
             assigncat = {'nmbr':[], 'retn':[], 'mnmx':[], 'mean':[], 'MAD3':[], 'lgnm':[], \
                          'bins':[], 'bsor':[], 'pwrs':[], 'pwr2':[], 'por2':[], 'bxcx':[], \
                          'addd':[], 'sbtr':[], 'mltp':[], 'divd':[], \
                          'log0':[], 'log1':[], 'logn':[], 'sqrt':[], 'rais':[], 'absl':[], \
                          'bnwd':[], 'bnwK':[], 'bnwM':[], 'bnwo':[], 'bnKo':[], 'bnMo':[], \
                          'bnep':[], 'bne7':[], 'bne9':[], 'bneo':[], 'bn7o':[], 'bn9o':[], \
                          'bkt1':[], 'bkt2':[], 'bkt3':[], 'bkt4':[], \
                          'nbr2':[], 'nbr3':[], 'MADn':[], 'MAD2':[], 'tlbn':[], \
                          'mnm2':[], 'mnm3':[], 'mnm4':[], 'mnm5':[], 'mnm6':[], \
                          'ntgr':[], 'ntg2':[], 'ntg3':[], 'mea2':[], 'mea3':[], 'bxc2':[], \
                          'dxdt':[], 'd2dt':[], 'd3dt':[], 'dxd2':[], 'd2d2':[], 'd3d2':[], \
                          'nmdx':[], 'nmd2':[], 'nmd3':[], 'mmdx':[], 'mmd2':[], 'mmd3':[], \
                          'shft':[], 'shf2':[], 'shf3':[], 'shf4':[], 'shf7':[], 'shf8':[], \
                          'bnry':[], 'onht':[], 'text':[], 'txt2':[], '1010':[], 'smth':[], \
                          'ordl':[], 'ord3':[], 'hash':[], 'hsh2':[], 'hs10':[], \
                          'Unht':[], 'Utxt':[], 'Utx2':[], 'Uor3':[], 'Uor6':[], 'U101':[], \
                          'splt':[], 'spl2':[], 'spl5':[], 'sp15':[], 'sp19':[], 'sbst':[], \
                          'spl8':[], 'spl9':[], 'sp10':[], 'sp16':[], 'sp20':[], 'sbs2':[], \
                          'srch':[], 'src2':[], 'src4':[], 'strn':[], 'lngt':[], 'aggt':[], \
                          'nmrc':[], 'nmr2':[], 'nmcm':[], 'nmc2':[], 'nmEU':[], 'nmE2':[], \
                          'nmr7':[], 'nmr8':[], 'nmc7':[], 'nmc8':[], 'nmE7':[], 'nmE8':[], \
                          'ors2':[], 'ors5':[], 'ors6':[], 'ors7':[], 'ucct':[], 'Ucct':[], \
                          'or15':[], 'or17':[], 'or19':[], 'or20':[], 'or21':[], 'or22':[], \
                          'date':[], 'dat2':[], 'dat6':[], 'wkdy':[], 'bshr':[], 'hldy':[], \
                          'wkds':[], 'wkdo':[], 'mnts':[], 'mnto':[], \
                          'yea2':[], 'mnt2':[], 'mnt6':[], 'day2':[], 'day5':[], \
                          'hrs2':[], 'hrs4':[], 'min2':[], 'min4':[], 'scn2':[], 'DPrt':[], \
                          'DPnb':[], 'DPmm':[], 'DPbn':[], 'DPod':[], 'DP10':[], 'DPoh':[], \
                          'qbt1':[], 'qbt2':[], 'qbt3':[], 'qbt4':[], 'nmqb':[], 'mmqb':[], \
                          'excl':[], 'exc2':[], 'exc3':[], 'exc4':[], 'exc5':[], \
                          'null':[], 'copy':[], 'shfl':[], 'eval':[], 'ptfm':[]}, \
             assignparam = {'default_assignparam' : {'(category)' : {'(parameter)' : 42}}, \
                                     '(category)' : {'(column)'   : {'(parameter)' : 42}}}, \
             assigninfill = {'stdrdinfill':[], 'MLinfill':[], 'zeroinfill':[], 'oneinfill':[], \
                             'adjinfill':[], 'meaninfill':[], 'medianinfill':[], \
                             'modeinfill':[], 'lcinfill':[], 'naninfill':[]}, \
             assignnan = {'categories':{}, 'columns':{}, 'global':[]}, \
             transformdict = {}, processdict = {}, evalcat = False, \
             privacy_encode = False, printstatus = True)
```

* df_train: a pandas dataframe or numpy array containing a structured 
dataset intended for use to subsequently train a machine learning model. 
The set at a minimum should be 'tidy' meaning a single column per feature 
and a single row per observation. If desired the set may include one are more
"ID" columns (intended to be carved out and consistently shuffled or partitioned
such as an index column) and zero or one column intended to be used as labels 
for a downstream training operation. The tool supports the inclusion of 
non-index-range column as index or multicolumn index (requires named index 
columns). Such index types are added to the returned "ID" sets which are 
consistently shuffled and partitioned as the train and test sets. For passed
numpy array any label column should be the final column.

* df_test: a pandas dataframe or numpy array containing a structured 
dataset intended for use to generate predictions from a downstream machine 
learning model trained from the automunge returned sets. The set must be 
consistently formatted as the train set with consistent column labels and/or
order of columns. (This set may optionally contain a labels column if one 
was included in the train set although it's inclusion is not required). If 
desired the set may include one or more ID column(s) or column(s) intended 
for use as labels. A user may pass False if this set is not available. The tool 
supports the inclusion of non-index-range column as index or multicolumn index 
(requires named index columns). Such index types are added to the returned 
"ID" sets which are consistently shuffled and partitioned as the train and 
test sets.

* labels_column: a string of the column title for the column from the
df_train set intended for use as labels in training a downstream machine
learning model. The function defaults to False for cases where the
train set does not include a label column. An integer column index may 
also be passed such as if the source dataset was a numpy array. A user can 
also pass True in which case the label set will be taken from the final
column of the train set (including cases of single column in train set).

* trainID_column: a string of the column title for the column from the
df_train set intended for use as a row identifier value (such as could
be sequential numbers for instance). The function defaults to False for
cases where the training set does not include an ID column. A user can 
also pass a list of string columns titles such as to carve out multiple
columns to be excluded from processing but consistently shuffled and 
partitioned. An integer column index or list of integer column indexes 
may also be passed such as if the source dataset was a numpy array. Note
this can be used in general to carve out any columns to be returned in ID 
sets consistently shuffled and partitioned.

* testID_column: a string of the column title for the column from the
df_test set intended for use as a row identifier value (such as could be
sequential numbers for instance). The function defaults to False for
cases where the training set does not include an ID column. A user can 
also pass a list of string columns titles such as to carve out multiple
columns to be excluded from processing but consistently shuffled and 
partitioned. An integer column index or list of integer column indexes 
may also be passed such as if the source dataset was a numpy array. Note
that if ID columns are same between a train and test set, can leave this
as False (or True) and trainID_column will be applied to test set automatically.

* valpercent: a float value between 0 and 1 which designates the percent
of the training data which will be set aside for the validation
set (generally used for hyperparameter tuning of a downstream model).
This value defaults to 0 for no validation set returned. Note that when
shuffletrain parameter is activated (which is default for train sets) validation
sets will contain random rows.  If shuffletrain parameter is set to False then any 
validation set will be pulled from the bottom sequential rows of the df_train dataframe.

* floatprecision: an integer with acceptable values of _16/32/64_ designating
the memory precision for returned float values. (A tradeoff between memory
usage and floating point precision, smaller for smaller footprint.) 
This currently defaults to 32 for 32-bit precision of float values. Note
that there may be energy efficiency benefits at scale to basing this to 16.
Note that integer data types are still retained with this option.

* shuffletrain: can be passed as one of _{True, False, 'traintest'}_ which 
indicates if the rows in df_train will be shuffled prior to carving out the 
validation sets.  This value defaults to True. Note that if this value is set to 
False then any validation sets will be pulled from the bottom x% sequential 
rows of the df_train dataframe. (Where x% is the sum of validation ratios.) 
Otherwise validation rows will be randomly selected. The third option 'traintest'
is comparable to True for the training set and shuffles the returned test sets
as well. Note that all corresponding returned sets are consistently shuffled 
(such as between train/labels/trainID sets).

* dupl_rows: can be passed as _(True/False/'traintest'/'test')_ which indicates
if duplicate rows will be consolidated to single instance in returned sets. (In
other words, if same row included more than once, it will only be returned once.)
Defaults to False for not activated. True applies consolidation to train set only,
'test' applies consolidation to test set only, 'traintest' applies consolidation 
to both train and test sets seperately. Note this is applied prior to 
TrainLabelFreqLevel if elected. As implemented this does not take into account
duplicate rows in train/test data which have different labels, only one verison
of features/label pair is returned.

* TrainLabelFreqLevel: can be passed as _(True/False/'traintest'/'test')_ 
which indicates if the TrainLabelFreqLevel method will be applied to prepare for 
oversampling training data associated with underrepresented labels (aka class 
imbalance). The method adds multiples of training data rows for those labels with 
lower frequency resulting in an (approximately) levelized frequency. This defaults 
to False. Note that this feature may be applied to numerical label sets if 
the processing applied to the set includes aggregated bins, such as for example
by passing a label column to the 'exc3' category in assigncat for pass-through
force to numeric with inclusion of standard deviation bins or to 'exc4' for 
inclusion of powers of ten bins. For cases where labels are included in the 
test set, this may also be passed as _'traintest'_ to apply levelizing to both 
train and test sets or be passed as _'test'_ to only apply levelizing to test set.
(If a label set includes multiple configurations of the labels, the levelizing
will be based on the first categoric / binned set (either one-hot or ordinal)
based on order of columns.)

* powertransform: _(False/True/'excl'/'exc2')_, defaults to False, when passed as 
True an evaluation will be performed of distribution properties to select between
box-cox, z-score, min-max scaling, or mean absolute deviation scaling normalization
of numerical data. Note that after application of box-cox transform child columns 
are generated for a subsequent z-score normalization. Please note that
I don't consider the current means of distribution property evaluation highly
sophisticated and we will continue to refine this method with further research
going forward. Additionally, powertransform may be passed as values 'excl' or 
'exc2', where for 'excl' columns not explicitly assigned to a root category in 
assigncat will be left untouched, or for 'exc2' columns not explicitly assigned 
to a root category in assigncat will be forced to numeric and subject to default 
modeinfill. (These two excl arguments may be useful if a user wants to experiment 
with specific transforms on a subset of the columns without incurring processing 
time of an entire set.) Note that powertransform not applied to label columns by
default, but can still be applied by passing label column to ptfm in assigncat.

* binstransform: a boolean identifier _(True/False)_ which indicates if all
default numerical sets will receive bin processing such as to generate child
columns with boolean identifiers for number of standard deviations from
the mean, with groups for values <-2, -2-1, -10, 01, 12, and >2. This value 
defaults to False.

* MLinfill: a boolean identifier _(True/False)_ which indicates if the ML
infill method will be applied as a default to predict infill for missing 
or improperly formatted data using machine learning models trained on the
rest of the df\_train set. This defaults to True. ML infill may alternatively
be assigned to distinct columns in assigninfill when MLinfill passed as False. Note that even if sets passed
to automunge(.) have no points needing infill, when MLinfill is activated 
machine learning models will still be trained for potential use of predicting 
infill to subsequent data passed through the postmunge(.) function. ML infill
by default applies scikit-learn random forest machine learning models to predict infill, 
which may be changed to other autoML frameworks via the ML_cmnd parameter.
Parameters and tuning may also be passed to the model training as demonstrated 
with ML_cmnd parameter below.

* infilliterate: an integer indicating how many applications of the ML
infill processing are to be performed for purposes of predicting infill.
The assumption is that for sets with high frequency of missing values
that multiple applications of ML infill may improve accuracy although
note this is not an extensively tested hypothesis. This defaults to 1.
Note that due to the sequence of model training / application, a comparable
set prepared in automunge and postmunge with this option may vary slightly in 
output (as automunge(.) will train separate models on each iteration and
postmunge will just apply the final model on each iteration).

* randomseed: a positive integer used as a seed for randomness throughout 
such as for repeatable data set shuffling, ML infill, and feature importance algorithms. 
This defaults to False indicating a random random seed will be applied (sampled between 0:2^32).

* eval_ratio: a 0-1 float or integer for number of rows, defaults to 0.5, serves
to reduce the overhead of the category evaluation functions under automation by only
evaluating this sampled ratio of rows instead from the full set. Makes automunge faster.
To accomodate small data sets, the convention is that eval_ratio is only applied
when training set has > 2,000 rows.

* numbercategoryheuristic: an integer used as a heuristic. When a 
categorical set has more unique values than this heuristic, it defaults 
to categorical treatment via hashing processing via 'hsh2', otherwise 
categorical sets default to binary encoding via '1010'. This defaults to 255.

* pandasoutput: a selector for format of returned sets. Defaults to _True_
for returned pandas dataframe. If set to _True_ returns pandas dataframes
(note that index is not always preserved, non-integer indexes are extracted to the ID sets,
and automunge(.) generates an application specific range integer index in ID sets 
corresponding to the order of rows as they were passed to function). If set to _False_
returns numpy arrays instead of dataframes.

* NArw_marker: a boolean identifier _(True/False)_ which indicates if the
returned sets will include columns with markers for source column entries subject to 
infill (columns with suffix '\_NArw'). This value defaults to True. Note 
that the properties of cells qualifying as candidate for infill are based
on the 'NArowtype' of the root category of transformations associated with 
the column, see Library of Transformations section below for catalog, the
various NArowtype options (such as justNaN, numeric, positivenumeric, etc)
are also further clarified below in discussion around the processdict parameter.

* featureselection: applied to activate a feature importance evaluation. 
Defaults to False, accepts {False, True, 'pct', 'metric', 'report'}. 
If selected automunge will return a summary of feature importance findings in the featureimportance
returned dictionary. False turns off, True turns on, 'pct' performs the evaluation followed by
a dimensionality reduction based on the featurethreshold parameter to retain a % of top features. 
'metric' performs the evaluation followed by a dimensionality reduction to retain features above a metric value based on featurethreshold parameter. 'report' performs the evluation and returns a report with no
further processing of data. Feature importance evaluation requires the inclusion of a
designated label column in the train set. Note that sorted 
feature importance results are returned in postprocess_dict['FS_sorted'], 
including columns sorted by metric and metric2. Note that feature importance 
model training inspects same ML_cmnd parameters as ML infill. (Note that any user-specified size of validationratios 
if passed are used in this method, otherwise defaults to 0.2.)

* featurethreshold: defaults to 0., accepts float in range of 0-1. Inspected when
featureselection passed as 'pct' or 'metric'. Used to designate the threshold for feature
importance dimensionality reduction. Where e.g. for 'pct' 0.9 would retain 90% of top
features, or e.g. for 'metric' 0.03 would retain features whose metric was >0.03. Note that
NArw columns are only retained for those sets corresponding to columns that "made the cut".

* inplace: defaults to False, when True the df_train (and df_test) passed to automunge(.)
are overwritten with the returned train and test sets. This reduces memory overhead.
For example, to take advantage with reduced memory overhead you could call automunge(.) as:
```
train, train_ID, labels, \
val, val_ID, val_labels, \
test, test_ID, test_labels, \
postprocess_dict = \
am.automunge(df_train, df_test=df_test, inplace=True)
```

* Binary: a dimensionality reduction technique whereby the set of columns
with boolean encodings are collectively encoded with binary encoding such
as may drastically reduce the column count. This has many benefits such as
memory bandwidth and energy cost for inference I suspect, however, there 
may be tradeoffs associated with ability of the model to handle outliers,
as for any new combination of boolean set in the test data the collection
will be subject to the infill. Pass _True_ to activate, defaults to _False_. 
Note that can also be passed as _'retain'_ to retain the boolean columns that 
served as basis for encoding instead of replacing them. To only convert a 
portion of the boolean columns, can pass this as a list of column headers, 
which may include source column headers and/or returned column headers (source 
headers convert all boolean columns derived from a source column, returned 
headers allow to only convert a portion, note it is ok to pass non-boolean 
columns they will be ignored). When passing a list of headers, the default is 
that the binary transform will replace those target columns. For a retain 
option user can pass False as first item in list, e.g. for partial set retention, 
can pass Binary = [False, 'target_column_1', 'target_column_2']. Note that
when applied a column named 'Binary' is used in derivation, thus this is a 
reserved column header when applying this transform.

* PCAn_components: defaults to False for no PCA dimensionality reduction performed.
If passed as _None_ not performed unless # features exceeds 0.5 # rows as a heuristic.
A user can pass _an integer_ to define the number of PCA derived features for 
purposes of dimensionality reduction, such integer to be less than the otherwise 
returned number of sets. Function will default to kernel PCA for all non-negative 
sets or otherwise Sparse PCA. Also if this value is passed as a _float <1.0_ then 
linear PCA will be applied such that the returned number of sets are the minimum 
number that can reproduce that percent of the variance. Note this can also be passed 
in conjunction with assigned PCA type or parameters in the ML_cmnd object. These methods
apply PCA with the scikit-learn library.

* PCAexcl: a _list_ of column headers for columns that are to be excluded from
any application of PCA

* excl_suffix: boolean selector _{True, False}_ for whether columns headers from 'excl' 
transform are returned with suffix appender '\_excl' included. Defaults to False for
no suffix. For advanced users setting this to True makes navigating data structures a 
little easier at small cost of aesthetics of any 'excl' pass-through column headers.

* ML_cmnd: 

```
ML_cmnd = {'autoML_type':'randomforest', \
           'MLinfill_cmnd':{'RandomForestClassifier':{}, 'RandomForestRegressor':{}}, \
           'PCA_type':'default', \
           'PCA_cmnd':{}}
```
The ML_cmnd allows a user to pass parameters to the predictive algorithms
used for ML infill / feature importance evaluation or PCA. (The default
option for 'autoML_type' is 'randomforest' which uses a Scikit-learn Random 
Forest implementation, other options are discussed below.)
For example, a user who doesn't mind a little extra training time for ML infill 
could increase the passed n_estimators beyond the scikit default of 100.

```
ML_cmnd = {'autoML_type':'randomforest', \
           'MLinfill_cmnd':{'RandomForestClassifier':{'n_estimators':1000}, \
                            'RandomForestRegressor':{'n_estimators':1000}}, \
           'PCA_type':'default', \
           'PCA_cmnd':{}}
           
```
A user can also perform hyperparameter tuning of the parameters passed to the
predictive algorithms by instead of passing distinct values passing lists or
range of values. The hyperparameter tuning defaults to grid search for cases 
where user passes parameters as lists or ranges, for example:
```
ML_cmnd = {'autoML_type':'randomforest', \
           'hyperparam_tuner':'gridCV', \
           'MLinfill_cmnd':{'RandomForestClassifier':{'max_depth':range(4,6)}, \
                            'RandomForestRegressor' :{'max_depth':[3,6,12]}}}
```
A user can also perform randomized search via ML_cmnd, and pass parameters as 
distributions via scipy stats module such as:
```
ML_cmnd = {'autoML_type':'randomforest', \
           'hyperparam_tuner' : 'randomCV', \
           'randomCV_n_iter'  : 15, \
           'MLinfill_cmnd':{'RandomForestClassifier':{'max_depth':stats.randint(3,6)}, \
                            'RandomForestRegressor' :{'max_depth':[3,6,12]}}}
```
There is an experimental option available to use an alternate autoML framework for ML infill 
via the AutoGluon library. Requires externally installing AutoGluon library. 
(If AutoGluon doesn't want to train a model for some particular column you can 
run again after assigning that column to a different infill in assigninfill.) Note that since AutoGluon
saves model properties in a local folder, when you process additional data with postmunge it will
need to be in a notebook saved in same directory as was used for automunge. Further information
on AutoGluon library available on arxiv as [AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data](https://arxiv.org/abs/2003.06505) by Nick Erickson et al.
```
#can activate AutoGluon for ML infill and feature importance by passing ML_cmnd as
ML_cmnd = {'autoML_type':'autogluon'}
```
Parameters can be passed to the AutoGluon fit operation by ML_cmnd['AutoGluon'], here we demonstrate passing 
presets=‘best_quality’, which may improve model accuracy but at cost of much larger memory footprint for 
properties saved to disk originating from using more models in the ensemble. Note that the random_seed 
parameter is already passed based on the automunge(.) randomseed.
```
#can activate AutoGluon for ML infill and feature importance by passing ML_cmnd as
ML_cmnd = {'autoML_type': 'autogluon', \
           'MLinfill_cmnd':{'AutoGluon'  : {'presets' : 'best_quality'}}}
```
Another autoML option for ML infill and feature importance is by the CatBoost library.
Requires externally installing CatBoost library. Uses early stopping by default for regression 
and no early stopping by default for classifier. Note that the random_seed 
parameter is already passed based on the automunge(.) randomseed.
```
#CatBoost available by passing ML_cmnd as 
ML_cmnd = {'autoML_type':'catboost'}
```
Can pass parameters to model initialization and fit operation as:
```
#example of turning on early stopping for classifier 
#by passing a eval_ratio for validation set which defaults to 0.15 for regressor
#note eval_ratio is an Automunge parameter, other parameters accepted are those from CatBoost library
ML_cmnd = {'autoML_type':'catboost', 
           'MLinfill_cmnd' : {'catboost_classifier_model' : {},
                              'catboost_classifier_fit'   : {'eval_ratio' : 0.15 },
                              'catboost_regressor_model'  : {},
                              'catboost_regressor_fit'    : {}}}
```
In general, accuracy performance of autoML options are expected as AutoGluon > CatBoost > Random Forest.
In general, latency performance of autoML options are expected as Random Forest > CatBoost > AutoGluon.
In general, memory performance of autoML options are expected as Random Forest > CatBoost > AutoGluon.
And where Random Forest and Catboost are more portable than AutoGluon since don't require a local model 
repository saved to hard drive. (For now retaining Random Forest as the default, of course a further
tradeoff is that Random Forest doesn't include GPU support.)

Another ML infill option is available by the FLAML library. 
```
#FLAML available by passing ML_cmnd as 
ML_cmnd = {'autoML_type':'flaml'}
```
Can pass parameters to fit operation as:
```
#example of setting time budget in seconds for training
ML_cmnd = {'autoML_type':'flaml', 
           'MLinfill_cmnd' : {'flaml_classifier_fit'   : {'time_budget' : 15 },
                              'flaml_regressor_fit'    : {'time_budget' : 15}}}
```

A user can also assign specific methods for PCA transforms. Current PCA_types
supported include 'PCA', 'SparsePCA', and 'KernelPCA', all via Scikit-Learn.
Note that the n_components are passed separately with the PCAn_components 
argument noted above. A user can also pass parameters to the PCA functions
through the PCA_cmnd, for example one could pass a kernel type for KernelPCA
as:
```
ML_cmnd = {'autoML_type':'randomforest', \
           'MLinfill_cmnd':{'RandomForestClassifier':{}, \
                            'RandomForestRegressor':{}}, \
           'PCA_type':'KernelPCA', \
           'PCA_cmnd':{'kernel':'sigmoid'}}
           
```
A user can also exclude returned boolean (0/1) columns from any PCA application by passing 
'PCA_cmnd':{'bool_PCA_excl':True}
or exclude returned boolean and ordinal columns from PCA application by
'PCA_cmnd':{'bool_ordl_PCAexcl':True}
such as could potentially result in memory savings.

* assigncat:

```
#Here are the current transformation options built into our library, which
#we are continuing to build out. A user may also define their own.

assigncat = {'nmbr':[], 'retn':[], 'mnmx':[], 'mean':[], 'MAD3':[], 'lgnm':[], \
             'bins':[], 'bsor':[], 'pwrs':[], 'pwr2':[], 'por2':[], 'bxcx':[], \
             'addd':[], 'sbtr':[], 'mltp':[], 'divd':[], \
             'log0':[], 'log1':[], 'logn':[], 'sqrt':[], 'rais':[], 'absl':[], \
             'bnwd':[], 'bnwK':[], 'bnwM':[], 'bnwo':[], 'bnKo':[], 'bnMo':[], \
             'bnep':[], 'bne7':[], 'bne9':[], 'bneo':[], 'bn7o':[], 'bn9o':[], \
             'bkt1':[], 'bkt2':[], 'bkt3':[], 'bkt4':[], \
             'nbr2':[], 'nbr3':[], 'MADn':[], 'MAD2':[], 'tlbn':[], \
             'mnm2':[], 'mnm3':[], 'mnm4':[], 'mnm5':[], 'mnm6':[], \
             'ntgr':[], 'ntg2':[], 'ntg3':[], 'mea2':[], 'mea3':[], 'bxc2':[], \
             'dxdt':[], 'd2dt':[], 'd3dt':[], 'dxd2':[], 'd2d2':[], 'd3d2':[], \
             'nmdx':[], 'nmd2':[], 'nmd3':[], 'mmdx':[], 'mmd2':[], 'mmd3':[], \
             'shft':[], 'shf2':[], 'shf3':[], 'shf4':[], 'shf7':[], 'shf8':[], \
             'bnry':[], 'onht':[], 'text':[], 'txt2':[], '1010':[], 'smth':[], \
             'ordl':[], 'ord3':[], 'hash':[], 'hsh2':[], 'hs10':[], \
             'Unht':[], 'Utxt':[], 'Utx2':[], 'Uor3':[], 'Uor6':[], 'U101':[], \
             'splt':[], 'spl2':[], 'spl5':[], 'sp15':[], 'sp19':[], 'sbst':[], \
             'spl8':[], 'spl9':[], 'sp10':[], 'sp16':[], 'sp20':[], 'sbs2':[], \
             'srch':[], 'src2':[], 'src4':[], 'strn':[], 'lngt':[], 'aggt':[], \
             'nmrc':[], 'nmr2':[], 'nmcm':[], 'nmc2':[], 'nmEU':[], 'nmE2':[], \
             'nmr7':[], 'nmr8':[], 'nmc7':[], 'nmc8':[], 'nmE7':[], 'nmE8':[], \
             'ors2':[], 'ors5':[], 'ors6':[], 'ors7':[], 'ucct':[], 'Ucct':[], \
             'or15':[], 'or17':[], 'or19':[], 'or20':[], 'or21':[], 'or22':[], \
             'date':[], 'dat2':[], 'dat6':[], 'wkdy':[], 'bshr':[], 'hldy':[], \
             'wkds':[], 'wkdo':[], 'mnts':[], 'mnto':[], \
             'yea2':[], 'mnt2':[], 'mnt6':[], 'day2':[], 'day5':[], \
             'hrs2':[], 'hrs4':[], 'min2':[], 'min4':[], 'scn2':[], 'DPrt':[], \
             'DPnb':[], 'DPmm':[], 'DPbn':[], 'DPod':[], 'DP10':[], 'DPoh':[], \
             'qbt1':[], 'qbt2':[], 'qbt3':[], 'qbt4':[], 'nmqb':[], 'mmqb':[], \
             'excl':[], 'exc2':[], 'exc3':[], 'exc4':[], 'exc5':[], \
             'null':[], 'copy':[], 'shfl':[], 'eval':[], 'ptfm':[]}
```         

Descriptions of these transformations are provided in document below (in section
titled "Library of Transformations"). [Library of Transformations](https://github.com/Automunge/AutoMunge#library-of-transformations)

A user may add column header identifier strings to each of these lists to assign 
a distinct specific processing approach to any column (including labels). Note 
that this processing category will serve as the "root" of the tree of transforms 
as defined in the transformdict. Note that additional categories may be passed if 
defined in the passed transformdict and processdict. An example of usage here 
could be if a user wanted to only process numerical columns 'nmbrcolumn1' and 
'nmbrcolumn2' with z-score normalization instead of the full range of numerical 
derivations when implementing the binstransform parameter they could pass 
```
assigncat = {'nbr2':['nmbrcolumn1', 'nmbrcolumn2']}
```
Note that for single entry column assignments a user can just pass the string or integer 
of the column header without the list brackets.

* assignparam
A user may pass column-specific or category specific parameters to those transformation 
functions that accept parameters. Any parameters passed to automunge(.) will be saved in
the postprocess_dict and consistently applied in postmunge(.). assignparam is 
a dictionary that should be formatted per following example:
```
#template:
assignparam = {'global_assignparam'  : {'(parameter)': 42}, \
               'default_assignparam' : {'(category)' : {'(parameter)' : 42}}, \
                        '(category)' : {'(column)'   : {'(parameter)' : 42}}}, \

#example:
assignparam = {'category1' : {'column1' : {'param1' : 123}, 'column2' : {'param1' : 456}}, \
               'cateogry2' : {'column3' : {'param2' : 'abc', 'param3' : 'def'}}}
```

In other words: The first layer keys are the transformation category for 
which parameters are intended. The second layer keys are string identifiers 
for the columns for which the parameters are intended. The third layer keys 
are the parameters whose values are to be passed. To specify new default
parameters for a given trasnformation category 'default_assignparam' can
be applied, or to specificy global parameters for all transfomation functions
'global_assignparam' can be applied. Transforms that do not accept a particular 
parameter will just ignore the specification.

As an example with actual parameters, consider the transformation category 
'splt' intended for 'column1', which accepts parameter 'minsplit' for minimum 
character length of detected overlaps. If we wanted to pass 4 instead of the 
default of 5:
```
assignparam = {'splt' : {'column1' : {'minsplit' : 4}}}
```
Note that the category identifier should be the category entry to the family 
tree primitive associated with the transform, which may be different than the 
root category of the family tree assigned in assigncat. The set of family 
tree definitions for root categories are included below for reference. 

As an example to demonstrate edge case for cases where transformation category does 
not match transformation function (based on entries to transformdict and 
processdict), if we want to pass a parameter to turn off UPCS transform included 
in or19 family tree and associuated with the or19 transformation category for 
instance, we would pass the parameter to or19 instead of UPCS because assignparam 
inspects the transformation category associated with the transformation function, 
and UPCS function is the processdict entry for or19 category entry in the family 
tree primitives associated with the or19 root category, even though 'activate' is 
an UPCS transform parameter. (This clarification intended for advanced users to 
avoid ambiguity.)
```
assignparam = {'or19' : {'column1' : {'activate' : False}}}
```
Note that column string identifiers may just be the source column string or may 
include the suffix appenders for downstream columns serving as input to the 
target trasnformation function, such as may be useful if multiple versions of 
the same transformation are applied within the same family tree. If more than 
one column identifier matches a column in assignparam entry to a transformation 
category (such as both the source column and the derived column serving as input 
to the transformation function), the derived column (such as may include suffix 
appenders) will take precedence.

Note that if a user wishes to overwrite the default parameters associated with a 
particular category for all columns without specifying them individually they can 
pass a 'default_assignparam' entry as follows (this only overwrites those parameters 
that are not otherwise specified in assignparam).
```
assignparam = {'category1' : {'column1' : {'param1' : 123}, 'column2' : {'param1' : 456}}, \
               'cateogry2' : {'column3' : {'param2' : 'abc', 'param3' : 'def'}}, \
               'default_assignparam' : {'category3' : {'param4' : 789}}}
```
Or to pass the same parameter to all transformations to all columns, can use the 
'global_assignparam'. The global_assignparam may be useful for instance to turn off 
inplace trasnformations such as to retain family tree column grouping correspondance 
in returned set. Transformations that do not accept a particular parameter will just 
ignore.
```
assignparam = {'global_assignparam' : {'inplace' : False}}
```
In order of precendence, parameters assigned to distinct category/column configurations 
take precedence to default_assignparam assigned to categories which take precendence to 
global_assignparam assigned to all transformations which take precendence to parameters 
set as defaultparams in processdict definition. 

See the Library of Transformations section below for those transformations that 
accept parameters.

* assigninfill 
```
#Here are the current infill options built into our library, which
#we are continuing to build out.
assigninfill = {'stdrdinfill':[], 'MLinfill':[], 'zeroinfill':[], 'oneinfill':[], \
                'adjinfill':[], 'meaninfill':[], 'medianinfill':[], \
                'modeinfill':[], 'lcinfill':[], 'naninfill':[]}
```
A user may add column identifier strings to each of these lists to designate the 
column-specific infill approach for missing or improperly formatted values. The
source column identifier strings may be passed for assignment of common infill 
approach to all columns derived from same source column, or derived column identifier
strings (including the suffix appenders from transformations) may be passed to assign 
infill approach to a specific derived column. Note that passed derived column headers 
take precedence in case of overlap with passed source column headers. Note that infill
defaults to MLinfill if nothing assigned and the MLinfill argument to automunge is set 
to True. Note that for single entry column assignments a user can just pass the string 
or integer of the column header without the list brackets. Note that the infilled cells
are based on the rows corresponding to activations from the NArw_marker parameter.
```
#  - stdrdinfill  : the default infill specified in the library of transformations for 
#                   each transform below. 
#  - MLinfill     : for MLinfill to distinct columns when MLinfill parameter not activated
#  - zeroinfill   : inserting the integer 0 to missing cells. 
#  - oneinfill    : inserting the integer 1. 
#  - adjinfill    : passing the value from the preceding row to missing cells. 
#  - meaninfill   : inserting the mean derived from the train set to numeric columns. 
#  - medianinfill : inserting the median derived from the train set to numeric columns. 
#                   (Note currently boolean columns derived from numeric are not supported 
#                   for mean/median and for those cases default to those infill from stdrdinfill.) 
#  - modeinfill   : inserting the most common value for a set, note that modeinfill 
#                   supports multi-column boolean encodings, such as one-hot encoded sets or 
#                   binary encoded sets. 
#  - lcinfill     : comparable to modeinfill but with least common value instead of most. 
#  - naninfill    : inserting NaN to missing cells. 

#an example of passing columns to assign infill via assigninfill:
#for source column 'column1', which hypothetically is returned through automunge(.) as
#'column1_nmbr', 'column1_mnmx', 'column1_bxcx_nmbr'
#we can assign MLinfill to 'column1_bxcx_nmbr' and meaninfill to the other two by passing 
#to an automunge call: 

assigninfill = {'MLinfill':['column1_bxcx_nmbr'], 'meaninfill':['column1']}
```
Please note that support of assigninfill to label columns is intended as a future extension.

* assignnan: for use to designate data set entries that will be targets for infill, such as 
may be entries not covered by NArowtype definitions from processdict. For example, we have 
general convention that NaN is a target for infill, but a data set may be passed with a custom 
string signal for infill, such as 'unknown'. This assignment operator saves the step of manual 
munging prior to passing data to functions by allowing user to specify custom targets for infill.

assignnan accepts following form, populated in first tier with any of 'categories'/'columns'/'global'
```
assignnan = {'categories':{}, 'columns':{}, 'global':[]}
```
Note that global takes entry as a list, while categories and columns take entries as a dictionary 
with values of the target assignments and corresponding lists of terms, which could be populated 
with entries as e.g.:
```
assignnan = {'categories' : {'cat1' : ['unknown1']}, \
             'columns' : {'col1' : ['unknown2']}, \
	     'global' : ['unknown3']}
```
Where 'cat1' is example of root category, 'col1' is example of source column header, and 'unknown1'/2/3 
are examples of entries intended for infill corresponding to each. In cases of redundant specification, 
global takes precendence over columns which takes precedence over categories. Note that lists of terms 
can also be passed as single values such as string / number for internal conversion to list.

Assignnan also supports stochastic and range based injections, such as to target for infill specific
segments of a set's distribution. 'injections' can be passed to assignnan as:
```
assignnan =     {'injections' : {'(column)' : {'inject_ratio' : (float), \
                                               'range' : {'ratio'  : (float), \
                                                          'ranges' : [[min1, max1], [min2, max2]]}, \
                                               'minmax_range' : {'ratio'  : (float), \
                                                                 'ranges' : [[min1, max1], [min2, max2]]}, \
                                               'entries' : ['(entry1)', '(entry2)'], \
                                               'entry_ratio' : {'(entry1)' : (float), \
                                                                '(entry2)' : (float)}
                                              }
                                }
                }
    
#where injections may be specified for each source column passed to automunge(.)
#- inject_ratio is uniform randomly injected nan points to ratio of entries
#- range is injection within a specified range based on ratio float defaulting to 1.0
#- minmax_range is injection within scaled range (accepting floats 0-1 based on received 
#column max and min (returned column is not scaled)
#- entries are full replacement of specific entries to a categoric set
#- entry_ratio are partial injection to specific entries to a categoric set per specified float ratio 
```

* transformdict: allows a user to pass a custom tree of transformations.
Note that a user may define their own (traditionally 4 character) string "root"
identifiers for a series of processing steps using the categories of processing 
already defined in our library and then assign columns in assigncat, or for 
custom processing functions this method should be combined with processdict 
which is only slightly more complex. For example, a user wishing to define a 
new set of transformations for numerical series 'newt' that combines NArows, 
min-max, box-cox, z-score, and standard deviation bins could do so by passing a 
trasnformdict as:
```
transformdict =  {'newt' : {'parents' : ['bxc4'], \
                            'siblings': [], \
                            'auntsuncles' : ['mnmx'], \
                            'cousins' : ['NArw'], \
                            'children' : [], \
                            'niecesnephews' : [], \
                            'coworkers' : [], \
                            'friends' : []}}
                                    
#Where since bxc4 is passed as a parent, this will result in pulling
#offspring keys from the bxc4 family tree, which has a nbr2 key as children.

#from automunge library:
    transform_dict.update({'bxc4' : {'parents' : ['bxcx'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : ['NArw'], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['nbr2'], \
                                     'friends' : []}})
                                     
#note that 'nbr2' is passed as a coworker primitive meaning no downstream 
#primitives would be accessed from the nbr2 family tree. If we wanted nbr2 to
#incorporate any offspring from the nbr2 tree we could instead assign as children
#or niecesnephews.

#Note  that optionally primitives without entries can be ommitted,
#and list brackets can be omitted for single entries to a primitive
#the following is an equivalent specification to the 'newt' entry above
transformdict =  {'newt' : {'parents'     : 'bxc4', \
                            'auntsuncles' : 'mnmx', \
                            'cousins'     : 'NArw'}}

```
Basically here 'newt' is the key and when passed to one of the family primitives
the corresponding process function is applied, and if it is passed to a family
primitive with downstream offspring then those offspring keys are pulled from
that key's family tree. For example, here mnmx is passed as an auntsuncles which
means the mnmx processing function is applied with no downstream offspring. The
bxcx key is passed as a parent which means the bxcx transform is applied coupled
with any downstream transforms from the bxcx key family tree, which we also show.
Note the family primitives tree of transformations can be summarized as:
```
'parents' :           upstream / first generation / replaces column / with offspring
'siblings':           upstream / first generation / supplements column / with offspring
'auntsuncles' :       upstream / first generation / replaces column / no offspring
'cousins' :           upstream / first generation / supplements column / no offspring
'children' :          downstream parents / offspring generations / replaces column / with offspring
'niecesnephews' :     downstream siblings / offspring generations / supplements column / with offspring
'coworkers' :         downstream auntsuncles / offspring generations / replaces column / no offspring
'friends' :           downstream cousins / offspring generations / supplements column / no offspring
```

![image](https://user-images.githubusercontent.com/44011748/76485331-a2a24380-63f2-11ea-8559-08bb1c3be395.png)

Note that a user should avoid redundant entries across a set of upstream or downstream primitives.
Since there is recursion involved a user should be careful of creating infinite loops from passing
downstream primitive entries with offspring whose own offspring coincide with an earlier generation.
(The presence of infinite loops is tested for to a max depth of 111 offspring, an arbitrary figure.)

Note that when we define a new transform such as 'newt' above, we also need 
to define a corresponding processdict entry for the new category, which we 
demonstrate here:

* processdict: allows a user to define their own processing functions and transformation category properties 
corresponding to new transformdict entries. All transformation categories used in transformdict, including
those used as root categories as well as transformation category entries to family tree primitives associated
with a root category, require a correpsonding entry in the processdict to define transformation category 
properties. Only in cases where a transformdict entry is being passed to overwrite an existing category internal 
to the library is a corresponding processdict entry not required. We'll describe the processdict entries here:
```
#for example, to populate a custom transformation category 'newt' that uses
#internal defined transformation functions:

processdict =  {'newt' : {'dualprocess' : am.process_mnmx, \
                          'singleprocess' : None, \
                          'postprocess' : am.postprocess_mnmx, \
                          'NArowtype' : 'numeric', \
                          'MLinfilltype' : 'numeric', \
                          'labelctgy' : 'mnmx'}}

#A user should pass either a pair of processing functions to both 
#dualprocess and postprocess, or alternatively just a single processing
#function to singleprocess, and pass None to those not used.
#For now, if just using the category as a root key and not as a family primitive, 
#can simply pass None to all the processing slots. We'll demonstrate their 
#composition and data structures for custom processing functions later in the
#section of this document "Custom Processing Functions".

#dualprocess: for passing a processing function in which normalization 
#             parameters are derived from properties of the training set
#             and jointly process the train set and if available test set

#singleprocess: for passing a processing function in which no normalization
#               parameters are needed from the train set to process the
#               test set, such that train and test sets processed separately

#postprocess: for passing a processing function in which normalization 
#             parameters originally derived from the train set are applied
#             to seperately process a test set

#NArowtype: can be entries of {'numeric', 'integer', 'justNaN', 'exclude', 
#                              'positivenumeric', 'nonnegativenumeric', 
#                              'nonzeronumeric', 'parsenumeric', 'datetime'}
# - 'numeric' for source columns with expected numeric entries
# - 'integer' for source columns with expected integer entries
# - 'justNaN' for source columns that may have expected entries other than numeric
# - 'exclude' for source columns that aren't needing NArow columns derived
# - 'positivenumeric' for source columns with expected positive numeric entries
# - 'nonnegativenumeric' for source columns with expected non-negative numeric (zero allowed)
# - 'nonzeronumeric' for source columns with allowed positive and negative but no zero
# - 'parsenumeric' marks for infill strings that don't contain any numeric characters
# - 'datetime' marks for infill cells that aren't recognized as datetime objects
# ** Note that NArowtype also is used as basis for metrics evaluated in drift assessment of source columns
# ** Note that by default any np.inf values are converted to NaN for infill
# ** Note that by default python None entries are treated as targets for infill

#MLinfilltype: can be entries {'numeric', 'singlct', 'binary', 'multirt', 'concurrent_act', 'concurrent_nmbr', 
#                              '1010', 'exclude', 'boolexclude', 'ordlexclude', 'totalexclude'}
#              'numeric' refers to columns where predictive algorithms treat
#                        as a regression for numeric sets
#              'singlct' single column sets with ordinal entries (integers)
#              'binary'  single column sets with boolean entries (0/1)
#              'multirt' refers to category returning multiple columns where 
#                        predictive algorithms treat as a multi modal classifier
#              'concurrent_act' for multicolumn sets with boolean entries as may have 
#                        multiple entries in the same row
#              'concurrent_nmbr' for multicolumn sets with numerical entries
#              '1010'   for multicolumn sets with binary encoding via 1010
#                        will be converted to onehot for ML
#              'exclude' for columns which will be excluded from ML infill
#              'boolexclude' boolean set suitable for Binary transform but excluded from all infill (eg NArw entries)
#              'ordlexclude' ordinal set exluded from infill
#              'totalexclude' for complete passthroughs (eg excl) without infill and excluded 
#                        from inf conversion and assignnan global option

#labelctgy: should be a string entry of a single transform category as recorded by the associated transformation
#function. Used to determine a basis of feature selection for cases where labels are returned in multiple configurations.
#Also used in label frequency levelizer.

#Note that NArowtype is associated with a category's use as a root category, such as may be assigned to a column in assigncat
#MLinfilltype is associated with a category's use as a transformation category entry to family tree primitives associated with a root category
#and labelctgy is associated with a category's use as a root category for a label column when conducting feature importance
```

Optionally, some additional values can be incorporated into the processdict to 
support inversion for a transformation category:
```
#for example 
processdict =  {'newt' : {'dualprocess' : am.process_mnmx, \
                          'singleprocess' : None, \
                          'postprocess' : am.postprocess_mnmx, \
                          'inverseprocess' : am.inverseprocess_mnmx, \
                          'info_retention' : False, \
                          'NArowtype' : 'numeric', \
                          'MLinfilltype' : 'numeric', \
                          'labelctgy' : 'mnmx'}}
			  
#Where 'inverseprocess' is a function to invert the forward pass transformation.
#And 'info_retention' is boolean to signal whether there will be any information
#loss to recovered data from inversion. For format of inverseprocess functions
#please refer to the code base.
```

Optionally, a user can set alternate default parameters to be passed to the associated
transfomation functions by including the 'defaultparams' key. These updates to default
parameters will still be overwritten if user manually specifies parameters in assignparam.
```
#for example
processdict =  {'DLmm' : {'dualprocess' : am.process_DPmm, \
                          'singleprocess' : None, \
                          'postprocess' : am.postprocess_DPmm, \
                          'inverseprocess' : am.inverseprocess_UPCS, \
                          'info_retention' : True, \
                          'defaultparams' : {'noisedistribution' : 'laplace'}, \
                          'NArowtype' : 'numeric', \
                          'MLinfilltype' : 'numeric', \
                          'labelctgy' : 'DLmm'}}
```

Since specification of transformation functions can be kind of cumbersome in order
to dig out from the codebase naming conventions for internally defined functions, a
simplification is available when populating a processdict for a user passed entry by
way of the 'functionpointer' entry. When a functionpointer category entry is included, 
the transformation functions and other entries are automatically populated based on entries found in 
processdict entries of the pointer, such as with entries for dualprocess, singleprocess, 
postprocess, inverseprocess, and info_retention, and also the other processdict entries. 
For cases where a functionpointer points to a processdict entry that itself has a functionpointer 
entry, chains of pointers are followed until an entry with defined processing functions is reached. 
defaultparam entries of each pointer link are also accessed for update, and if the new category 
specification contains any redundant defaultparam entries with those found in a pointer 
category the new category entries take precedence. Similarly for chains of pointers the nearer
links of other entries take precedence.

In other words, if you are populating a new processdict transformation 
category and you want the transformation functions and other entries to match an existing category, you 
can simply pass the existing category as a functionpointer entry to the new category. 
Here is an example if we want to match the DLmm category demonstrated above for a new 
category 'newt', such as would be useful if we wanted to define an alternate DLmm family 
tree in a corresponding newt transformdict entry.
```
processdict =  {'newt' : {'functionpointer' : 'DLmm', \
                          'NArowtype' : 'numeric', \
                          'MLinfilltype' : 'numeric', \
                          'labelctgy' : 'DLmm'}}
			  
#or an even simpler approach if no overwrites are desired could just be to copy everything
processdict =  {'newt' : {'functionpointer' : 'DLmm'}}
```
Note that when passing a processdict entry to overwrite an internally defined processdict entry, you
can pass the functionpointer to point to itself, and then only have to populate the entries you are overwriting.

Note that many of the transformation functions in the library have support for distinguishing between 
inplace operations vs returning a column copied from the input. Inplace operations are expected to 
reduce memory overhead. When populating a processdict entry, iff the transformation function supports 
inplace operations and you want to apply inplace operations when available based on the order of family 
tree primitive entries (inplace when available is performed on the final replacement primitve entry 
for a given generation), you need to specify with an 'inplace_option' processdict entry such as:
```
#for example 
processdict =  {'newt' : {'functionpointer' : 'mnmx', \
                          'inplace_option' : True, \
                          'NArowtype' : 'numeric', \
                          'MLinfilltype' : 'numeric', \
                          'labelctgy' : 'mnmx'}}
```

As an asterisk for advanced users:
Note that when populating a processdict for a transformation category, the
inversion function should be consistent with the transformation category that
was populated in the column_dict data structure as part of the forward pass
transformations in the dualprocess or singleprocess processdict entries.

* evalcat: modularizes the automated evaluation of column properties for assignment 
of root transformation categories, allowing user to pass custom functions for this 
purpose. Passed functions should follow format:

```
def evalcat(df, column, randomseed, eval_ratio, numbercategoryheuristic, powertransform, labels = False):
  """
  #user defined function that takes as input a dataframe df and column id string column
  #evaluates the contents of cells and classifies the column for root category of 
  #transformation (e.g. comparable to categories otherwise assigned in assigncat)
  #returns category id as a string
  """
  ...
  return category
```
And could then be passed to automunge function call such as:
```
evalcat = evalcat
```
I recommend using the evalcategory function defined in master file as starting point. 
(Minus the 'self' parameter since defining external to class.) Note that the 
parameters eval_ratio, numbercategoryheuristic, powertransform, and labels are passed as user 
parameters in automunge(.) call and only used in evalcategory function, so if user wants 
to repurpose them totally can do so. (They default to .5, 255, False, False.) Note evalcat 
defaults to False to use built-in evalcategory function. Note evalcat will only be 
applied to columns not assigned in assigncat. (Note that columns assigned to 'eval' / 'ptfm'
in assigncat will be passed to this function for evaluation with powertransform = False / True
respectively.) Note that function currently uses python collections library and datetime as dt.

* privacy_encode: a boolean marker _{True, False}_ defaults to False. For cases where sets 
are returned as pandas dataframe, a user may desire privacy preserving encodings in which
column headers of received data are anonymized. This parameter when activated replaces
headers and suffixes with integers, including distinct sets of integers between train, labels,
and ID sets. Note that conversion information is available in returned postprocess_dict under
privacy reports. Note that when activated consistent convention is applied in postmunge
and inversion is supported.

* printstatus: user can pass _True/False_ indicating whether the function will print 
status of processing during operation. Defaults to True.

Ok well we'll demonstrate further below how to build custom transformation functions,
for now you should have sufficient tools to build sets of transformation categories 
using the family tree primitives and etc.

...

# postmunge(.)

The postmunge(.) function is intended to consistently prepare subsequently available
and consistently formatted train or test data with just a single function call. It 
requires passing the postprocess_dict object returned from the original application 
of automunge and that the passed test data have consistent column header labeling as 
the original train set (or for Numpy arrays consistent order of columns). Processing
data with postmunge(.) is considerably more efficient than automunge(.) since it does 
not require the overhead of the evaluation methods, the derivation of transformation 
normalization parameters, and/or the training of models for ML infill.

```

#for postmunge(.) function to prepare subsequently available data
#using the postprocess_dict object returned from original automunge(.) application

#Remember to initialize automunge
from Automunge import *
am = AutoMunge()


#Then we can run postmunge function as:

test, test_ID, test_labels, \
postreports_dict = \
am.postmunge(postprocess_dict, df_test, \
             testID_column = False, labelscolumn = False, \
             pandasoutput = True, printstatus = True, inplace = False, \
             dupl_rows = False, TrainLabelFreqLevel = False, \
	     featureeval = False, traindata = False, \
             driftreport = False, inversion = False, \
             returnedsets = True, shuffletrain = False)
```

Or to run postmunge(.) with default parameters we simply need the postprocess_dict
object returned from the corresponding automunge(.) call and a consistently formatted
additional data set.

```
test, test_ID, test_labels, \
postreports_dict \
= am.postmunge(postprocess_dict, df_test)
```          

## postmunge(.) returned sets:
Here now are descriptions for the returned sets from postmunge, which
will be followed by descriptions of the parameters which can be passed to
the function. Default is that returned sets are pandas dataframes, with
single column sets returned as pandas series.

* test: the set of features, consistently encoded and normalized as the
training data, that can be used to generate predictions from a model
trained with the train set from automunge.

* test_ID: the set of ID values corresponding to the test set. Also included 
in this set is a derived column titled 'Automunge_index', 
this column serves as an index identifier for order of rows as they were 
received in passed data, such as may be beneficial when data is shuffled.

* test_labels: a set of numerically encoded labels corresponding to the
test set if a label column was passed. Note that the function
assumes the label column is originally included in the train set. Note
that if the labels set is a single column a returned numpy array is 
flattened (e.g. [[1,2,3]] converted to [1,2,3] )

* postreports_dict: a dictionary containing entries for following:
  - postreports_dict['featureimportance']: results of optional feature 
  importance evaluation based on parameter featureeval. (See automunge(.) 
  notes above for feature importance printout methods.)
  - postreports_dict['finalcolumns_test']: list of columns returned from 
  postmunge
  - postreports_dict['driftreport']: results of optional drift report 
  evaluation tracking properties of postmunge data in comparison to the 
  original data from automunge call associated with the postprocess_dict 
  presumably used to train a model. Results aggregated by entries for the
  original (pre-transform) list of columns, and include the normalization
  parameters from the automunge call saved in postprocess_dict as well
  as the corresponding parameters from the new data consistently derived 
  in postmunge
  - postreports_dict['sourcecolumn_drift']: results of optional drift report
  evaluation tracking properties of postmunge data derived from source 
  columns in comparison to the original data from automunge(.) call associated 
  with the postprocess_dict presumably used to train a model. 
  
```
#the results of a postmunge driftreport assessment are returned in the postreports_dict 
#object returned from a postmunge call, as follows:

postreports_dict = \
{'featureimportance':{(not shown here for brevity)},
'finalcolumns_test':[(derivedcolumns)],
'driftreport': {(sourcecolumn) : {'origreturnedcolumns_list':[(derivedcolumns)], 
                           'newreturnedcolumns_list':[(derivedcolumns)],
                           'drift_category':(category),
                           'orignotinnew': {(derivedcolumn):{'orignormparam':{(stats)}},
                           'newnotinorig': {(derivedcolumn):{'newnormparam':{(stats)}},
                           'newreturnedcolumn':{(derivedcolumn):{'orignormparam':{(stats)},
                                                                 'newnormparam':{(stats)}}}},
'rowcount_basis': {'automunge_train_rowcount':#, 'postmunge_test_rowcount':#},
'sourcecolumn_drift': {'orig_driftstats': {(sourcecolumn) : (stats)}, 
                       'new_driftstats' : {(sourcecolumn) : (stats)}}}
		       
#the driftreport stats for derived columns are based on the normalization_dict entries from the
#corresponding processing function associated with that column's derivation

#here is an example of source column drift assessment statistics for a positive numeric root category:
postreports_dict['sourcecolumn_drift']['new_driftstats'] = \
{(sourcecolumn) : {'max'         : (stat),
                   'quantile_99' : (stat),
                   'quantile_90' : (stat),
                   'quantile_66' : (stat),
                   'median'      : (stat),
                   'quantile_33' : (stat),
                   'quantile_10' : (stat),
                   'quantile_01' : (stat),
                   'min'         : (stat),
                   'mean'        : (stat),
                   'std'         : (stat),
                   'MAD'         : (stat),
                   'skew'        : (stat),
                   'shapiro_W'   : (stat),
                   'shapiro_p'   : (stat),
                   'nonpositive_ratio' : (stat),
                   'nan_ratio'   : (stat)}} 
```

...


## postmunge(.) passed parameters

```

#for postmunge(.) function on subsequently available test data
#using the postprocess_dict object returned from original automunge(.) application

#Remember to initialize automunge
from Automunge import *
am = AutoMunge()


#Then we can run postmunge function as:

test, test_ID, test_labels, \
postreports_dict = \
am.postmunge(postprocess_dict, df_test, \
             testID_column = False, labelscolumn = False, \
             pandasoutput = True, printstatus = True, inplace = False, \
             dupl_rows = False, TrainLabelFreqLevel = False, \
	     featureeval = False, traindata = False, \
             driftreport = False, inversion = False, \
             returnedsets = True, shuffletrain = False)
```

* postprocess_dict: this is the dictionary returned from the initial
application of automunge(.) which included normalization parameters to
facilitate consistent processing of additional train or test data to the 
original processing of the train set. This requires a user to remember 
to download the dictionary at the original application of automunge, 
otherwise if this dictionary is not available a user can feed this 
subsequent test data to the automunge along with the original train data 
exactly as was used in the original automunge(.) call.

* df_test: a pandas dataframe or numpy array containing a structured 
dataset intended for use to generate predictions from a machine learning 
model trained from the automunge returned sets. The set must be consistently 
formatted as the train set with consistent order of columns and if labels are
included consistent labels. If desired the set may include an ID column. The 
tool supports the inclusion of non-index-range column as index or multicolumn 
index (requires named index columns). Such index types are added to the 
returned "ID" sets which are consistently shuffled and partitioned as the 
train and test sets. If numpy array passed any ID columns from train set should
be included.

* testID_column: a string of the column title for the column from the
df_test set intended for use as a row identifier value (such as could be
sequential numbers for instance). The function defaults to False for
cases where the training set does not include an ID column. A user can 
also pass a list of string columns titles such as to carve out multiple
columns to be excluded from processing but consistently shuffled and 
partitioned. An integer column index or list of integer column indexes 
may also be passed such as if the source dataset was a numpy array. This
can also be passed as True (or False) when ID columns are same as automunge 
train set and will be automatically recognized.

* labelscolumn: default to _False_ indicates that a labels column is not 
included in the test set passed to postmunge. A user can either pass
_True_ or the string ID of the labels column, noting that it is a requirement
that the labels column header string must be consistent with that from
the original train set. An integer column index may also be passed such
as if the source dataset was a numpy array. A user should take care to set 
this parameter if they are passing data with labels. Note that True signals
presence of consistent labels column header as was passed to automunge(.).
Note that if a label column is included consistent with label column from
automunge(.) call it will be automatically applied as labelscolumn.

* pandasoutput: a selector for format of returned sets. Defaults to _True_
for returned pandas dataframes. If set to _True_ returns pandas dataframes
(note that index is not preserved, non-range indexes are extracted 
to the ID sets, and automunge(.) generates an application specific range 
integer index in ID sets corresponding to the order of rows as they were 
passed to function). If set to _False_ returns numpy arrays instead of dataframes.

* printstatus: user can pass _True/False_ indicating whether the function 
will print status of processing during operation. Defaults to True.

* inplace: defaults to False, when True the df_test passed to postmunge(.)
is overwritten with the returned test set. This reduces memory overhead.
For example, to take advantage with reduced memory overhead you could call postmunge(.) as:
```
test, test_ID, test_labels, \
postreports_dict = \
am.postmunge(postprocess_dict, df_test, inplace = False)
```

* dupl_rows: can be passed as _(True/False\)_ which indicates
if duplicate rows will be consolidated to single instance in returned sets. (In
other words, if same row included more than once, it will only be returned once.)
Defaults to False for not activated. True applies consolidation to test set. Note 
this is applied prior to TrainLabelFreqLevel if elected. As implemented this does 
not take into account duplicate rows in test data which have different labels, 
only one verison of features/label pair is returned.

* TrainLabelFreqLevel: a boolean identifier _(True/False)_ which indicates
if the TrainLabelFreqLevel method will be applied to oversample test
data associated with underrepresented labels. The method adds multiples
to test data rows for those labels with lower frequency resulting in
an (approximately) levelized frequency. This defaults to False. Note that
this feature may be applied to numerical label sets if the assigncat processing
applied to the set in automunge(.) had included aggregated bins, such
as for example 'exc3' for pass-through numeric with standard deviation bins,
or 'exc4' for pass-through numeric with powers of ten bins. Note this 
method requires the inclusion of a designated label column.

* featureeval: a boolean identifier _(True/False)_ to activate a feature
importance evaluation, comparable to one performed in automunge but based on the 
test set passed to postmunge. Defaults to False. The results are returned in the
postreports_dict object returned from postmunge as postreports_dict['featureimportance']. 
The results will also be printed out if printstatus is activated.  Note that sorted 
feature importance results are returned in postreports_dict['FS_sorted'], including 
columns sorted by metric and metric2. Relies on ML_cmnd parameters from original
automunge(.) call.

* driftreport: activates a drift report evaluation, in which the normalization 
parameters are recalculated for the columns of the test data passed to postmunge 
for comparison to the original normalization parameters derived from the corresponding 
columns of the automunge train data set. The results are returned in the
postreports_dict object returned from postmunge as postreports_dict['driftreport']. 
The results will also be printed out if printstatus is activated. Defaults to _False_, and:
  - _False_ means no postmunge drift assessment is performed
  - _True_ means an assessment is performed for both the source column and derived column 
  stats
  - _'efficient'_ means that a postmunge drift assessment is only performed on the source 
  columns (less information but much more energy efficient)
  - _'report_effic'_ means that the efficient assessment is performed and returned with 
  no processing of data
  - _'report_full'_ means that the full assessment is performed and returned with no 
  processing of data

* inversion: defaults to False, may be passed as one of {False, ‘test’, ‘labels’, 'denselabels', or a list}, 
where ‘test’ or ‘labels’ activate an inversion operation to recover, by a set of transformations 
mirroring the inversion of those applied in automunge(.), the form of test data or labels 
data to consistency with the source columns as were originally passed to automunge(.). When 
passed as a list, accepts list of source column or returned column headers for inversion target. 
'denselabels' is for label set inversion in which labels were prepared in multiple formats,
such as to recover the original form on each basis for comparison.
The inversion operation is supported by the optional process_dict entries ‘info_retention’ and 
‘inverseprocess’. Note that columns are only returned for those sets in which a path of 
inversion was available by processdict inverseprocess entries. Note that the path of 
inversion is prioritized to those returned sets with information retention and availability 
of inverseprocess functions. Note that both feature importance and Binary dimensionality 
reduction is supported, support is not expected for PCA. Note that recovery of label 
sets with label smoothing is supported. Note that during an inversion operation the 
postmunge function only considers the parameters postprocess_dict, df_test, inversion, 
pandasoutput, and/or printstatus. Note that in an inversion operation the 
postmunge(.) function returns three sets: a recovered set, a list of recovered columns, and 
a dictionary logging results of the path selection process.

Here is an example of a postmunge call with inversion.
```
df_invert, recovered_list, inversion_info_dict = \
am.postmunge(postprocess_dict, test_labels, inversion='labels', \
             pandasoutput=True, printstatus=True)
```

Here is an example of a process_dict entry with the optional inversion entries included, such 
as may be defined by user for custom functions and passed to automunge(.) in the processdict 
parameter:
```
process_dict.update({'mnmx' : {'dualprocess'    : self.process_mnmx, \
                               'singleprocess'  : None, \
                               'postprocess'    : self.postprocess_mnmx, \
                               'inverseprocess' : self.inverseprocess_mnmx, \
                               'info_retention' : True, \
                               'NArowtype'      : 'numeric', \
                               'MLinfilltype'   : 'numeric', \
                               'labelctgy'      : 'mnmx'}})
```

And here is an example of the convention for inverseprocess functions, such as may be passed 
to a process_dict entry:
```
  def inverseprocess_mnmx(self, df, categorylist, postprocess_dict):
    """
    #inverse transform corresponding to process_mnmx
    #assumes any relevant parameters were saved in normalization_dict
    #does not perform infill, assumes clean data
    """
    
    normkey = categorylist[0]
    
    minimum = \
    postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['minimum']
    maximum = \
    postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['maximum']
    maxminusmin = \
    postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['maxminusmin']
    
    inputcolumn = postprocess_dict['column_dict'][normkey]['inputcolumn']
    
    df[inputcolumn] = df[normkey] * maxminusmin + minimum
    
    return df, inputcolumn
```

* traindata: boolean _{True, False}_, defaults to False. Only inspected when a transformation
is called that treats train data different than test data (currently only relevant to 
DP family of transforms for noise injection to train sets or label smoothing transforms in smth family). When passed 
as True treats df_test as a train set for purposes of these specific transforms, otherwise
default of False treats df_test as a test set (which turns off noise injection for DP transforms).

* returnedsets: Can be passed as one of _{True, False, 'test_ID', 'test_labels', 'test_ID_labels'}_. 
Designates the composition of the sets returned
from a postmunge(.) call. Defaults to True for the full composition of five returned sets.
With other options postmunge(.) only returns a single set, where for False that set consists 
of the test set, or for the other options returns the test set concatenated with the ID, 
labels, or both. For example:

```
#in default of returnedsets=True, postmunge(.) returns five sets, such as this call:
test, test_ID, test_labels, \
postreports_dict = \
am.postmunge(postprocess_dict, df_test, returnedsets = True)

#for other returnedset options, postmunge(.) returns just a single set, the test set:
test = \
am.postmunge(postprocess_dict, df_test, returnedsets = False)

#Note that if you want to access the column labels for an appended ID or labels set,
#They can be accessed in the postprocess_dict under entries for 
postprocess_dict['finalcolumns_labels']
postprocess_dict['finalcolumns_trainID']
```

* shuffletrain: can be passed as one of _{True, False}_ which indicates if the rows in 
the returned sets will be (consistently) shuffled. This value defaults to False. 


## Default Transformations

When root categories of transformations are not assigned for a given column in
assigncat, automunge performs an evaluation of data properties to infer 
appropriate means of feature engineering and numerical encoding. The default
categories of transformations are as follows:
- nmbr: for numerical data, columns are treated with z-score normalization. If 
binstransform parameter was activated this will be supplemented by a collection
of bins indicating number of standard deviations from the mean.
- 1010: for categorical data excluding special cases described following, columns are 
subject to binarization encoding via '1010'. If the 
number of unique entries in the column exceeds the parameter 'numbercategoryheuristic'
(which defaults to 255), the encoding will instead be by hashing.
- bnry: for categorical data of <=2 unique values excluding infill (eg NaN), the 
column is encoded to 0/1. Note that numerical sets with <= 2 unique values in train
set default to bnry.
- hsh2: for categorical data, if the number of unique entries in the column exceeds 
the parameter 'numbercategoryheuristic' (which defaults to 255), the encoding will 
instead be by 'hsh2' which is an ordinal (integer) encoding based on hashing.
hsh2 is excluded from ML infill.
- hash: for all unique entry categoric sets (based on sets with >75% unique entries), 
the encoding will be by hash which extracts distinct words within entries returned in 
a set of columns with an integer hashing. hash is excluded from ML infill. Note that for edge
cases with large string entries resulting in too high dimensionality, the max_column_count
parameter can be passed to default_assignparam in assignparam to put a cap on returned column count.
- dat6: for time-series data, a set of derivations are performed returning
'year', 'mdsn', 'mdcs', 'hmss', 'hmsc', 'bshr', 'wkdy', 'hldy' (these are defined 
in next section)
- null: for columns without any valid values in training set (e.g. all NaN) column is deleted

For label sets, we use a distinct set of root categories under automation. These are in
some cases comparable to those listed above for training data, but differ in that the label 
sets will not include a returned 'NArw' (infill marker) even when parameter NArw_marker 
passed as True.
- lbnm: for numerical data, a label set is treated with an 'exc2' pass-through transform (without normalization).
- lbor: for categoric data of >2 unique values, a label set is treated with an 'ordl' ordinal encoding (alphabetical order of encodings).

Other label categories are available for assignment in assigncat, described below in the 
library of transforms section for label set encodings.

Note that if a user wishes to avoid the automated assignment of default transformations,
such as to leave those columns not specifically assigned to transformation categories in 
assigncat as unchanged, the powertransform parameter may be passed as values 'excl' or 
'exc2', where for 'excl' columns not explicitly assigned to a root category in assigncat 
will be left untouched, or for 'exc2' columns not explicitly assigned to a root category 
in assigncat will be forced to numeric and subject to default modeinfill. (These two excl
arguments may be useful if a user wants to experiment with specific transforms on a 
subset of the columns without incurring processing time of an entire set.)

Note that for columns designated for label sets as a special case categorical data will
default to 'ordl' (ordinal encoding) instead of '1010'. Also, numerical data will default
to 'excl2' (pass-through) instead of 'nmbr'.

- powertransform: if the powertransform parameter is activated, a statistical evaluation
will be performed on numerical sets to distinguish between columns to be subject to
bxcx, nmbr, or mnmx. Please note that we intend to further refine the specifics of this
process in future implementations. Additionally, powertransform may be passed as values 'excl' 
or 'exc2', where for 'excl' columns not explicitly assigned to a root category in assigncat 
will be left untouched, or for 'exc2' columns not explicitly assigned to a root category in 
assigncat will be forced to numeric and subject to default modeinfill. (These two excl 
arguments may be useful if a user wants to experiment with specific transforms on a subset of 
the columns without incurring processing time of an entire set for instance.)

- floatprecision: parameter indicates the precision of floats in returned sets (16/32/64)
such as for memory considerations.

In all cases, if the parameter NArw_marker is activated returned sets will be
supplemented with a NArw column indicating rows that were subject to infill. Each 
transformation category has a default infill approach detailed below.

Note that default transformations can be overwritten within an automunge(.) call by way
of passing custom transformdict family tree definitions which overwrite the family tree 
of the default root categories listed above. For instance, if a user wishes to process 
numerical columns with a default mean scaling ('mean') instead of z-score 
normalization ('nmbr'), the user may copy the transform_dict entries from the code-base 
for 'mean' root category and assign as a definition of the 'nmbr' root category, and then 
pass that defined transformdict in the automunge call. (Note that we don't need to 
overwrite the processdict for nmbr if we don't intend to overwrite it's use as an entry 
in other root category family trees. Also it's good practice to retain any downstream 
entries such as in case the default for nmbr is used as an entry in some other root 
category's family tree.) Here's a demonstration.

```
#create a transformdict that overwrites the root category definition of nmbr with mean:
#(assumes that we want to include NArw indicating presence of infill)
transformdict = {'nmbr' : {'parents' : [], \
                           'siblings': [], \
                           'auntsuncles' : ['mean'], \
                           'cousins' : ['NArw'], \
                           'children' : [], \
                           'niecesnephews' : [], \
                           'coworkers' : [], \
                           'friends' : []}}
                           
#And then we can simply pass this transformdict to an automunge(.) call.

train, train_ID, labels, \
val, val_ID, val_labels, \
test, test_ID, test_labels, \
postprocess_dict = \
am.automunge(df_train, \
             transformdict = transformdict)

```

Note if any of default transformation automation categories (nmbr/1010/ord3/text/bnry/dat6/null)
are overwritten in this fashion, a user can still assign original default categories to distinct
columns in assigncat by using corresponding alternates of (nmbd/101d/ordd/texd/bnrd/datd/nuld).

...

## Library of Transformations

### Library of Transformations Subheadings:
* [Intro](https://github.com/Automunge/AutoMunge/blob/master/README.md#intro)
* [Label Set Encodings](https://github.com/Automunge/AutoMunge/blob/master/README.md#label-set-encodings)
* [Numerical Set Normalizations](https://github.com/Automunge/AutoMunge/blob/master/README.md#numerical-set-normalizations)
* [Numerical Set Transformations](https://github.com/Automunge/AutoMunge/blob/master/README.md#numerical-set-transformations)
* [Numercial Set Bins and Grainings](https://github.com/Automunge/AutoMunge/blob/master/README.md#numercial-set-bins-and-grainings)
* [Sequential Numerical Set Transformations](https://github.com/Automunge/AutoMunge/blob/master/README.md#sequential-numerical-set-transformations)
* [Categorical Set Encodings](https://github.com/Automunge/AutoMunge/blob/master/README.md#categorical-set-encodings)
* [Date-Time Data Normalizations](https://github.com/Automunge/AutoMunge/blob/master/README.md#date-time-data-normalizations)
* [Date-Time Data Bins](https://github.com/Automunge/AutoMunge/blob/master/README.md#date-time-data-bins)
* [Differential Privacy Noise Injections](https://github.com/Automunge/AutoMunge/blob/master/README.md#differential-privacy-noise-injections)
* [Misc. Functions](https://github.com/Automunge/AutoMunge/blob/master/README.md#misc-functions)
* [Parsed Categoric Encodings](https://github.com/Automunge/AutoMunge/blob/master/README.md#Parsed-Categoric-Encodings)
* [More Efficient Parsed Categoric Encodings](https://github.com/Automunge/AutoMunge/blob/master/README.md#more-efficient-Parsed-Categoric-Encodings)
* [Multi-tier Parsed-Categoric-Encodings](https://github.com/Automunge/AutoMunge/blob/master/README.md#multi-tier-Parsed-Categoric-Encodings)
* [List of Root Categories](https://github.com/Automunge/AutoMunge/blob/master/README.md#list-of-root-categories)
* [List of Suffix Appenders](https://github.com/Automunge/AutoMunge/blob/master/README.md#list-of-suffix-appenders)
* [Other Reserved Strings](https://github.com/Automunge/AutoMunge/blob/master/README.md#other-reserved-strings)
* [Root Category Family Tree Definitions](https://github.com/Automunge/AutoMunge/blob/master/README.md#root-category-family-tree-definitions)
 ___ 
### Intro
Automunge has a built in library of transformations that can be passed for
specific columns with assigncat. (A column if left unassigned will defer to
the automated default methods to evaluate properties of the data to infer 
appropriate methods of numerical encoding.)  For example, a user can pass a 
min-max scaling method to a list of specific columns with headers 'column1',
'column2' with: 
```
assigncat = {'mnmx':['column1', 'column2']}
```
When a user assigns a column to a specific category, that category is treated
as the root category for the tree of transformations. Each key has an 
associated transformation function (where the root category transformation function 
is only applied if the root key is also found in the tree of family primitives). 
The tree of family primitives, as introduced earlier, applies first the keys found 
in upstream primitives i.e. parents/siblings/auntsuncles/cousins. If a transform 
is applied for a primitive that includes downstream offspring, such as parents/
siblings, then the family tree for that key with offspring is inspected to determine
downstream offspring categories, for example if we have a parents key of 'mnmx',
then any children/niecesnephews/coworkers/friends in the 'mnmx' family tree will
be applied as parents/siblings/auntsuncles/cousins, respectively. Note that the
designation for supplements/replaces refers purely to the question of whether the
column to which the transform is being applied is kept in place or removed.

Now we'll start here by listing again the family tree primitives for those root 
categories built into the automunge library. After that we'll give a quick 
narrative for each of the associated transformation functions. First here again
are the family tree primitives.

```
'parents' :           
upstream / first generation / replaces column / with offspring

'siblings':           
upstream / first generation / supplements column / with offspring

'auntsuncles' :       
upstream / first generation / replaces column / no offspring

'cousins' :           
upstream / first generation / supplements column / no offspring

'children' :          
downstream parents / offspring generations / replaces column / with offspring

'niecesnephews' :     
downstream siblings / offspring generations / supplements column / with offspring

'coworkers' :         
downstream auntsuncles / offspring generations / replaces column / no offspring

'friends' :           
downstream cousins / offspring generations / supplements column / no offspring
```

Here is a quick description of the transformation functions associated 
with each key which can either be assigned to a family tree primitive (or used 
as a root key). We're continuing to build out this library of transformations.
In some cases different transformation categories may be associated with the
same set of transformation functions, but may be distinguished by different
family tree aggregations of transformation category sets.

Note the design philosophy is that any transform can be applied to any type 
of data and if the data is not suited (such as applying a numeric transform
to a categorical set) the transform will just return all zeros. Note the 
default infill refers to the infill applied under 'standardinfill'. Note the
default NArowtype refers to the categories of data that won't be subject to 
infill.

### Label Set Encodings
Label set encodings are unique in that they don't include an aggregated NArw missing data markers
based on NArw_marker parameter. Missing data in label sets are subject to row deletions. Note that inversion of 
label set encodings is support by the postmunge(.) inversion parameter.
* lbnm: for numeric label sets, entries are given a pass-through transform via 'exc2' (the numeric default under automation)
* lbnb: for numeric label sets, entries are given a z-score normalization via 'nmbr'
* lbor: for categoric label sets, entries are given an ordinal encoding via 'ordl' (the categoric default under automation)
* lb10: for categoric label sets, entries are given a binary encoding via '1010'
* lbos: for categoric label sets, entries are given an ordinal encoding via 'ordl' followed by a conversion to
string by 'strg' (some ML libraries prefer string encoded labels to recognize the classification application)
* lbte: for categoric label sets, entries are given a one-hot encoding (this has some interpretabiltiy benefits over ordinal)
* lbbn: for categoric label sets with 2 unique values, entries are given a binarization via 'bnry'
* lbsm: for categoric encoding with smoothed labels (i.e. "label smoothing"), further described in smth transform below (accepts activation parameter for activation threshold)
* lbfs: for categoric encoding with fitted smoothed labels (i.e. fitted label smoothing), further described in fsmh transform below (accepts activation parameter for activation threshold)
* lbda: for date-time label sets, entries are encoded comparable to 'dat6' described further below

### Numerical Set Normalizations
* nmbr/nbr2/nbr3/nmdx/nmd2/nmd3: z-score normalization<br/>
(x - mean) / (standard deviation)
  - useful for: normalizing numeric sets of unknown distribution
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_nmbr'
  - assignparam parameters accepted:  
    - 'cap' and 'floor', default to False for no floor or cap, 
      True means floor/cap based on training set min/max, otherwise passed values serve as floor/cap to scaling, 
      noting that if cap<max then max reset to cap and if floor>min then min reset to floor
      cap and floor based on pre-transform values
    - 'muilitplier' and 'offset' to apply multiplier and offset to posttransform values, default to 1,0,
      note that multiplier is applied prior to offset
    - 'adjinfill', boolean, defaults to False, True makes default infill adjacent cell instead of mean imputation
  - driftreport postmunge metrics: mean / std / max / min
  - inversion available: yes with full recovery
* mean/mea2/mea3: mean normalization (like z-score in the numerator and min-max in the denominator)<br/>
(x - mean) / (max - min)
My intuition says z-score has some benefits but really up to the user which they prefer.
  - useful for: similar to z-score except data remains in fixed range
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_mean'
  - assignparam parameters accepted:
    - 'cap' and 'floor', default to False for no floor or cap, 
      True means floor/cap based on training set min/max, otherwise passed values serve as floor/cap to scaling, 
      noting that if cap<max then max reset to cap and if floor>min then min reset to floor
      cap and floor based on pre-transform values
    - 'muilitplier' and 'offset' to apply multiplier and offset to posttransform values, default to 1,0,
      note that multiplier is applied prior to offset
    - 'adjinfill', boolean, defaults to False, True makes default infill adjacent cell instead of mean imputation
  - driftreport postmunge metrics: minimum / maximum / mean / std
  - inversion available: yes with full recovery
* mnmx/mnm2/mnm5/mmdx/mmd2/mmd3: vanilla min-max scaling<br/>
(x - min) / (max - min)
  - useful for: normalizing numeric sets where all non-negative output is preferred
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_mnmx'
  - assignparam parameters accepted: 
    - 'cap' and 'floor', default to False for no floor or cap, 
      True means floor/cap based on training set min/max, otherwise passed values serve as floor/cap to scaling, 
      noting that if cap<max then max reset to cap and if floor>min then min reset to floor
      cap and floor based on pre-transform values
    - 'adjinfill', boolean, defaults to False, True makes default infill adjacent cell instead of mean imputation
  - driftreport postmunge metrics: minimum / maximum / maxminusmin / mean / std / cap / floor
  - inversion available: yes with full recovery
* mnm3/mnm4: min-max scaling with outliers capped at 0.01 and 0.99 quantiles
  - useful for: normalizing numeric sets where all non-negative output is preferred, and outliers capped
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_mnm3'
  - assignparam parameters accepted: 
    - qmax or qmin to change the quantiles from 0.99/0.01
    - 'adjinfill', boolean, defaults to False, True makes default infill adjacent cell instead of mean imputation
  - driftreport postmunge metrics: quantilemin / quantilemax / mean / std
  - inversion available: yes
* mnm6: min-max scaling with test floor set capped at min of train set (ensures
test set returned values >= 0, such as might be useful for kernel PCA for instance)
  - useful for: normalizing numeric sets where all non-negative output is preferred, guarantees nonnegative in postmunge
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_mnmx'
  - assignparam parameters accepted:
    - 'adjinfill', boolean, defaults to False, True makes default infill adjacent cell instead of mean imputation
  - driftreport postmunge metrics: minimum / maximum / mean / std
  - inversion available: yes
* retn: related to min/max scaling but retains +/- of values, based on conditions
if max>=0 and min<=0, x=x/(max-min), elif max>=0 and min>=0 x=(x-min)/(max-min),
elif max<=0 and min<=0 x=(x-max)/(max-min)
  - useful for: normalization with sign retention for iterpretability
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_retn'
  - assignparam parameters accepted:  
    - 'cap' and 'floor', default to False for no floor or cap, 
      True means floor/cap based on training set min/max, otherwise passed values serve as floor/cap to scaling, 
      noting that if cap<max then max reset to cap and if floor>min then min reset to floor
      cap and floor based on pre-transform values
    - 'muilitplier' and 'offset' to apply multiplier and offset to posttransform values, default to 1,0,
      note that multiplier is applied prior to offset
    - 'divisor' to select between default of 'minmax' or 'mad, 'std', where minmax means scaling by divisor of max-min
	std based on scaling by divisor of standard deviation and mad by median absolute deviation
    - 'adjinfill', boolean, defaults to False, True makes default infill adjacent cell instead of mean imputation
  - driftreport postmunge metrics: minimum / maximum / mean / std
  - inversion available: yes with full recovery
* rtbn: retain normalization supplemented by ordinal encoded standard deviation bins
* rtb2: retain normalization supplemented by one-hot encoded standard deviation bins
* MADn/MAD2: mean absolute deviation normalization, subtract set mean <br/>
(x - mean) / (mean absolute deviation)
  - useful for: normalizing sets with fat-tailed distribution
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_MADn'
  - assignparam parameters accepted:
    - 'adjinfill', boolean, defaults to False, True makes default infill adjacent cell instead of mean imputation
  - driftreport postmunge metrics: mean / MAD / maximum / minimum
  - inversion available: yes with full recovery
* MAD3: mean absolute deviation normalization, subtract set maximum<br/>
(x - maximum) / (mean absolute deviation)
  - useful for: normalizing sets with fat-tailed distribution
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_MAD3'
  - assignparam parameters accepted:
    - 'adjinfill', boolean, defaults to False, True makes default infill adjacent cell instead of mean imputation
  - driftreport postmunge metrics: mean / MAD / datamax / maximum / minimum
  - inversion available: yes with full recovery
* mxab: max absolute scaling normalization (just including this one for completeness, retn is a much better option to ensure consistent scaling between sets)<br/>
(x) / max absolute
  - useful for: normalizing sets by dividing by max, commonly used in some circles
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_mxab'
  - assignparam parameters accepted:
    - 'adjinfill', boolean, defaults to False, True makes default infill adjacent cell instead of mean imputation
  - driftreport postmunge metrics: minimum / maximum / maxabs / mean / std
  - inversion available: yes with full recovery
* lgnm: normalization intended for lognormal distributed numerical sets,
achieved by performing a logn transform upstream of a nmbr normalization.
  - useful for: normalizing sets within proximity of lognormal distribution
  - default infill: mean
  - default NArowtype: positivenumeric
  - suffix appender: '_logn_nmbr'
  - assignparam parameters accepted: can pass params to nmbr consistent with nmbr documentation above
  - driftreport postmunge metrics: consistent with both logn and nmbr
  - inversion available: yes with full recovery
### Numerical Set Transformations
* bxcx/bxc2/bxc3/bxc4/bxc5: performs Box-Cox power law transformation. Applies infill to 
values <= 0. Note we currently have a test for overflow in returned results and if found 
set to 0. Please note that this method makes use of scipy.stats.boxcox. Please refer to
family trees below for full set of transfomration categories asscoiated with these roots.
  - useful for: translates power law distributions to closer approximate gaussian
  - default infill: mean (i.e. mean of values > 0)
  - default NArowtype: positivenumeric
  - suffix appender: '_bxcx'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: trnsfrm_mean / bxcx_lmbda / bxcxerrorcorrect / mean
  - inversion available: no
* log0/log1: performs logarithmic transform (base 10). Applies infill to values <= 0.
  - useful for: sets with mixed range of large and small values
  - default infill: meanlog
  - default NArowtype: positivenumeric
  - suffix appender: '_log0'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: meanlog
  - inversion available: yes with full recovery
* logn: performs natural logarithmic transform (base e). Applies infill to values <= 0.
  - useful for: sets with mixed range of large and small values
  - default infill: meanlog
  - default NArowtype: positivenumeric
  - suffix appender: '_logn'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: meanlog
  - inversion available: yes with full recovery
* sqrt: performs square root transform. Applies infill to values < 0.
  - useful for: common mathematic transform
  - default infill: mean
  - default NArowtype: nonnegativenumeric
  - suffix appender: '_sqrt'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: meansqrt
  - inversion available: yes with full recovery
* addd: performs addition of an integer or float to a set
  - useful for: common mathematic transform
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_addd'
  - assignparam parameters accepted: 'add' for value added (default to 1)
  - driftreport postmunge metrics: mean, add
  - inversion available: yes with full recovery
* sbtr: performs subtraction of an integer or float to a set
  - useful for: common mathematic transform
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_sbtr'
  - assignparam parameters accepted: 'subtract' for value subtracted (default to 1)
  - driftreport postmunge metrics: mean, subtract
  - inversion available: yes with full recovery
* mltp: performs multiplication of an integer or float to a set
  - useful for: common mathematic transform
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_mltp'
  - assignparam parameters accepted: 'multiply' for value multiplied (default to 2)
  - driftreport postmunge metrics: mean, multiply
  - inversion available: yes with full recovery
* divd: performs division of an integer or float to a set
  - useful for: common mathematic transform
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_divd'
  - assignparam parameters accepted: 'divide' for value subtracted (default to 2)
  - driftreport postmunge metrics: mean, divide
  - inversion available: yes with full recovery
* rais: performs raising to a power of an integer or float to a set
  - useful for: common mathematic transform
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_rais'
  - assignparam parameters accepted: 'raiser' for value raised (default to 2)
  - driftreport postmunge metrics: mean, raiser
  - inversion available: yes with full recovery
* absl: performs absolute value transform to a set
  - useful for: common mathematic transform
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_absl'
  - assignparam parameters accepted: (none)
  - driftreport postmunge metrics: mean
  - inversion available: yes with partial recovery

Q Notation family of transforms return a multicolumn binary encoded set with registers for sign, integers, and fractionals.
Transforms accept parameters integer_bits / fractional_bits / sign_bit for register sizes, care should be taken for 
adequate registers to avoid overflow (overflow entries have values replaced with max or min capacity based on register sizes). 
Default register sizes were selected to accomodate z-score normalized data with +/-6 
standard deviations from mean and approximately 4 significant figures in decimals. For example, with default parameters an input column 'floats' will return columns: ['floats_qbt1_sign', 'floats_qbt1_2^2', 'floats_qbt1_2^1', 'floats_qbt1_2^0', 'floats_qbt1_2^-1', 'floats_qbt1_2^-2', 'floats_qbt1_2^-3', 'floats_qbt1_2^-4', 'floats_qbt1_2^-5', 'floats_qbt1_2^-6', 'floats_qbt1_2^-7', 'floats_qbt1_2^-8', 'floats_qbt1_2^-9', 'floats_qbt1_2^-10', 'floats_qbt1_2^-11', 'floats_qbt1_2^-12'].
* qbt1: binary encoded signed floats with registers for sign, integers, and fractionals, default overflow at +/- 8.000
  - useful for: feeding normalized floats to quantum circuits
  - default infill: zero
  - default NArowtype: numeric
  - suffix appender: '_qbt1_2^#' where # integer associated with register and also '_qbt1_sign'
  - assignparam parameters accepted: 
    - suffix: defaults to 'qbt1'
    - sign_bit: boolean defaults to True to include sign register
    - integer_bits: defaults to 3 for number of bits in register
    - fractional_bits: defaults to 12 for number of bits in register
  - driftreport postmunge metrics: maximum, minimum, mean, stdev
  - inversion available: yes with full recovery
* qbt2: binary encoded signed integers with registers for sign and integers, default overflow at +/-32,767
  - useful for: feeding floats to quantum circuits
  - default infill: zero
  - default NArowtype: numeric
  - suffix appender: '_qbt2_2^#' where # integer associated with register and also '_qbt2_sign'
  - assignparam parameters accepted: 
    - suffix: defaults to 'qbt2'
    - sign_bit: boolean defaults to True to include sign register
    - integer_bits: defaults to 15 for number of bits in register
    - fractional_bits: defaults to 0 for number of bits in register
  - driftreport postmunge metrics: maximum, minimum, mean, stdev
  - inversion available: yes with full recovery
* qbt3: binary encoded unsigned floats with registers for integers and fractionals, default overflow at 8.000 and <0
  - useful for: feeding unsigned normalized floats to quantum circuits
  - default infill: zero
  - default NArowtype: numeric
  - suffix appender: '_qbt3_2^#' where # integer associated with register
  - assignparam parameters accepted: 
    - suffix: defaults to 'qbt3'
    - sign_bit: boolean defaults to False, activate to include sign register
    - integer_bits: defaults to 3 for number of bits in register
    - fractional_bits: defaults to 13 for number of bits in register
  - driftreport postmunge metrics: maximum, minimum, mean, stdev
  - inversion available: yes with full recovery
* qbt4: binary encoded unsigned integers with registers for integers, default overflow at 65,535 and <0
  - useful for: feeding unsigned floats to quantum circuits
  - default infill: zero
  - default NArowtype: numeric
  - suffix appender: '_qbt4_2^#' where # integer associated with register
  - assignparam parameters accepted: 
    - suffix: defaults to 'qbt4'
    - sign_bit: boolean defaults to False, activate to include sign register
    - integer_bits: defaults to 16 for number of bits in register
    - fractional_bits: defaults to 0 for number of bits in register
  - driftreport postmunge metrics: maximum, minimum, mean, stdev
  - inversion available: yes with full recovery

Other Q Notation root categories:
- nmqb has upstream z score to qbt1 and z score not retained
- nmq2 has upstream z score to qbt1 and z score is retained
- mmqb has upstream min max to qbt3 and min max not retained
- mmq3 has upstream min max to qbt3 and min max is retained
- lgnr logarithmic number representation, registers 1 for sign, 1 for log sign, 4 log integer registers, 3 log fractional registers

### Numercial Set Bins and Grainings
* pwrs: bins groupings by powers of 10 (for values >0)
  - useful for: featuere engineering for linear models, also for oversampling bins with TrainFreqLevelizer parameter
  - default infill: no activation
  - default NArowtype: positivenumeric
  - suffix appender: '_10^#' where # is integer indicating target powers of 10 for column
  - assignparam parameters accepted: 
    - 'negvalues', boolean defaults to False, True bins values <0
      (recomend using pwr2 instead of this parameter since won't update NArowtype)
  - driftreport postmunge metrics: powerlabelsdict / meanlog / maxlog / 
	                           <column> + '_ratio' (column specific)
  - inversion available: yes with partial recovery
* pwr2: bins groupings by powers of 10 (comparable to pwrs with negvalues parameter activated for values >0 & <0)
  - useful for: featuere engineering for linear models, also for oversampling bins with TrainFreqLevelizer parameter
  - default infill: no activation
  - default NArowtype: nonzeronumeric
  - suffix appender: '\_10^#' or '\_-10^#' where # is integer indicating target powers of 10 for column
  - assignparam parameters accepted: 
    - 'negvalues', boolean defaults to True, True bins values <0
      (recomend using pwrs instead of this parameter since won't update NArowtype)
  - driftreport postmunge metrics: powerlabelsdict / labels_train / missing_cols / 
			           <column> + '_ratio' (column specific)
  - inversion available: yes with partial recovery
* pwor: for numerical sets, outputs an ordinal encoding indicating where a
value fell with respect to powers of 10
  - useful for: ordinal version of pwrs
  - default infill: zero
  - default NArowtype: positivenumeric
  - suffix appender: '_pwor'
  - assignparam parameters accepted: 
    - 'negvalues', boolean defaults to False, True bins values <0
  - driftreport postmunge metrics: meanlog / maxlog / ordl_activations_dict
  - inversion available: yes with partial recovery
* por2: for numerical sets, outputs an ordinal encoding indicating where a
value fell with respect to powers of 10 (comparable to pwor with negvalues parameter activated)
  - useful for: ordinal version of pwr2
  - default infill: zero (a distinct encoding)
  - default NArowtype: nonzeronumeric
  - suffix appender: '_por2'
  - assignparam parameters accepted: 
    - 'negvalues', boolean defaults to True, True bins values <0
  - driftreport postmunge metrics: train_replace_dict / test_replace_dict / ordl_activations_dict
  - inversion available: yes with partial recovery
* pwbn: comparable to pwor but followed by a binary encoding, such as may be useful for data with 
high variability
  - useful for: ordinal version of pwrs
  - default infill: zero (a distinct encoding)
  - default NArowtype: nonzeronumeric
  - suffix appender: '_pwor_1010_#' (where # is integer for binary encoding activation number) 
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: train_replace_dict / test_replace_dict / ordl_activations_dict
  - inversion available: yes with partial recovery
* por3: comparable to por2 but followed by a binary encoding, such as may be useful for data with 
high variability
  - useful for: ordinal version of pwr2
  - default infill: zero (a distinct encoding)
  - default NArowtype: nonzeronumeric
  - suffix appender: '_por2_1010_#' (where # is integer for binary encoding activation number) 
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: train_replace_dict / test_replace_dict / ordl_activations_dict
  - inversion available: yes with partial recovery
* bins: for numerical sets, outputs a set of 6 columns indicating where a
value fell with respect to number of standard deviations from the mean of the
set (i.e. integer suffix represent # from mean as <-2:0, -2-1:1, -10:2, 01:3, 12:4, >2:5)
Note this can be activated to supplment numeric sets with binstransform automunge parameter.
  - useful for: featuere engineering for linear models, also for oversampling bins with TrainFreqLevelizer parameter
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '\_bins\_#' where # is integer identifier of bin
  - assignparam parameters accepted: 
    - bincount integer for number of bins, defaults to 6
  - driftreport postmunge metrics: binsmean / binsstd / <column> + '_ratio' (column specific)
  - inversion available: yes with partial recovery
* bsor: for numerical sets, outputs an ordinal encoding indicating where a
value fell with respect to number of standard deviations from the mean of the
set (i.e. integer encoding represent # from mean as <-2:0, -2-1:1, -10:2, 01:3, 12:4, >2:5)
  - useful for: ordinal version of bins
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_bsor'
  - assignparam parameters accepted: 
    - bincount as integer for # of bins (defaults to 6)
  - driftreport postmunge metrics: ordinal_dict / ordl_activations_dict / binsmean / binsstd
  - inversion available: yes with partial recovery
* bnwd/bnwK/bnwM: for numerical set graining to fixed width bins for one-hot encoded bins 
(columns without activations in train set excluded in train and test data). 
bins default to width of 1/1000/1000000 eg for bnwd/bnwK/bnwM
  - useful for: bins for sets with known recurring demarkations
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '\_bnwd\_#1\_#2' where #1 is the width and #2 is the bin identifier (# from min)
                     and 'bnwd' as bnwK or bnwM based on variant
  - assignparam parameters accepted: 
    - 'width' to set bin width
  - driftreport postmunge metrics: binsmean / bn_min / bn_max / bn_delta / bn_count / bins_id / 
			           bins_cuts / bn_width_bnwd (or bnwK/bnwM) / textcolumns / 
                                   <column> + '_ratio' (column specific)
  - inversion available: yes with partial recovery
* bnwo/bnKo/bnMo: for numerical set graining to fixed width bins for ordinal encoded bins 
(integers without train set activations still included in test set). 
bins default to width of 1/1000/1000000 eg for bnwd/bnwK/bnwM
  - useful for: ordinal version of preceding
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_bnwo' (or '_bnKo', '_bnMo')
  - assignparam parameters accepted: 
    - 'width' to set bin width
  - driftreport postmunge metrics: binsmean / bn_min / bn_max / bn_delta / bn_count / bins_id / 
			           bins_cuts / bn_width / ordl_activations_dict
  - inversion available: yes with partial recovery
* bnep/bne7/bne9: for numerical set graining to equal population bins for one-hot encoded bins. 
bin count defaults to 5/7/9 eg for bnep/bne7/bne9
  - useful for: bins for sets with unknown demarkations
  - default infill: no activation
  - default NArowtype: numeric
  - suffix appender: '\_bnep\_#1' where #1 is the bin identifier (# from min) (or bne7/bne9 instead of bnep)
  - assignparam parameters accepted: 'bincount' to set number of bins
  - driftreport postmunge metrics: binsmean / bn_min / bn_max / bn_delta / bn_count / bins_id / 
                                   bins_cuts / bincount_bnep (or bne7/bne9) / textcolumns / 
                                   <column> + '_ratio' (column specific)
  - inversion available: yes with partial recovery
* bneo/bn7o/bn9o: for numerical set graining to equal population bins for ordinal encoded bins. 
bin count defaults to 5/7/9 eg for bneo/bn7o/bn9o
  - useful for: ordinal version of preceding
  - default infill: adjacent cell
  - default NArowtype: numeric
  - suffix appender: '\_bneo' (or bn7o/bn9o)
  - assignparam parameters accepted: 
    - 'bincount' to set number of bins
  - driftreport postmunge metrics: binsmean / bn_min / bn_max / bn_delta / bn_count / bins_id / 
			           bins_cuts / bincount / ordl_activations_dict
  - inversion available: yes with partial recovery
* bkt1: for numerical set graining to user specified encoded bins. First and last bins unconstrained.
  - useful for: bins for sets with known irregular demarkations
  - default infill: no activation
  - default NArowtype: numeric
  - suffix appender: '\_bkt1\_#1' where #1 is the bin identifier (# from min)
  - assignparam parameters accepted: 
    - 'buckets', a list of numbers, to set bucket boundaries (leave out +/-'inf')
      defaults to [0,1,2] (arbitrary plug values), can also pass buckets values as percent of range by framing as a set instead of list eg {0,0.25,0.50,1}
  - driftreport postmunge metrics: binsmean / buckets_bkt1 / bins_cuts / bins_id / textcolumns / 
					   <column> + '_ratio' (column specific)
  - inversion available: yes with partial recovery
* bkt2: for numerical set graining to user specified encoded bins. First and last bins bounded.
  - useful for: bins for sets with known irregular demarkations, similar to preceding but first and last bins bounded
  - default infill: no activation
  - default NArowtype: numeric
  - suffix appender: '\_bkt2\_#1' where #1 is the bin identifier (# from min)
  - assignparam parameters accepted: 
    - 'buckets', a list of numbers, to set bucket boundaries
      defaults to [0,1,2] (arbitrary plug values), can also pass buckets values as percent of range by framing as a set instead of list eg {0,0.25,0.50,1}
  - driftreport postmunge metrics: binsmean / buckets_bkt2 / bins_cuts / bins_id / textcolumns / 
					   <column> + '_ratio' (column specific)
  - inversion available: yes with partial recovery
* bkt3: for numerical set graining to user specified ordinal encoded bins. First and last bins unconstrained.
  - useful for: ordinal version of bkt1
  - default infill: unique activation
  - default NArowtype: numeric
  - suffix appender: '_bkt3'
  - assignparam parameters accepted: 
    - 'buckets', a list of numbers, to set bucket boundaries (leave out +/-'inf')
      defaults to [0,1,2] (arbitrary plug values), can also pass buckets values as percent of range by framing as a set instead of list eg {0,0.25,0.50,1}
  - driftreport postmunge metrics: binsmean / buckets / bins_cuts / bins_id / ordl_activations_dict
  - inversion available: yes with partial recovery
* bkt4: for numerical set graining to user specified ordinal encoded bins. First and last bins bounded.
  - useful for: ordinal version of bkt2
  - default infill: unique activation
  - default NArowtype: numeric
  - suffix appender: '_bkt4'
  - assignparam parameters accepted: 
    - 'buckets', a list of numbers, to set bucket boundaries
      defaults to [0,1,2] (arbitrary plug values), can also pass buckets values as percent of range by framing as a set instead of list eg {0,0.25,0.50,1}
  - driftreport postmunge metrics: binsmean / buckets / bins_cuts / bins_id / ordl_activations_dict
  - inversion available: yes with partial recovery
* note that bins each have variants for one-hot vs ordinal vs binary encodings
one-hot : bkt1, bkt2, bins, bnwd, bnwK, bnwM, bnep, bne7, bne9, pwrs, pwr2
ordinal : bkt3, bkt4, bsor, bnwo, bnKo, bnMo, bneo, bn7o, bn9o, pwor, por2
binary  : bkb3, bkb4, bsbn, bnwb, bnKb, bnMb, bneb, bn7b, bn9b, pwbn, por3
* tlbn: returns equal population bins in separate columns with activations replaced by min-max scaled 
values within that segment's range (between 0-1) and other values subject to an infill of -1 
(intended for use to evaluate feature importance of different segments of a numerical set's distribution
with metric2 results from a feature importance evaluation)
  - useful for: evaluating relative feature importance between different segments of a numeric set distribution
  - default infill: no activation (this is the recommended infill for this transform)
  - default NArowtype: numeric
  - suffix appender: '\_tlbn\_#' where # is the bin identifier,  and max# is right tail / min# is left tail
  - assignparam parameters accepted: 
    - 'bincount' to set number of bins (defaults to 9)
  - driftreport postmunge metrics: binsmean / bn_min / bn_max / bn_delta / bn_count / bins_id / 
			           bins_cuts / bincount_tlbn / textcolumns / <column> + '_ratio' (column specific)
  - inversion available: no
### Sequential Numerical Set Transformations
Please note that sequential transforms assume the forward progression of time towards direction of bottom of dataframe.
Please note that only stdrdinfill (adjinfill) are supported for shft transforms.
* dxdt/d2dt/d3dt/d4dt/d5dt/d6dt: rate of change (row value minus value in preceding row), high orders 
return lower orders (eg d2dt returns original set, dxdt, and d2dt), all returned sets include 'retn' 
normalization which scales data with min/max while retaining +/- sign
  - useful for: time series data, also bounding sequential sets
  - default infill: adjacent cells
  - default NArowtype: numeric
  - suffix appender: '_dxdt'
  - assignparam parameters accepted: 
    - 'periods' sets number of time steps offset to evaluate, defaults to 1
  - driftreport postmunge metrics: positiveratio / negativeratio / zeroratio / minimum / maximum / mean / std
  - inversion available: no
* dxd2/d2d2/d3d2/d4d2/d5d2/d6d2: denoised rate of change (average of last two or more rows minus average
of preceding two or more rows), high orders return lower orders (eg d2d2 returns original set, dxd2, 
and d2d2), all returned sets include 'retn' normalization
  - useful for: time series data, also bounding sequential sets
  - default infill: adjacent cells
  - default NArowtype: numeric
  - suffix appender: '_dxd2'
  - assignparam parameters accepted: 
    - 'periods' sets number of time steps offset to evaluate, defaults to 2
  - driftreport postmunge metrics: positiveratio / negativeratio / zeroratio / minimum / maximum / mean / std
  - inversion available: no
* nmdx/nmd2/nmd3/nmd4/nmd5/nmd6: comparable to dxdt but includes upstream of sequential transforms a 
nmrc numeric string parsing top extract numbers from string sets
* mmdx/mmd2/mmd3/mmd4/mmd5/mmd6: comparable to dxdt but uses z-score normalizaitons via 'nbr2' instead of 'retn'
* dddt/ddd2/ddd3/ddd4/ddd5/ddd6: comparable to dxdt but no normalizations applied
* dedt/ded2/ded3/ded4/ded5/ded6: comparable to dxd2 but no normalizations applied
  - inversion available: no
* shft/shf2/shf3: shifted data forward by a period number of time steps defaulting to 1/2/3. Note that NArw aggregation
not supported for shift transforms, infill only available as adjacent cell
  - useful for: time series data, carrying prior time steps forward
  - default infill: adjacent cells
  - default NArowtype: numeric
  - suffix appender: '_shft' / '_shf2' / '_shf3'
  - assignparam parameters accepted: 
    - 'periods' sets number of time steps offset to evaluate, defaults to 1/2/3
    - 'suffix' sets the suffix appender of returned column
      as may be useful to disginguish if applying this multiple times
  - driftreport postmunge metrics: positiveratio / negativeratio / zeroratio / minimum / maximum / mean / std
  - inversion available: yes
### Categorical Set Encodings
* bnry: converts sets with two values to boolean identifiers. Defaults to assigning
1 to most common value and 0 to second most common, unless 1 or 0 is already included
in most common of the set then defaults to maintaining those designations. If applied 
to set with >2 entries applies infill to those entries beyond two most common. 
  - useful for: binarizing sets with two unique values (differs from 1010 in that distinct encoding isn't registered for missing data to return single column)
  - default infill: most common value
  - default NArowtype: justNaN
  - suffix appender: '_bnry'
  - assignparam parameters accepted:
    - 'adjinfill', boolean, defaults to False, True makes default infill adjacent cell
    - 'str_convert', boolean defaults as False for distinct encodings between numbers and string equivalents
      e.g. 2 != '2', when passed as True e.g. 2 == '2'
  - driftreport postmunge metrics: missing / 1 / 0 / extravalues / oneratio / zeroratio
  - inversion available: yes with full recovery
* bnr2: (Same as bnry except for default infill.)
  - useful for: similar to bnry preceding but with different default infill
  - default infill: least common value
  - default NArowtype: justNaN
  - suffix appender: '_bnr2'
  - assignparam parameters accepted:
    - 'adjinfill', boolean, defaults to False, True makes default infill adjacent cell
    - 'str_convert', boolean defaults as False for distinct encodings between numbers and string equivalents
      e.g. 2 != '2', when passed as True e.g. 2 == '2'
  - driftreport postmunge metrics: missing / 1 / 0 / extravalues / oneratio / zeroratio
  - inversion available: yes with full recovery
* text/txt2: converts categorical sets to one-hot encoded set of boolean identifiers 
(consistently encodings numbers and numerical string equivalents due to column labeling convention, e.g. 12 == '12')
  - useful for: one hot encoding, returns distinct column activation per unique entry
  - default infill: all entries zero
  - default NArowtype: justNaN
  - suffix appender: '_(category)' where category is the categoric entry target of column activations
  - assignparam parameters accepted:
    - 'adjinfill', boolean, defaults to False, True makes default infill adjacent cell
  - driftreport postmunge metrics: textlabelsdict_text / <column> + '_ratio' (column specific)
  - inversion available: yes with full recovery
* onht: converts categorical sets to one-hot encoded set of boolean identifiers 
(like text but different convention for returned column headers and distinct encodings for numbers and numerical string equivalents)
  - useful for: similar to text transform preceding but with numbered column header convention
  - default infill: all entries zero
  - default NArowtype: justNaN
  - suffix appender: '_onht\_#' where # integer corresponds to the target entry of a column
  - assignparam parameters accepted:
    - 'adjinfill', boolean, defaults to False, True makes default infill adjacent cell
    - 'str_convert', boolean defaults as False for distinct encodings between numbers and string equivalents
      e.g. 2 != '2', when passed as True e.g. 2 == '2'
  - driftreport postmunge metrics: textlabelsdict_text / <column> + '_ratio' (column specific)
			           text_categorylist is key between columns and target entries
  - inversion available: yes with full recovery
* ordl/ord2/ord5: converts categoric sets to ordinal integer encoded set, encodings sorted alphabetically
  - useful for: categoric sets with high cardinality where one-hot or binarization may result in high dimensionality. Also default for categoric labels.
  - default infill: plug value 'zzzinfill'
  - default NArowtype: justNaN
  - suffix appender: '_ordl'
  - assignparam parameters accepted:
    - 'adjinfill', boolean, defaults to False, True makes default infill adjacent cell
    - 'ordered_overide', boolean defaults True, when True instects for Pandas ordered categorical and 
      if found integer encoding order defers to that basis
    - 'str_convert', boolean defaults as False for distinct encodings between numbers and string equivalents
      e.g. 2 != '2', when passed as True e.g. 2 == '2'
  - driftreport postmunge metrics: ordinal_dict / ordinal_overlap_replace / ordinal_activations_dict
  - inversion available: yes with full recovery
* ord3: converts categoric sets to ordinal integer encoded set, sorted first by frequency of category 
occurrence, second basis for common count entries is alphabetical
  - useful for: similar to ordl preceding but activations are sorted by entry frequency instead of alphabetical
  - default infill: plug value 'zzzinfill'
  - default NArowtype: justNaN
  - suffix appender: '_ord3'
  - assignparam parameters accepted:
    - 'adjinfill', boolean, defaults to False, True makes default infill adjacent cell
    - 'ordered_overide', boolean defaults True, when True instects for Pandas ordered categorical and 
      if found integer encoding order defers to that basis
    - 'str_convert', boolean defaults as False for distinct encodings between numbers and string equivalents
      e.g. 2 != '2', when passed as True e.g. 2 == '2'
  - driftreport postmunge metrics: ordinal_dict / ordinal_overlap_replace / ordinal_activations_dict
  - inversion available: yes with full recovery
* ord4: derived by an ord3 transform followed by a mnmx transform. Useful as a scaled metric
(numeric in range 0-1) which ranks any redundant entries by frequency of occurance.
* lbos: an ord3 encoding followed by downstream conversion to string dtype. This may be useful for
label sets passed to downstream libraries to ensure they treat labels as target for classification instead
of regression.
* 1010: converts categorical sets of >2 unique values to binary encoding (more memory 
efficient than one-hot encoding)
  - useful for: our default categoric encoding for sets with number of entries below numbercategoryheustic (defaulting to 255)
  - default infill: plug value 'zzzinfill'
  - default NArowtype: justNaN
  - suffix appender: '\_1010\_#' where # is integer indicating order of 1010 columns
  - assignparam parameters accepted:
    - 'adjinfill', boolean, defaults to False, True makes default infill adjacent cell
    - 'str_convert', boolean defaults as False for distinct encodings between numbers and string equivalents
      e.g. 2 != '2', when passed as True e.g. 2 == '2'
  - driftreport postmunge metrics: _1010_binary_encoding_dict / _1010_overlap_replace / 
	                           _1010_binary_column_count / _1010_activations_dict
  (for example if 1010 encoded to three columns based on number of categories <8,
  it would return three columns with suffix appenders 1010_1, 1010_2, 1010_3)
  - inversion available: yes with full recovery
* maxb / matx / ma10: categoric encodings that allow user to cap the number activations in the set. 
maxb (ordinal), matx (one hot), and ma10 (binary). 
  - useful for: categoric sets where some outlier entries may not occur with enough frequency for training purposes
  - default infill: plug value 'zzzinfill'
  - default NArowtype: justNaN
  - suffix appender: '\_maxb'
  - assignparam parameters accepted:
    - 'maxbincount': set a maximum number of activations (integer) default False
    - 'minentrycount': set a minimum number of entries in train set to register an activation (integer) default False
    - 'minentryratio': set a minimum ratio of entries in train set to register an activation (float between 0-1)
  - driftreport postmunge metrics: new_maxactivation / consolidation_count
  - inversion available: yes with full recovery
* ucct: converts categorical sets to a normalized float of unique class count,
for example, a 10 row train set with two instances of 'circle' would replace 'circle' with 0.2
and comparable to test set independent of test set row count
  - useful for: supplmenting categoric sets with a proxy for activation frequency
  - default infill: ratio of infill in train set
  - default NArowtype: justNaN
  - suffix appender: '_ucct'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: ordinal_dict / ordinal_overlap_replace / ordinal_activations_dict
  - inversion available: no
* lngt, lnlg: returns string length of categoric entries (lngt followed by min/max, lnlg by log)
  - useful for: supplementing categoric sets with a proxy for information content (based on string length)
  - default infill: plug value of 3 (based on len(str(np.nan)) )
  - default NArowtype: justNaN
  - suffix appender: '_lngt'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: maximum, minimum, mean, std
  - inversion available: no
* aggt: consolidate categoric entries based on user passed aggregate parameter
  - useful for: performing upstream of categoric encoding when some entries are redundant
  - default infill: none
  - default NArowtype: justNaN
  - suffix appender: '_aggt'
  - assignparam parameters accepted: 
    - 'aggregate' as a list or as a list of lists of aggregation sets
  - driftreport postmunge metrics: aggregate
  - inversion available: yes with partial recovery
* smth: applies a one-hot encoding followed by a label smoothing operation to reduce activation value and increase null value. The smoothing is applied to train data but not validation or test data. Smoothing can be applied to test data in postmunge(.) by activating the traindata parameter.
  - useful for: label smoothing, speculate there may be benefit for categoric encodings with noisy entries of some error rate
  - default infill: none
  - default NArowtype: justNaN
  - suffix appender: '\_smth\_#' where # is integer
  - assignparam parameters accepted: 
    - 'activation' defaults to 0.9, a float between 0.5-1 to designate activation value
    - 'LSfit' defaults to False, when True applies fitted label smoothing (consistent with fsmh)
    - 'testsmooth' defaults to False, when True applies smoothing to test data in both automunge and postmunge
  - driftreport postmunge metrics: comparable to onht
  - inversion available: yes with full recovery
* fsmh: comparable to smth but applies by default a fitted label smoothing, in which null values are fit to ratio of activations corresponding to current activation. The smoothing is applied to train data but not validation or test data. Smoothing can be applied to test data in postmunge(.) by activating the traindata parameter.
* hash: applies "the hashing trick" to convert high cardinality categoric sets to set of columns with integer word encodings
e.g. for an entry "Three word quote" may return three columns with integers corresponding to each of three words
where integer is determined by hashing, and also based on passed parameter vocab_size.
Note that hash strips out special characters. Uhsh is available if upstream uppercase conversion desired. Note that there is a possibility
of encoding overlap between entries with this transform. Also note that hash is excluded from ML infill
vocab_size calculated based on number of unique words found in train set times a multiplier (defaulting to 2), where if that 
is greater than cap then reverts to cap. The hashing transforms are intended as an alternative to other categoric 
encodings which doesn't require a conversion dictionary assmebly for consistent processing of subsequent data, as 
may benefit sets with high cardinatility (i.e. high number of unique entries). The tradeoff is that inversion
is not supported as there is possibility of redundant encodings for different unique entries.
  - useful for: categoric sets with very high cardinality, default for categoric sets with (nearly) all unique entries
  - default infill: none
  - default NArowtype: justNaN
  - suffix appender: '\_hash\_#'
  - assignparam parameters accepted:
    - 'heuristic_multiplier', float defaults to 2
    - 'heuristic_cap', integer defaults to 1024
    - 'vocab_size', integer defaults to False, when assigned overrides heuristic
    - 'space', defaults to ' ', this is used to extract words by space seperator
    - 'excluded_characters', defaults to [',', '.', '?', '!', '(', ')'], these characters are stripped prior to enconding
    - 'salt', arbitrary string, defaults to empty string '', appended to entries to perturb encoding basis for privacy
    - 'hash_alg', defaults to 'hash' for use of native python hash function for speed, 'md5' uses hashlib md5 function instead
    - 'max_column_count', defaults to False, can pass as integer to cap the number of returned columns, in which case when
      words are extracted the final entrty within cap will contain all remaining word and space characters
  - driftreport postmunge metrics: col_count (number of columns), vocab_size
  - inversion available: no
* hsh2: similar to hash but does not partition entries by space seperator, so only returns one column. Note this version doesn't scrub special characters prior to encoding.
  - useful for: categoric sets with very high cardinality, default for categoric sets with number of entries exceeding numbercategoryheuristic (defaulting to 255)
  - default infill: none
  - default NArowtype: justNaN
  - suffix appender: '_hash'
  - assignparam parameters accepted: 
    - 'heuristic_multiplier', float defaults to 2
    - 'heuristic_cap', integer defaults to 1024
    - 'vocab_size', integer defaults to False, when assigned overrides heuristic
    - 'excluded_characters', a list of strings, defaults to [] (an empty set), these characters are stripped prior to enconding
    - 'salt', arbitrary string, defaults to empty string '', appended to entries to perturb encoding basis for privacy
    - 'hash_alg', defaults to 'hash' for use of native python hash function for speed, 'md5' uses hashlib md5 function instead
  - driftreport postmunge metrics: col_count (number of columns), vocab_size
  - inversion available: no
* hs10: similar to hsh2 but returns activations in a set of columns with binary encodings, similar to 1010
  - useful for: binary version of hsh2
  - default infill: none
  - default NArowtype: justNaN
  - suffix appender: '\_hs10\_#'
  - assignparam parameters accepted: 
    - 'heuristic_multiplier', float defaults to 2
    - 'heuristic_cap', integer defaults to 1024
    - 'vocab_size', integer defaults to False, when assigned overrides heuristic
    - 'excluded_characters', a list of strings, defaults to [] (an empty set), these characters are stripped prior to enconding
    - 'salt', arbitrary string, defaults to empty string '', appended to entries to perturb encoding basis for privacy
    - 'hash_alg', defaults to 'hash' for use of native python hash function for speed, 'md5' uses hashlib md5 function instead
  - driftreport postmunge metrics: col_count (number of columns), vocab_size
  - inversion available: no
* UPCS: convert string entries to all uppercase characters
  - useful for: performing upstream of categoric encodings when case configuration is irelevant
  - default infill: none
  - default NArowtype: justNaN
  - suffix appender: '_UPCS'
  - assignparam parameters accepted: 
    - 'activate', boolean defaults to True, False makes this a passthrough without conversion
  - driftreport postmunge metrics: activate
  - inversion available: yes with partial recovery
* new processing functions Unht / Utxt / Utx2 / Utx3 / Uord / Uor2 / Uor3 / Uor6 / U101 / Ucct / Uhsh / Uhs2 / Uh10
  - comparable to functions onht / text / txt2 / txt3 / ordl / ord2 / ord3 / ors6 / 1010 / ucct / hash / hsh2 / hs10
  - but upstream conversion of all strings to uppercase characters prior to encoding
  - (e.g. 'USA' and 'usa' would be consistently encoded)
  - default infill: in uppercase conversion NaN's are assigned distinct encoding 'NAN'
  - and may be assigned other infill methods in assigninfill
  - default NArowtype: 'justNaN'
  - suffix appender: '_UPCS'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: comparable to functions text / txt2 / txt3 / ordl / ord2 / ord3 / ors6 / 1010 / ucct
  - inversion available: yes
* ntgr/ntg2/ntg3: sets of transformations intended for application to integer sets of unknown interpretation
(such as may be continuous variables, discrete relational variables, or categoric). The ntgr family encodes
in multiple forms appropriate for each of these different types, such as to allow the ML training to identify
which is most useful. Reference the family trees below for composition details (can do a control-F search for ntgr etc).
  - useful for: encoding integer sets of unknown interpretation
  - default NArowtype: 'integer'
  - ntgr set includes: ord4, retn, 1010, ordl
  - ntg2 set includes: ord4, retn, 1010, ordl, pwr2
  - ntg3 set includes: ord4, retn, ordl, por2
### Date-Time Data Normalizations
Date time processing transforms are implementations of two master functions: time and tmcs, which accept
various parameters associated with suffix, time scale, and sin/cos periodicity, etc. They segment time stamps by
time scale returned in seperate columns. If a particular time scale is not present in training data it is omitted.
* date/dat2: for datetime formatted data, segregates data by time scale to multiple
columns (year/month/day/hour/minute/second) and then performs z-score normalization
  - useful for: datetime entries of mixed time scales where periodicity is not relevant
  - default infill: mean
  - default NArowtype: datetime
  - suffix appender: includes appenders for (_year, _mnth, _days, _hour, _mint, _scnd)
  - assignparam parameters accepted:
    - timezone: defaults to False as passthrough, otherwise can pass time zone abbreviation 
      (useful to consolidate different time zones such as for bus hr bins)
      for list of pandas accepted abbreviations see pytz.all_timezones
  - driftreport postmunge metrics: meanyear / stdyear / meanmonth / stdmonth / meanday / stdday / 
			           meanhour / stdhour / meanmint / stdmint / meanscnd / stdscnd
  - inversion available: pending
* year/mnth/days/hour/mint/scnd: segregated by time scale and z-score normalization
  - useful for: datetime entries of single time scale where periodicity is not relevant
  - default infill: mean
  - default NArowtype: datetime
  - suffix appender: includes appenders for (_year, _mnth, _days, _hour, _mint, _scnd)
  - driftreport postmunge metrics: timemean / timemax / timemin / timestd
  - inversion available: pending
* mnsn/mncs/dysn/dycs/hrsn/hrcs/misn/mics/scsn/sccs: segregated by time scale and 
dual columns with sin and cos transformations for time scale period (eg 12 months, 24 hrs, 7 days, etc)
  - useful for: datetime entries of single time scale where periodicity is relevant
  - default infill: mean
  - default NArowtype: datetime
  - suffix appender: includes appenders for (mnsn/mncs/dysn/dycs/hrsn/hrcs/misn/mics/scsn/sccs)
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: timemean / timemax / timemin / timestd
  - inversion available: pending
* mdsn/mdcs: similar sin/cos treatment, but for combined month/day, note that periodicity is based on 
number of days in specific months, including account for leap year, with 12 month periodicity
  - useful for: datetime entries of single time scale combining months and days where periodicity is relevant
  - default infill: mean
  - default NArowtype: datetime
  - suffix appender: includes appenders for (mdsn/mdcs)
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: timemean / timemax / timemin / timestd
  - inversion available: pending
* dhms/dhmc: similar sin/cos treatment, but for combined day/hour/min, with 7 day periodicity

  - default infill: mean
  - default NArowtype: datetime
  - suffix appender: includes appenders for (dhms/dhmc)
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: timemean / timemax / timemin / timestd
  - inversion available: pending
* hmss/hmsc: similar sin/cos treatment, but for combined hour/minute/second, with 24 hour periodicity
  - useful for: datetime entries of single time scale combining time scales where periodicity is relevant
  - default infill: mean
  - default NArowtype: datetime
  - suffix appender: includes appenders for (hmss/hmsc)
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: timemean / timemax / timemin / timestd
  - inversion available: pending
* mssn/mscs: similar sin/cos treatment, but for combined minute/second, with 1 hour periodicity
  - useful for: datetime entries of single time scale combining time scales below minute threshold where periodicity is relevant
  - default infill: mean
  - default NArowtype: datetime
  - suffix appender: includes appenders for (hmss/hmsc)
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: timemean / timemax / timemin / timestd
  - inversion available: pending
* dat6: default transformation set for time series data, returns:
'year', 'mdsn', 'mdcs', 'hmss', 'hmsc', 'bshr', 'wkdy', 'hldy'
  - useful for: datetime entries of multiple time scales where periodicity is relevant, default date-time encoding, includes bins for holidays, business hours, and weekdays
  - default infill: mean
  - default NArowtype: datetime
  - suffix appender: includes appenders for ('year', 'mdsn', 'mdcs', 'hmss', 'hmsc', 'bshr', 'wkdy', 'hldy')
  - assignparam parameters accepted:
    - timezone: defaults to False as passthrough, otherwise can pass time zone abbreviation 
      (useful to consolidate different time zones such as for bus hr bins)
      for list of pandas accepted abbreviations see pytz.all_timezones
  - driftreport postmunge metrics: meanyear / stdyear / mean_mdsn / mean_mdcs / mean_hmss / mean_hmsc
  - inversion available: pending
### Date-Time Data Bins
* wkdy: boolean identifier indicating whether a datetime object is a weekday
  - useful for: supplementing datetime encodings with weekday bins
  - default infill: none
  - default NArowtype: datetime
  - suffix appender: '_wkdy'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: activationratio
  - inversion available: pending
* wkds/wkdo: encoded weekdays 0-6, 'wkds' for one-hot via 'text', 'wkdo' for ordinal via 'ord3'
  - useful for: ordinal version of preceding wkdy
  - default infill: 7 (eg eight days a week)
  - default NArowtype: datetime
  - suffix appender: '_wkds'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: mon_ratio / tue_ratio / wed_ratio / thr_ratio / fri_ratio / sat_ratio / 
	  sun_ratio / infill_ratio
  - inversion available: pending
* mnts/mnto: encoded months 1-12, 'mnts' for one-hot via 'text', 'mnto' for ordinal via 'ord3'
  - useful for: supplementing datetime encodings with month bins
  - default infill: 0
  - default NArowtype: datetime
  - suffix appender: '_mnts'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: infill_ratio / jan_ratio / feb_ratio / mar_ratio / apr_ratio / may_ratio / 
	  jun_ratio / jul_ratio / aug_ratio / sep_ratio / oct_ratio / nov_ratio / dec_ratio
  - inversion available: pending
* bshr: boolean identifier indicating whether a datetime object falls within business
hours (9-5, time zone unaware)
  - useful for: supplementing datetime encodings with business hour bins
  - default infill: datetime
  - default NArowtype: justNaN
  - suffix appender: '_bshr'
  - assignparam parameters accepted: 
    - 'start' and 'end', which default to 9 and 17
  - driftreport postmunge metrics: activationratio
  - inversion available: pending
* hldy: boolean identifier indicating whether a datetime object is a US Federal
holiday
  - useful for: supplementing datetime encodings with holiday bins
  - default infill: none
  - default NArowtype: datetime
  - suffix appender: '_hldy'
  - assignparam parameters accepted: 
    - 'holiday_list', should be passed as a list of strings of dates of additional holidays to be recognized 
      e.g. ['2020/03/30']
  - driftreport postmunge metrics: activationratio
  - inversion available: pending
### Differential Privacy Noise Injections
The DP family of transforms are special in that they treat train sets different than test sets, 
specifically they apply a noise injection to train sets such as may benefit differential privacy.
Note that if desired to treat data passed to postmunge as a train set can apply the traindata
parameter to postmunge. Note that when passing parameters to these functions, the transformation
category associated with the transformation function may be different than the root category.
* DPnb: applies a z-score normalization followed by a noise injection to train data sampled
from a Gaussian which defaults to 0 mu and 0.06 sigma, but only to a subset of the data based
on flip_prob parameter.
  - useful for: noise injection for data augmentation, model perturbation for ensembles, differential privacy
  - default infill: the DP function does not apply a default infill assume upstream nmbr cleans data
  - default NArowtype: numeric
  - suffix appender: '_nmbr_DPnb'
  - assignparam parameters accepted: 
    - 'noisedistribution' as {'normal', 'laplace'}, defaults to normal
    - 'flip_prob' for percent of entries recieving noise injection, defaults to 0.03
    - 'mu' for noise mean, defaults to 0
    - 'sigma' for noise scale, defaults to 0.06
    - 'testnoise' defaults to False, when True noise is injected to test data in both automunge and postmunge by default
    - parameters should be passed to 'DPnb' transformation category from family tree
  - driftreport postmunge metrics: mu, sigma for DPnm, upstream z score via nmbr for others
  - inversion available: yes
* DPmm: applies a min-max scaling followed by a noise injection to train data sampled
from a Gaussian which defaults to 0 mu and 0.03 sigma. Note that noise is scaled to ensure output
remains in range 0-1 (by scaling neg noise when input <0.5 and scaling pos noise when input >0.5)
  - useful for: noise injection for data augmentation, model perturbation for ensembles, differential privacy
  - default infill: the DP function does not apply a default infill assume upstream mnmx cleans data
  - default NArowtype: numeric
  - suffix appender: '_mnmx_DPmm'
  - assignparam parameters accepted: 
    - 'noisedistribution' as {'normal', 'laplace'}, defaults to normal
    - 'flip_prob' for percent of entries recieving noise injection, defaults to 0.03
    - 'mu' for noise mean, defaults to 0
    - 'sigma' for noise scale, defaults to 0.03
    - 'testnoise' defaults to False, when True noise is injected to test data in both automunge and postmunge by default
    - parameters should be passed to 'DPmm' transformation category from family tree
  - driftreport postmunge metrics: mu, sigma for DPnm, upstream minmax via mnmx for others
  - inversion available: yes
* DPrt: applies a retn normalization with a noise injection to train data sampled
from a Gaussian which defaults to 0 mu and 0.03 sigma. Note that noise is scaled to ensure output
remains in range 0-1 (by scaling neg noise when input <0.5 and scaling pos noise when input >0.5)
  - useful for: noise injection for data augmentation, model perturbation for ensembles, differential privacy
  - default infill: comparable to retn with mean (calculated before noise injection)
  - suffix appender: '_DPrt'
  - assignparam parameters accepted: 
    - parameters comparable to retn divisor / offset / multiplier / 
    - cap / floor defaulting to 'minmax'/0/1/False/False, also
    - 'adjinfill' to change default infill from mean imputation to adjacent cell
    - 'noisedistribution' as {'normal', 'laplace'}, defaults to normal
    - 'mu' for noise mean, defaults to 0, 
    - 'sigma' for noise scale, defautls to 0.03
    - 'flip_prob' for percent of entries recieving noise injection, defaults to 0.03
    - 'testnoise' defaults to False, when True noise is injected to test data in both automunge and postmunge by default
    - Parameters should be passed to 'DPrt' transformation category from family tree.
  - driftreport postmunge metrics: mu, sigma, flip_prob for DPrt, also metrics comparable to retn
  - inversion available: yes
* DLmm/DLnb/DLrt: comparable to DPmm/DPnb/DPrt but applies laplace distributed noise instead of gaussian
with same parameters accepted (where mu is center of noise, sigma is scale, and flip-prob is ratio)
and with same default parameter values
* DPbn: applies a two value binary encoding (bnry) followed by a noise injection to train data which
flips the activation per parameter flip_prob which defaults to 0.03
  - useful for: noise injection for data augmentation, model perturbation for ensembles, differential privacy
  - default infill: the DP function does not apply a default infill assume upstream bnry cleans data
  - default NArowtype: justNaN
  - suffix appender: '_bnry_DPbn'
  - assignparam parameters accepted: 
    - 'flip_prob' for percent of activation flips (defaults to 0.03), 
    - 'testnoise' defaults to False, when True noise is injected to test data in both automunge and postmunge by default
    - parameters should be passed to 'DPbn' transformation category from family tree
  - driftreport postmunge metrics: flip_prob for DPbn, upstream binary via bnry for others
  - inversion available: yes
* DPod: applies an ordinal encoding (ord3) followed by a noise injection to train data which
flips the activations per parameter flip_prob which defaults to 0.03 to a random draw from the
set of activations (including the current activation so actual flip percent is < flip_prob based
on number of activations)
  - useful for: noise injection for data augmentation, model perturbation for ensembles, differential privacy
  - default infill: the DP function does not apply a default infill assume upstream ord3 cleans data
  - default NArowtype: justNaN
  - suffix appender: '_ord3_DPod'
  - assignparam parameters accepted: 
    - 'flip_prob' for percent of activation flips (defaults to 0.03), 
    - 'testnoise' defaults to False, when True noise is injected to test data in both automunge and postmunge by default
    - parameters should be passed to 'DPod' transformation category from family tree
  - driftreport postmunge metrics: flip_prob for DPod, upstream ordinal via ord3 for others
  - inversion available: yes
* DPoh: applies an ordinal encoding (ord3) followed by a noise injection to train data which
flips the activations per parameter flip_prob which defaults to 0.03 to a random draw from the
set of activations (including the current activation so actual flip percent is < flip_prob based
on number of activations), followed by a one-hot encoding
  - useful for: noise injection for data augmentation, model perturbation for ensembles, differential privacy
  - default infill: the DP function does not apply a default infill assume upstream ord3 cleans data
  - default NArowtype: justNaN
  - suffix appender: '\_ord3\_DPod\_onht\_#' where # is integer for each categoric entry
  - assignparam parameters accepted: 
    - 'flip_prob' for percent of activation flips (defaults to 0.03), 
    - 'testnoise' defaults to False, when True noise is injected to test data in both automunge and postmunge by default
    - parameters should be passed to 'DPo2' transformation category from family tree
  - driftreport postmunge metrics: flip_prob for DPod, upstream ordinal via ord3 and downstream onht for others
  - inversion available: yes
* DP10: applies an ordinal encoding (ord3) followed by a noise injection to train data which
flips the activations per parameter flip_prob which defaults to 0.03 to a random draw from the
set of activations (including the current activation so actual flip percent is < flip_prob based
on number of activations), followed by a 1010 binary encoding
  - useful for: noise injection for data augmentation, model perturbation for ensembles, differential privacy
  - default infill: the DP function does not apply a default infill assume upstream ord3 cleans data
  - default NArowtype: justNaN
  - suffix appender: '\_ord3\_DPod\_1010\_#' where # is integer for each column which collectively encode categoric entries
  - assignparam parameters accepted: 
    - 'flip_prob' for percent of activation flips (defaults to 0.03), 
    - 'testnoise' defaults to False, when True noise is injected to test data in both automunge and postmunge by default
    - parameters should be passed to 'DPo3' transformation category from family tree
  - driftreport postmunge metrics: flip_prob for DPod, upstream ordinal via ord3 and downstream 1010 for others
  - inversion available: yes
### Misc. Functions
* null: deletes source column
  - default infill: none
  - default NArowtype: exclude
  - no suffix appender, column deleted
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: none
  - inversion available: no
* excl: passes source column un-altered, no transforms or infill. (Note that returned data may not be 
numeric and predictive methods like ML infill and feature selection may not work for that scenario.)
Note that for assignnan designation of infill designations, excl is excluded from 'global' assignments
(although may still be assigned explicitly under assignnan columns or categories entries).
  - useful for: passthrough sets
  - default infill: none
  - default NArowtype: exclude
  - suffix appender: None or '_excl' (dependant on automunge(.) excl_suffix parameter)
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: none
  - inversion available: yes
* exc2/exc3/exc4: passes source column unaltered other than force to numeric, mode infill applied
(exc3 and exc4 have downstream standard deviation or power of 10 bins aggregated such as may be beneficial
when applying TrainLabelFreqLevel to a numeric label set)
  - useful for: passthrough sets where all numeric entries desired, exc3 and exc4 useful for oversampling with numeric labels by TrainFreqLevelizer
  - default infill: mode
  - default NArowtype: numeric
  - suffix appender: '_exc2'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: none
  - inversion available: yes
* exc5: passes source column unaltered other than force to numeric, mode infill applied for non-integers
  - useful for: passthrough sets where all numeric entries desired
  - default infill: mode
  - default NArowtype: integer
  - suffix appender: '_exc5'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: none
  - inversion available: yes
* eval: performs data property evaluation consistent with default automation to designated column
  - useful for: applying automated evaluation to distinct columns for cases where default automated evaluation turned off by powertransform='excl'
  - default infill: based on evaluation
  - default NArowtype: based on evaluation
  - suffix appender: based on evaluation
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: none
  - inversion available: contingent on result
* ptfm: performs distribution property evaluation consistent with the automunge powertransform 
parameter activated to designated column
  - useful for: applying automated powertransform evaluation to distinct columns
  - default infill: based on evaluation
  - default NArowtype: based on evaluation
  - suffix appender: based on evlauation
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: none
  - inversion available: contingent on result				  
* copy: create new copy of column, useful when applying the same transform to same column more
than once with different parameters. Does not prepare column for ML on it's own.
  - default infill: exclude
  - default NArowtype: exclude
  - suffix appender: '_copy'
  - assignparam parameters accepted: 'suffix' for custom suffix appender
  - driftreport postmunge metrics: none
  - inversion available: yes
* shfl: shuffles the values of a column based on passed randomseed (Note that returned data may not 
be numeric and predictive methods like ML infill and feature selection may not work for that scenario
unless an additional transform is applied downstream.)
  - useful for: shuffle useful to negate feature from influencing inference
  - default infill: exclude
  - default NArowtype: justNAN
  - suffix appender: '_shfl'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: none
  - inversion available: no
* NArw: produces a column of boolean identifiers for rows in the source
column with missing or improperly formatted values. Note that when NArw
is assigned in a family tree it bases NArowtype on the root category, 
when NArw is passed as the root category it bases NArowtype on default.
  - useful for: supplmenting any transform with marker for missing entries. On by default by NArw_marker parameter
  - default infill: not applicable
  - default NArowtype: justNaN
  - suffix appender: '_NArw'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: pct_NArw
  - inversion available: no
* NAr2: produces a column of boolean identifiers for rows in the source
column with missing or improperly formatted values.
  - useful for: similar to NArw but different default NArwtype for when used as a root category
  - default infill: not applicable
  - default NArowtype: numeric
  - suffix appender: '_NArw'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: pct_NArw
  - inversion available: no
* NAr3: produces a column of boolean identifiers for rows in the source
column with missing or improperly formatted values.
  - useful for: similar to NArw but different default NArwtype for when used as a root category
  - default infill: not applicable
  - default NArowtype: positivenumeric
  - suffix appender: '_NArw'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: pct_NArw
  - inversion available: no
* NAr4: produces a column of boolean identifiers for rows in the source
column with missing or improperly formatted values.
  - useful for: similar to NArw but different default NArwtype for when used as a root category
  - default infill: not applicable
  - default NArowtype: nonnegativenumeric
  - suffix appender: '_NArw'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: pct_NArw
  - inversion available: no
* NAr5: produces a column of boolean identifiers for rows in the source
column with missing or improperly formatted values.
  - useful for: similar to NArw but different default NArwtype for when used as a root category
  - default infill: not applicable
  - default NArowtype: integer
  - suffix appender: '_NArw'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: pct_NArw
  - inversion available: no
### Parsed Categoric Encodings
Please note I recommend caution on using splt/spl2/spl5/spl6 transforms on categorical
sets that may include scientific units for instance, as prefixes will not be noted
for overlaps, e.g. this wouldn't distinguish between kilometer and meter for instance.
Note that overlap lengths below 5 characters are ignored unless that value is overridden
by passing 'minsplit' parameter through assignparam.
* splt: searches categorical sets for overlaps between string character subsets and returns new boolean column
for identified overlap categories. Note this treats numeric values as strings eg 1.3 = '1.3'.
Note that priority is given to overlaps of higher length, and by default overlap go down to 5 character length.
  - useful for: extracting grammatical strucutre shared between entries
  - default infill: none
  - default NArowtype: justNaN
  - suffix appender: '\_splt\_##*##' where ##*## is target identified string overlap 
  - assignparam parameters accepted: 
    - 'minsplit': indicating lowest character length for recognized overlaps 
    - 'space_and_punctuation': True/False, defaults to True, when passed as
      False character overlaps are not recorded which include space or punctuation
      based on characters in excluded_characters parameter
    - 'excluded_characters': a list of strings which are excluded from overlap 
      identification when space_and_punctuation set as False, defaults to
      `[' ', ',', '.', '?', '!', '(', ')']`
    - 'concurrent_activations': defaults as False, True makes comaprable to sp15, 
      although recomend using sp15 instead for correct MLinfilltype
    - 'suffix': returned column suffix appender, defaults to '_splt'
    - 'int_headers': True/False, defaults as False, when True returned column headers 
      are encoded with integers, such as for privacy preserving of data contents
    - 'test_same_as_train': defaults False, True makes this comparable to spl8
  - driftreport postmunge metrics: overlap_dict / splt_newcolumns_splt / minsplit
  - inversion available: yes with partial recovery
* sp15: similar to splt, but allows concurrent activations for multiple detected overlaps (spelled sp-fifteen)
Note that this version runs risk of high dimensionality of returned data in comparison to splt.
  - useful for: extracting grammatical strucutre shared between entries with increased information retention vs splt
  - default infill: none
  - default NArowtype: justNaN
  - suffix appender: '\_sp15\_##*##' where ##*## is target identified string overlap 
  - assignparam parameters accepted: 
    - comparable to splt, with concurrent_activations as True
  - driftreport postmunge metrics: overlap_dict / splt_newcolumns_sp15 / minsplit
  - inversion available: yes with partial recovery
* sp19: comaprable to sp15, but with returned columns aggregated by a binary encoding to reduce dimensionality
  - useful for: extracting grammatical strucutre shared between entries with decreased dimensionality vs sp15
  - default infill: distinct encoding
  - default NArowtype: justNaN
  - suffix appender: '\_sp19\_#' where # is integer associated with the encoding
  - assignparam parameters accepted: comparable to sp15
  - driftreport postmunge metrics: comparable to sp15 with addition of _1010_activations_dict for activation ratios
  - inversion available: yes with partial recovery
* sbst: similar to sp15, but only detects string overlaps shared between full unique entries and subsets of longer character length entries
  - useful for: exgtracting cases of overlap between full entries and subsets of other entries
  - default infill: none
  - default NArowtype: justNaN
  - suffix appender: '\_sbst\_##*##' where ##*## is target identified string overlap 
  - assignparam parameters accepted: 
    - 'int_headers': True/False, defaults as False, when True returned column headers 
      are encoded with integers, such as for privacy preserving of data contents
    - 'minsplit': indicating lowest character length for recognized overlaps, defaults to 1
    - 'concurrent_activations':  True/False, defaults to True, when True
      entries may have activations for multiple simultaneous overlaps
    - 'test_same_as_train': defaults False, True makes this comaprable to sbs2
    - 'suffix': returned column suffix appender, defaults to '_sbst'
  - driftreport postmunge metrics: overlap_dict / splt_newcolumns_sbst / minsplit
  - inversion available: yes with partial recovery
* sbs3: comaprable to sbst, but with returned columns aggregated by a binary encoding to reduce dimensionality
  - useful for: binary version of sbst for reduced dimensionality
  - default infill: distinct encoding
  - default NArowtype: justNaN
  - suffix appender: '\_sbs3\_#' where # is integer associated with the encoding
  - assignparam parameters accepted: comparable to sbst
  - driftreport postmunge metrics: comparable to sbst with addition of _1010_activations_dict for activation ratios
  - inversion available: yes with partial recovery
* spl2/ors2/ors6/txt3: similar to splt, but instead of creating new column identifier it replaces categorical 
entries with the abbreviated string overlap
  - useful for: similar to splt but returns single column, used in aggregations like or19
  - default infill: none
  - default NArowtype: justNaN
  - suffix appender: '_spl2'
  - assignparam parameters accepted: 
    - 'minsplit': indicating lowest character length for recognized overlaps 
    - 'space_and_punctuation': True/False, defaults to True, when passed as
      False character overlaps are not recorded which include space or punctuation
      based on characters in excluded_characters parameter
    - 'excluded_characters': a list of strings which are excluded from overlap 
      identification when space_and_punctuation set as False, defaults to
      `[' ', ',', '.', '?', '!', '(', ')']`
    - 'test_same_as_train': defaults False, True makes this comaprable to spl9
    - 'suffix': returned column suffix appender, defaults to '_spl2'
    - 'consolidate_nonoverlaps': defaults to False, True makes this comparble to spl5
  - driftreport postmunge metrics: overlap_dict / spl2_newcolumns / spl2_overlap_dict / spl2_test_overlap_dict / 
                                   minsplit
  - inversion available: yes with partial recovery
* spl5/spl6/ors5: similar to spl2, but those entries without identified string overlap are set to 0,
(used in ors5 in conjunction with ord3)
  - useful for: final tier of spl2 aggregations such as in or19
  - default infill: none
  - default NArowtype: justNaN
  - suffix appender: '_spl5'
  - assignparam parameters accepted:
    - comaprable to spl2, consolidate_nonoverlaps as True
  - driftreport postmunge metrics: overlap_dict / spl2_newcolumns / spl2_overlap_dict / spl2_test_overlap_dict / 
                                   spl5_zero_dict / minsplit
  - inversion available: yes with partial recovery
* spl6: similar to spl5, but with a splt performed downstream for identification of overlaps
within the overlaps
  - useful for: just a variation on parsing aggregations
  - default infill: none
  - default NArowtype: justNaN
  - suffix appender: '_spl5'
  - assignparam parameters accepted:
    - comparable to spl2
  - driftreport postmunge metrics: overlap_dict / spl2_newcolumns / spl2_overlap_dict / spl2_test_overlap_dict / 
                                   spl5_zero_dict / minsplit
  - inversion available: yes with partial recovery
* spl7: similar to spl5, but recognizes string character overlaps down to minimum 2 instead of 5
  - useful for: just a variation on parsing aggregations
  - default infill: none
  - default NArowtype: justNaN
  - suffix appender: '_spl5'
  - assignparam parameters accepted: 
    - comparble to spl5, minsplit defaults to 2
  - driftreport postmunge metrics: overlap_dict / srch_newcolumns_srch / search
  - inversion available: yes with partial recovery
* srch: searches categorical sets for overlaps with user passed search string and returns new boolean column
for identified overlap entries.
  - useful for: identifying specific entry character subsets by search
  - default infill: none
  - default NArowtype: justNaN
  - suffix appender: '\_srch\_##*##' where ##*## is target identified search string
  - assignparam parameters accepted: 
    - 'search': a list of strings, defaults as empty set
      (note search parameter list can included embedded lists of terms for 
      aggregated activations of terms in the sublist)
    - 'case': bool to indicate case sensitivity of search, defaults True
  - driftreport postmunge metrics: overlap_dict / splt_newcolumns_splt / minsplit
  - inversion available: yes with partial recovery
* src2: comparable to srch but expected to be more efficient when target set has narrow range of entries
  - useful for: similar to srch slight variation on implementation
  - default infill: none
  - default NArowtype: justNaN
  - suffix appender: '\_src2_##*##' where ##*## is target identified search string
  - assignparam parameters accepted: 
    - 'search': a list of strings, defaults as empty set
      (note search parameter list can included embedded lists of terms for 
      aggregated activations of terms in the sublist)
  - driftreport postmunge metrics: overlap_dict / splt_newcolumns_splt / minsplit
  - inversion available: yes with partial recovery
* src3: comparable to src2 with additional support for test set entries not found in train set
* src4: searches categorical sets for overlaps with user passed search string and returns ordinal column
for identified overlap entries. (Note for multiple activations encoding priority given to end of list entries).
  - useful for: ordinal version of srch
  - default infill: none
  - default NArowtype: justNaN
  - suffix appender: '\_src4'
  - assignparam parameters accepted: 
    - 'search': a list of strings, defaults as empty set
      (note search parameter list can included embedded lists of terms for 
      aggregated activations of terms in the sublist)
    - 'case': bool to indicate case sensitivity of search, defaults True
  - driftreport postmunge metrics: overlap_dict / splt_newcolumns_splt / minsplit
  - inversion available: yes with partial recovery
* nmrc/nmr2/nmr3: parses strings and returns any number groupings, prioritized by longest length
  - useful for: extracting numeric character subsets of entries
  - default infill: mean
  - default NArowtype: parsenumeric
  - suffix appender: '_nmrc'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: overlap_dict / mean / maximum / minimum
  - inversion available: yes with full recovery
* nmcm/nmc2/nmc3: similar to nmrc, but recognizes numbers with commas, returns numbers stripped of commas
  - useful for: extracting numeric character subsets of entries, recognizes commas
  - default infill: mean
  - default NArowtype: parsenumeric
  - suffix appender: '_nmcm'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: overlap_dict / mean / maximum / minimum
  - inversion available: yes with full recovery
* nmEU/nmE2/nmE3: similar to nmcm, but recognizes numbers with period or space thousands deliminator and comma decimal
  - useful for: extracting numeric character subsets of entries, recognizes EU format
  - default infill: mean
  - default NArowtype: parsenumeric
  - suffix appender: '_nmEU'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: overlap_dict / mean / maximum / minimum
  - inversion available: yes with full recovery
* strn: parses strings and returns any non-number groupings, prioritized by longest length
  - useful for: extracting nonnumeric character subsets of entries
  - default infill: 'zzzinfill'
  - default NArowtype: justNaN
  - suffix appender: '_strn'
  - assignparam parameters accepted: none
  - driftreport postmunge metrics: overlap_dict
  - inversion available: pending
### More Efficient Parsed Categoric Encodings
* new processing functions nmr4/nmr5/nmr6/nmc4/nmc5/nmc6/nmE4/nmE5/nmE6/spl8/spl9/sp10 (spelled sp"ten")/sp16/src2/sbs2/sp20/sbs4:
  - comparable to functions nmrc/nmr2/nmr3/nmcm/nmc2/nmc3/nmEU/nmE2/nmE3/splt/spl2/spl5/sp15/srch/sbst/sp19/sbs3
  - but make use of new assumption that set of unique values in test set is same or a subset of those values 
    from the train set, which allows for a more efficient application (no more string parsing of test sets)
  - default infill: comparable
  - default NArowtype: comparable
  - suffix appender: same format, updated per the new category
  - assignparam parameters accepted: comparable
  - driftreport postmunge metrics: comparable
  - inversion available: yes
* new processing functions nmr7/nmr8/nmr9/nmc7/nmc8/nmc9/nmE7/nmE8/nmE9:
  - comparable to functions nmrc/nmr2/nmr3/nmcm/nmc2/nmc3/nmEU/nmE2/nmE3
  - but implements string parsing only for unique test set entries not found in train set
  - for more efficient test set processing in automunge and postmunge
  - (less efficient than nmr4/nmc4 etc but captures outlier points as may not be unusual in continuous distributions)
  - default infill: comparable
  - default NArowtype: comparable
  - suffix appender: same format, updated per the new category
  - assignparam parameters accepted: comparable
  - driftreport postmunge metrics: overlap_dict / mean / maximum / minimum / unique_list / maxlength
  - inversion available: no
### Multi-tier Parsed Categoric Encodings
The following are a few variations of parsed categoric encoding aggregations. We recomend the or19 variant and 
have written about in paper [Parsed Categoric Encodings with Automunge](https://medium.com/automunge/string-theory-acbd208eb8ca).
* new processing root categories or11 / or12 / or13 / or14 / or15 / or16 / or17 / or18 / or19 / or20
  - or11 / or13 intended for categorical sets that may include multiple tiers of overlaps 
  and include base binary encoding via 1010 supplemented by tiers of string parsing for 
  overlaps using spl2 and spl5, or11 has two tiers of overlap string parsing, or13 has three, 
  each parsing returned with an ordinal encoding sorted by frequency (ord3)
  - or12 / or14 are comparable to or11 / or13 but include an additional supplemental 
  transform of string parsing for numerical entries with nmrc followed by a z-score normalization 
  of returned numbers via nmbr
  - or15 / or16 / or17 / or18 comparable to or11 / or12 / or13 / or14 but incorporate an
  UPCS transform upstream and make use of spl9/sp10 instead of spl2/spl5 for assumption that
  set of unique values in test set is same or subset of train set for more efficient postmunge
  - or19 / or20 comparable to or16 / or18 but replace the 'nmrc' string parsing for numeric entries
  with nmc8 which allows comma characters in numbers and makes use of consistent assumption to
  spl9/sp10 that set of unique values in test set is same or subset of train for efficient postmunge
  - or21 / or22 comparable to or19 / or20 but use spl2/spl5 instead of spl9/sp10, 
  which allows string parsing to handle test set entries not found in the train set
  - assignparam parameters accepted: 'minsplit': indicating lowest character length for recognized overlaps 
  (note that parameter has to be assigned to specific categories such as spl2/spl5 etc), also other parameters
associated with constituent functions
  - driftreport postmunge metrics: comparable to constituent functions
  - inversion available: yes with full recovery

 ___ 
### List of Root Categories
Here are those root categories presented again in a concise sorted list, intended as reference so user can
avoid unintentional duplication.
- '1010',
- '101d',
- 'DP10',
- 'DPb2',
- 'DPbn',
- 'DPm2',
- 'DPmm',
- 'DPn2',
- 'DPn3',
- 'DPnb',
- 'DPnm',
- 'DPo2',
- 'DPo3',
- 'DPo4',
- 'DPo5',
- 'DPo6',
- 'DPod',
- 'DPoh',
- 'DPrt',
- 'MAD2',
- 'MAD3',
- 'MADn',
- 'NAr2',
- 'NAr3',
- 'NAr4',
- 'NAr5',
- 'NArw',
- 'U101',
- 'Ucct',
- 'Uh10',
- 'Uhs2',
- 'Uhsh',
- 'UPCS',
- 'Unht',
- 'Uor2',
- 'Uor3',
- 'Uor6',
- 'Uord',
- 'Utx2',
- 'Utx3',
- 'Utxt',
- 'absl',
- 'addd',
- 'aggt',
- 'bins',
- 'bkb3',
- 'bkb4',
- 'bkt1',
- 'bkt2',
- 'bkt3',
- 'bkt4',
- 'bn7b',
- 'bn7o',
- 'bn9b',
- 'bn9o',
- 'bnKo',
- 'bnMo',
- 'bne7',
- 'bne9',
- 'bneb',
- 'bneo',
- 'bnep',
- 'bnKb',
- 'bnMb',
- 'bnr2',
- 'bnrd',
- 'bnry',
- 'bnwb',
- 'bnwK',
- 'bnwM',
- 'bnwd',
- 'bnwo',
- 'bsbn',
- 'bshr',
- 'bsor',
- 'bxc2',
- 'bxc3',
- 'bxc4',
- 'bxc5',
- 'bxcx',
- 'copy',
- 'd2d2',
- 'd2dt',
- 'd3d2',
- 'd3dt',
- 'd4d2',
- 'd4dt',
- 'd5d2',
- 'd5dt',
- 'd6d2',
- 'd6dt',
- 'dat2',
- 'dat3',
- 'dat4',
- 'dat5',
- 'dat6',
- 'datd',
- 'date',
- 'day2',
- 'day3',
- 'day4',
- 'day5',
- 'days',
- 'ddd2',
- 'ddd3',
- 'ddd4',
- 'ddd5',
- 'ddd6',
- 'dddt',
- 'ded2',
- 'ded3',
- 'ded4',
- 'ded5',
- 'ded6',
- 'dedt',
- 'dhmc',
- 'dhms',
- 'divd',
- 'dxd2',
- 'dxdt',
- 'dycs',
- 'dysn',
- 'exc2',
- 'exc3',
- 'exc4',
- 'exc5',
- 'excl',
- 'fsmh',
- 'hash',
- 'hldy',
- 'hmsc',
- 'hmss',
- 'hour',
- 'hrcs',
- 'hrs2',
- 'hrs3',
- 'hrs4',
- 'hrsn',
- 'hs10',
- 'hsh2',
- 'lb10',
- 'lbbn',
- 'lbda',
- 'lbfs',
- 'lbnm',
- 'lbo5',
- 'lbor',
- 'lbos',
- 'lbsm',
- 'lbte',
- 'lgn2',
- 'lgnm',
- 'lgnr',
- 'lngt',
- 'lnlg',
- 'log0',
- 'log1',
- 'logn',
- 'ma10',
- 'matx',
- 'maxb',
- 'mdcs',
- 'mdsn',
- 'mea2',
- 'mea3',
- 'mean',
- 'mics',
- 'min2',
- 'min3',
- 'min4',
- 'mint',
- 'misn',
- 'mltp',
- 'mmd2',
- 'mmd3',
- 'mmd4',
- 'mmd5',
- 'mmd6',
- 'mmdx',
- 'mmor',
- 'mmq2',
- 'mmqb',
- 'mncs',
- 'mnm2',
- 'mnm3',
- 'mnm4',
- 'mnm5',
- 'mnm6',
- 'mnm7',
- 'mnmx',
- 'mnsn',
- 'mnt2',
- 'mnt3',
- 'mnt4',
- 'mnt5',
- 'mnt6',
- 'mnth',
- 'mnto',
- 'mnts',
- 'mscs',
- 'mssn',
- 'mxab',
- 'nbr2',
- 'nbr3',
- 'nmbd',
- 'nmbr',
- 'nmc2',
- 'nmc3',
- 'nmc4',
- 'nmc5',
- 'nmc6',
- 'nmc7',
- 'nmc8',
- 'nmc9',
- 'nmcm',
- 'nmd2',
- 'nmd3',
- 'nmd4',
- 'nmd5',
- 'nmd6',
- 'nmdx',
- 'nmE2',
- 'nmE3',
- 'nmE4',
- 'nmE5',
- 'nmE6',
- 'nmE7',
- 'nmE8',
- 'nmE9',
- 'nmEm',
- 'nmq2',
- 'nmqb',
- 'nmr2',
- 'nmr3',
- 'nmr4',
- 'nmr5',
- 'nmr6',
- 'nmr7',
- 'nmr8',
- 'nmr9',
- 'nmrc',
- 'ntg2',
- 'ntg3',
- 'ntgr',
- 'nuld',
- 'null',
- 'om10',
- 'onht',
- 'or10',
- 'or11',
- 'or12',
- 'or13',
- 'or14',
- 'or15',
- 'or16',
- 'or17',
- 'or18',
- 'or19',
- 'or20',
- 'or21',
- 'or22',
- 'or23',
- 'or3b',
- 'or3c',
- 'or3d',
- 'ord2',
- 'ord3',
- 'ord4',
- 'ord5',
- 'ordd',
- 'ordl',
- 'ors2',
- 'ors5',
- 'ors6',
- 'ors7',
- 'por2',
- 'por3',
- 'pwbn',
- 'pwor',
- 'pwr2',
- 'pwrs',
- 'qbt1',
- 'qbt2',
- 'qbt3',
- 'qbt4',
- 'qbt5',
- 'rais',
- 'retn',
- 'rtb2',
- 'rtbn',
- 'sbs2',
- 'sbs3',
- 'sbs4',
- 'sbst',
- 'sbtr',
- 'sccs',
- 'scn2',
- 'scnd',
- 'scsn',
- 'sgn1',
- 'sgn2',
- 'sgn3',
- 'sgn4',
- 'shf2',
- 'shf3',
- 'shf4',
- 'shf5',
- 'shf6',
- 'shf7',
- 'shf8',
- 'shfl',
- 'shft',
- 'smth',
- 'sp10',
- 'sp11',
- 'sp12',
- 'sp13',
- 'sp14',
- 'sp15',
- 'sp16',
- 'sp17',
- 'sp18',
- 'sp19',
- 'sp20',
- 'spl2',
- 'spl5',
- 'spl6',
- 'spl7',
- 'spl8',
- 'spl9',
- 'splt',
- 'sqrt',
- 'src2',
- 'src3',
- 'src4',
- 'srch',
- 'strn',
- 'strg',
- 'texd',
- 'text',
- 'tlbn',
- 'tmzn',
- 'txt2',
- 'txt3',
- 'ucct', 
- 'wkdo',
- 'wkds',
- 'wkdy',
- 'yea2',
- 'year'
 ___ 
### List of Suffix Appenders
The convention is that each transform returns a derived column or set of columns which are distinguished 
from the source column by suffix appenders to the header strings. Note that in cases of root categories 
whose family trees include multiple generations, there may be multiple inclusions of different suffix 
appenders in a single returned column. Provided here is a concise sorted list of all suffix appenders so 
that any user passing a custom defined transformation can avoid any unintentional duplication. Note that
the transformation functions test for suffix overlap error from creating new column with headers already
present in dataframe and return results in postprocess_dict['miscparameters_results']['suffixoverlap_results'].
(Or for comparable validation results for PCA, Binary, and excl transforms see 'PCA_suffixoverlap_results', 
'Binary_suffixoverlap_results', 'excl_suffixoverlap_results'.)

- '\_-10^'
- '\_-10^' + i (where i is an integer corresponding to the source number power of ten)
- '\_10^' + i (where i is an integer corresponding to the source number power of ten)
- '\_1010_' + i (where i is an integer corresponding to the ith digit of the binary encoding)
- '_absl'
- '_addd'
- '_aggt'
- '\_bins\_#' where # is an integer
- '\_bkt1_' + i (where i is identifier of bin)
- '\_bkt2_' + i (where i is identifier of bin)
- '\_bkt3_' + i (where i is identifier of bin)
- '\_bkt4_' + i (where i is identifier of bin)
- '_bn7o'
- '_bn9o'
- '\_bne7_' + i (where i is identifier of bin)
- '\_bne9_' + i (where i is identifier of bin)
- '_bneo'
- '\_bnep_' + i (where i is identifier of bin)
- '_bnKo'
- '_bnMo'
- '_bnr2'
- '_bnry'
- '\_bnwd_' + i + '_' + j (where i is bin width and j is identifier of bin)
- '\_bnwK_' + i + '_' + j (where i is bin width and j is identifier of bin)
- '\_bnwM_' + i + '_' + j (where i is bin width and j is identifier of bin)
- '_bnwo'
- '_bshr'
- '_bsor'
- '_bxcx'
- '_copy'
- '_days'
- '_dhmc'
- '_dhms'
- '_divd'
- '_DP10'
- '_DPbn'
- '_DPmm'
- '_DPmm_tmp1'
- '_DPnb'
- '_DPnm'
- '_DPod'
- '_DPod_tmp1'
- '_DPod_tmp2'
- '_DPoh'
- '_DPrt'
- '_DPrt_tmp1'
- '_DPrt_tmp2'
- '_dxd2'
- '_dxdt'
- '_dycs'
- '_dysn'
- '_exc2'
- '_exc5'
- '_excl'
- '\_hash\_#' (where # is integer)
- '\_hash'
- '\_hs10\_#' (where # is integer)
- '_hldy'
- '_hmsc'
- '_hmss'
- '_hour'
- '_hrcs'
- '_hrsn'
- '_lgnr'
- '_lngt'
- '_log0'
- '_logn'
- '_MAD3'
- '_MADn'
- '_maxb'
- '_mdcs'
- '_mdsn'
- '_mean'
- '_mics'
- '_mint'
- '_misn'
- '_mltp'
- '_mncs'
- '_mnm3'
- '_mnmx'
- '_mnsn'
- '_mnth'
- '_mnts'
- '_mscs'
- '_mssn'
- '_mxab'
- '_NArows'
- '_NArw'
- '_nmbr'
- '_nmc4'
- '_nmc7'
- '_nmcm'
- '_nmE4'
- '_nmE7'
- '_nmEU'
- '_nmr4'
- '_nmr7'
- '_nmrc'
- '\_onht'
- '\_onht_' + # (where # is integer associated with entry for activations)
- '_ord3'
- '_ordl'
- '_por2'
- '_pwor'
- '\_qbt1\_sign' and '\_qbt1\_2^2#' where # is integer
- '\_qbt2\_sign' and '\_qbt2\_2^2#' where # is integer
- '\_qbt3\_2^2#' where # is integer
- '\_qbt4\_2^2#' where # is integer
- '_rais'
- '_retn'
- '\_sbs2_' + string (where string is an identified overlap of characters between categorical entries)
- '\_sbs3_' + i (where i is ineteger)
- '\_sbs4_' + i (where i is ineteger)
- '\_sbst_' + string (where string is an identified overlap of characters between categorical entries)
- '_sbtr'
- '_sccs'
- '_scnd'
- '_scsn'
- '_shf2'
- '_shf3'
- '_shf4'
- '_shf5'
- '_shf6'
- '_shfl'
- '_shft'
- '_sp10'
- '\_sp15_' + string (where string is an identified overlap of characters between categorical entries)
- '\_sp16_' + string (where string is an identified overlap of characters between categorical entries)
- '\_sp19_' + i (where i is ineteger)
- '\_sp20_' + i (where i is ineteger)
- '_spl2'
- '_spl5'
- '_spl7'
- '\_spl8_' + string (where string is an identified overlap of characters between categorical entries)
- '_spl9'
- '\_splt_' + string (where string is an identified overlap of characters between categorical entries)
- '_sqrt'
- '\_src2_' + string (where string is an identified overlap of characters with user passed search string)
- '\_src3_' + string (where string is an identified overlap of characters with user passed search string)
- '_src4'
- '\_srch_' + string (where string is an identified overlap of characters with user passed search string)
- '_strn'
- '_strg'
- '\_tlbn_' + i (where i is identifier of bin)
- '\_text_' + string (where string is a categorical entry in one-hot encoded set)
- '_tmzn'
- '_ucct'
- '_UPCS'
- '_wkds'
- '_wkdy'
- '_year'
 ___ 
### Other Reserved Strings
- 'zzzinfill': a reserved string entry to data sets that is used in many places as an infill values such as for categorical encodings.
Note that when inversion is performed those entries without recovery are returned with this value.
- 'Binary': a reserved column header for cases where a Binary transform is applied with the automunge(.) Binary parameter. 
- 'Binary_1010_#': The columns returned from Binary transform have headers per this convention.
- 'PCAcol#': when PCA dimensionality reduction is performed, returned columns have headers per this convention.
- 'Automunge_index': a reserved column header for index columns returned in ID sets. When automunge(.) is run the returned ID sets are
populated with an index matching order of rows from original returned set, note that if this string is already present in the ID sets
it will instead populate as 'Automunge_index_' + a 12 digit random integer associated with the application number.

Note that results of various validation checks such as for column header overlaps and other potential bugs are returned from 
automunge(.) in the postprocess_dict as postprocess_dict['miscparameters_results'], and returned from postmunge(.) in the postreports_dict
as postreports_dict['pm_miscparameters_results']. (If the function fails to compile check the printouts.)
 ___ 
### Root Category Family Tree Definitions
And here are the family tree definitions for root categories currently built into the internal 
library. Basically providing this as a reference, not really expecting anyone to read this line 
by line or anything. (Note that the NArw transformation without quotation marks (eg NArw 
vs 'NArw') will not be activated if user passes the automunge(.) parameter as NArw_marker=False.)
If you want to skip to the next section you can click here: [Custom Transformation Functions](https://github.com/Automunge/AutoMunge#custom-transformation-functions)

```
    transform_dict.update({'nmbr' : {'parents'       : ['nmbr'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : [bint]}})
    
    transform_dict.update({'dxdt' : {'parents'       : ['dxdt'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})
    
    transform_dict.update({'d2dt' : {'parents'       : ['d2dt'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['dxdt'], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})
    
    transform_dict.update({'d3dt' : {'parents'       : ['d3dt'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['d2dt'], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})

    transform_dict.update({'d4dt' : {'parents'       : ['d4dt'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['d3dt'], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})

    transform_dict.update({'d5dt' : {'parents'       : ['d5dt'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['d4dt'], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})

    transform_dict.update({'d6dt' : {'parents'       : ['d6dt'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['d5dt'], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})
    
    transform_dict.update({'dxd2' : {'parents'       : ['dxd2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})
    
    transform_dict.update({'d2d2' : {'parents'       : ['d2d2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['dxd2'], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})
    
    transform_dict.update({'d3d2' : {'parents'       : ['d3d2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['d2d2'], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})

    transform_dict.update({'d4d2' : {'parents'       : ['d4d2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['d3d2'], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})

    transform_dict.update({'d5d2' : {'parents'       : ['d5d2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['d4d2'], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})

    transform_dict.update({'d6d2' : {'parents'       : ['d6d2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['d5d2'], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})
    
    transform_dict.update({'nmdx' : {'parents'       : ['nmdx'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['dxdt'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})
    
    transform_dict.update({'nmd2' : {'parents'       : ['nmd2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['d2dt'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})
    
    transform_dict.update({'nmd3' : {'parents'       : ['nmd3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['d3dt'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})

    transform_dict.update({'nmd4' : {'parents'       : ['nmd4'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['d4dt'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})

    transform_dict.update({'nmd5' : {'parents'       : ['nmd5'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['d5dt'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})

    transform_dict.update({'nmd6' : {'parents'       : ['nmd6'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['d6dt'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})
    
    transform_dict.update({'mmdx' : {'parents'       : ['mmdx'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['nbr2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['nbr2'], \
                                     'friends'       : []}})
    
    transform_dict.update({'mmd2' : {'parents'       : ['mmd2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['nbr2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['mmdx'], \
                                     'coworkers'     : ['nbr2'], \
                                     'friends'       : []}})
    
    transform_dict.update({'mmd3' : {'parents'       : ['mmd3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['nbr2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['mmd2'], \
                                     'coworkers'     : ['nbr2'], \
                                     'friends'       : []}})

    transform_dict.update({'mmd4' : {'parents'       : ['mmd4'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['nbr2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['mmd3'], \
                                     'coworkers'     : ['nbr2'], \
                                     'friends'       : []}})

    transform_dict.update({'mmd5' : {'parents'       : ['mmd5'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['nbr2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['mmd4'], \
                                     'coworkers'     : ['nbr2'], \
                                     'friends'       : []}})

    transform_dict.update({'mmd6' : {'parents'       : ['mmd6'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['nbr2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['mmd5'], \
                                     'coworkers'     : ['nbr2'], \
                                     'friends'       : []}})
    
    transform_dict.update({'dddt' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['dddt', 'exc2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'ddd2' : {'parents'       : ['ddd2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['exc2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['dddt'], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'ddd3' : {'parents'       : ['ddd3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['exc2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['ddd2'], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'ddd4' : {'parents'       : ['ddd4'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['exc2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['ddd3'], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'ddd5' : {'parents'       : ['ddd5'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['exc2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['ddd4'], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'ddd6' : {'parents'       : ['ddd6'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['exc2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['ddd5'], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'dedt' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['dedt', 'exc2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'ded2' : {'parents'       : ['ded2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['exc2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['dedt'], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'ded3' : {'parents'       : ['ded3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['exc2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['ded2'], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'ded4' : {'parents'       : ['ded4'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['exc2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['ded3'], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'ded5' : {'parents'       : ['ded5'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['exc2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['ded4'], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'ded6' : {'parents'       : ['ded6'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['exc2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['ded5'], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'shft' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['shft'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'shf2' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['shf2'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'shf3' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['shf3'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'shf4' : {'parents'       : ['shf4'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})
  
    transform_dict.update({'shf5' : {'parents'       : ['shf5'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})
    
    transform_dict.update({'shf6' : {'parents'       : ['shf6'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})

    transform_dict.update({'shf7' : {'parents'       : ['shf4', 'shf5'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})
    
    transform_dict.update({'shf8' : {'parents'       : ['shf4', 'shf5', 'shf6'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['retn'], \
                                     'friends'       : []}})

    transform_dict.update({'bnry' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bnry'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'bnr2' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bnr2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'onht' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['onht'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'text' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['text'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'txt2' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['text'], \
                                     'cousins'       : [NArw, 'splt'], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'txt3' : {'parents'       : ['txt3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['text'], \
                                     'friends'       : []}})

    transform_dict.update({'smth' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['smth'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'fsmh' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['fsmh'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'lngt' : {'parents'       : ['lngt'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['mnmx'], \
                                     'friends'       : []}})
  
    transform_dict.update({'lnlg' : {'parents'       : ['lnlg'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['log0'], \
                                     'friends'       : []}})

    transform_dict.update({'UPCS' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['UPCS'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'Unht' : {'parents'       : ['Unht'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['onht'], \
                                     'friends'       : []}})
  
    transform_dict.update({'Utxt' : {'parents'       : ['Utxt'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['text'], \
                                     'friends'       : []}})
    
    transform_dict.update({'Utx2' : {'parents'       : ['Utx2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['text'], \
                                     'friends'       : ['splt']}})

    transform_dict.update({'Utx3' : {'parents'       : ['Utx3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['txt3'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'Ucct' : {'parents'       : ['Ucct'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['ucct', 'ord3'], \
                                     'friends'       : []}})
    
    transform_dict.update({'Uord' : {'parents'       : ['Uord'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['ordl'], \
                                     'friends'       : []}})
        
    transform_dict.update({'Uor2' : {'parents'       : ['Uor2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['ord2'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'Uor3' : {'parents'       : ['Uor3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['ord3'], \
                                     'friends'       : []}})
    
    transform_dict.update({'Uor6' : {'parents'       : ['Uor6'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['spl6'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['ord3'], \
                                     'friends'       : []}})
    
    transform_dict.update({'U101' : {'parents'       : ['U101'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})
    
    transform_dict.update({'splt' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['splt'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'spl2' : {'parents'       : ['spl2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['ord3'], \
                                     'friends'       : []}})
    
    transform_dict.update({'spl5' : {'parents'       : ['spl5'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['ord3'], \
                                     'friends'       : []}})
    
    transform_dict.update({'spl6' : {'parents'       : ['spl6'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['splt'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : ['ord3']}})
    
    transform_dict.update({'spl7' : {'parents'       : ['spl7'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['ord3'], \
                                     'friends'       : []}})

    transform_dict.update({'spl8' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['spl8'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'spl9' : {'parents'       : ['spl9'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['ord3'], \
                                     'friends'       : []}})

    transform_dict.update({'sp10' : {'parents'       : ['sp10'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['ord3'], \
                                     'friends'       : []}})
    
    
    transform_dict.update({'sp11' : {'parents'       : ['sp11'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['spl5'], \
                                     'coworkers'     : ['ord3'], \
                                     'friends'       : []}})
    
    transform_dict.update({'sp12' : {'parents'       : ['sp12'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['sp11'], \
                                     'coworkers'     : ['ord3'], \
                                     'friends'       : []}})
    
    transform_dict.update({'sp13' : {'parents'       : ['sp13'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['sp10'], \
                                     'coworkers'     : ['ord3'], \
                                     'friends'       : []}})
    
    transform_dict.update({'sp14' : {'parents'       : ['sp14'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['sp13'], \
                                     'coworkers'     : ['ord3'], \
                                     'friends'       : []}})
    
    transform_dict.update({'sp15' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['sp15'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'sp16' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['sp16'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'sp17' : {'parents'       : ['sp17'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['spl5'], \
                                     'coworkers'     : ['ord3'], \
                                     'friends'       : []}})
    
    transform_dict.update({'sp18' : {'parents'       : ['sp18'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : ['sp17'], \
                                     'coworkers'     : ['ord3'], \
                                     'friends'       : []}})

    transform_dict.update({'sp19' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['sp19'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'sp20' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['sp20'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'sbst' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['sbst'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'sbs2' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['sbs2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'sbs3' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['sbs3'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'sbs4' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['sbs4'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'hash' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['hash'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'hsh2' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['hsh2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'hs10' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['hs10'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'Uhsh' : {'parents'       : ['Uhsh'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['hash'], \
                                     'friends'       : []}})

    transform_dict.update({'Uhs2' : {'parents'       : ['Uhs2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['hsh2'], \
                                     'friends'       : []}})
    
    transform_dict.update({'Uh10' : {'parents'       : ['Uh10'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['hs10'], \
                                     'friends'       : []}})
    
    transform_dict.update({'srch' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['srch'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'src2' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['src2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'src3' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['src3'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'src4' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['src4'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'aggt' : {'parents'       : ['aggt'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['ord3'], \
                                     'friends'       : []}})
    
    transform_dict.update({'strn' : {'parents'       : ['strn'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['ord3'], \
                                     'friends'       : []}})

  
    transform_dict.update({'strg' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['strg'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'nmrc' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['nmrc'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'nmr2' : {'parents'       : ['nmr2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'nmr3' : {'parents'       : ['nmr3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['mnmx'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'nmr4' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['nmr4'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'nmr5' : {'parents'       : ['nmr5'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'nmr6' : {'parents'       : ['nmr6'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['mnmx'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'nmr7' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['nmr7'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'nmr8' : {'parents'       : ['nmr8'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'nmr9' : {'parents'       : ['nmr9'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['mnmx'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'nmcm' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['nmcm'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'nmc2' : {'parents'       : ['nmc2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'nmc3' : {'parents'       : ['nmc3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['mnmx'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'nmc4' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['nmc4'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'nmc5' : {'parents'       : ['nmc5'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'nmc6' : {'parents'       : ['nmc6'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['mnmx'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'nmc7' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['nmc7'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'nmc8' : {'parents'       : ['nmc8'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'nmc9' : {'parents'       : ['nmc9'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['mnmx'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'nmEU' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['nmEU'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'nmE2' : {'parents'       : ['nmE2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'nmE3' : {'parents'       : ['nmE3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['mnmx'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'nmE4' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['nmE4'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'nmE5' : {'parents'       : ['nmE5'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'nmE6' : {'parents'       : ['nmE6'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['mnmx'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'nmE7' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['nmE7'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'nmE8' : {'parents'       : ['nmE8'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'nmE9' : {'parents'       : ['nmE9'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['mnmx'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'ors7' : {'parents'       : ['spl6', 'nmr2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['ord3'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'ors5' : {'parents'       : ['spl5'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['ord3'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'ors6' : {'parents'       : ['spl6'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['ord3'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'ordl' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['ordl'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
        
    transform_dict.update({'ord2' : {'parents'       : ['ord2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['mnmx'], \
                                     'friends'       : []}})
    
    transform_dict.update({'ord3' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['ord3'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'ord5' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['ord5'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'maxb' : {'parents'       : ['or3b'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'or3b' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['or3b'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['maxb'], \
                                     'friends'       : []}})
  
    transform_dict.update({'matx' : {'parents'       : ['or3c'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['onht'], \
                                     'friends'       : []}})
    
    transform_dict.update({'or3c' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['or3c'], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['matx'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'ma10' : {'parents'       : ['or3d'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})
    
    transform_dict.update({'or3d' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['or3d'], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['ma10'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'ucct' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['ucct'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
        
    transform_dict.update({'ord4' : {'parents'       : ['ord4'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['mnmx'], \
                                     'friends'       : []}})
    
    transform_dict.update({'ors2' : {'parents'       : ['spl2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['ord3'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'or10' : {'parents'       : ['ord4'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['1010'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['mnmx'], \
                                     'friends'       : []}})
    
    transform_dict.update({'or11' : {'parents'       : ['sp11'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['1010'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'or12' : {'parents'       : ['nmr2'], \
                                     'siblings'      : ['sp11'], \
                                     'auntsuncles'   : ['1010'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'or13' : {'parents'       : ['sp12'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['1010'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'or14' : {'parents'       : ['nmr2'], \
                                     'siblings'      : ['sp12'], \
                                     'auntsuncles'   : ['1010'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'or15' : {'parents'       : ['or15'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['sp13'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})
  
    transform_dict.update({'or16' : {'parents'       : ['or16'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['nmr2'], \
                                     'niecesnephews' : ['sp13'], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})
    
    transform_dict.update({'or17' : {'parents'       : ['or17'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['sp14'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})
    
    transform_dict.update({'or18' : {'parents'       : ['or18'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['nmr2'], \
                                     'niecesnephews' : ['sp14'], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})

    transform_dict.update({'or19' : {'parents'       : ['or19'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['nmc8'], \
                                     'niecesnephews' : ['sp13'], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})
    
    transform_dict.update({'or20' : {'parents'       : ['or20'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['nmc8'], \
                                     'niecesnephews' : ['sp14'], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})
    
    transform_dict.update({'or21' : {'parents'       : ['or21'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['nmc8'], \
                                     'niecesnephews' : ['sp17'], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})
    
    transform_dict.update({'or22' : {'parents'       : ['or22'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['nmc8'], \
                                     'niecesnephews' : ['sp18'], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})

    transform_dict.update({'or23' : {'parents'       : ['or23'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['nmcm', 'sp19', 'ord3'], \
                                     'friends'       : []}})
    
    transform_dict.update({'om10' : {'parents'       : ['ord4'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['1010', 'mnmx'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['mnmx'], \
                                     'friends'       : []}})

    transform_dict.update({'mmor' : {'parents'       : ['ord4'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mnmx'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'1010' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['1010'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'null' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['null'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'NArw' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['NArw'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'NAr2' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['NAr2'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'NAr3' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['NAr3'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'NAr4' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['NAr4'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'NAr5' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['NAr5'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'nbr2' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['nmbr'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'nbr3' : {'parents'       : ['nbr3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : ['bint']}})
    
    transform_dict.update({'MADn' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['MADn'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'MAD2' : {'parents'       : ['MAD2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'MAD3' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['MAD3'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mnmx' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mnmx'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mnm2' : {'parents'       : ['nmbr'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mnmx'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mnm3' : {'parents'       : ['nmbr'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mnm3'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mnm4' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mnm3'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mnm5' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mnmx'], \
                                     'cousins'       : ['nmbr', NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mnm6' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mnm6'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mnm7' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mnmx', 'bins'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'mxab' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mxab'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'retn' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'rtbn' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn', 'bsor'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'rtb2' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn', 'bins'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'mean' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mean'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mea2' : {'parents'       : ['nmbr'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mean'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mea3' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mean', 'bins'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'tmzn' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['tmzn'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'date' : {'parents'       : ['date'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['year', 'mnth', 'days', 'hour', 'mint', 'scnd'], \
                                     'friends'       : []}})
  
    transform_dict.update({'dat2' : {'parents'       : ['dat2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['bshr', 'wkdy', 'hldy'], \
                                     'friends'       : []}})
    
    transform_dict.update({'dat3' : {'parents'       : ['dat3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['year', 'mnsn', 'mncs', 'dysn', 'dycs', 'hrsn', 'hrcs', 'misn', 'mics', 'scsn', 'sccs'], \
                                     'friends'       : []}})
    
    transform_dict.update({'dat4' : {'parents'       : ['dat4'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['year', 'mdsn', 'mdcs', 'hmss', 'hmsc'], \
                                     'friends'       : []}})
    
    transform_dict.update({'dat5' : {'parents'       : ['dat5'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['year', 'mdsn', 'mdcs', 'dysn', 'dycs', 'hmss', 'hmsc'], \
                                     'friends'       : []}})
    
    transform_dict.update({'dat6' : {'parents'       : ['dat6'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['year', 'mdsn', 'mdcs', 'hmss', 'hmsc', 'bshr', 'wkdy', 'hldy'], \
                                     'friends'       : []}})
    
    transform_dict.update({'year' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['year'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'yea2' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['year', 'yrsn', 'yrcs', 'mdsn', 'mdcs'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'yrcs' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['yrcs'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'yrsn' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['yrsn'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mnth' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mnth'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'mnt2' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mnsn', 'mncs'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mnt3' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mnsn', 'mncs', 'dysn', 'dycs', 'hrsn', 'hrcs', 'misn', 'mics', 'scsn', 'sccs'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mnt4' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mdsn', 'mdcs'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mnt5' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mdsn', 'mdcs', 'hmss', 'hmsc'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mnt6' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mdsn', 'mdcs', 'dysn', 'dycs', 'hmss', 'hmsc'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mnsn' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mnsn'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mncs' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mncs'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mdsn' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mdsn'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mdcs' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mdcs'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'days' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['days'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'day2' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['dysn', 'dycs'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'day3' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['dysn', 'dycs', 'hrsn', 'hrcs', 'misn', 'mics', 'scsn', 'sccs'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'day4' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['dhms', 'dhmc'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'day5' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['dhms', 'dhmc', 'hmss', 'hmsc'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'dysn' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['dysn'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'dycs' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['dycs'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'dhms' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['dhms'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'dhmc' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['dhmc'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'hour' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['hour'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'hrs2' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['hrsn', 'hrcs'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'hrs3' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['hrsn', 'hrcs', 'misn', 'mics', 'scsn', 'sccs'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'hrs4' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['hmss', 'hmsc'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'hrsn' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['hrsn'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'hrcs' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['hrcs'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'hmss' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['hmss'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'hmsc' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['hmsc'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mint' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mint'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'min2' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['misn', 'mics'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'min3' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['misn', 'mics', 'scsn', 'sccs'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'min4' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mssn', 'mscs'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'misn' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['misn'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mics' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mics'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mssn' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mssn'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mscs' : {'parents'       : [], \
                                     'siblings': [], \
                                     'auntsuncles'   : ['mscs'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'scnd' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['scnd'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'scn2' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['scsn', 'sccs'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'scsn' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['scsn'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'sccs' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['sccs'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'bxcx' : {'parents'       : ['bxcx'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'bxc2' : {'parents'       : ['bxc2'], \
                                     'siblings'      : ['nmbr'], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'bxc3' : {'parents'       : ['bxc3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'bxc4' : {'parents'       : ['bxc4'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['nbr2'], \
                                     'friends'       : []}})

    transform_dict.update({'bxc5' : {'parents'       : ['bxc5'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mnmx'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['nbr2', 'bins'], \
                                     'friends'       : []}})

    transform_dict.update({'ntgr' : {'parents'       : ['ntgr'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn', '1010', 'ordl'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['mnmx'], \
                                     'friends'       : []}})
    
    transform_dict.update({'ntg2' : {'parents'       : ['ntg2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn', '1010', 'ordl', 'pwr2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['mnmx'], \
                                     'friends'       : []}})
    
    transform_dict.update({'ntg3' : {'parents'       : ['ntg3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['retn', 'ordl', 'por2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['mnmx'], \
                                     'friends'       : []}})
    
    transform_dict.update({'pwrs' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['pwrs'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'pwr2' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['pwr2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'log0' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['log0'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'log1' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['log0', 'pwr2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'logn' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['logn'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'lgnm' : {'parents'       : ['lgnm'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['nmbr'], \
                                     'friends'       : []}})
    
    transform_dict.update({'sqrt' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['sqrt'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'addd' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['addd'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'sbtr' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['sbtr'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'mltp' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['mltp'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'divd' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['divd'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'rais' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['rais'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'absl' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['absl'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'bkt1' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bkt1'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'bkt2' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bkt2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'bkt3' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bkt3'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'bkt4' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bkt4'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'wkdy' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['wkdy'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'bshr' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bshr'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'hldy' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['hldy'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'wkds' : {'parents'       : ['wkds'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['text'], \
                                     'friends'       : []}})
  
    transform_dict.update({'wkdo' : {'parents'       : ['wkdo'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['ordl'], \
                                     'friends'       : []}})
    
    transform_dict.update({'mnts' : {'parents'       : ['mnts'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['text'], \
                                     'friends'       : []}})
  
    transform_dict.update({'mnto' : {'parents'       : ['mnto'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['ordl'], \
                                     'friends'       : []}})
    
    transform_dict.update({'bins' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bins'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'bint' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bint'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'bsor' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bsor'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'btor' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['btor'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'bnwd' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bnwd'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'bnwK' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bnwK'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'bnwM' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bnwM'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'bnwo' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bnwo'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'bnKo' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bnKo'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'bnMo' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bnMo'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})    
    
    transform_dict.update({'bnep' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bnep'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'bne7' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bne7'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'bne9' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bne9'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'bneo' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bneo'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'bn7o' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bn7o'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'bn9o' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bn9o'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'tlbn' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['tlbn'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'pwor' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['pwor'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'por2' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['por2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'por3' : {'parents'       : ['por3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})

    transform_dict.update({'bkb3' : {'parents'       : ['bkb3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})
  
    transform_dict.update({'bkb4' : {'parents'       : ['bkb4'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})
    
    transform_dict.update({'bsbn' : {'parents'       : ['bsbn'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})
    
    transform_dict.update({'bnwb' : {'parents'       : ['bnwb'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})
    
    transform_dict.update({'bnKb' : {'parents'       : ['bnKb'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})

    transform_dict.update({'bnMb' : {'parents'       : ['bnMb'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})
    
    transform_dict.update({'bneb' : {'parents'       : ['bneb'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})

    transform_dict.update({'bn7b' : {'parents'       : ['bn7b'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})
    
    transform_dict.update({'bn9b' : {'parents'       : ['bn9b'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})
    
    transform_dict.update({'pwbn' : {'parents'       : ['pwbn'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})

    transform_dict.update({'DPnb' : {'parents'       : ['DPn3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'DPn3' : {'parents'       : ['DPn3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['DPnb'], \
                                     'friends'       : []}})

    transform_dict.update({'DPmm' : {'parents'       : ['DPm2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'DPm2' : {'parents'       : ['DPm2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['DPmm'], \
                                     'friends'       : []}})

    transform_dict.update({'DPrt' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['DPrt'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'DLnb' : {'parents'       : ['DLn3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'DLn3' : {'parents'       : ['DLn3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['DLnb'], \
                                     'friends'       : []}})

    transform_dict.update({'DLmm' : {'parents'       : ['DLm2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'DLm2' : {'parents'       : ['DLm2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['DLmm'], \
                                     'friends'       : []}})

    transform_dict.update({'DLrt' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['DLrt'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'DPbn' : {'parents'       : ['DPb2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'DPb2' : {'parents'       : ['DPb2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['DPbn'], \
                                     'friends'       : []}})
    
    transform_dict.update({'DPod' : {'parents'       : ['DPo4'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'DPo4' : {'parents'       : ['DPo4'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['DPod'], \
                                     'friends'       : []}})
    
    transform_dict.update({'DPoh' : {'parents'       : ['DPo5'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['onht'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'DPo5' : {'parents'       : ['DPo5'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['DPo2'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'DPo2' : {'parents'       : ['DPo2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['onht'], \
                                     'friends'       : []}})
    
    transform_dict.update({'DP10' : {'parents'       : ['DPo6'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['1010'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'DPo6' : {'parents'       : ['DPo6'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['DPo3'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'DPo3' : {'parents'       : ['DPo3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['1010'], \
                                     'friends'       : []}})

    transform_dict.update({'qbt1' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['qbt1'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'qbt2' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['qbt2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'qbt3' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['qbt3'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'qbt4' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['qbt4'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'nmqb' : {'parents'       : ['nmqb'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['qbt1'], \
                                     'friends'       : []}})
  
    transform_dict.update({'nmq2' : {'parents'       : ['nmq2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : ['qbt1']}})
  
    transform_dict.update({'mmqb' : {'parents'       : ['mmqb'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['qbt3'], \
                                     'friends'       : []}})
    
    transform_dict.update({'mmq2' : {'parents'       : ['mmq2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : ['qbt3']}})
    
    transform_dict.update({'copy' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['copy'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'excl' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['excl'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'exc2' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['exc2'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'exc3' : {'parents'       : ['exc3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : ['bins']}})
    
    transform_dict.update({'exc4' : {'parents'       : ['exc4'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : ['pwr2']}})
    
    transform_dict.update({'exc5' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['exc5'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    #exc6 was removed from library, is now same as excl, just including here so no printout for prior code demonstrations 
    transform_dict.update({'exc6' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['excl'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'shfl' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['shfl'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'nmbd' : {'parents'       : ['nmbr'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : [bint]}})

    transform_dict.update({'101d' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['1010'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'ordd' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['ord3'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'texd' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['text'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'bnrd' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bnry'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'datd' : {'parents'       : ['datd'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['year', 'mdsn', 'mdcs', 'hmss', 'hmsc', 'bshr', 'wkdy', 'hldy'], \
                                     'friends'       : []}})
    
    transform_dict.update({'nuld' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['null'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'lbnm' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['exc2'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'lbnb' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['nmbr'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'lb10' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['1010'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'lbor' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['ordl'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'lbos' : {'parents'       : ['lbos'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['strg'], \
                                     'friends'       : []}})
    
    transform_dict.update({'lbte' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['text'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'lbbn' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['bnry'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'lbsm' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['lbsm'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'lbfs' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['lbfs'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'lbda' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['year', 'mdsn', 'mdcs', 'hmss', 'hmsc', 'bshr', 'wkdy', 'hldy'], \
                                     'cousins'       : [], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'lgnr' : {'parents'       : ['lgnr', 'sgn3'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['lgn2'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
  
    transform_dict.update({'lgn2' : {'parents'       : ['lgn2'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['qbt5'], \
                                     'friends'       : []}})
    
    transform_dict.update({'sgn1' : {'parents'       : ['sgn1'], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : [], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : ['sgn2'], \
                                     'friends'       : []}})
    
    transform_dict.update({'qbt5' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['qbt5'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
    
    transform_dict.update({'sgn2' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['sgn2'], \
                                     'cousins'       : [NArw], \
                                     'children'      : [], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'sgn3' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['sgn3'], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['sgn4'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})

    transform_dict.update({'sgn4' : {'parents'       : [], \
                                     'siblings'      : [], \
                                     'auntsuncles'   : ['sgn4'], \
                                     'cousins'       : [NArw], \
                                     'children'      : ['sgn1'], \
                                     'niecesnephews' : [], \
                                     'coworkers'     : [], \
                                     'friends'       : []}})
```

...

## Custom Transformation Functions

Ok final item on the agenda, we're going to demonstrate methods to create custom
transformation functions, such that a user may customize the feature engineering
while building on all of the extremely useful built in features of automunge such
as infill methods including ML infill, feature importance, dimensionality reduction,
preparation for class imbalance oversampling, and perhaps most importantly the 
simplest possible way for consistent processing of additional data with just a single 
function call. The transformation functions will need to be channeled through pandas 
and incorporate a handful of simple data structures, which we'll demonstrate below.

Let's say we want to recreate the mnm3 category which caps outliers at 0.01 and 0.99
quantiles, but instead make it the 0.001 and 0.999 quantiles. (Of course that is already
possible in easier fashion by just passing parameters to the transformation function, 
this is just to demonstrate.) Well we'll call this new category mnm8. So in order to 
pass a custom transformation function, first we'll need to define a new root category 
transformdict and a corresponding processdict.

```
#Let's create a really simple family tree for the new root category mnmn8 which
#simply creates a column identifying any rows subject to infill (NArw), performs 
#a z-score normalization, and separately performs a version of the new transform
#mnm8 which we'll define below.

transformdict = {'mnm8' : {'parents' : [], \
                           'siblings': [], \
                           'auntsuncles' : ['mnm8', 'nmbr'], \
                           'cousins' : ['NArw'], \
                           'children' : [], \
                           'niecesnephews' : [], \
                           'coworkers' : [], \
                           'friends' : []}}

#Note that since this mnm8 requires passing normalization parameters derived
#from the train set to process the test set, we'll need to create two seperate 
#transformation functions, the first a "dualprocess" function that processes
#both the train and if available a test set simultaneously, and the second
#a "postprocess" that only processes the test set on it's own.

#So what's being demonstrated here is that we're passing the functions under
#dualprocess and postprocess that we'll define below.

processdict = {'mnm8' : {'dualprocess' : process_mnm8, \
                         'singleprocess' : None, \
                         'postprocess' : postprocess_mnm8, \
                         'NArowtype' : 'numeric', \
                         'MLinfilltype' : 'numeric', \
                         'labelctgy' : 'mnm8'}}

#Now we have to define the custom processing functions which we are passing through
#the processdict to automunge.

#Here we'll define a "dualprocess" function intended to process both a train and
#test set simultaneously. We'll also need to create a seperate "postprocess"
#function intended to just process a subsequent test set.

#define the function
def process_mnm8(mdf_train, mdf_test, column, category, \
                       postprocess_dict, params = {}):
  #where
  #mdf_train is the train data set (pandas dataframe)
  #mdf_test is the consistently formatted test dataset (if no test data 
  #set is passed to automunge a small dummy set will be passed in it's place)
  #column is the string identifying the column header
  #category is the (traditionally 4 character) string category identifier, here is 
  #will be 'mnm8', 
  #postprocess_dict is an object we pass to share data between 
  #functions and later returned from automunge
  #and params are any column specific parameters to be passed by user in assignparam
  
  #first, if this function accepts any parameters (it doesn't but just to demonstrate)
  #we'll access those parameters from params, otherwise assign default values
  #if 'parameter1' in params:
  #  mnm8_parameter = params['parameter1']
  #else:
  #  mnm8_parameter = (some default value)
  
  #we'll initialize an item to store results from a type of validation on copy operation
  #to detect suffix overlap error
  suffixoverlap_results = {}

  #create the new column, using the category key as a suffix identifier
  #the new column can be created with a copy operation, or to ensure no overlap with 
  #existing columns can make use of following internal functions
  #the first copies parallel to the validation, the second is just the validation
  
  #mdf_train, suffixoverlap_results = \
  #am.df_copy_train(mdf_train, column, column + '_mnm8', suffixoverlap_results)
  
  #or to run validation independant of copy operation could also run
  #suffixoverlap_results = \
  #am.df_check_suffixoverlap(mdf_train, [column + '_mnm8'], suffixoverlap_results)
  #(using am. for externally defined functions or self. for internally defined)
  
  #or just copy source column into new column
  mdf_train[column + '_mnm8'] = mdf_train[column].copy()
  mdf_test[column + '_mnm8'] = mdf_test[column].copy()
  
  #perform an initial (default) infill method, here we use mean as a plug, automunge
  #may separately perform a infill method per user specifications elsewhere
  #convert all values to either numeric or NaN
  mdf_train[column + '_mnm8'] = pd.to_numeric(mdf_train[column + '_mnm8'], errors='coerce')
  mdf_test[column + '_mnm8'] = pd.to_numeric(mdf_test[column + '_mnm8'], errors='coerce')

  #if we want to collect any statistics for the driftreport we could do so prior
  #to transformations and save them in the normalization dictionary below with the
  #other normalization parameters, e.g.
  min = mdf_train[column + '_mnm8'].min()
  max = mdf_train[column + '_mnm8'].max()
  
  #Now we do the specifics of the processing function, here we're demonstrating
  #the min-max scaling method capping values at 0.001 and 0.999 quantiles
  #in some cases we would address infill first, here to preserve the quantile evaluation
  #we'll do that first
  
  #get high quantile of training column for min-max scaling
  quantilemax = mdf_train[column + '_mnm8'].quantile(.999)
  
  #outlier scenario for when data wasn't numeric (nan != nan)
  if quantilemax != quantilemax:
    quantilemax = 0

  #get low quantile of training column for min-max scaling
  quantilemin = mdf_train[column + '_mnm8'].quantile(.001)
  
  if quantilemax != quantilemax:
    quantilemax = 0

  #replace values > quantilemax with quantilemax for both train and test data
  mdf_train.loc[mdf_train[column + '_mnm8'] > quantilemax, (column + '_mnm8')] \
  = quantilemax
  mdf_test.loc[mdf_train[column + '_mnm8'] > quantilemax, (column + '_mnm8')] \
  = quantilemax
  
  #replace values < quantilemin with quantilemin for both train and test data
  mdf_train.loc[mdf_train[column + '_mnm8'] < quantilemin, (column + '_mnm8')] \
  = quantilemin
  mdf_test.loc[mdf_train[column + '_mnm8'] < quantilemin, (column + '_mnm8')] \
  = quantilemin


  #note the infill method is now completed after the quantile evaluation / replacement
  #get mean of training data for infill
  mean = mdf_train[column + '_mnm8'].mean()
  
  if mean != mean:
    mean = 0
     
  #replace missing data with training set mean
  mdf_train[column + '_mnm8'] = mdf_train[column + '_mnm8'].fillna(mean)
  mdf_test[column + '_mnm8'] = mdf_test[column + '_mnm8'].fillna(mean)
    
  #this is to avoid outlier div by zero when max = min
  maxminusmin = quantilemax - quantilemin
  if maxminusmin == 0:
    maxminusmin = 1

  #perform min-max scaling to train and test sets using values derived from train
  mdf_train[column + '_mnm8'] = (mdf_train[column + '_mnm8'] - quantilemin) / \
                                (maxminusmin)
  mdf_test[column + '_mnm8'] = (mdf_test[column + '_mnm8'] - quantilemin) / \
                               (maxminusmin)


  #ok here's where we populate the data structures

  #create list of columns (here it will only be one column returned)
  nmbrcolumns = [column + '_mnm8']
  
  #The normalization dictionary is how we pass values between the "dualprocess"
  #function and the "postprocess" function. This is also where we save any metrics
  #we want to track such as to track drift in the postmunge driftreport.
  
  #Here we populate the normalization dictionary with any values derived from
  #the train set that we'll need to process the test set.
  #note that if we're returning a multicolumn set we'll need one of these
  #for each column_dict entry populated below, using that column as the key
  #note any stats collected for driftreport are also saved here.
  nmbrnormalization_dict = {column + '_mnm8' : {'quantilemin' : quantilemin, \
                                                'quantilemax' : quantilemax, \
                                                'mean' : mean, \
                                                'minimum' : min, \
                                                'maximum' : max}}

  #the column_dict_list is returned from the function call and supports the 
  #automunge methods. We populate it as follows:
  
  #initialize
  column_dict_list = []
  
  #where we're storing following
  #{'category' : 'mnm8', \ -> identifier of the category fo transform applied
  # 'origcategory' : category, \ -> category of original column in train set, passed in function call
  # 'normalization_dict' : nmbrnormalization_dict, \ -> normalization parameters of train set
  # 'origcolumn' : column, \ -> ID of original column in train set (just pass as column)
  # 'inputcolumn' : column, \ -> column serving as input to this transform
  # 'columnslist' : nmbrcolumns, \ -> a list of columns created in this transform, 
  #                                  later fleshed out to include all columns derived from same source column
  # 'categorylist' : [nc], \ -> a list of columns created in this transform
  # 'infillmodel' : False, \ -> populated elsewhere, for now enter False
  # 'infillcomplete' : False, \ -> populated elsewhere, for now enter False
  # 'suffixoverlap_results' : suffixoverlap_results, \ -> validation results for suffix overlap error
  # 'deletecolumn' : False}} -> populated elsewhere, for now enter False
  
  #for column in nmbrcolumns
  for nc in nmbrcolumns:

    column_dict = { nc : {'category' : 'mnm8', \
                          'origcategory' : category, \
                          'normalization_dict' : nmbrnormalization_dict, \
                          'origcolumn' : column, \
                          'inputcolumn' : column, \
                          'columnslist' : nmbrcolumns, \
                          'categorylist' : nmbrcolumns, \
                          'infillmodel' : False, \
                          'infillcomplete' : False, \
                          'suffixoverlap_results' : suffixoverlap_results, \
                          'deletecolumn' : False}}

    column_dict_list.append(column_dict.copy())



  return mdf_train, mdf_test, column_dict_list
  
  #where mdf_train and mdf_test now have the new column incorporated
  #and column_dict_list carries the data structures supporting the operation 
  #of automunge. (If the original column was intended for replacement it 
  #will be stricken elsewhere)
  
  
#and then since this is a method that passes values between the train
#and test sets, we'll need to define a corresponding "postprocess" function
#intended for use on just the test set

def postprocess_mnm3(mdf_test, column, postprocess_dict, columnkey, params={}):
  #where mdf_test is a dataframe of the test set
  #column is the string of the column header
  #postprocess_dict is how we carry packets of data between the 
  #functions in automunge and postmunge
  #columnkey is a key used to access stuff in postprocess_dict if needed
  #(columnkey is only valid for upstream primitive entries, if you also want to use function
  #as a downstream category we have to recreate a columnkey such as follows for normkey)
  #and params are any column specific parameters to be passed by user in assignparam

  #retrieve normalization parameters from postprocess_dict
  #normkey is the column returned from original transformation, a key used to access parameters
  normkey = column + '_mnm8'

  mean = \
  postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['mean']

  quantilemin = \
  postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['quantilemin']

  quantilemax = \
  postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['quantilemax']
  
  #(note that for cases where you might not know the suffix that was appended in advance,
  #there are methods to retrieve a normkey using properties of data structures, contact
  #the author and I can point you to them.)

  #copy original column for implementation
  mdf_test[column + '_mnm8'] = mdf_test[column].copy()


  #convert all values to either numeric or NaN
  mdf_test[column + '_mnm8'] = pd.to_numeric(mdf_test[column + '_mnm8'], errors='coerce')

  #get mean of training data
  mean = mean  

  #replace missing data with training set mean
  mdf_test[column + '_mnm8'] = mdf_test[column + '_mnm8'].fillna(mean)
  
  #this is to avoid outlier div by zero when max = min
  maxminusmin = quantilemax - quantilemin
  if maxminusmin == 0:
    maxminusmin = 1

  #perform min-max scaling to test set using values from train
  mdf_test[column + '_mnm8'] = (mdf_test[column + '_mnm8'] - quantilemin) / \
                               (maxminusmin)


  return mdf_test

#Voila

#One more demonstration, note that if we didn't need to pass any properties
#between the train and test set, we could have just processed one at a time,
#and in that case we wouldn't need to define separate functions for 
#dualprocess and postprocess, we could just define what we call a singleprocess 
#function incorporating similar data structures but passing only a single dataframe.

#Such as:
def process_mnm8(df, column, category, postprocess_dict, params = {}):
  
  #etc
  
  return df, column_dict_list
```

## Conclusion

And there you have it, you now have all you need to prepare data for 
machine learning with the Automunge platform. Feedback is welcome.

...

As a citation, please note that the Automunge package makes use of 
the Pandas, Scikit-learn, SciPy stats, and NumPy libraries. In addition 
to the default of Scikit-learn's Random Forest predictive models, 
Automunge also has options for ML infill using the AutoGluon, CatBoost, or FLAML libraries.

Wes McKinney. Data Structures for Statistical Computing in Python,
Proceedings of the 9th Python in Science Conference, 51-56 (2010)
[publisher
link](http://conference.scipy.org/proceedings/scipy2010/mckinney.html)

Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel,
Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer,
Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David
Cournapeau, Matthieu Brucher, Matthieu Perrot, Édouard Duchesnay.
Scikit-learn: Machine Learning in Python, Journal of Machine Learning
Research, 12, 2825-2830 (2011) [publisher
link](http://jmlr.org/papers/v12/pedregosa11a.html)

Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler 
Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren 
Weckesser, Jonathan Bright, St ́efan J. van der Walt, Matthew Brett, 
Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, 
Eric Jones, Robert Kern, Eric Larson, CJ Carey, Ilhan Polat, Yu Feng, 
Eric W. Moore, Jake Vand erPlas, Denis Laxalde, Josef Perktold, Robert 
Cim- rman, Ian Henriksen, E. A. Quintero, Charles R Harris, Anne M. 
Archibald, Antˆonio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and 
SciPy 1. 0 Contributors. SciPy 1.0: Fundamental Algorithms for Scientific 
Computing in Python. Nature Methods, 17:261– 272, 2020. 
doi: https://doi.org/10.1038/s41592-019-0686-2.

S. van der Walt, S. Colbert, and G. Varoquaux. The numpy array: A 
structure for efficient numerical computation. Computing in Science 
& Engineering, 13:22–30, 2011.

Nick Erickson, Jonas Mueller, Alexander Shirkov, Hang Zhang, Pedro Larroy, 
Mu Li, and Alexander Smola. AutoGluon-Tabular: Robust and Accurate AutoML 
for Structured Data [arxiv:2003.06505](https://arxiv.org/abs/2003.06505)

Anna Veronika Dorogush, Vasily Ershov, Andrey Gulin. CatBoost: gradient 
boosting with categorical features support [arXiv:1810.11363](https://arxiv.org/abs/1810.11363)

Chi Wang, Qingyun Wu, Markus Weimer, Erkang Zhu. FLAML: A Fast and Lightweight AutoML Library
[arXiv:1911.04706](https://arxiv.org/abs/1911.04706)
...

Have fun munging!!

...

You can read more about the tool through the blog posts documenting the
development online [here](https://medium.com/automunge) or for more
writing my collection of essays is titled "From
the Diaries of John Henry" which is also available online at
[turingsquared.com](https://turingsquared.com).

The Automunge website is helpfully located at 
[automunge.com](https://automunge.com).

...

This file is part of Automunge which is released under GNU General Public License v3.0.
See file LICENSE or go to https://github.com/Automunge/AutoMunge for full license details.

contact available via [automunge.com](https://automunge.com)

Copyright (C) 2018, 2019, 2020, 2021 - All Rights Reserved

Patent Pending, applications 16552857, 17021770
