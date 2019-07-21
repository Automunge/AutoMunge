# Automunge Package
# 
Automunge is a tool for automating the final steps of data wrangling of
structured (tabular) data prior to the application of machine learning.
The automunge(.) function takes as input structured training data intended 
to train a machine learning model with any corresponding labels if available 
included in the set, and also if available consistently formatted test  data 
that can then be used to generate predictions from that trained model. When 
fed pandas dataframes for these sets the function returns a series of 
transformed numpy arrays (or pandas dataframes depending on selection) which 
are numerically encoded and suitable for the direct application of machine 
learning algorithms. A user has an option between default feature engineering 
based on inferred properties of the data with feature transformations such as 
z score normalization, standard deviation bins for numerical sets, box-cox 
power law transform for all positive numerical sets, one-hot encoding for 
categorical sets, and more (full documentation below), assigning specific 
column feature engineering methods using a built in library of feature 
engineering transformations, or alternatively the passing of user-defined 
custom transformation functions incorporating simple data structures such as 
to allow custom methods to each column while still making use of all of the 
built-in features of the tool (such as ML infill, feature importance, 
dimensionality reduction, and most importantly the simplest way for the 
consistent processing of subsequently available data using just a single 
function call of the postmunge(.) function). Missing data points in the sets 
are also available to be addressed by either assigning distinct methods to 
each column or alternatively by the automated "ML infill" method which 
predicts infill using machine learning models trained on the rest of the set 
in a fully generalized and automated fashion. automunge(.) returns a python 
dictionary which can be used as an input along with a subsequent test data 
set to the function postmunge(.) for  consistent processing of test data which
wasn't available for the initial address.

In addition to it's use for feature engineering transformations, automunge(.) 
also can serve an evaluatory purpose by way of a feature importance evaluation 
through the derivation of two metrics which provide an indication for the 
importance of original and derived features towards the accuracy of a predictive 
model.

If elected, a user can also use the tool to perform a dimensionality reduction via 
principle component analysis (a type of entity embedding via unsupervised learning) 
of the data sets with the automunge(.) function or consistently for subsequently 
available data with the postmunge(.) function.

AutoMunge is now available for free pip install for your open source
python data-wrangling

```
pip install AutoMunge-pkg
```

Once installed, run this in a local session to initialize:

```
from AutoMunge_pkg import AutoMunge
am = AutoMunge.AutoMunge()
```

Where eg for train/test set processing run:

```
train, trainID, labels, \
validation1, validationID1, validationlabels1, \
validation2, validationID2, validationlabels2, \
test, testID, testlabels, \
testlabelsencoding_dict, finalcolumns_train, finalcolumns_test, \
featureimportance, postprocess_dict \
= am.automunge(df_train, df_test, etc)
```

or for subsequent consistant processing of test data, using the
dictionary returned from original application of automunge(.), run:

```
test, testID, testlabels, \
labelsencoding_dict, finalcolumns_test \ =
am.postmunge(postprocess_dict, df_test)
```

I find it helpful to pass these functions with the full range of arguments
included for reference, thus a user may simply copy and past this form.

```
#for automunge(.) function on original train and test data

train, trainID, labels, \
validation1, validationID1, validationlabels1, \
validation2, validationID2, validationlabels2, \
test, testID, testlabels, \
labelsencoding_dict, finalcolumns_train, finalcolumns_test, \
featureimportance, postprocess_dict = \
am.automunge(df_train, df_test = False, labels_column = False, trainID_column = False, \
            testID_column = False, valpercent1=0.20, valpercent2 = 0.10, \
            shuffletrain = False, TrainLabelFreqLevel = False, powertransform = False, \
            binstransform = False, MLinfill = True, infilliterate=1, randomseed = 42, \
            numbercategoryheuristic = 0.0, pandasoutput = True, \
            featureselection = True, featurepct = 1.0, featuremetric = .02, \
            featuremethod = 'pct', PCAn_components = None, PCAexcl = [], \
            ML_cmnd = {'MLinfill_type':'default', \
                       'MLinfill_cmnd':{'RandomForestClassifier':{}, 'RandomForestRegressor':{}}, \
                       'PCA_type':'default', \
                       'PCA_cmnd':{}}, \
            assigncat = {'mnmx':[], 'mnm2':[], 'mnm3':[], 'mnm4':[], 'mnm5':[], 'mnm6':[], \
                         'nmbr':[], 'nbr2':[], 'nbr3':[], 'MADn':[], 'MAD2':[], \
                         'bins':[], 'bint':[], \
			 'bxcx':[], 'bxc2':[], 'bxc3':[], 'bxc4':[], \
                         'bnry':[], 'text':[], \
                         'date':[], 'dat2':[], 'wkdy':[], 'bshr':[], 'hldy':[], \
                         'excl':[], 'exc2':[], 'exc3':[], 'null':[]}, \
            assigninfill = {'stdrdinfill':[], 'MLinfill':[], 'zeroinfill':[], \
                            'adjinfill':[], 'meaninfill':[], 'medianinfill':[]}, \
            transformdict = {}, processdict = {})

```

Please remember to save the automunge(.) returned object postprocess_dict 
such as using pickle library, which can then be later passed to the postmunge(.) 
function to consistently process subsequently available data.

```
#for postmunge(.) function on subsequently available test data
#using the postprocess_dict object returned from original automunge(.) application

test, testID, testlabels, \
labelsencoding_dict, finalcolumns_test = \
am.postmunge(postprocess_dict, df_test, testID_column = False, \
             labelscolumn = False, pandasoutput=True)
```



The functions depend on pandas dataframe formatted train and test data
with consistent column labels (column headers are required). The functions 
return Numpy arrays numerically encoded and normalized such as to make 
them suitable for
direct application to a machine learning model in the framework of a
user's choice, including sets for the various activities of a generic
machine learning project such as training, hyperparameter tuning
validation (validation1), final  validation (validation2), or data intended 
for use in generation of predictions from the trained model (test set). The
functions also return a few other sets such as labels, column headers,
ID sets, and etc if elected - a full list of returned arrays is below.

When left to automation, the function works by inferring a category of 
data based on properties of each column to select the type of processing 
function to apply, for example whether a column is a numerical, categorical,
binary, or time-series set. Alternately, a user can pass column header IDs to 
assign specific processing functions to distinct columns - which processing functions
may be pulled from the internal library of transformations or alternately user
defined.
Normalization parameters from the initial automunge application are
saved to a returned dictionary for subsequent consistent processing of test data
that wasn't available at initial address with the postmunge(.) function. 

The feature engineering transformations are recorded with a series of suffixes 
appended to the column header title in the returned sets, for example the 
application of z-score normalization returns a column with header 'origname' + '_nmbr'. 
The function allows feature engineering methods for the training data, test data,
and any column designated for labels if included with the sets.

In automation, for numerical data, the functions generate a series of derived
transformations resulting in multiple child columns, with
transformations including z-score normalization (nmbr), and standard
deviation bins for values in range <-2, -2-1, -10, 01, 12, >2 from the
mean. For numerical sets with all positive values the functions also optionally can 
return a power-law transformed set using the box-cox method, along with
a corresponding set with z-score normalization applied. For time-series
data the model segregates the data by time-scale (year, month, day,
hour, minute, second) and returns a set for each with z-score
normalization applied. For binary categorical data the functions
return a single column with 1/0 designation. For multimodal categorical
data the functions return one-hot encoded sets using the naming
convention 'origname' + '_category'. (I believe this automation of the
one-hot encoding method to be a particularily useful feature of the
tool.) For all cases the functions generate a supplemental column (NArw)
with a boolean identifier for cells that were subject to infill due to
missing or improperly formatted data.

The functions also include a method we call 'ML infill' which if elected
predicts infill for missing values in both the train and test sets using
machine learning models trained on the rest of the set in a fuly
generalized and automated fashion. The ML infill works by initially
applying infill using traditional methods such as mean for a numerical
set, most common value for a binary set, and a boolean identifier for
categorical. The functions then generate a column specific set of
training data, labels, and feature sets for the derivation of infill.
The column's trained model is included in the outputted dictionary for
application of the same model in the postmunge function. Alternately, a
user can pass column headers to assign different infill methods to distinct 
columns.

The automunge(.) function also includes a method for feature importance 
evaluation, in which metrics are derived to measure the impact to predictive 
accuracy of original source columns as well as relative importance of 
derived columns using a permutation importance method. Permutation importance 
method was inspired by a fast.ai lecture and more information can be found in 
the paper "Beware Default Random Forest Importances" by Terrence Parr, Kerem 
Turgutlu, Christopher Csiszar, and Jeremy Howard. This method currently makes 
use of Scikit-Learns Random Forest predictors.

The function also includes a method we call 'LabelFreqLevel' which
if elected applies multiples of the feature sets associated with each
label category in the returned training data so as to enable
oversampling of those labels which may be underrepresented in the
training data. This method is available for categorical labels or also
for numerical labels when the label processing includes standard deviation
bins. This method is expected to improve downstream model
accuracy for training data with uneven distribution of labels. For more
on the class imbalance problem see "A systematic study of the class imbalance 
problem in convolutional neural networks" - Buda, Maki, Mazurowski.

The function also can perform dimensionality reduction of the sets via 
principle component analysis (PCA). The function automatically performs a 
transformation when the number of features is more than 50% of the number
of observations in the train set (this is a somewhat arbitrary heuristic).
Alternately, the user can pass a desired number of features and their 
preference of type and parameters between linear PCA, Sparse PCA, or Kernel 
PCA - all currently implemented in Scikit-Learn.

The application of the automunge and postmunge functions requires the
assignment of the function to a series of named sets. We suggest using
consistent naming convention as follows:

```
train, trainID, labels, \
validation1, validationID1, validationlabels1, \
validation2, validationID2, validationlabels2, \ 
test, testID, testlabels, \
labelsencoding_dict, finalcolumns_train, finalcolumns_test, \
featureimportance, postprocess_dict \
= am.automunge(df_train, ...)
```

The full set of arguments available to be passed are given here, with
explanations provided below: 

```
am.automunge(df_train, df_test = False, labels_column = False, trainID_column = False, \
            testID_column = False, valpercent1=0.20, valpercent2 = 0.10, \
            shuffletrain = False, TrainLabelFreqLevel = False, powertransform = False, \
            binstransform = False, MLinfill = True, infilliterate=1, randomseed = 42, \
            numbercategoryheuristic = 0.0, pandasoutput = True, \
            featureselection = True, featurepct = 1.0, featuremetric = .02, \
            featuremethod = 'pct', PCAn_components = None, PCAexcl = [], \
            ML_cmnd = {'MLinfill_type':'default', \
                       'MLinfill_cmnd':{'RandomForestClassifier':{}, 'RandomForestRegressor':{}}, \
                       'PCA_type':'default', \
                       'PCA_cmnd':{}}, \
            assigncat = {'mnmx':[], 'mnm2':[], 'mnm3':[], 'mnm4':[], 'mnm5':[], 'mnm6':[], \
                         'nmbr':[], 'nbr2':[], 'nbr3':[], 'MADn':[], 'MAD2':[], \
                         'bins':[], 'bint':[], \
			 'bxcx':[], 'bxc2':[], 'bxc3':[], 'bxc4':[], \
                         'bnry':[], 'text':[], \
                         'date':[], 'dat2':[], 'wkdy':[], 'bshr':[], 'hldy':[], \
                         'excl':[], 'exc2':[], 'exc3':[], 'null':[]}, \
            assigninfill = {'stdrdinfill':[], 'MLinfill':[], 'zeroinfill':[], \
                            'adjinfill':[], 'meaninfill':[], 'medianinfill':[]}, \
            transformdict = {}, processdict = {})
```

Or for the postmunge function:

```
#for postmunge(.) function on subsequentlky available test data
#using the postprocess_dict object returned from original automunge(.) application

test, testID, testlabels, \
labelsencoding_dict, finalcolumns_test = \
```

With the full set of arguments available to be passed as:

```
am.postmunge(postprocess_dict, df_test, testID_column = False, \
             labelscolumn = False, pandasoutput=True)
```

Note that the only required argument to the automunge function is the
train set dataframe, the other arguments all have default values if
nothing is passed. The postmunge function requires as minimum the
postprocess_dict object (a python dictionary returned from the application of
automunge) and a dataframe test set consistently formatted as those sets
that were originally applied to automunge.

...

Here now are descriptions for the returned sets from automunge, which
will be followed by descriptions of the arguments which can be passed to
the function, followed by similar treatment for postmunge returned sets
and arguments.

...

## automunge returned sets:

* train: a numerically encoded set of data intended to be used to train a
downstream machine learning model in the framework of a user's choice

* trainID: the set of ID values corresponding to the train set if a ID
column(s) was passed to the function. This set may be useful if the shuffle
option was applied.

* labels: a set of numerically encoded labels corresponding to the
train set if a label column was passed. Note that the function
assumes the label column is originally included in the train set. Note
that if the labels set is a single column a returned numpy array is 
flattened (e.g. [[1,2,3]] converted to [1,2,3] )

* validation1: a set of training data carved out from the train set
that is intended for use in hyperparameter tuning of a downstream model.

* validationID1: the set of ID values coresponding to the validation1
set

* validationlabels1: the set of labels coresponding to the validation1
set

* validation2: the set of training data carved out from the train set
that is intended for the final validation of a downstream model (this
set should not be applied extensively for hyperparameter tuning).

* validationID2: the set of ID values coresponding to the validation2
set.

* validationlabels2: the set of labels coresponding to the validation2
set

* test: the set of features, consistently encoded and normalized as the
training data, that can be used to generate predictions from a
downstream model trained with train. Note that if no test data is
available during initial address this processing will take place in the
postmunge(.) function. 

* testID: the set of ID values coresponding to the test set.

* testlabels: a set of numerically encoded labels corresponding to the
test set if a label column was passed. Note that the function
assumes the label column is originally included in the train set.

* labelsencoding_dict: a dictionary that can be used to reverse encode
predictions that were generated from a downstream model (such as to
convert a one-hot encoded set back to a single categorical set).

* finalcolumns_train: a list of the column headers corresponding to the
training data. Note that the inclusion of suffix appenders is used to
identify which feature engineering transformations were applied to each
column.

* finalcolumns_test: a list of the column headers corresponding to the
test data. Note that the inclusion of suffix appenders is used to
identify which feature engineering transformations were applied to each
column. Note that this list should match the one preceeding.

* featureimportance: a dictionary containing summary of feature importance
ranking and metrics for each of the derived sets. Note that the metric
value provides an indication of the importance of the original source
column such that larger value suggests greater importance, and the metric2 
value provides an indication of the relative importance of columns derived
from the original source column such that smaller metric2 value suggests 
greater relative importance. One can print the values here such as with
this code:

```
#to inspect values returned in featureimportance object one could run
for keys,values in featureimportance.items():
    print(keys)
    print('metric = ', values['metric'])
    print('metric2 = ', values['metric2'])
    print()
```


* postprocess_dict: a returned python dictionary that includes
normalization parameters and trained machine learning models used to
generate consistent processing of test data that wasn't available at
initial address of automunge. It is recommended that this dictionary be
saved on each application used to train a downstream model so that it may
be passed to postmunge(.) to consistently process subsequently available
test data.

...

## automunge(.) passed arguments

```
am.automunge(df_train, df_test = False, labels_column = False, trainID_column = False, \
            testID_column = False, valpercent1=0.20, valpercent2 = 0.10, \
            shuffletrain = False, TrainLabelFreqLevel = False, powertransform = False, \
            binstransform = False, MLinfill = True, infilliterate=1, randomseed = 42, \
            numbercategoryheuristic = 0.0, pandasoutput = True, \
            featureselection = True, featurepct = 1.0, featuremetric = .02, \
            featuremethod = 'pct', PCAn_components = None, PCAexcl = [], \
            ML_cmnd = {'MLinfill_type':'default', \
                       'MLinfill_cmnd':{'RandomForestClassifier':{}, 'RandomForestRegressor':{}}, \
                       'PCA_type':'default', \
                       'PCA_cmnd':{}}, \
            assigncat = {'mnmx':[], 'mnm2':[], 'mnm3':[], 'mnm4':[], 'mnm5':[], 'mnm6':[], \
                         'nmbr':[], 'nbr2':[], 'nbr3':[], 'MADn':[], 'MAD2':[], \
                         'bins':[], 'bint':[], \
			 'bxcx':[], 'bxc2':[], 'bxc3':[], 'bxc4':[], \
                         'bnry':[], 'text':[], \
                         'date':[], 'dat2':[], 'wkdy':[], 'bshr':[], 'hldy':[], \
                         'excl':[], 'exc2':[], 'exc3':[], 'null':[]}, \
            assigninfill = {'stdrdinfill':[], 'MLinfill':[], 'zeroinfill':[], \
                            'adjinfill':[], 'meaninfill':[], 'medianinfill':[]}, \
            transformdict = {}, processdict = {})
```

* df_train: a pandas dataframe containing a structured dataset intended
for use to subsequently train a machine learning model. The set at a
minimum should be 'tidy' meaning a single column per feature and a
single row per observation. If desired the set may include a row ID
column and a column intended to be used as labels for a downstream
training operation. The set must have string column headers. The tool
supports the inclusion of non-index-range column as index or multicolumn 
index (requires named index columns). Such index types are added to the 
returned "ID" sets which are consistently shuffled and partitioned as 
the train and test sets.

* df_test: a pandas dataframe containing a structured dataset intended for
use to generate predictions from a downstream machine learning model
trained from the automunge returned sets. The set must be consistantly
formated as the train set with consistent column labels (This set may 
optionally contain a labels column if one was included in the train set 
although it's inclusion is not required). If desired the set may include 
a row ID column or a column intended for use as labels. A user may pass 
False if this set not available. The tool supports the inclusion of non-
index-range column as index or multicolumn index (requires named index 
columns). Such index types are added to the returned "ID" sets which are 
consistently shuffled and partitioned as the train and test sets.

* labels_column: a string of the column title for the column from the
df_train set intended for use as labels in training a downstream machine
learning model. The function defaults to False for cases where the
training set does not include a label column.

* trainID_column: a string of the column title for the column from the
df_train set intended for use as a row identifier value (such as could
be sequential numbers for instance). The function defaults to False for
cases where the training set does not include an ID column. A user can 
also pass a list of string columns titles such as to carve out multiple
columns to be excluded from processing but consistently partitioned.

* testID_column: a string of the column title for the column from the
df_test set intended for use as a row identifier value (such as could be
sequential numbers for instance). The function defaults to False for
cases where the training set does not include an ID column. A user can 
also pass a list of string columns titles such as to carve out multiple
columns to be excluded from processing but consistently partitioned.

* valpercent1: a float value between 0 and 1 which designates the percent
of the training data which will be set aside for the first validation
set (generally used for hyperparameter tuning of a downstream model).
Note the default here is set at 20% but that is fairly an arbitrary
value and a user may wish to deviate for different size sets. Note that
this value may be set to 0 if no validation set is needed (such as may
be the case for k-means validation).

* valpercent2: a float value between 0 and 1 which designates the percent
of the training data which will be set aside for the second validation
set (generally used for final validation of a model prior to release).
Note the default here is set at 10% but that is fairly an arbitrary
value and a user may wish to deviate for different size sets. This value
may also be set to 0 if desired.

* shuffletrain: a boolean identifier (True/False) which indicates if the
rows in df_train will be shuffled prior to carving out the validation
sets. Note that if this value is set to False then the validation sets
will be pulled from the bottom x% sequential rows of the dataframe.
(Where x% is the sum of validation ratios.) Note that if this value is
set to False although the validations will be pulled from sequential
rows, the split between validaiton1 and validation2 sets will be
randomized.

* TrainLabelFreqLevel: a boolean identifier (True/False) which indicates
if the TrainLabelFreqLevel method will be applied to oversample training
data associated with underrepresented labels. The method adds multiples
to training data rows for those labels with lower frequency resulting in
an (approximately) levelized frequency. This defaults to False. Note that
this feature may be applied to numerical label sets if the processing
applied to the set includes standard deviation bins.

* powertransform: a boolean identifier (True/False) which indicates if the
box-cox power law transform will be included for default application to
numerical sets with all positive values. This defaults to False. Note
that after application of box-cox transform child columns are generated
for a subsequent z-score normalization as well as a set of bins
associated with number of standard deviations from the mean. For more on
box-cox transform I'd recommend a google search or something.

* binstransform: a boolean identifier (True/False) which indicates if the
numerical sets will receive bin processing such as to generate child
columns with boolean identifiers for number of standard deviations from
the mean, with groups for values <-2, -2-1, -10, 01, 12, and >2 . Note
that the bins and bint transformations are the same, only difference is
that the bint transform assumes the column has already been normalized
while the bins transform does not. This value defaults to True.

* MLinfill: a boolean identifier (True/False) which indicates if the ML
infill method will be applied as a default to predict infill for missing 
or improperly formatted data using machine learning models trained on the
rest of the set.

* infilliterate: an integer indicating how many applications of the ML
infill processing are to be performed for purposes of predicting infill.
The assumption is that for sets with high frequency of missing values
that multiple applications of ML infill may improve accuracy although
note this is not an extensively tested hypothesis.

* randomseed: a postitive integer used as a seed for randomness in data
set shuffling, ML infill, and fearture importance  algorithms. This 
defaults to 42, a nice round number.

* forcetocategoricalcolumns: a list of string identifiers of column titles
for those columns which are to be treated as categorical to allow
one-hot encoding. This may be useful e.g. for numerically encoded
categorical sets such as like zip codes or phone numbers or something
which would otherwise be evaluated as numerical and subject to
normalization. *update this aregument no longer supported, a user can
instead assign distinct methods to each column with assigncat per below, 
such as assigning a column to category 'text' for categorical.

* numbercategoryheuristic: a float value between 0 and 1 which will be
used as a heuristic to identify numerically encoded columns which are to
be treated as categorical sets, for example a value of 0.10 here would
indicate that if the number of distinct values in a column is less than
10% of the total number of rows then this should be treated as a
categorical set and receive one-hot encoding. I may later change this
heuristic to just a integer number of distinct values, in hindsight that
may have made more sense. This defaults to 0.00 meaning the heuristic is
not applied.

* pandasoutput: a selector for format of returned sets. Defaults to False
for returned Numpy arrays. If set to True returns pandas dataframes
(note that index is not preserved in the train/validation split, an ID
column may be passed for index identification).

* featureselection: a boolean identifier telling the function whether to
perform a feature importance evaluation. If selected automunge will
return a summary of feature importance findings in the featureimportance
returned dictionary. This also activates the trimming of derived sets
that did not meet the importance threshold if [featurepct < 1.0 and 
featuremethod = 'pct'] or if [fesaturemetric > 0.0 and featuremethod = 
'metric'].

* featurepct: the percentage of derived sets that are kept in the output
based on the feature importance evaluation. Note that NArw columns are
excluded from the trimming for now (the inclusion of NArws in trimming
will likely be included in a future expansion). This item only used if
featuremethod passed as 'pct' (the default).

* featuremetric: the feature importance metric below which derived sets
are trimmed from the output. Note that this item only used if
featuremethod passed as 'metric'.

* featuremethod: can be passed as either 'pct' or 'metric' to select which
feature importance method is used for trimming the derived sets.

* PCAn_components: a user can pass an integer to define the number of PCA
derived features for purposes of dimensionality reduction, such integer to 
be less than the otherwise returned number of sets. Function will default 
to kernel PCA for all non-negative sets or otherwise Sparse PCA. Also if
this values passed as a float <1.0 then linear PCA will be applied such 
that the returned number of sets are the minimum number that can reproduce
that percent of the variance. Note this can also be passed in conjunction 
with assigned PCA type or parameters in the ML_cmnd object.

* PCAexcl: a list of column headers for columns that are to be excluded from
any application of PCA

* ML_cmnd: 

```
ML_cmnd = {'MLinfill_type':'default', \
           'MLinfill_cmnd':{'RandomForestClassifier':{}, 'RandomForestRegressor':{}}, \
           'PCA_type':'default', \
           'PCA_cmnd':{}}, \
```
The ML_cmnd allows a user to pass parameters to the predictive algorithms
used for ML infill and feature importance evaluation. Currently the only
option for 'MLinfill_type' is default which uses Scikit-Learn's Random 
Forest implementation, the intent is to add other options in a future extension.
For example, a user wishing to pass a custom parameter of max_depth for to the 
Random Forest algorithms could pass:
_
```
ML_cmnd = {'MLinfill_type':'default', \
           'MLinfill_cmnd':{'RandomForestClassifier':{'max_depth':4}, \
                            'RandomForestRegressor':{'max_depth':4}}, \
           'PCA_type':'default', \
           'PCA_cmnd':{}}, \
           
#(note that currently unable to pass RF parameters to criterion and n_jobs)
```
A user can also assign specific methods for PCA transforms. Current PCA_types
supported include 'PCA', 'SparsePCA', and 'KernelPCA', all via Scikit-Learn.
Note that the n_components are passed seperately with the PCAn_components 
argument noted above. A user can also pass parameters to the PCA functions
through the PCA_cmnd, for example one could pass a kernel type for KernelPCA
as:
```
ML_cmnd = {'MLinfill_type':'default', \
           'MLinfill_cmnd':{'RandomForestClassifier':{}, \
                            'RandomForestRegressor':{}}, \
           'PCA_type':'KernelPCA', \
           'PCA_cmnd':{'kernel':'sigmoid'}}, \
           
#Also note that SparsePCA currenlty doesn't have available
#n_jobs or normalize_components, and similarily KernelPCA 
#doesn't have available n_jobs.
```
Note that the PCA is currently defaulted to active for cases where the 
train set number of features is >0.50 the number of rows. A user can 
change this ratio by passing 'PCA_cmnd':{'col_row_ratio':0.22}} for 
instance. Also a user can simply turn off default PCA transforms by 
passing 'PCA_cmnd':{'PCA_type':'off'}. A user can also exclude returned
boolean (0/1) columns from any PCA application by passing 
'PCA_cmnd':{'bool_PCA_excl':True}
such as could potentially result in memory savings.


* assigncat:

```
#Here are the current trasnformation options built into our library, which
#we are continuing to build out. A user may also define their own.

        assigncat = {'mnmx':[], 'mnm2':[], 'mnm3':[], 'mnm4':[], 'mnm5':[], 'mnm6':[], \
                     'nmbr':[], 'nbr2':[], 'nbr3':[], 'MADn':[], 'MAD2':[], \
                     'bins':[], 'bint':[], \
		     'bxcx':[], 'bxc2':[], 'bxc3':[], 'bxc4':[], \
                     'bnry':[], 'text':[], \
                     'date':[], 'dat2':[], 'wkdy':[], 'bshr':[], 'hldy':[], \
                     'excl':[], 'exc2':[], 'exc3':[], 'null':[]}, \
```         

A user may add column identifier strings to each
of these lists to designate this specific processing approach. Note that
this processing category will serve as the "root" of the tree of
transforms as defined in the transformdict. Note that additional
categories may be passed if defined in the passed transformdict and
processdict. An example of usage here could be if a user wanted to only
process numerical columns 'nmbrcolumn1' and 'nmbrcolumn2' with z-score
normalization instead of the full range of numerical derivations they
could pass assigncat = {'nbr2':['nmbrcolumn1'], ...}. We'll provide 
details on each of the built-in library of transformations below.

* assigninfill 
```
#Here are the current infill options built into our library, which
#we are continuing to build out.
assigninfill = {'stdrdinfill':[], 'MLinfill':[], 'zeroinfill':[], \
                'adjinfill':[], 'meaninfill':[], 'medianinfill':[]}, \
```
A user may add column identifier strings to each of these lists to 
designate the column-specific infill approach for missing or
improperly formated values. Note that this infill category defaults to
MLinfill if nothing assigned and the MLinfill argument to automunge is
set to True. stdrdinfill mean mean for numeric sets, most common for 
binary, and new column boolean for categorical. zeroinfill means inserting 
the integer 0 to missing cells. adjinfill means passing the value from the 
preceding row to missing cells. meaninfill means inserting the mean 
derived from the train set to numeric columns. medianinfill means 
inserting the median derived from the train set to numeric columns. (Note
currently boolean columns derived from numeric are not supported for 
mean/median and for those cases default to those infill from stdrdinfill.)

* transformdict: allows a user to pass a custom tree of transformations.
Note that a user may define their own 4 character string "root"
identifiers for a series of processing steps using the categories 
of processing already defibned in our library and then assign columns 
in assigncat, or for custom processing functions this method should 
be combined with processdict which is only slightly more complex. 
For example, a user wishing to define a new set of transformations 
for numerical series 'newt' that combines NArows, min-max, box-cox, z-score, 
and standard deviation bins could do so by passing a trasnformdict as:
```
transformdict =  {'newt' : {'greatgrandparents' : [], \
                            'grandparents' : ['NArw'], \
                            'parents' : ['bxc4'], \
                            'siblings': [], \
                            'auntsuncles' : ['mnmx'], \
                            'cousins' : [], \
                            'children' : [], \
                            'niecesnephews' : [], \
                            'coworkers' : [], \
                            'friends' : []}}
                                    
#Where since bxc4 is passed as a parent, this will result in pulling
#ofspring keys from the bxcx family tree, which has a nbr2 key as children.

#from automunge library:
    transform_dict.update({'bxc4' : {'greatgrandparents' : [], \
                                     'grandparents' : ['NArw'], \
                                     'parents' : ['bxcx'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [], \
                                     'children' : ['nbr2'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
                                     
#note that 'nmbr' is passed as a children primitize meaning if nbr2 key
#has any offspring those will be produced as well.


```
Basically here 'newt' is the key and when passed to one of the family primitives
the corresponding process function is applied, and if it is passed to a family
primitive with downstream offspring then those offspring keys are pulled from
that key's family tree. For example, here mnmx is passed as an auntsuncles which
means the mnmx processing function is applied with no downstream offspring. The
bxcx key is passed as a parent which means the bxcx trasnform is applied coupled
with any downstream transforms from the bxcx key family tree, which we also show.
Note the family primitives tree of transformations can be summarized as:
```
'greatgrandparents' : upstream / first generation / suplements column / with offspring
'grandparents' :      upstream / first generation / supplements column / no offspring
'parents' :           upstream / all generations / replaces column / with offspring
'siblings':           upstream / all generations / supplements column / with offspring
'auntsuncles' :       upstream / all generations / replaces column / no offspring
'cousins' :           upstream / all generations / supplements column / no offspring
'children' :          downstream parents / all generations / replaces column / with offspring
'niecesnephews' :     downstream siblings / all generations / supplements column / with offspring
'coworkers' :         downstream auntsuncles / all generations / replaces column / no offspring
'friends' :           downstream cousins / all generations / supplements column / no offspring
```
Note that when we define a new transform such as 'newt' above, we also need 
to define a corresponding processdict entry for the new category, which we 
demonstrate here:


* processdict: allows a user to define their own processing functions 
corresponding to new transformdict keys. We'll describe the entries here:
```
#for example 
processdict =  {'newt' : {'dualprocess' : None, \
			  'singleprocess' : None, \
			  'postprocess' : None, \
        		  'NArowtype' : 'numeric', \
      			  'MLinfilltype' : 'numeric', \
           		  'labelctgy' : 'mnmx'}}

#A user should pass either a pair of processing functions to both 
#dualprocess and postprocess, or alternatively just a single processing
#function to singleprocess, and pass None to those not used.
#For now, if just using the category as a root key and not as a family primitive, 
#can simply pass None to all the processing slots. We'll demonstrate their 
#composition and data structures for custom processing functions later in this 
#document.

#dualprocess: for passing a processing function in which normalization 
#             parameters are derived from properties of the training set
#             and jointly process the train set and if available test set

#singleprocess: for passing a processing function in which no normalization
#               parameters are needed from the train set to process the
#               test set, such that train and test sets processed seperately

#postprocess: for passing a processing function in which normalization 
#             parameters originally derived from the train set are applied
#             to seperately process a test set

#NArowtype: can be entries of either 'numeric', 'justNaN', or 'exclude' where
#			'numeric' refers to columns where non-numeric entries are subject
#					  to infill
#			'justNaN' refers to columns where only NaN entries are subject
#			          to infill
#			'exclude' refers to columns where no infill will be performed

#MLinfilltype: can be entries of 'numeric', 'singlct', 'multirt', 'exclude'
#              'multisp', 'exclude', or 'label' where
#			   'numeric' refers to columns where predictive algorithms treat
#			   as a regression for numeric sets
#			   'singlect' refers to columns where category gives a single
#			   column where predictive algorithms treat as a boolean classifier
#			   'multirt' refers to category returning multiple columns where 
#			   predictive algorithms treat as a multi modal classifier
#			   'exclude' refers to categories excluded from predcitive address
#			   'multisp' tbh I think this is just a duplicate of multirt, a
#			   future update may strike this one
#			   'label' refers to categories specifically intended for label
#			   processing

```

Ok well we'll demonstrate further below how to build custom processing functions,
for now this just gives you sufficient tools to build sets of processing using
the built in sets in the library.

...

# postmunge

The postmunge(.) function is intended to consistently process subsequently available
and consistently formatted test data with just a single function call. It requires 
passing the postprocess_dict object returned from the original application of automunge 
and that the passed test data have consistent column header labeling as the original 
train set.

```

#for postmunge(.) function on subsequently available test data
#using the postprocess_dict object returned from original automunge(.) application

#Remember to initialize automunge
from AutoMunge_pkg import AutoMunge
am = AutoMunge.AutoMunge()


#Then we can run postmunge function as:

test, testID, testlabels, \
labelsencoding_dict, finalcolumns_test = \
am.postmunge(postprocess_dict, df_test, testID_column = False, \
             labelscolumn = False, pandasoutput=True)
```
             


## postmunge(.) returned sets:
Here now are descriptions for the returned sets from postmunge, which
will be followed by descriptions of the arguments which can be passed to
the function. 

* test: the set of features, consistently encoded and normalized as the
training data, that can be used to generate predictions from a model
trained with the np_train set from automunge.

* testID: the set of ID values coresponding to the test set.

* testlabels: a set of numerically encoded labels corresponding to the
test set if a label column was passed. Note that the function
assumes the label column is originally included in the train set. Note
that if the labels set is a single column a returned numpy array is 
flattened (e.g. [[1,2,3]] converted to [1,2,3] )

* labelsencoding_dict: this is the same labelsencoding_dict returned from
automunge, it's used in case one wants to reverse encode predicted labels

* finalcolumns_test: a list of the column headers corresponding to the
test data. Note that the inclusion of suffix appenders is used to
identify which feature engineering transformations were applied to each
column. Note that this list should match the one from automunge.

...


## postmunge(.) passed arguments

```

#for postmunge(.) function on subsequently available test data
#using the postprocess_dict object returned from original automunge(.) application

#Remember to initialize automunge
from AutoMunge_pkg import AutoMunge
am = AutoMunge.AutoMunge()


#Then we can run postmunge function as:

test, testID, testlabels, \
labelsencoding_dict, finalcolumns_test = \
am.postmunge(postprocess_dict, df_test, testID_column = False, \
             labelscolumn = False, pandasoutput=True)
```

* postprocess_dict: this is the dictionary returned from the initial
application of automunge which included normalization parameters to
facilitate consistent processing of test data to the original processing
of the train set. This requires a user to remember to download the
dictionary at the original application of automunge, otherwise if this
dictionary is not available a user can feed this subsequent test data to
the automunge along with the original train data exactly as was used in
the original automunge call.

* df_test: a pandas dataframe containing a structured dataset intended for
use to generate predictions from a machine learning model trained from
the automunge returned sets. The set must be consistantly formated as
the train set with consistent column labels. If desired the set may
include an ID column. The tool supports the inclusion of non-index-range 
column as index or multicolumn index (requires named index columns). Such 
index types are added to the returned "ID" sets which are consistently 
shuffled and partitioned as the train and test sets.

* testID_column: a string of the column title for the column from the
df_test set intended for use as a row identifier value (such as could be
sequential numbers for instance). The function defaults to False for
cases where the training set does not include an ID column. A user can 
also pass a list of string columns titles such as to carve out multiple
columns to be excluded from processing but consistently partitioned.

* labelscolumn: default to False indicates that a labels column is not 
included in the test set passed to postmunge. A user can either pass
True or the string ID of the labels column, noting that it is a requirement
that the labels column header string must be consistent with that from
the original train set.

* pandasoutput: a selector for format of returned sets. Defaults to False
for returned Numpy arrays. If set to True returns pandas dataframes
(note that index is not preserved, an ID column may be passed for index
identification).

...

## Library of Transformations

Automunge has a built in library of transformations that can be passed for
specific columns with assigncat. A column if left unassigned will defer to
the automated default methods.  For example, a user can pass a min-max
scaling method to a specific column 'col1' with: 
```
assigncat = {'mnmx':['col1']}
```
When a user assigns a column to a specific category, that category is treated
as the root category for the tree of transformations. Each key has an 
associated transformation function, and that transformation function is only
applied if the root key is also found in the tree of family primitives. The
tree of family primitives, as introduced earlier, applies first the first
generation transforms of greatgrandparents and grandparents specific to the 
original root key, and then any transforms for keys found in upstream primitives
i.e. parents/siblings/auntsuncles/cousins. If a transform is applied for a 
primitive that includes downstream offspring, such as greatgrandparents/parents/
siblings, then the family tree for that key with offspring is inspected to determine
downstream offspring categories, for example if we have a parents key of 'mnmx',
then any children/niecesnephews/coworkers/friends in the 'mnmx' family tree will
be applied as parents/siblings/auntsuncles/cousins, respectively. Note that the
designation for supplements/replaces refers purely to the question of whether the
column to which the trasnform is being applied is kept in place or removed. Please
note that it is a quirck of the function that no original column can be left in 
place without the application of some transformation such as to allow the building
of the apppripriate data structures, thus at least one replacement primitive must
always be included. If a user does wish to leave a column in place unaltered, they 
can simply assign that column to the 'excl' root category.

Now we'll start here by listing again the family tree primitives for those root 
categories built into the automunge library. After that we'll give a quick 
narrative for each of the associated transformation functions. First here again
are the family tree primitives.

```
'greatgrandparents' : upstream / first generation / suplements column / with offspring
'grandparents' :      upstream / first generation / supplements column / no offspring
'parents' :           upstream / all generations / replaces column / with offspring
'siblings':           upstream / all generations / supplements column / with offspring
'auntsuncles' :       upstream / all generations / replaces column / no offspring
'cousins' :           upstream / all generations / supplements column / no offspring
'children' :          downstream parents / all generations / replaces column / with offspring
'niecesnephews' :     downstream siblings / all generations / supplements column / with offspring
'coworkers' :         downstream auntsuncles / all generations / replaces column / no offspring
'friends' :           downstream cousins / all generations / supplements column / no offspring
```

And here arethe series of family trees currently built into the internal library.

```
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
    transform_dict.update({'nbr3' : {'greatgrandparents' : [], \
                                     'grandparents' : ['NArw'], \
                                     'parents' : ['nmbr'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : ['bint']}})
    transform_dict.update({'MADn' : {'greatgrandparents' : [], \
                                     'grandparents' : ['NArw'], \
                                     'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['MADn'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    transform_dict.update({'MAD2' : {'greatgrandparents' : [], \
                                     'grandparents' : ['NArw'], \
                                     'parents' : ['MAD2'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : ['bint']}})
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
    transform_dict.update({'mnm6' : {'greatgrandparents' : [], \
                                     'grandparents' : ['NArw'], \
                                     'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mnm6'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    transform_dict.update({'mnm7' : {'greatgrandparents' : [], \
                                     'grandparents' : [], \
                                     'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mnmx', 'bins'], \
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
    transform_dict.update({'dat2' : {'greatgrandparents' : [], \
                                     'grandparents' : ['NArw'], \
                                     'parents' : [], \
                                     'siblings': ['bshr', 'wkdy', 'hldy'], \
                                     'auntsuncles' : ['date'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    transform_dict.update({'bxc2' : {'greatgrandparents' : [], \
                                     'grandparents' : ['NArw'], \
                                     'parents' : ['bxc2'], \
                                     'siblings': ['nmbr'], \
                                     'auntsuncles' : [], \
                                     'cousins' : [], \
                                     'children' : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    transform_dict.update({'bxc3' : {'greatgrandparents' : [], \
                                     'grandparents' : ['NArw'], \
                                     'parents' : ['bxc3'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [], \
                                     'children' : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    transform_dict.update({'bxc4' : {'greatgrandparents' : [], \
                                     'grandparents' : ['NArw'], \
                                     'parents' : ['bxcx'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [], \
                                     'children' : ['nbr2'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    transform_dict.update({'wkdy' : {'greatgrandparents' : [], \
                                     'grandparents' : [], \
                                     'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['wkdy'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    transform_dict.update({'bshr' : {'greatgrandparents' : [], \
                                     'grandparents' : [], \
                                     'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bshr'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    transform_dict.update({'hldy' : {'greatgrandparents' : [], \
                                     'grandparents' : [], \
                                     'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['hldy'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    transform_dict.update({'bins' : {'greatgrandparents' : [], \
                                     'grandparents' : [], \
                                     'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bins'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    transform_dict.update({'bint' : {'greatgrandparents' : [], \
                                     'grandparents' : [], \
                                     'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bint'], \
                                     'cousins' : [], \
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
    transform_dict.update({'exc2' : {'greatgrandparents' : [], \
                                     'grandparents' : [], \
                                     'parents' : ['exc2'], \
                                     'siblings': [], \
                                     'auntsuncles' : ['exc2'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['bins'], \
                                     'friends' : []}})
    transform_dict.update({'exc3' : {'greatgrandparents' : [], \
                                     'grandparents' : [], \
                                     'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['exc2'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    

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
```

Here is a quick description of the transformation functions associated 
with each key which can be assigned to a primitive (and not just used as 
a root key). We're continuing to build out this library of transformations.

* NArw: produces a column of boolean identifiers for rows in the source
column with missing or improperly formatted values.
* nmbr/nbr2/nbr3: z-score normalization
* mnmx/mnm2/mnm5: vanilla min-max scaling
* mnm3/mnm4: min-max scaling with outliers capped at 0.01 and 0.99 quantiles
* mnm6: min-max scaling with test set capped at min/max of train set
* bnry: converts sets with two values to boolean identifiers
* text: converts categorical sets to one-hot encoded set of boolean identifiers
* bxcx/bxc2/bxc3/bxc4: performs Box-Cox power law transformation
* date/dat2: for datetime formatted data, segregates data by time scale to multiple
columns (year/month/day/hour/minute/second) and then performs z-score normalization
* wkdy: boolean identifier indicating whether a datetime object is a weekday
* bshr: boolean identifier indicating whether a datetime object is a business
hour (9-5, time zone unaware)
* hldy: boolean identifier indicating whether a datetime object is a US Federal
holiday
* bins: for numerical sets, outputs a set of 6 columns indicating where a
value fell with respect to number of standard deviations from the mean of the
set (i.e. <-2, -2-1, -10, 01, 12, >2)
* bint: comparable to bins except assumes that source data was already normalized
* null: deletes source column
* excl: passes source column un-altered
* exc2: passes source column unaltered except for infill


...

## Custom Transformation Functions

Ok final item on the agenda, we're going to demonstrate methods to create custom
transformation functions, such that a user may customize the feature engineering
while building on all of the extremely useful built in features of automunge such
as infill methods including ML infill, feature importance, dimensionality reduction,
and perhaps most importantly the simplest possible way for consistent processing 
of subsequently available data with just a single function call. The transformation
functions will need to be channeled through pandas and incorproate a handful of 
simple data structures, which we'll demonstrate below.

Let's say we want to recreate the mm3 category which caps outliers at 0.01 and 0.99
quantiles, but instead make it the 0.001 and 0.999 quantiles. Well we'll call this 
cateogry mnm8. So in order to pass a custom transformation function, first we'll need to 
define a new root category trasnformdict and a corresponding processdict.

```
#Let's creat ea really simple family tree for the new root category mnmn8 which
#simply creates a column identifying any rows subject to infill (NArw), performs 
#a z-score normalization, and seperately performs a version of the new transform
#mnm8 which we'll define below.

transformdict = {'mnm8' : {'greatgrandparents' : [], \
                           'grandparents' : ['NArw'], \
                           'parents' : [], \
                           'siblings': [], \
                           'auntsuncles' : ['mnm8', 'nmbr'], \
                           'cousins' : [], \
                           'children' : [], \
                           'niecesnephews' : [], \
                           'coworkers' : [], \
                           'friends' : []}, \

#Note that since this mnm8 requires passing normalization parameters derived
#from the train set to process the test set, we'll need to create twop sep[erate 
#trasnformations functions, the first a "dualprocess" function that processes
#both the train and if available a test set swimultaneously, and the second
#a "postprocess" that only processes the test set on it's own.

#So what's being demosnrtated here is that we're passing the functions under
#dualprocess and postprocess that we'll define below.

processdict = {'mnm8' : {'dualprocess' : process_mnm8_class, \
                         'singleprocess' : None, \
                         'postprocess' : postprocess_mnm8_class, \
                         'NArowtype' : 'numeric', \
                         'MLinfilltype' : 'numeric', \
                         'labelctgy' : 'mnm8'}}

#Now we have to define the custom processing functions which we are passing through
#the processdict to automunge.

#Insterad of demosntrating the full functions, I'll just demonstrate the
#requirements


#Here we'll define a "dualprocess" function intended to process both a train and
#test set simulateously. We'll also need to create a seperate "postprocess"
#function intended to just process the test set.

#define the function
def process_mnm8_class(mdf_train, mdf_test, column, category, \
                       postprocess_dict):
  #where
  #mdf_train is the train data set (pandas dataframe)
  #mdf_test is the consistently formatted test dataset (if no test data 
  #set is available a dummy set will be passed in it's place)
  #column is the string identifying the column header
  #category is the 4 charcter string category identifier, here is will be 'mnm8'
  #postprocess_dict is an object we pass to share data between functions if needed

  #create thee new column, using the catehgory key as a suffix identifier
  
  #copy source column into new column
  mdf_train[column + '_mnm8'] = mdf_train[column].copy()
  mdf_test[column + '_mnm8'] = mdf_test[column].copy()
  
  
  #perform an initial infill method, here we use mean as a plug, automunge
  #will seperately perform a infill method per user specifications elsewhere
  #convert all values to either numeric or NaN
  mdf_train[column + '_mnm8'] = pd.to_numeric(mdf_train[column + '_mnm8'], errors='coerce')
  mdf_test[column + '_mnm8'] = pd.to_numeric(mdf_test[column + '_mnm8'], errors='coerce')


  
  #Now we do the specifics of the processing function, here we're demonstrating
  #the min-max scaling method capping values at 0.001 and 0.999 quantiles
  
  #get maximum value of training column
  quantilemax = mdf_train[column + '_mnm8'].quantile(.999)

  #get minimum value of training column
  quantilemin = mdf_train[column + '_mnm8'].quantile(.001)

  #replace values > quantilemax with quantilemax
  mdf_train.loc[mdf_train[column + '_mnm8'] > quantilemax, (column + '_mnm8')] \
  = quantilemax
  mdf_test.loc[mdf_train[column + '_mnm8'] > quantilemax, (column + '_mnm8')] \
  = quantilemax
  #replace values < quantile10 with quantile10
  mdf_train.loc[mdf_train[column + '_mnm8'] < quantilemin, (column + '_mnm8')] \
  = quantilemin
  mdf_test.loc[mdf_train[column + '_mnm8'] < quantilemin, (column + '_mnm8')] \
  = quantilemin


  #note the infill method is now completed after the quantile evaluation / replacement
  #get mean of training data
  mean = mdf_train[column + '_mnm8'].mean()    
  #replace missing data with training set mean
  mdf_train[column + '_mnm8'] = mdf_train[column + '_mnm8'].fillna(mean)
  mdf_test[column + '_mnm8'] = mdf_test[column + '_mnm8'].fillna(mean)


  #perform min-max scaling to train and test sets using values from train
  mdf_train[column + '_mnm8'] = (mdf_train[column + '_mnm8'] - quantilemin) / \
                                (quantilemax - quantilemin)
  mdf_test[column + '_mnm8'] = (mdf_test[column + '_mnm8'] - quantilemin) / \
                               (quantilemax - quantilemin)


  #ok here's where we populate the data structures

  #create list of columns (here it will only be one column returned)
  nmbrcolumns = [column + '_mnm8']
  
  #The normalization dictionary is how we pass values between the "dualprocess"
  #function and the "postprocess" function
  
  #Here we populate the normalization dictionary with any values derived from
  #the train set that we'll need to process the test set.
  nmbrnormalization_dict = {column + '_mnm8' : {'quantilemin' : quantilemin, \
                                                'quantilemax' : quantilemax, \
                                                'mean' : mean}}

  #the column_dict_list is returned from the function call and supports the 
  #automunge methods. We populate it as follows:
  
  #initialize
  column_dict_list = []
  
  #where we're storing following
  #{'category' : 'mnm8', \ -> identifier of the category fo transform applied
  # 'origcategory' : category, \ -> category of original column in train set, passed in function call
  # 'normalization_dict' : nmbrnormalization_dict, \ -> normalization parameters of train set
  # 'origcolumn' : column, \ -> ID of original column in train set
  # 'columnslist' : nmbrcolumns, \ -> a list of columns created in this transform, 
  #                                  later fleshed out to include all columns derived from same source column
  # 'categorylist' : [nc], \ -> a list of columns created in this transform
  # 'infillmodel' : False, \ -> populated elsewhere, for now enter False
  # 'infillcomplete' : False, \ -> populated elsewhere, for now enter False
  # 'deletecolumn' : False}} -> populated elsewhere, for now enter False
  
  for nc in nmbrcolumns:

    if nc[-5:] == '_mnm8':

      column_dict = { nc : {'category' : 'mnm8', \
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
  
  #where mdf_train and mdf_test now have the new column incorporated
  #and column_dict_list carries the data structures supporting the operation 
  #of automunge. (If the original columkjn was intended for replacement it 
  #will be stricken elsewhere)
  
  
#and then since this is a method that passes values between the train
#and test sets, we'll need to define a corresponding "postproces" function
#intended for use on just the test set

def postprocess_mnm3_class(mdf_test, column, postprocess_dict, columnkey):
  #where mdf_test is a dataframe fo the test set
  #column is the string of the column header
  #postprocess_dict is how we carry packets of datra between the 
  #functions in automunge
  #columnkey is a key used to access stuff in postprocess_dict if needed


  #retrieve normalization parameters from postprocess_dict
  normkey = column + '_mnm8'

  mean = \
  postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['mean']

  quantilemin = \
  postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['quantilemin']

  quantilemax = \
  postprocess_dict['column_dict'][normkey]['normalization_dict'][normkey]['quantilemax']

  #copy original column for implementation
  mdf_test[column + '_mnm8'] = mdf_test[column].copy()


  #convert all values to either numeric or NaN
  mdf_test[column + '_mnm8'] = pd.to_numeric(mdf_test[column + '_mnm8'], errors='coerce')

  #get mean of training data
  mean = mean  

  #replace missing data with training set mean
  mdf_test[column + '_mnm8'] = mdf_test[column + '_mnm8'].fillna(mean)

  #perform min-max scaling to test set using values from train
  mdf_test[column + '_mnm8'] = (mdf_test[column + '_mnm8'] - quantilemin) / \
                               (quantilemax - quantilemin)


  return mdf_test

#Voila

#One more demonstration, note that if we didn't need to pass any properties
#between the train and test set, we could have just processed one at a time,
#and in that case we wouldn't need to define seperate functions for 
#dualprocess and postprocess, we could just define what we call a singleprocess 
#function incorproating similar data strucures but without only a single dataframe 
#passed

#Such as:
def process_mnm4_class(df, column, category, postprocess_dict):
  
  #etc
  
  return return df, column_dict_list

#For a full demonstration check out my essay 
"Automunge 1.79: An Open Source Platform for Feature Engineering"


```

And there you have it, you now have all you need to wrangle data on the 
automunge platform. Feedback is welcome.


...

As a citation, please note that the Automunge package makes use of both
the Pandas and Sciki-learn libraries.

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

...

Have fun munging!

...

You can read more about the tool through the blog posts documenting the
development on medium [here](https://medium.com/automunge) or for more
writing I recently completed my first collection of essays titled "From
the Diaries of John Henry" which is also available on Medium
[here](https://medium.com/from-the-diaries-of-john-henry).

The AutoMunge website is helpfully located at URL
[automunge.com](automunge.com).

...

Patent Pending
