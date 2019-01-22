# AutoMunge Package

AutoMunge is a tool for automating the final steps of data wrangling prior to the 
application of machine learning. The automunge(.) function processes structured training 
data and if available consistently formatted test data that can then be used to generate 
predictions from a trained downstream model. When fed pandas dataframes for these sets the
function returns transformed numpy arrays or pandas dataframes (depending on selection) 
which are numerically encoded, with feature transformations such as z score normalization, 
standard deviation bins for numerical sets, box-cox power law transform for all positive 
numerical sets, one-hot encoding for categorical sets, and more (full documentation below). 
Missing data points in the set are also addressed by the "ML infill" method which predict 
infill using machine learning models trained on the rest of the set in a fully generalized 
and automated fashion. automunge(.) also returns a python dictionary which can be used as 
an input along with a subsequent test data set to the function postmunge(.) for  consistent 
processing of test data which wasn't available for the initial address.

AutoMunge is now available for free pip install for your open source python data-wrangling

pip install AutoMunge-pkg

Once installed:

from AutoMunge_pkg import AutoMunge

am = AutoMunge.AutoMunge()

Where eg for train/test set processing run:

np_train, np_trainID, np_labels, np_validation1, np_validationID1, \
np_validationlabels1, np_validation2, np_validationID2, np_validationlabels2, \
np_test, np_testID, labelsencoding_dict, finalcolumns_train, finalcolumns_test,  \
postprocess_dict \
= am.automunge(df_train, df_test, etc)

or for subsequent consistant processing of test data, using the dictionary returned from
automunge(.), run:

np_test, np_testID, labelsencoding_dict, finalcolumns_test \
= am.postmunge(postprocess_dict, df_test)

One can save the dictionary "postprocess_dict" returned from automunge(.) with pickle for 
instance.


The functions depend on pandas dataframe formatted train and test data with consistent 
column labels. The functions return Numpy arrays numerically encoded and normalized such 
as to make them suitable for direct application to a machine learning model in the 
framework of a user's choice, including sets for the various activities of a generic 
machine learning project such as training, hyperparameter tuning validation (validation1), 
or final  validation (validation2). The functions also return a few other sets such as 
labels, column headers, ID sets, and etc if elected - a full list of returned arrays is 
below.

The functions work by inferring a category of data based on 
properties of each column to select the type of processing function to apply. 
Normalization parameters from the initial automunge application are saved to a dictionary
for subsequent consistent processing of test data that wasn't available at initial 
address. The feature engineering transformations are recorded with a series of suffixes
appended to the column header title, for example the application of z-score normalization
returns a column with header 'origname' + '_nmbr'. The function provides consistent 
processing between training data and the column designated for labels if included with 
the train set.

For numerical data, the functions generate a series of derived transformations resulting 
in multiple child columns, with transformations including z-score normalization (nmbr), 
and standard deviation bins for values in range <-2, -2-1, -10, 01, 12, >2 from the mean. 
For numerical sets with all positive values the functions also returns a power-law 
transformed set using the box-cox method, along with a corresponding set with z-score 
normalization applied. For time-series data the model segregates the data by time-scale 
(year, month, day, hour, minute, second) and returns a set for each with z-score 
normalization applied (I suspect this method could probably be improved by adding bins
for normal business hours, holidays, weekends, etc - potential future extension). For 
binary categorical data the functions return a single column with 1/0 designation. For 
multimodal categorical data the functions return one-hot encoded sets using the naming 
convention 'origname' + '_category'. (I believe this automation of the one-hot encoding 
method to be a particularily useful feature of the tool.) For all cases the functions 
generate a supplemental column (NArw) with a boolean identifier for cells that were 
subject to infill due to missing or improperly formatted data.

The functions also include a method we call 'ML infill' which if elected predicts infill 
for missing values in both the train and test sets using machine learning models trained 
on the rest of the set in a fuly generalized and automated fashion. The ML infill works by 
initially applying infill using traditional methods such as mean for a numerical set, most
common value for a binary set, and a boolean identifier for categorical. The functions 
then generate a column specific set of training data, labels, and feature sets for the 
derivation of infill. The column's trained model is included in the outputted dictionary
for application of the same model in the postmunge function.

The function also includes a method we call 'TrainLabelFreqLevel' which if elected applies
multiples of the feature sets associated with each label category in the returned training
data so as to enable oversampling of those labels which may be underrepresented in the 
training data. This method is expected to improve downstream model accuracy for training 
data with uneven distribution of labels. For more on this method see "A systematic study 
of the class imbalance problem in convolutional neural networks" - Buda, Maki, Mazurowski.

The application of the automunge and postmunge functions requires the assignment of the 
function to a series of named sets. We suggest using consistent naming convention as
follows:

np_train, np_trainID, np_labels, np_validation1, np_validationID1, \
np_validationlabels1, np_validation2, np_validationID2, np_validationlabels2, \
np_test, np_testID, labelsencoding_dict, finalcolumns_train, finalcolumns_test,  \
postprocess_dict \
= am.automunge(df_train, ...)

The full set of arguments available to be passed are given here, with explanations 
provided below:
am.automunge(df_train, df_test = False, labels_column = False, trainID_column = False, \
            testID_column = False, valpercent1=0.20, valpercent2 = 0.10, \
            shuffletrain = True, TrainLabelFreqLevel = True, powertransform = True, \
            binstransform = True, MLinfill = True, infilliterate=1, randomseed = 42, \
            forcetocategoricalcolumns = [], numbercategoryheuristic = 0.000, \
            excludetransformscolumns = [], pandasoutput = False):


Or for the postmunge function:

np_test, np_testID, labelsencoding_dict, finalcolumns_test \
= am.postmunge(postprocess_dict, df_test, ...)

With the full set of arguments available to be passed as:

am.postmunge(postprocess_dict, df_test, testID_column = False, pandasoutput = False)

Note that the only required argument to the automunge function is the train set dataframe,
the other arguments all have default values if nothing is passed. The postmunge function
requires as minimum the postprocess_dict, a python dictionary returned from the 
application of automunge, and a dataframe test set consistently formatted as those sets 
that were originally applied to automunge.

...

Here now are descriptions for the returned sets from automunge, which will be followed by 
descriptions of the arguments which can be passed to the function, followed by similar 
treatment for postmunge returned sets and arguments.

...

automunge returned sets:

np_train: a numerically encoded set of data intended to be used to train a downstream
machine learning model in the framework of a user's choice

np_trainID: the set of ID values corresponding to the np-train set if a ID column was
passed to the function. This set may be useful if the shuffle option was applied.

np_labels: a set of numerically encoded labels corresponding to the np_train set if a 
label column was passed. Note that the function assumes the label column is originally 
included in the train set. The encoding of the labels is consistent to the methods used 
for the training data, including supplemental feature engineering derived columns.

np_validation1: a set of training data carved out from the np_train set that is intended
for use in hyperparameter tuning of a downstream model. 

np_validationID1: the set of ID values coresponding to the np_validation1 set

np_validationlabels1: the set of labels coresponding to the np_validation1 set

np_validation2: the set of training data carved out from the np_train set that is 
intended for the final validation of a downstream model (this set should not be applied
extensively for hyperparameter tuning).

np_validationID2: the set of ID values coresponding to the np_validation2 set.

np_validationlabels2: the set of labels coresponding to the np_validation2 set

np_test: the set of features, consistently encoded and normalized as the training data, 
that can be used to generate predictions from a downstream model trained with np_train. 
Note that if no test data is available during initial address this processing will 
take place in the postmunge(.) function.

np_testID: the set of ID values coresponding to the np_test set.

labelsencoding_dict: a dictionary that can be used to reverse encode predictions that 
were generated from a downstream model (such as to convert a one-hot encoded set back to
a single categorical set).

finalcolumns_train: a list of the column headers corresponding to the training data. Note 
that the inclusion of suffix appenders is used to identify which feature engineering 
transformations were applied to each column.

finalcolumns_test: a list of the column headers corresponding to the test data. Note 
that the inclusion of suffix appenders is used to identify which feature engineering 
transformations were applied to each column. Note that this list should match the one 
preceeding.

postprocess_dict: a returned python dictionary that includes normalization parameters 
and trained machine learning models used to generate consistent processing of test data 
that wasn't available at initial address of automunge. It is recommended that this 
dictionary be saved on each application used to train a downstream model.

...

automunge(.) passed arguments

am.automunge(df_train, df_test = False, labels_column = False, trainID_column = False, \
            testID_column = False, valpercent1=0.20, valpercent2 = 0.10, \
            shuffletrain = True, TrainLabelFreqLevel = True, powertransform = True, \
            binstransform = True, MLinfill = True, infilliterate=1, randomseed = 42, \
            forcetocategoricalcolumns = [], numbercategoryheuristic = 0.000, \
            excludetransformscolumns = [], pandasoutput = False):
            
df_train: a pandas dataframe containing a structured dataset intended for use to train a 
downstream machine learning model. The set at a minimum should be 'tidy' meaning a single 
column per feature and a single row per observation. If desired the set may include a row 
ID column and a column intended to be used as labels for a downstream training operation. 

df_test: a pandas dataframe containing a structured dataset intended for use to generate 
predictions from a downstream machine learning model trained from the automunge returned 
sets. The set must be consistantly formated as the train set with consistent column 
labels (save for any designated labels column from the train set). If desired the set may 
include a row ID column.

labels_column: a string of the column title for the column from the df_train set intended 
for use as labels in training a downstream machine learning model. The function defaults 
to False for cases where the training set does not include a label column.

trainID_column: a string of the column title for the column from the df_train set intended
for use as a row identifier value (such as could be sequential numbers for instance). The 
function defaults to False for cases where the training set does not include an ID column.

testID_column: a string of the column title for the column from the df_test set intended
for use as a row identifier value (such as could be sequential numbers for instance). The 
function defaults to False for cases where the training set does not include an ID column.

valpercent1: a float value between 0 and 1 which designates the percent of the training 
data which will be set aside for the first validation set (generally used for 
hyperparameter tuning of a downstream model). Note the default here is set at 20% but that
is fairly an arbitrary value and a user may wish to deviate for different size sets. Note 
that this value may be set to 0 if no validation set is needed (such as may be the case 
for k-means validation).

valpercent2: a float value between 0 and 1 which designates the percent of the training 
data which will be set aside for the second validation set (generally used for final 
validation of a model prior to release). Note the default here is set at 10% but that is 
fairly an arbitrary value and a user may wish to deviate for different size sets. This 
value may also be set to 0 if desired. 

shuffletrain: a boolean identifier (True/False) which indicates if the rows in df_train 
will be shuffled prior to carving out the validation sets. Note that if this value is set 
to False then the validation sets will be pulled from the bottom x% sequential rows of the
dataframe. (Where x% is the sum of validation ratios.) Note that if this value is set to 
False although the validations will be pulled from sequential rows, the split between 
validaiton1 and validation2 sets will be randomized.

TrainLabelFreqLevel: a boolean identifier (True/False) which indicates if the 
TrainLabelFreqLevel method will be applied to oversample training data associated with 
underrepresented labels. The method adds multiples to training data rows for those labels 
with lower frequency resulting in an (approximately) levelized frequency. This defaults to
True.

powertransform: a boolean identifier (True/False) which indicates if the box-cox power law
transform will be included for application of numerical sets with all positive values. 
This defaults to True. Note that after application of box-cox transform child columns are
generated for a subsequent z-score normalization as well as a set of bins associated with 
number of standard deviations from the mean. For more on box-cox transform I'd recommend a
google search or something.

binstransform: a boolean identifier (True/False) which indicates if the numerical sets 
will receive bin processing such as to generate child columns with boolean identifiers for
number of standard deviations from the mean, with groups for values <-2, -2-1, -10, 01, 12,
and >2 . Note that the bins and bint transformations are the same, only difference is that
the bint transform assumes the column has already been normalized while the bins transform
does not. This value defaults to True.

MLinfill: a boolean identifier (True/False) which indicates if the ML infill method will 
be applied to predict infill for missing or improperly formatted data using machine 
learning models trained on the rest of the set.

infilliterate: an integer indicating how many applications of the ML infill processing are
to be performed for purposes of predicting infill. The assumption is that for sets with 
high frequency of missing values that multiple applications of ML infill may improve 
accuracy although note this is not an extensively tested hypothesis.

randomseed: a postitive integer used as a seed for randomness in data set shuffling as 
well as the ML infill algorithms. This defaults to 42, a nice round number.

forcetocategoricalcolumns: a list of string identifiers of column titles for those columns
which are to be treated as categorical to allow one-hot encoding. This may be useful e.g. 
for numerically encoded categorical sets such as like zip codes or phone numbers or 
something which would otherwise be evaluated as numerical and subject to normalization.

numbercategoryheuristic: a float value between 0 and 1 which will be used as a heuristic 
to identify numerically encoded columns which are to be treated as categorical sets, for
example a value of 0.10 here would indicate that if the number of distinct values in a 
column is less than 10% of the total number of rows then this should be treated as a 
categorical set and receive one-hot encoding. I may later change this heuristic to just a
integer number of distinct values, in hindsight that may have made more sense. This 
defaults to 0.00 meaning the heuristic is not applied.

excludetransformscolumns: a list of string identifiers of column titles for those columns
which are to be excluded from processing. This may be useful for columns that were already
subject to other feature engineering methods. Note that these excluded from transform 
columns will need to be numericallly encoded if the ML infill methods are to be applied to
the other columns.

pandasoutput: a selector for format of returned sets. Defaults to False for returned Numpy
arrays. If set to True returns pandas dataframes (note that index is not preserved in the 
train/validation split, an ID column may be passed for index identification).

...

postmunge returned sets:

np_test: the set of features, consistently encoded and normalized as the training data, 
that can be used to generate predictions from a model trained with the np_train set from 
automunge.

np_testID: the set of ID values coresponding to the test set.

labelsencoding_dict: this is the same labelsencoding_dict returned from automunge, it's 
just returned here again in case you need it. Kind of redundant.

finalcolumns_test: a list of the column headers corresponding to the test data. Note 
that the inclusion of suffix appenders is used to identify which feature engineering 
transformations were applied to each column. Note that this list should match the one 
from automunge.

...


postmunge(.) passed arguments

am.postmunge(postprocess_dict, df_test, testID_column = False, pandasoutput = False)

postprocess_dict: this is the dictionary returned from the initial application of 
automunge which included normalization parameters to facilitate consistent processing of 
test data to the original processing of the train set. This requires a user to remember to
download the dictionary at the original application of automunge, otherwise if this 
dictionary is not available a user can feed this subsequent test data to the automunge 
along with the original train data exactly as was used in the original automunge call.

df_test: a pandas dataframe containing a structured dataset intended for use to generate 
predictions from a machine learning model trained from the automunge returned sets. The 
set must be consistantly formated as the train set with consistent column labels. If 
desired the set may include a row ID number.

testID_column: a string of the column title for the column from the df_test set intended
for use as a row identifier value (such as could be sequential numbers for instance). The 
function defaults to False for cases where the training set does not include an ID column.

pandasoutput: a selector for format of returned sets. Defaults to False for returned Numpy
arrays. If set to True returns pandas dataframes (note that index is not preserved, an ID 
column may be passed for index identification).

...

Have fun munging!

...

You can read more about the tool through the blog posts documenting the development on 
medium [here](https://medium.com/automunge) or for more writing I recently completed my 
first collection of essays titled "From the Diaries of John Henry" which is also available
on Medium [here](https://medium.com/from-the-diaries-of-john-henry).

The AutoMunge website is helpfully located at URL [automunge.com](automunge.com).

...

Patent Pending
