# Automunge
# 
Automunge is a platform for preparing tabular data for machine learning. A user
has options between automated inference of column properties for application of
appropriate simple feature engineering methods, or may also assign to distinct 
columns custom feature engineering transformations, custom sets (e.g. "family 
trees") of feature engineering transformations, or custom infill methods. The 
feature engineering transformation functions may be accessed from the internal 
library of transformation categories (aka a "feature store"), or may also 
be user defined with minimal requirements of simple data structures for 
incorporation into the platform. The tool includes options for automated 
feature importance evaluation, automated derivation of infill predictions
using machine learning models trained on the set in a fully generalized and
automated fashion, automated preparation for oversampling for class imbalance, 
automated dimensionality reductions such as based on feature importance or 
principle component analysis, automated evaluation of data property drift 
between training data and subsequent data, and perhaps most importantly the 
simplest means for consistent processing of additional data with just a single 
function call. In short, we make machine learning easy.

The automunge(.) function takes as input structured training data intended 
to train a machine learning model with any corresponding labels if available 
included in the set, and also if available consistently formatted test data 
that can then be used to generate predictions from that trained model. When 
fed pandas dataframes or numpy arrays for these sets the function returns a 
series of transformed numpy arrays or pandas dataframes (per selection) which 
are numerically encoded and suitable for the direct application of machine 
learning algorithms. A user has an option between default feature engineering 
based on inferred properties of the data with feature transformations such as 
z-score normalization, standard deviation bins for numerical sets, box-cox 
power law transform for all positive numerical sets, binary encoding for 
categorical sets, time series agregation to sin and cos transforms (with bins
for business hours, weekdays, and holidays), and more (full documentation 
below); assigning specific column feature engineering methods using a built-in 
library of feature engineering transformations; or alternatively the passing 
of user-defined custom transformation functions incorporating simple data 
structures such as to allow custom methods to each column while still making 
use of all of the built-in features of the tool (such as ML infill, feature 
importance, dimensionality reduction, and most importantly the simplest way 
for the consistent processing of subsequently available data using just a 
single function call of the postmunge(.) function). Missing data points in the 
sets are also available to be addressed by either assigning distinct methods 
to each column or alternatively by the automated "ML infill" method which 
predicts infill using machine learning models trained on the rest of the set 
in a fully generalized and automated fashion. automunge(.) returns a python 
dictionary which can be used as an input along with a subsequent test data 
set to the function postmunge(.) for  consistent processing of test data 
which wasn't available for the initial address.

In addition to it's use for feature engineering transformations, automunge(.) 
also can serve an evaluatory purpose by way of a feature importance evaluation 
through the derivation of a series of metrics which provide an indication for 
the importance of original and derived features towards the accuracy of a 
predictive model.

If elected, a user can also use the tool to perform a dimensionality reduction 
via principle component analysis (a type of entity embedding via unsupervised 
learning) of the data sets with the automunge(.) function or consistently for 
subsequently available data with the postmunge(.) function.

AutoMunge is now available for free pip install for your open source
python data-wrangling

```
pip install Automunge
```

```
#or to upgrade (we currently roll out upgrades pretty frequently)
pip install Automunge --upgrade
```

Once installed, run this in a local session to initialize:

```
from Automunge import Automunger
am = Automunger.AutoMunge()
```

Where eg for train set processing  with default parameters run:

```
train, trainID, labels, \
validation1, validationID1, validationlabels1, \
validation2, validationID2, validationlabels2, \
test, testID, testlabels, \
testlabelsencoding_dict, finalcolumns_train, finalcolumns_test, \
featureimportance, postprocess_dict \
= am.automunge(df_train)
```

or for subsequent consistant processing of test data, using the
dictionary returned from original application of automunge(.), run:

```
test, testID, testlabels, \
labelsencoding_dict, postreports_dict \
= am.postmunge(postprocess_dict, df_test)
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
            testID_column = False, valpercent1=0.0, valpercent2 = 0.0, floatprecision = 32, \
            shuffletrain = False, TrainLabelFreqLevel = False, powertransform = False, \
            binstransform = False, MLinfill = False, infilliterate=1, randomseed = 42, \
	    LabelSmoothing_train = False, LabelSmoothing_test = False, LabelSmoothing_val = False, \
            LSfit = False, numbercategoryheuristic = 63, pandasoutput = True, NArw_marker = True, \
            featureselection = False, featurepct = 1.0, featuremetric = .02, featuremethod = 'default', \
            Binary = False, PCAn_components = None, PCAexcl = [], \
            ML_cmnd = {'MLinfill_type':'default', \
                       'MLinfill_cmnd':{'RandomForestClassifier':{}, 'RandomForestRegressor':{}}, \
                       'PCA_type':'default', \
                       'PCA_cmnd':{}}, \
            assigncat = {'mnmx':[], 'mnm2':[], 'mnm3':[], 'mnm4':[], 'mnm5':[], 'mnm6':[], \
	                 'mean':[], 'mea2':[], 'mea3':[], \
		         'nmbr':[], 'nbr2':[], 'nbr3':[], 'MADn':[], 'MAD2':[], 'MAD3':[], \
		         'dxdt':[], 'd2dt':[], 'd3dt':[], 'dxd2':[], 'd2d2':[], 'd3d2':[], \
		         'nmdx':[], 'nmd2':[], 'nmd3':[], 'mmdx':[], 'mmd2':[], 'mmd3':[], \
		         'bins':[], 'bint':[], 'bsor':[], 'pwr2':[], 'por2':[], \
			 'bnwd':[], 'bnwK':[], 'bnwM':[], 'bnwo':[], 'bnKo':[], 'bnMo':[], \
			 'bnep':[], 'bne7':[], 'bne9':[], 'bneo':[], 'bn7o':[], 'bn9o':[], \
		         'bxcx':[], 'bxc2':[], 'bxc3':[], 'bxc4':[], \
		         'log0':[], 'log1':[], 'sqrt':[], \
		         'bnry':[], 'text':[], 'txt2':[], 'txt3':[], '1010':[], 'or10':[], \
		         'ordl':[], 'ord2':[], 'ord3':[], 'ord4':[], 'om10':[], 'mmor':[], \
			 'Utxt':[], 'Utx2':[], 'Utx3':[], 'Uor3':[], 'Uor6':[], 'U101':[], \
                         'splt':[], 'spl2':[], 'spl3':[], 'spl4':[], 'spl5':[], \
                         'nmrc':[], 'nmr2':[], 'nmr3':[], 'nmcm':[], 'nmc2':[], 'nmc3':[], \
			 'nmr7':[], 'nmr8':[], 'nmr9':[], 'nmc7':[], 'nmc8':[], 'nmc9':[], \
                         'ors2':[], 'ors5':[], 'ors6':[], 'ors7':[], \
			 'or11':[], 'or12':[], 'or15':[], 'or17':[], 'or19':[], 'or20':[], \
		         'date':[], 'dat2':[], 'dat6':[], 'wkdy':[], 'bshr':[], 'hldy':[], \
		         'yea2':[], 'mnt2':[], 'mnt6':[], 'day2':[], 'day5':[], \
		         'hrs2':[], 'hrs4':[], 'min2':[], 'min4':[], 'scn2':[], \
		         'excl':[], 'exc2':[], 'exc3':[], 'null':[], 'eval':[], 'copy':[]}, \
            assigninfill = {'stdrdinfill':[], 'MLinfill':[], 'zeroinfill':[], 'oneinfill':[], \
                            'adjinfill':[], 'meaninfill':[], 'medianinfill':[], 'modeinfill':[]}, \
            assignparam = {}, transformdict = {}, processdict = {}, evalcat = False, \
            printstatus = True)
```

Please remember to save the automunge(.) returned object postprocess_dict 
such as using pickle library, which can then be later passed to the postmunge(.) 
function to consistently process subsequently available data.

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

test, testID, testlabels, \
labelsencoding_dict, postreports_dict = \
am.postmunge(postprocess_dict, df_test, testID_column = False, \
             labelscolumn = False, pandasoutput=True, printstatus = True, \
             TrainLabelFreqLevel = False, featureeval = False, driftreport = False, ]\
	     LabelSmoothing = False, LSfit = False)
```


The functions depend on pandas dataframe formatted train and test data
or numpy arrays with consistent order of columns between train and test data. 
The functions return numpy arrays or pandas dataframes numerically encoded 
and normalized such as to make them suitable for direct application to a 
machine learning model in the framework of a user's choice, including sets for 
the various activities of a generic machine learning project such as training, 
hyperparameter tuning validation (validation1), final  validation (validation2), 
or data intended for use in generation of predictions from the trained model 
(test set). The functions also return a few other sets such as labels, column 
headers, ID sets, and etc if elected - a full list of returned arrays is below.

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
application of z-score normalization returns a column with header origname + '_nmbr'. 
As another example, for one-hot encoded sets the set of columns are returned with
header origname + '_category' where category is the category from the set indicated 
by a column. Each transformation category has a unique suffix appender.

In automation, for numerical data, the functions generate a series of derived
transformations resulting in multiple child columns. For numerical data, if the
powertransform option is selected distribution properties are evaluated for 
potential application of z-score normalization, min-max scaling, power law transform 
via box-cox method, or mean absolute deviation scaling. Otherwise numerical data 
defaults to z-score, with z-score normalization options for standard
deviation bins for values in range <-2, -2-1, -10, 01, 12, >2 from the
mean. For numerical sets with all positive values the functions also optionally
can return a power-law transformed set using the box-cox method, along with
a corresponding set with z-score normalization applied. For time-series
data the model segregates the data by time-scale (year, month, day, hour, minute, 
second) and returns year z-score normalized, a pair of sets for combined month/day 
and combined hour / minute / second with sin and cos transformations at period of 
time-scale, and also returns binned sets identifying business hours, weekdays, and 
US holidays. For binary categorical data the functions return a single column with 
1/0 designation. For multimodal categorical data the functions return one-hot 
encoded sets using the naming convention origname + _ + category. (I believe this 
automation of the one-hot encoding method to be a particularily useful feature of 
the tool.) For all cases the functions generate a supplemental column (NArw)
with a boolean identifier for cells that were subject to infill due to missing or 
improperly formatted data. (Please note that I don't consider the current methods 
of numerical set distribution evaluation highly sophisticated and have some work to 
do here). 

The functions also include a method we call 'ML infill' which if elected
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
default. Extension into more sophisticated methods such as that may employ 
automated hyperparameter tuning for instance is intended for a future extension.

The automunge(.) function also includes a method for feature importance 
evaluation, in which metrics are derived to measure the impact to predictive 
accuracy of original source columns as well as relative importance of 
derived columns using a permutation importance method. Permutation importance 
method was inspired by a fast.ai lecture and more information can be found in 
the paper "Beware Default Random Forest Importances" by Terrence Parr, Kerem 
Turgutlu, Christopher Csiszar, and Jeremy Howard. This method currently makes 
use of Scikit-Learns Random Forest predictors. I believe the metric we refer to
as metric2 which evaluates relative importance between features derived from the 
same source column is a unique approach.

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

The function also can perform dimesnionality reduction of the sets via
the Binary option which takes the set of columns with boolean {1/0} encodings
and applies a binary transform to reduce the number of columns.

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
= am.automunge(df_train)
```

The full set of arguments available to be passed are given here, with
explanations provided below: 

```
train, trainID, labels, \
validation1, validationID1, validationlabels1, \
validation2, validationID2, validationlabels2, \
test, testID, testlabels, \
labelsencoding_dict, finalcolumns_train, finalcolumns_test, \
featureimportance, postprocess_dict = \
am.automunge(df_train, df_test = False, labels_column = False, trainID_column = False, \
            testID_column = False, valpercent1=0.0, valpercent2 = 0.0, floatprecision = 32, \
            shuffletrain = False, TrainLabelFreqLevel = False, powertransform = False, \
            binstransform = False, MLinfill = False, infilliterate=1, randomseed = 42, \
	    LabelSmoothing_train = False, LabelSmoothing_test = False, LabelSmoothing_val = False, \
            LSfit = False, numbercategoryheuristic = 63, pandasoutput = True, NArw_marker = True, \
            featureselection = False, featurepct = 1.0, featuremetric = .02, featuremethod = 'default', \
            Binary = False, PCAn_components = None, PCAexcl = [], \
            ML_cmnd = {'MLinfill_type':'default', \
                       'MLinfill_cmnd':{'RandomForestClassifier':{}, 'RandomForestRegressor':{}}, \
                       'PCA_type':'default', \
                       'PCA_cmnd':{}}, \
            assigncat = {'mnmx':[], 'mnm2':[], 'mnm3':[], 'mnm4':[], 'mnm5':[], 'mnm6':[], \
	                 'mean':[], 'mea2':[], 'mea3':[], \
		         'nmbr':[], 'nbr2':[], 'nbr3':[], 'MADn':[], 'MAD2':[], 'MAD3':[], \
		         'dxdt':[], 'd2dt':[], 'd3dt':[], 'dxd2':[], 'd2d2':[], 'd3d2':[], \
		         'nmdx':[], 'nmd2':[], 'nmd3':[], 'mmdx':[], 'mmd2':[], 'mmd3':[], \
		         'bins':[], 'bint':[], 'bsor':[], 'pwr2':[], 'por2':[], \
			 'bnwd':[], 'bnwK':[], 'bnwM':[], 'bnwo':[], 'bnKo':[], 'bnMo':[], \
			 'bnep':[], 'bne7':[], 'bne9':[], 'bneo':[], 'bn7o':[], 'bn9o':[], \
		         'bxcx':[], 'bxc2':[], 'bxc3':[], 'bxc4':[], \
		         'log0':[], 'log1':[], 'sqrt':[], \
		         'bnry':[], 'text':[], 'txt2':[], 'txt3':[], '1010':[], 'or10':[], \
		         'ordl':[], 'ord2':[], 'ord3':[], 'ord4':[], 'om10':[], 'mmor':[], \
			 'Utxt':[], 'Utx2':[], 'Utx3':[], 'Uor3':[], 'Uor6':[], 'U101':[], \
                         'splt':[], 'spl2':[], 'spl3':[], 'spl4':[], 'spl5':[], \
                         'nmrc':[], 'nmr2':[], 'nmr3':[], 'nmcm':[], 'nmc2':[], 'nmc3':[], \
			 'nmr7':[], 'nmr8':[], 'nmr9':[], 'nmc7':[], 'nmc8':[], 'nmc9':[], \
                         'ors2':[], 'ors5':[], 'ors6':[], 'ors7':[], \
			 'or11':[], 'or12':[], 'or15':[], 'or17':[], 'or19':[], 'or20':[], \
		         'date':[], 'dat2':[], 'dat6':[], 'wkdy':[], 'bshr':[], 'hldy':[], \
		         'yea2':[], 'mnt2':[], 'mnt6':[], 'day2':[], 'day5':[], \
		         'hrs2':[], 'hrs4':[], 'min2':[], 'min4':[], 'scn2':[], \
		         'excl':[], 'exc2':[], 'exc3':[], 'null':[], 'eval':[], 'copy':[]}, \
            assigninfill = {'stdrdinfill':[], 'MLinfill':[], 'zeroinfill':[], 'oneinfill':[], \
                            'adjinfill':[], 'meaninfill':[], 'medianinfill':[], 'modeinfill':[]}, \
            assignparam = {}, transformdict = {}, processdict = {}, evalcat = False, \
            printstatus = True)
```

Or for the postmunge function:

```
#for postmunge(.) function on additional or subsequently available train or test data
#using the postprocess_dict object returned from original automunge(.) application

test, testID, testlabels, \
labelsencoding_dict, postreports_dict = \
am.postmunge(postprocess_dict, df_test)
```

With the full set of arguments available to be passed as:

```
am.postmunge(postprocess_dict, df_test, testID_column = False, \
             labelscolumn = False, pandasoutput=True, printstatus = True, \
             TrainLabelFreqLevel = False, featureeval = False, driftreport = False, ]\
	     LabelSmoothing = False, LSfit = False):
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
generate consistent processing of additional train or test data such as 
may not have been available at initial application of automunge. It is 
recommended that this dictionary be externally saved on each application 
used to train a downstream model so that it may be passed to postmunge(.) 
to consistently process subsequently available test data, such as 
demonstrated with the pickle library above.

...

## automunge(.) passed arguments

```
am.automunge(df_train, df_test = False, labels_column = False, trainID_column = False, \
            testID_column = False, valpercent1=0.0, valpercent2 = 0.0, floatprecision = 32, \
            shuffletrain = False, TrainLabelFreqLevel = False, powertransform = False, \
            binstransform = False, MLinfill = False, infilliterate=1, randomseed = 42, \
	    LabelSmoothing_train = False, LabelSmoothing_test = False, LabelSmoothing_val = False, \
            LSfit = False, numbercategoryheuristic = 63, pandasoutput = True, NArw_marker = True, \
            featureselection = False, featurepct = 1.0, featuremetric = .02, featuremethod = 'default', \
            Binary = False, PCAn_components = None, PCAexcl = [], \
            ML_cmnd = {'MLinfill_type':'default', \
                       'MLinfill_cmnd':{'RandomForestClassifier':{}, 'RandomForestRegressor':{}}, \
                       'PCA_type':'default', \
                       'PCA_cmnd':{}}, \
            assigncat = {'mnmx':[], 'mnm2':[], 'mnm3':[], 'mnm4':[], 'mnm5':[], 'mnm6':[], \
	                 'mean':[], 'mea2':[], 'mea3':[], \
		         'nmbr':[], 'nbr2':[], 'nbr3':[], 'MADn':[], 'MAD2':[], 'MAD3':[], \
		         'dxdt':[], 'd2dt':[], 'd3dt':[], 'dxd2':[], 'd2d2':[], 'd3d2':[], \
		         'nmdx':[], 'nmd2':[], 'nmd3':[], 'mmdx':[], 'mmd2':[], 'mmd3':[], \
		         'bins':[], 'bint':[], 'bsor':[], 'pwr2':[], 'por2':[], \
			 'bnwd':[], 'bnwK':[], 'bnwM':[], 'bnwo':[], 'bnKo':[], 'bnMo':[], \
			 'bnep':[], 'bne7':[], 'bne9':[], 'bneo':[], 'bn7o':[], 'bn9o':[], \
		         'bxcx':[], 'bxc2':[], 'bxc3':[], 'bxc4':[], \
		         'log0':[], 'log1':[], 'sqrt':[], \
		         'bnry':[], 'text':[], 'txt2':[], 'txt3':[], '1010':[], 'or10':[], \
		         'ordl':[], 'ord2':[], 'ord3':[], 'ord4':[], 'om10':[], 'mmor':[], \
			 'Utxt':[], 'Utx2':[], 'Utx3':[], 'Uor3':[], 'Uor6':[], 'U101':[], \
                         'splt':[], 'spl2':[], 'spl3':[], 'spl4':[], 'spl5':[], \
                         'nmrc':[], 'nmr2':[], 'nmr3':[], 'nmcm':[], 'nmc2':[], 'nmc3':[], \
			 'nmr7':[], 'nmr8':[], 'nmr9':[], 'nmc7':[], 'nmc8':[], 'nmc9':[], \
                         'ors2':[], 'ors5':[], 'ors6':[], 'ors7':[], \
			 'or11':[], 'or12':[], 'or15':[], 'or17':[], 'or19':[], 'or20':[], \
		         'date':[], 'dat2':[], 'dat6':[], 'wkdy':[], 'bshr':[], 'hldy':[], \
		         'yea2':[], 'mnt2':[], 'mnt6':[], 'day2':[], 'day5':[], \
		         'hrs2':[], 'hrs4':[], 'min2':[], 'min4':[], 'scn2':[], \
		         'excl':[], 'exc2':[], 'exc3':[], 'null':[], 'eval':[], 'copy':[]}, \
            assigninfill = {'stdrdinfill':[], 'MLinfill':[], 'zeroinfill':[], 'oneinfill':[], \
                            'adjinfill':[], 'meaninfill':[], 'medianinfill':[], 'modeinfill':[]}, \
            assignparam = {}, transformdict = {}, processdict = {}, evalcat = False, \
            printstatus = True)
```

* df_train: a pandas dataframe or numpy array containing a structured 
dataset intended for use to subsequently train a machine learning model. 
The set at a minimum should be 'tidy' meaning a single column per feature 
and a single row per observation. If desired the set may include one are more
"ID" columns (intended to be carved out and consitently shuffled or partitioned
such as an index column) and one or more columns intended to be used as labels 
for a downstream training operation. The tool supports the inclusion of 
non-index-range column as index or multicolumn index (requires named index 
columns). Such index types are added to the returned "ID" sets which are 
consistently shuffled and partitioned as the train and test sets. 

* df_test: a pandas dataframe or numpy array containing a structured 
dataset intended for use to generate predictions from a downstream machine 
learning model trained from the automunge returned sets. The set must be 
consistantly formated as the train set with consistent column labels and/or
order of columns. (This set may optionally contain a labels column if one 
was included in the train set although it's inclusion is not required). If 
desired the set may include one or more ID column(s) or column(s) intended 
for use as labels. A user may pass False if this set not available. The tool 
supports the inclusion of non-index-range column as index or multicolumn index 
(requires named index columns). Such index types are added to the returned 
"ID" sets which are consistently shuffled and partitioned as the train and 
test sets.

* labels_column: a string of the column title for the column from the
df_train set intended for use as labels in training a downstream machine
learning model. The function defaults to False for cases where the
training set does not include a label column. An integer column index may 
also be passed such as if the source dataset was numpy array. If the df_train
set passed to automunge is a single column intended for a label set, a user
can pass True here instead of the column name.

* trainID_column: a string of the column title for the column from the
df_train set intended for use as a row identifier value (such as could
be sequential numbers for instance). The function defaults to False for
cases where the training set does not include an ID column. A user can 
also pass a list of string columns titles such as to carve out multiple
columns to be excluded from processing but consistently partitioned. An 
integer column index or list of integer column indexes may also be passed 
such as if the source dataset was numpy array.

* testID_column: a string of the column title for the column from the
df_test set intended for use as a row identifier value (such as could be
sequential numbers for instance). The function defaults to False for
cases where the training set does not include an ID column. A user can 
also pass a list of string columns titles such as to carve out multiple
columns to be excluded from processing but consistently partitioned. An 
integer column index or list of integer column indexes may also be passed 
such as if the source dataset was numpy array.

* valpercent1: a float value between 0 and 1 which designates the percent
of the training data which will be set aside for the first validation
set (generally used for hyperparameter tuning of a downstream model).
This value defaults to 0. (Previously the default here was set at 0.20 but 
that is fairly an arbitrary value and a user may wish to deviate for 
different size sets.) Note that this value may be set to 0 if no validation 
set is needed (such as may be the case for k-means validation). Please see 
also the note below for the shuffletrain parameter.

* valpercent2: a float value between 0 and 1 which designates the percent
of the training data which will be set aside for the second validation
set (generally used for final validation of a model prior to release).
This value defaults to 0. (Previously the default was set at 0.10 but that 
is fairly an arbitrary value and a user may wish to deviate for different 
size sets.)

* floatprecision: an integer with acceptable values of 16/32/64 designating
the memory precision for returned float values. (A tradeoff between memory
usage and floating point precision, smaller for smaller footprint.)

* shuffletrain: a boolean identifier (True/False) which indicates if the
rows in df_train will be shuffled prior to carving out the validation
sets. Note that if this value is set to False then the validation sets
will be pulled from the bottom x% sequential rows of the dataframe.
(Where x% is the sum of validation ratios.) Note that if this value is
set to False although the validations will be pulled from sequential
rows, the split between validaiton1 and validation2 sets will be
randomized. This value defaults to False.

* TrainLabelFreqLevel: a boolean identifier (True/False) which indicates
if the TrainLabelFreqLevel method will be applied to prepare for oversampling 
training data associated with underrepresented labels (aka class imbalance). 
The method adds multiples to training data rows for those labels with lower 
frequency resulting in an (approximately) levelized frequency. This defaults 
to False. Note that this feature may be applied to numerical label sets if 
the processing applied to the set includes standard deviation bins.

* powertransform: a boolean identifier (True/False) which indicates if an
evaluation will be performed of distribution properties to select between
box-cox, z-score, min-max scaling, or mean absolute deviaiton scaling 
normalization. Note that after application of box-cox transform child columns 
are generated for a subsequent z-score normalization as well as a set of bins
associated with number of standard deviations from the mean. Please note that
I don't consider the current means of distribution property evaluation very
sophisticated and we will continue to refine this method with further research
going forward. This defaults to False. Additionally, powertransform may be 
passed as values 'excl' or 'exc2', where for 'excl' columns not explicitly 
assigned to a root category in assigncat will be left untouched, or for 'exc2'
columns not explicitly assigned to a root category in assigncat will be forced 
to numeric and subject to default modeinfill. (These two excl arguments may be 
useful if a user wants to experiment with specific transforms on a subset of
the columns without incurring processing time of an entire set.)

* binstransform: a boolean identifier (True/False) which indicates if the
numerical sets will receive bin processing such as to generate child
columns with boolean identifiers for number of standard deviations from
the mean, with groups for values <-2, -2-1, -10, 01, 12, and >2 . Note
that the bins and bint transformations are the same, only difference is
that the bint transform assumes the column has already been normalized
while the bins transform does not. This value defaults to False.

* MLinfill: a boolean identifier (True/False) which indicates if the ML
infill method will be applied as a default to predict infill for missing 
or improperly formatted data using machine learning models trained on the
rest of the set. This defaults to False. Note that ML infill may alternatively
be assigned to distinct columns in assigninfill.

* infilliterate: an integer indicating how many applications of the ML
infill processing are to be performed for purposes of predicting infill.
The assumption is that for sets with high frequency of missing values
that multiple applications of ML infill may improve accuracy although
note this is not an extensively tested hypothesis. This defaults to 1.

* randomseed: a postitive integer used as a seed for randomness throughout 
such as for data set shuffling, ML infill, and feature importance  algorithms. 
This defaults to 42, a nice round number.

* LabelSmoothing_train / LabelSmoothing_test / LabelSmoothing_val: each of these
parameters accept float values in range 0.0-1.0 or the default value of False to 
turn off. train is for the train set labels, test is for the test set labels, and
val is for the validation set labels. Label Smoothing refers to the regularaization
tactic of transforming boolean encoded labels from 1/0 designations to some mix of
reduced/increased threshold - for example passing the float 0.9 would result in the
conversion from 1/0 to 0.9/#, where # is a function of the number of cateogries in 
the label set - for example for a boolean label it would convert 1/0 to 0.9/0.1, or 
for the one-hot encoding of a three label set it would be convert 1/0 to 0.9/0.05.
Hat tip for the concept to "Rethinking the Inception Architecture for Computer Vision"
by Szegedy et al. Note that I believe not all predictive classifigation libraries 
uniformily accept smoothed labels, but when available the method can at times be useful. 
Note that a user can pass True to either of LabelSmoothing_test / LabelSmoothing_val 
which will consistently encode to LabelSmoothing_train. Please note that if multiple
one-hot encoded transformations originate from the same labels source column, the
application of Label Smoothing will be applied to each set individually.

* LSfit: a True/False indication for basis of label smoothing parameter K. The default
of False means the assumption will be for level distribution of labels, passing True
means any label smoothing will evluate distribution fo label activations such as to fit
the smoothing factor K to specific cells based on the activated column and target column.
The LSfit parameters of transformations will be based on properteis dervied from the
train set labels, such as for consistent encoding to the other sets (test and validaiton).

* numbercategoryheuristic: an integer used as a heuristic. When a 
categorical set has more unique values than this heuristic, it defaults 
to categorical treatment via ordinal processing via 'ord3', otherwise 
categorical sets default to binary encoding via '1010'. This defaults to 63.

* pandasoutput: a selector for format of returned sets. Defaults to False
for returned Numpy arrays. If set to True returns pandas dataframes
(note that index is not preserved in the train/validation split, an ID
column may be passed for index identification).

* NArw_marker: a boolean identifier (True/False) which indicates if the
returned sets will include columns with markers for rows subject to 
infill (columns with suffix 'NArw'). This value defaults to True.

* featureselection: a boolean identifier telling the function whether to
perform a feature importance evaluation. If selected automunge will
return a summary of feature importance findings in the featureimportance
returned dictionary. This also activates the trimming of derived sets
that did not meet the importance threshold if [featurepct < 1.0 and 
featuremethod = 'pct'] or if [fesaturemetric > 0.0 and featuremethod = 
'metric']. Note this defaults to False because it cannot operate without
a designated label column in the train set. (Note that any user-specified
size of validationratios if passed are used in this method, otherwise 
defaults to 0.33.)

* featurepct: the percentage of derived sets that are kept in the output
based on the feature importance evaluation. Note that NArw columns are
excluded from the trimming for now (the inclusion of NArws in trimming
will likely be included in a future expansion). This item only used if
featuremethod passed as 'pct' (the default).

* featuremetric: the feature importance metric below which derived sets
are trimmed from the output. Note that this item only used if
featuremethod passed as 'metric'.

* featuremethod: can be passed as either 'pct' or 'metric' to select which
feature importance method is used for trimming the derived sets. Or can pass
as 'default' for ignoring the featurepct/featuremetric parameters or can 
pass as 'report' to return the featureimportance results with no further
processing (other returned sets are empty).

* Binary: a dimensionality reduction technique whereby the set of columns
with binary encodings are collectively encoded with binary encoding such
as may drastically reduce the column count. This has many benefits such as
memory bandwidth and energy cost for inference I suspect, however, there 
may be tradeoffs associated with ability of the model to handle outliers,
as for any new combination of boolean set in the test data the collection
will be subject to the infill. Pass True to activate, defaults to False.

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
used for ML infill / feature importance evaluation or PCA. Currently the only
option for 'MLinfill_type' is default which uses Scikit-learn's Random 
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
           
#Also note that SparsePCA currently doesn't have available
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
or exclude returned boolean and ordinal columns from PCA application by
'PCA_cmnd':{'bool_ordl_PCAexcl':True}
such as could potentially result in memory savings.


* assigncat:

```
#Here are the current trasnformation options built into our library, which
#we are continuing to build out. A user may also define their own.

    assigncat = {'mnmx':[], 'mnm2':[], 'mnm3':[], 'mnm4':[], 'mnm5':[], 'mnm6':[], \
                 'mean':[], 'mea2':[], 'mea3':[], \
		 'nmbr':[], 'nbr2':[], 'nbr3':[], 'MADn':[], 'MAD2':[], 'MAD3':[], \
		 'dxdt':[], 'd2dt':[], 'd3dt':[], 'dxd2':[], 'd2d2':[], 'd3d2':[], \
		 'nmdx':[], 'nmd2':[], 'nmd3':[], 'mmdx':[], 'mmd2':[], 'mmd3':[], \
		 'bins':[], 'bint':[], 'bsor':[], 'pwr2':[], 'por2':[], \
		 'bnwd':[], 'bnwK':[], 'bnwM':[], 'bnwo':[], 'bnKo':[], 'bnMo':[], \
		 'bnep':[], 'bne7':[], 'bne9':[], 'bneo':[], 'bn7o':[], 'bn9o':[], \
		 'bxcx':[], 'bxc2':[], 'bxc3':[], 'bxc4':[], \
		 'log0':[], 'log1':[], 'sqrt':[], \
		 'bnry':[], 'text':[], 'txt2':[], 'txt3':[], '1010':[], 'or10':[], \
		 'ordl':[], 'ord2':[], 'ord3':[], 'ord4':[], 'om10':[], 'mmor':[], \
		 'Utxt':[], 'Utx2':[], 'Utx3':[], 'Uor3':[], 'Uor6':[], 'U101':[], \
                 'splt':[], 'spl2':[], 'spl3':[], 'spl4':[], 'spl5':[], \
                 'nmrc':[], 'nmr2':[], 'nmr3':[], 'nmcm':[], 'nmc2':[], 'nmc3':[], \
		 'nmr7':[], 'nmr8':[], 'nmr9':[], 'nmc7':[], 'nmc8':[], 'nmc9':[], \
                 'ors2':[], 'ors5':[], 'ors6':[], 'ors7':[], \
		 'or11':[], 'or12':[], 'or15':[], 'or17':[], 'or19':[], 'or20':[], \
		 'date':[], 'dat2':[], 'dat6':[], 'wkdy':[], 'bshr':[], 'hldy':[], \
		 'yea2':[], 'mnt2':[], 'mnt6':[], 'day2':[], 'day5':[], \
		 'hrs2':[], 'hrs4':[], 'min2':[], 'min4':[], 'scn2':[], \
		 'excl':[], 'exc2':[], 'exc3':[], 'null':[], 'eval':[], 'copy':[]}, \
```         

Descriptions of these transformations are provided in document below (in section
titled "Library of Transformations").

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

* assigninfill 
```
#Here are the current infill options built into our library, which
#we are continuing to build out.
assigninfill = {'stdrdinfill':[], 'MLinfill':[], 'zeroinfill':[], 'oneinfill':[], \
                'adjinfill':[], 'meaninfill':[], 'medianinfill':[], 'modeinfill':[]}, \
```
A user may add column identifier strings to each of these lists to 
designate the column-specific infill approach for missing or
improperly formated values. Note that this infill category defaults to
MLinfill if nothing assigned and the MLinfill argument to automunge is
set to True. stdrdinfill means: mean for numeric sets, most common for 
binary, and new column boolean for categorical. zeroinfill means inserting 
the integer 0 to missing cells. oneinfill means inserting the integer 1.
adjinfill means passing the value from the preceding row to missing cells. 
meaninfill means inserting the mean derived from the train set to numeric 
columns. medianinfill means inserting the median derived from the train 
set to numeric columns. (Note currently boolean columns derived from 
numeric are not supported for mean/median and for those cases default to 
those infill from stdrdinfill.) modeinfill means inserting the most common
value for a set, note that modeinfill supports one-hot encoded sets.

* assignparam
A user may pass column-specific parameters to those transformation functions
that accept parameters. Any parameters passed to automunge(.) will be saved in
the postprocess_dict and consistently applied in postmunge(.). assignparam is 
a dictionary that should be formatted per following example:
```
assignparam = {'category1' : {'column1' : {'param1' : 123}, 'column2' : {'param1' : 456}}, \
               'cateogry2' : {'column3' : {'param2' : 'abc', 'param3' : 'def'}}}

#In other words:
#The first layer keys are the transformation category for which parameters are intended
#The second layer keys are string identifiers for the columns for which the parameters are intended
#The third layer keys are the parameters whose values are to be passed.

#As an example with actual parameters, consider the trasnformation category 'splt' intended for 'column1',
#which accepts parameter 'minsplit' for minimum character length of detected overlaps. If we wanted to
pass 4 instead of the default of 5:
assignparam = {'splt' : {'column1' : {'minsplit' : 4}}

#Note that column string identifiers may just be the source column string or may include the
#suffix appenders such as if multiple versiuons of transformations are applied within the same family tree
#If more than one column identifier matches a column, the longest character length key which matches
#will be applied (such as may include suffixc appenders).

#Note that if a user wishes to overwrite the default parameters for all columns without specifying
#them individually they can pass a 'default_assignparam' entry as follows (this only overwirtes those 
#parameters that are not otherwise specified in assignparam)
assignparam = {'category1' : {'column1' : {'param1' : 123}, 'column2' : {'param1' : 456}}, \
               'cateogry2' : {'column3' : {'param2' : 'abc', 'param3' : 'def'}}, \
	       'default_assignparam' : {'category3' : {'param4' : 789}

```
See the Library of Transformations section below for those trasnformations that accept parameters.


* transformdict: allows a user to pass a custom tree of transformations.
Note that a user may define their own (traditionally 4 character) string "root"
identifiers for a series of processing steps using the categories of processing 
already defibned in our library and then assign columns in assigncat, or for 
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
'parents' :           upstream / first generation / replaces column / with offspring
'siblings':           upstream / first generation / supplements column / with offspring
'auntsuncles' :       upstream / first generation / replaces column / no offspring
'cousins' :           upstream / first generation / supplements column / no offspring
'children' :          downstream parents / offspring generations / replaces column / with offspring
'niecesnephews' :     downstream siblings / offspring generations / supplements column / with offspring
'coworkers' :         downstream auntsuncles / offspring generations / replaces column / no offspring
'friends' :           downstream cousins / offspring generations / supplements column / no offspring
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

#NArowtype: can be entries of either 'numeric', 'positivenumeric', 'justNaN', 
#or 'exclude' where
# - 'numeric' for source columns with expected numeric entries
# - 'justNaN' for source columns that may have expected entries other than numeric
# - 'exclude' for source columns that aren't needing NArow columns derived
# - 'positivenumeric' for source columns with expected positive numeric entries
# - 'nonnegativenumeric' for source columns with expected non-nbegative numeric (zero allowed)
# - 'nonzeronumeric' for source columns with allowed postiive and negative but no zero
# - 'parsenumeric' marks for infill strings that don't contain any numeric character entries
# - 'parsenumeric_commas' marks for infill strings that don't contain any numeric character entries, recognizes commas
# - 'datetime' marks for infill cells that arent' recognized as datetime objects

#MLinfilltype: can be entries of 'numeric', 'singlct', 'multirt', 'exclude'
#              'multisp', 'exclude', or 'label' where
#	       'numeric' refers to columns where predictive algorithms treat
#			 as a regression for numeric sets
#	       'singlect' refers to columns where category gives a single column
#			 where predictive algorithms treat as a classification target
#		'multirt' refers to category returning multiple columns where 
#			 predictive algorithms treat as a multi modal classifier
#		'exclude' refers to categories excluded from predcitive address
#		'multisp' for bins multicolumn sets with boolean entries
#                         (similar to multirt but treated differently in levelizer)
#		'label' refers to categories specifically intended for label
#			   processing
#               'binary'  for multicolumn sets with boolean entries as may have 
#                         multiple entries in the same row (not currently used, future extension)
#               '1010'   for multicolumn sets with binary encoding via 1010
#                         will be converted to onehot for ML

```

* evalcat: modularizes the automated evaluation of column properties for assignment 
of root transformation categories, allowing user to pass custom functions for this 
purpose. Passed functions should follow format:

```
def evalcat(df, column, numbercategoryheuristic, powertransform, labels = False):
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
I recomend using the evalcategory function defined in master file as starting point. 
(Minus the 'self' parameter since defining external to class.) Note that the 
parameters numbercategoryheuristic, powertransform, and labels are passed as user 
parameters in automunge call and only used in evalcategory function, so if user wants 
to repurpose them totally can do so. (They default to 15, False.) Note evalcat defaults 
to False to use built-in evalcategory function. Note evalcat will only be applied to 
columns not assigned in assigncat. (Note that columns assigned to 'eval' in assigncat
will be passed to this function for evaluation with powertransform = True.)

* printstatus: user can pass True/False indicating whether the function will print 
status of processing during operation. Defaults to True.

Ok well we'll demonstrate further below how to build custom processing functions,
for now this just gives you sufficient tools to build sets of processing using
the built in sets in the library.

...

# postmunge

The postmunge(.) function is intended to consistently process subsequently available
and consistently formatted test data with just a single function call. It requires 
passing the postprocess_dict object returned from the original application of automunge 
and that the passed test data have consistent column header labeling as the original 
train set (or for Numpy arrays consistent order of columns).

```

#for postmunge(.) function on subsequently available test data
#using the postprocess_dict object returned from original automunge(.) application

#Remember to initialize automunge
from Automunge import Automunger
am = Automunger.AutoMunge()


#Then we can run postmunge function as:

test, testID, testlabels, \
labelsencoding_dict, postreports_dict = \
am.postmunge(postprocess_dict, df_test, testID_column = False, \
             labelscolumn = False, pandasoutput=True, printstatus = True, \
             TrainLabelFreqLevel = False, featureeval = False, driftreport = False, \
	     LabelSmoothing = False, LSfit = False)
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

* postreports_dict: a dictionary containing entries for following:
- postreports_dict['featureimportance']: results of optional feature 
importance evaluation based on parameter featureeval
- postreports_dict['driftreport']: results of optional drift report 
evaluation tracking properties of psotmunge data in comparision to the 
original data from automunge call associated with the postprocess_dict 
presumably used to train a model. Results aggregated by entries for the
original (pre-transform) list of columns, and include the normailzaiton
parameters from the automunge call saved in postprocess_dict as well
as the corresponding parameters from the new data consistently derived 
in postmunge
- postreports_dict['finalcolumns_test']: list of columns returned from 
postmunge

...


## postmunge(.) passed arguments

```

#for postmunge(.) function on subsequently available test data
#using the postprocess_dict object returned from original automunge(.) application

#Remember to initialize automunge
from Automunge import Automunger
am = Automunger.AutoMunge()


#Then we can run postmunge function as:

test, testID, testlabels, \
labelsencoding_dict, finalcolumns_test = \
am.postmunge(postprocess_dict, df_test, testID_column = False, \
             labelscolumn = False, pandasoutput=True, printstatus = True, \
             TrainLabelFreqLevel = False, featureeval = False, driftreport = False, \
	     LabelSmoothing = False, LSfit = False)
```

* postprocess_dict: this is the dictionary returned from the initial
application of automunge which included normalization parameters to
facilitate consistent processing of test data to the original processing
of the train set. This requires a user to remember to download the
dictionary at the original application of automunge, otherwise if this
dictionary is not available a user can feed this subsequent test data to
the automunge along with the original train data exactly as was used in
the original automunge call.

* df_test: a pandas dataframe or numpy array containing a structured 
dataset intended for use to generate predictions from a machine learning 
model trained from the automunge returned sets. The set must be consistantly 
formatted as the train set with consistent order of columns and if labels are
included consistent labels. If desired the set may include an ID column. The 
tool supports the inclusion of non-index-range column as index or multicolumn 
index (requires named index columns). Such index types are added to the 
returned "ID" sets which are consistently shuffled and partitioned as the 
train and test sets.

* testID_column: a string of the column title for the column from the
df_test set intended for use as a row identifier value (such as could be
sequential numbers for instance). The function defaults to False for
cases where the training set does not include an ID column. A user can 
also pass a list of string columns titles such as to carve out multiple
columns to be excluded from processing but consistently partitioned. An 
integer column index or list of integer column indexes may also be passed 
such as if the source dataset was numpy array.

* labelscolumn: default to False indicates that a labels column is not 
included in the test set passed to postmunge. A user can either pass
True or the string ID of the labels column, noting that it is a requirement
that the labels column header string must be consistent with that from
the original train set. An integer column index may also be passed such
as if the source dataset was numpy array. A user should take care to set 
this parameter if they are passing data with labels.

* pandasoutput: a selector for format of returned sets. Defaults to False
for returned Numpy arrays. If set to True returns pandas dataframes
(note that index is not preserved, an ID column may be passed for index
identification).

* printstatus: user can pass True/False indicating whether the function 
will print status of processing during operation. Defaults to True.

* TrainLabelFreqLevel: a boolean identifier (True/False) which indicates
if the TrainLabelFreqLevel method will be applied to oversample test
data associated with underrepresented labels. The method adds multiples
to test data rows for those labels with lower frequency resulting in
an (approximately) levelized frequency. This defaults to False. Note that
this feature may be applied to numerical label sets if the processing
applied to the set in automunge had included standard deviation bins. Note 
this requires the inclusion of a designated labels column.

* featureeval: a boolean identifier (True/False) to activate a feature
importance evaluation, comparable to one performed in automunge but based
on the test set passed to postmunge. The results are returned in the
postreports_dict object returned from postmunge as postreports_dict['featureimportance']. 
The results will also be printed out if printstatus is activated.

* driftreport: a boolean identifier (True/False) to activate a drift report 
evaluation, in which the normalization parameters are recalculated for the 
columns of the test data passed to postmunge for comparison to the original 
normalization parameters derived from the coresponding columns of the 
automunge train data set. The results are returned in the
postreports_dict object returned from postmunge as postreports_dict['driftreport']. 
The results will also be printed out if printstatus is activated. Documentation of
the various metrics tracked in driftreport assembly is forthcoming.

* LabelSmoothing: accepts float values in range 0.0-1.0 or the default value of False
to turn off Label Smoothing. Note that a user can pass True to LabelSmoothing which 
will consistently encode to LabelSmoothing_train from the corresponding automunge(.) 
call, including any application of LSfit based on parameters of transformations 
derived from the train set labels.

* LSfit: a True/False indication for basis of label smoothing parameter K. The default
of False means the assumption will be for level distribution of labels, passing True
means any label smoothing will evluate distribution fo label activations such as to fit
the smoothing factor K to specific cells based on the activated column and target column.
Note that if LabelSmoothing passed as True the LSfit will be based on the basis from
the correspondign automunge(.) call (will override this one passed to postmunge).

...

## Default Transformations

When root categories of transformations are not assigned for a given column in
assigncat, automunge performs an evaluation of data properties to infer 
appropriate means of feature engineering and numerical encoding. The default
categories of transformations are as follows:
- nmbr: for numerical data, columns are treated with z-score normalization. If 
binstransform parameter was activated this will be supplemented by a collection
of bins indicating number of standard deviations from the mean.
- 1010: for categorical data, columns are subject to binary encoding. If the 
number of unique entries in the column exceeds the parameter 'numbercategoryheuristic'
(which defaults to 63), the encoding will instead be by 'ord3' which is an ordinal
(integer) encoding sorted by most common value. Note that numerical sets with 3
unique values in train set default to categorical encoding via 'text'.
- bnry: for categorical data of <=2 unique values excluding infill (eg NaN), the 
column is encoded to 0/1. Note that numerical sets with <= 2 unique values in train
set default to bnry.
- dat6: for time-series data, a set of derivations are performed returning
'year', 'mdsn', 'mdcs', 'hmss', 'hmsc', 'bshr', 'wkdy', 'hldy' (these are defined 
in next section)
- null: for columns with single entry column is deleted

Note that for columns designated for label sets as a special case categorical data will
default to 'text' (one-hot encoding) instead of '1010'. Also, numerical data will default
to 'excl2' (pass-through) instead of 'nmbr'. Also, if label smoothing is applied, label 
columns evaluated as 'bnry' (two iunique values) will default to 'text' instead of 'bnry'
as label smoothing requires one-hot encoding.

- PCA: if the number of features exceeds 0.5 the number of rows (an arbitrary heuristic)
a default PCA transform is applied defaulting to kernel if all positive or sparse Otherwise
using scikit library. Note that this heuristic ratio can be changed or PCA turned off
in the ML_cmnd.

- powertransform: if the powertransform parameter is activated, a statistical evaluation
will be performed on numerical sets to distinguish between columns to be subject to
bxcx, nmbr, or mnmx. Please note that we intend to further refine the specifics of this
process in future implementations. 

- floatprecision: parameter indicates the precision of floats in returned sets (16/32/64)
such as for memory considerations.

In all cases, if the parameter NArw_marker is activated returned sets will be
supplemented with a NArw column indicating rows that were subject to infill. Each 
transformation category has a default infill approach detailed below.

Note that default transformations can be overwritten within an automunge(.) call by way
of passing custom transformdict family tree defintions which overwrite the family tree 
of the default root categories listed above. For instance, if a user wishes to process 
numerical columns with a default mean scaling ('mean') instead of z-score 
normalization ('nmbr'), the user may copy the transform_dict entries from the code-base 
for mean root category and assign as a definition of the nmbr root category, and then 
pass that defined transformdict in the automunge call. (Note that we don't need to 
overwrite the processdict for nnmbr if we don't intend to overwrite it's use as an entry 
in other root category family trees. Also it's good practice to retain any downstream 
entries such as in case the default for nmbr is used as an entry in some other root 
category's family tree.) Here's a demonstration.

```
#create a transformdict that overwrites the root category definition of nmbr with mean:
transformdict = {'nmbr' : {'parents' : [], \
                           'siblings': [], \
                           'auntsuncles' : ['mean'], \
                           'cousins' : [], \
                           'children' : [], \
                           'niecesnephews' : [], \
                           'coworkers' : [], \
                           'friends' : []}}
                           
#And then we can simply pass this transformdict to an automunge(.) call.

(returned sets) = \
am.automunge(df_train, \
             transformdict = transformdict)

```

...

## Library of Transformations

Automunge has a built in library of transformations that can be passed for
specific columns with assigncat. (A column if left unassigned will defer to
the automated default methods to evaluate properties of the data to infer 
appropriate methods of numerical encoding.)  For example, a user can pass a 
min-max scaling method to a specific column 'col1' with: 
```
assigncat = {'mnmx':['col1']}
```
When a user assigns a column to a specific category, that category is treated
as the root category for the tree of transformations. Each key has an 
associated transformation function, and that transformation function is only
applied if the root key is also found in the tree of family primitives. The
tree of family primitives, as introduced earlier, applies first the keys found 
in upstream primitives i.e. parents/siblings/auntsuncles/cousins. If a transform 
is applied for a primitive that includes downstream offspring, such as parents/
siblings, then the family tree for that key with offspring is inspected to determine
downstream offspring categories, for example if we have a parents key of 'mnmx',
then any children/niecesnephews/coworkers/friends in the 'mnmx' family tree will
be applied as parents/siblings/auntsuncles/cousins, respectively. Note that the
designation for supplements/replaces refers purely to the question of whether the
column to which the transform is being applied is kept in place or removed. Please
note that it is a quirck of the function that no original column can be left in 
place without the application of some transformation such as to allow the building
of the apppropriate data structures, thus at least one replacement primitive must
always be included. If a user does wish to leave a column in place unaltered, they 
can simply assign that column to the 'excl' root category.

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
with each key which can be assigned to a primitive (and not just used as 
a root key). We're continuing to build out this library of transformations.

Note the design philosophy is that any transform can be applied to any type 
of data and if the data is not suited (such as applying a numeric transform
to a categorical set) the transform will just return all zeros. Note the 
default infill refers to the infill applied under 'standardinfill'. Note the
default NArowtype refers to the categories of data that won't be subject to 
infill.

* nmbr/nbr2/nbr3/nmdx/nmd2/nmd3: z-score normalization
(x - mean) / (standard deviation)
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_nmbr'
  - assignparam parameters accepted: none
* dxdt/d2dt/d3dt: rate of change (row value minus value in preceding row)
  - default infill: adjacent cells
  - default NArowtype: numeric
  - suffix appender: '_dxdt'
  - assignparam parameters accepted: 'periods' sets number of time steps offset to evaluate
  defaults to 1
* dxd2/d2d2/d3d2: denoised rate of change (average of last two rows minus average
of preceding two rows)
  - default infill: adjacent cells
  - default NArowtype: numeric
  - suffix appender: '_dxd2'
  - assignparam parameters accepted: 'periods' sets number of time steps offset to evaluate
  defaults to 2
* MADn/MAD2: mean absolute deviation normalization, subtract set mean
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_MADn'
  - assignparam parameters accepted: none
* MAD3: mean absolute deviation normalization, subtract set maximum
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_MAD3'
  - assignparam parameters accepted: none
* mnmx/mnm2/mnm5/mmdx/mmd2/mmd3: vanilla min-max scaling
(x - min) / (max - min)
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_mnmx'
  - assignparam parameters accepted: none
* mean/mea2/mea3: mean normalization (like z-score in the numerator and min-max in the denominator)
(x - mean) / (max - mean)
Note this is what Andrew Ng suggested as default in his MOOC. My intuition says z-score has some 
benefits but really up to the user which they prefer.
* mnm3/mnm4: min-max scaling with outliers capped at 0.01 and 0.99 quantiles
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_mnm3'
  - assignparam parameters accepted: none
* mnm6: min-max scaling with test floor set capped at min of train set (ensures
test set returned values >= 0, such as might be useful for kernel PCA for instance)
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_mnm6'
  - assignparam parameters accepted: none
* bnry: converts sets with two values to boolean identifiers. Defaults to assiging
1 to most common value and 0 to second most common, unless 1 or 0 is already included
in most common of the set then defaults to maintaining those designations. If applied 
to set with >2 entries applies infill to those entries beyond two most common. 
  - default infill: most common value
  - default NArowtype: justNaN
  - suffix appender: '_bnry'
  - assignparam parameters accepted: none
* text/txt2: converts categorical sets to one-hot encoded set of boolean identifiers
  - default infill: all entries zero
  - default NArowtype: justNaN
  - suffix appender: '_(category)' where category is the target of the column
  - assignparam parameters accepted: none
* ordl/ord2: converts categorical sets to ordinally encoded set of integer identifiers
  - default infill: plug value 'zzzinfill'
  - default NArowtype: justNaN
  - suffix appender: '_ordl'
  - assignparam parameters accepted: none
* ord3/ord4: converts categorical sets to ordinally encoded set of integer identifiers
sorted by frequency of category occurance
  - default infill: plug value 'zzzinfill'
  - default NArowtype: justNaN
  - suffix appender: '_ord3'
  - assignparam parameters accepted: none
* 1010: converts categorical sets of >2 unique values to binary encoding (more memory 
efficent than one-hot encoding)
  - default infill: plug value 'zzzinfill'
  - default NArowtype: justNaN
  - suffix appender: '_1010_#' where # is integer indicating order of 1010 columns
  - assignparam parameters accepted: none
  (for example if 1010 encoded to three columns based on number of categories <8,
  it would retuyrn three columns with suffix appenders 1010_1, 1010_2, 1010_3)
* bxcx/bxc2/bxc3/bxc4: performs Box-Cox power law transformation. Applies infill to values 
<= 0. Note we currently have a test for overflow in returned results and if found set to 0.
  - default infill: mean
  - default NArowtype: positivenumeric
  - suffix appender: '_bxcx'
  - assignparam parameters accepted: none
* log0/log1: performs logarithmic transform (base 10). Applies infill to values <= 0.
  - default infill: mean
  - default NArowtype: positivenumeric
  - suffix appender: '_log0'
  - assignparam parameters accepted: none
* sqrt: performs square root transform. Applies infill to values < 0.
  - default infill: mean
  - default NArowtype: nonnegativenumeric
  - suffix appender: '_sqrt'
  - assignparam parameters accepted: none
* pwrs: bins groupings by powers of 10
  - default infill: mean (ie log(mean))
  - default NArowtype: positivenumeric
  - suffix appender: '_10^#' where # is integer indicating target powers of 10 for column
* pwr2: bins groupings by powers of 10
  - default infill: no activation
  - default NArowtype: nonzeronumeric
  - suffix appender: '_10^#' or '_-10^#' where # is integer indicating target powers of 10 for column
  - assignparam parameters accepted: none
* pwor: for numerical sets, outputs an ordinal encoding indicating where a
value fell with respect to powers of 10
  - default infill: zero
  - default NArowtype: positivenumeric
  - suffix appender: '_pwor'
* por2: for numerical sets, outputs an ordinal encoding indicating where a
value fell with respect to powers of 10
  - default infill: zero (a distinct encoding)
  - default NArowtype: nonzeronumeric
  - suffix appender: '_por2'
  - assignparam parameters accepted: none
* bins: for numerical sets, outputs a set of 6 columns indicating where a
value fell with respect to number of standard deviations from the mean of the
set (i.e. <-2, -2-1, -10, 01, 12, >2)
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_bins_####' where #### is one of set (s<-2, s-21, s-10, s+01, s+12, s>+2)
  which indicate column target for number of standard deviations from the mean
  - assignparam parameters accepted: none
* bint: comparable to bins but assumes data has already been z-score normalized
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_bint_####' where #### is one of set (t<-2, t-21, t-10, t+01, t+12, t>+2)
  which indicate column target for number of standard deviations from the mean
  - assignparam parameters accepted: none
* bsor: for numerical sets, outputs an ordinal encoding indicating where a
value fell with respect to number of standard deviations from the mean of the
set (i.e. <-2:0, -2-1:1, -10:2, 01:3, 12:4, >2:5)
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_bsor'
  - assignparam parameters accepted: none
* bnwd/bnwK/bnwM: for numerical set graining to fixed width bins for one-hot encoded bins 
(columns without activations in train set excluded in train and test data)
bins default to width of 1/1000/1000000 eg for bnwd/bnwK/bnwM
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_bswd_#1_#2' where #1 is the width and #2 is the bin identifier (# from min)
  - assignparam parameters accepted: 'width' to set bin width
* bnwo/bnKo/bnMo: for numerical set graining to fixed width bins for ordinal encoded bins 
(integers without train set activations still included in test set)
bins default to width of 1/1000/1000000 eg for bnwd/bnwK/bnwM
  - default infill: mean
  - default NArowtype: numeric
  - suffix appender: '_bnwo' (or '_bnKo', '_bnMo')
  - assignparam parameters accepted: 'width' to set bin width
* bnep/bne7/bne9: for numerical set graining to equal population bins for one-hot encoded bins 
bin count defaults to 5/7/9 eg for bnep/bne7/bne9
  - default infill: no activation
  - default NArowtype: numeric
  - suffix appender: '_bnep_#1' where #1 is the bin identifier (# from min) (or bne7/bne9)
  - assignparam parameters accepted: 'bincount' to set number of bins
* bneo/bn7o/bn9o: for numerical set graining to equal population bins for ordinal encoded bins 
bin count defaults to 5/7/9 eg for bne0/bn7o/bn9o
  - default infill: adjacent cell
  - default NArowtype: numeric
  - suffix appender: '_bnep_#1' where #1 is the bin identifier (# from min) (or bn7o/bn9o)
  - assignparam parameters accepted: 'bincount' to set number of bins
* date/dat2: for datetime formatted data, segregates data by time scale to multiple
columns (year/month/day/hour/minute/second) and then performs z-score normalization
  - default infill: mean
  - default NArowtype: datetime
  - suffix appender: includes appenders for (_year, _mnth, _days, _hour, _mint, _scnd)
  - assignparam parameters accepted: none
* wkdy: boolean identifier indicating whether a datetime object is a weekday
  - default infill: none
  - default NArowtype: datetime
  - suffix appender: '_wkdy'
  - assignparam parameters accepted: none
* bshr: boolean identifier indicating whether a datetime object falls within business
hours (9-5, time zone unaware)
  - default infill: datetime
  - default NArowtype: justNaN
  - assignparam parameters accepted: none
* hldy: boolean identifier indicating whether a datetime object is a US Federal
holiday
  - default infill: none
  - default NArowtype: datetime
  - suffix appender: '_hldy'
  - assignparam parameters accepted: none
* year/mnth/days/hour/mint/scnd: segregated by time scale and z-score normalization
  - default infill: mean
  - default NArowtype: datetime
  - suffix appender: includes appenders for (_year, _mnth, _days, _hour, _mint, _scnd)
* mnsn/mncs/dysn/dycs/hrsn/hrcs/misn/mics/scsn/sccs: segregated by time scale and 
dual columns with sin and cos transformations for time scale period (eg 12 months, 24 hrs, 7 days, etc)
  - default infill: mean
  - default NArowtype: datetime
  - suffix appender: includes appenders for (mnsn/mncs/dysn/dycs/hrsn/hrcs/misn/mics/scsn/sccs)
  - assignparam parameters accepted: none
* mdsn/mdcs: similar sin/cos treatment, but for combined month/day
  - default infill: mean
  - default NArowtype: datetime
  - suffix appender: includes appenders for (mdsn/mdcs)
  - assignparam parameters accepted: none
* hmss/hmsc: similar sin/cos treatment, but for combined hour/minute/second
  - default infill: mean
  - default NArowtype: datetime
  - suffix appender: includes appenders for (hmss/hmsc)
  - assignparam parameters accepted: none
* dat6: default transformation set for time series data, returns:
'year', 'mdsn', 'mdcs', 'hmss', 'hmsc', 'bshr', 'wkdy', 'hldy'
  - default infill: mean
  - default NArowtype: datetime
  - suffix appender: includes appenders for ('year', 'mdsn', 'mdcs', 'hmss', 'hmsc', 'bshr', 'wkdy', 'hldy')
  - assignparam parameters accepted: none
* null: deletes source column
  - default infill: none
  - default NArowtype: exclude
  - no suffix appender, column deleted
  - assignparam parameters accepted: none
* excl: passes source column un-altered
  - default infill: none
  - default NArowtype: exclude
  - suffix appender: '_excl'
  - assignparam parameters accepted: none
* exc2/exc3: passes source column unaltered other than force to numeric, mode infill applied
  - default infill: mode
  - default NArowtype: numeric
  - suffix appender: '_exc2'
  - assignparam parameters accepted: none
* eval: performs distribution property evaluation consistent with the automunge
'powertransform' parameter activated to designated column
  - default infill: based on evaluation
  - default NArowtype: based on evaluation
  - suffix appender: based on evlauation
  - assignparam parameters accepted: none
* copy: create new copy of column, useful when applying the same transform to same column more
than once with different parameters. Does not prepare column for ML on it's own.
  - default infill: exclude
  - default NArowtype: exclude
  - suffix appender: '_copy'
  - assignparam parameters accepted: 'suffix' for custom suffix appender
* NArw: produces a column of boolean identifiers for rows in the source
column with missing or improperly formatted values. Note that when NArw
is assigned in a family tree it bases NArowtype on the root category, 
when NArw is passed as the root category it bases NArowtype on default.
  - default infill: not applicable
  - default NArowtype: justNaN
  - suffix appender: '_NArw'
  - assignparam parameters accepted: none
* NAr2: produces a column of boolean identifiers for rows in the source
column with missing or improperly formatted values.
  - default infill: not applicable
  - default NArowtype: numeric
  - suffix appender: '_NArw'
  - assignparam parameters accepted: none
* NAr3: produces a column of boolean identifiers for rows in the source
column with missing or improperly formatted values.
  - default infill: not applicable
  - default NArowtype: positivenumeric
  - suffix appender: '_NArw'
  - assignparam parameters accepted: none

Please note I recommend caution on using splt/spl2/spl5/spl6 transforms on categorical
sets that may include scientific units for instance, as prefixes will not be noted
for overlaps, e.g. this wouldn't distinguish between kilometer and meter for instance.
Note that overlap lengths below 5 characters are ignored.
* splt: searches categorical sets for overlaps between strings and returns new boolean column
for identified overlap categories. Note this treats numeric values as strings eg 1.3 = '1.3'.
Note that priority is given to overlaps of higher length, and by default overlap searches
start at 20 character length and go down to 5 character length.
  - default infill: none
  - default NArowtype: justNaN
  - suffix appender: '_splt_##*##' where ##*## is target idenbtified string overlap 
  - assignparam parameters accepted: 'minsplit': indicating lowest character length for recognized overlaps 
* spl2/spl3/spl4/ors2/txt3: similar to splt, but instead of creating new column identifier it replaces categorical 
entries with the abbreviated string overlap
  - default infill: none
  - default NArowtype: justNaN
  - suffix appender: '_spl2'
  - assignparam parameters accepted: 'minsplit': indicating lowest character length for recognized overlaps 
* spl5/spl6/ors5/ors6: similar to spl2, but those entries without idenitified string overlap are set to 0,
(used in ors5 in conjunction with ord3)
  - default infill: none
  - default NArowtype: justNaN
  - suffix appender: '_spl5'
  - assignparam parameters accepted: 'minsplit': indicating lowest character length for recognized overlaps 
* spl6: similar to spl5, but with a splt performed downstream for identification of overlaps
within the overlaps
  - default infill: none
  - default NArowtype: justNaN
  - suffix appender: '_spl5'
  - assignparam parameters accepted: 'minsplit': indicating lowest character length for recognized overlaps 
* spl7: similar to spl5, but recognizes string character overlaps down to minimum 2 instead of 5
  - default infill: none
  - default NArowtype: justNaN
  - suffix appender: '_spl5'
  - assignparam parameters accepted: 'minsplit': indicating lowest character length for recognized overlaps 
* nmrc/nmr2/nmr3: parses strings and returns any number groupings, prioritized by longest length
  - default infill: mean
  - default NArowtype: parsenumeric
  - suffix appender: '_nmrc'
* nmcm/nmc2/nmc3: similar to nmrc, but recognizes numbers with commas, returns numbers stripped of commas
  - default infill: mean
  - default NArowtype: parsenumeric_commas
  - suffix appender: '_nmcm'
  - assignparam parameters accepted: none
  
* new processing functions nmr4/nmr5/nmr6/nmc4/nmc5/nmc6/spl8/spl9/sp10 (spelled sp"ten"):
  - comparable to functions nmrc/nmr2/nmr3/nmcm/nmc2/nmc3/splt/spl2/spl5
  - but make use of new assumption that set of unique values in test set is same or a subset of those values 
    from the train set, which allows for a more efficient application (no more string parsing of test sets)
  - default infill: comparable
  - default NArowtype: comparable
  - suffix appender: same format, updated per the new category
  - assignparam parameters accepted: none

* new processing functions nmr7/nmr8/nmr9/nmc7/nmc8/nmc9:
  - comparable to functions nmrc/nmr2/nmr3/nmcm/nmc2/nmc3
  - but implements string parsing only for unique test set entries not found in train set
  - for more efficient test set processing in automunge and postmunge
  - (less efficient than nmr4/nmc4 etc but captures outlier points as may not be unusual in continuous distributions)
  - default infill: comparable
  - default NArowtype: comparable
  - suffix appender: same format, updated per the new category
  - assignparam parameters accepted: none

* new processing functions Utxt / Utx2 / Utx3 / Uord / Uor2 / Uor3 / Uor6 / U101
  - comparable to functions text / txt2 / txt3 / ordl / ord2 / ord3 / ors6 / 1010
  - but upstream conversion of all strings to uppercase characters prior to encoding
  - (e.g. 'USA' and 'usa' would be consistently encoded)
  - default infill: in uppercase conversion NaN's are assigned distinct encoding 'NAN'
  - and may be assigned other 9infill methods in assigninfill
  - default NArowtype: 'justNaN'
  - suffix appender: '_UPCS'
  - assignparam parameters accepted: none
  
* new processing root categories or11 / or12 / or13 / or14 / or15 / or16 / or17 / or18 / or19 / or20
  - or11 / or13 intended for categorical sets that may include multiple tiers of overlaps 
  and include base binary encoding via 1010 suppplemented by tiers of string parsing for 
  overlaps using spl2 and spl5, or11 has two tiers of overlap string parsing, or13 has three, 
  each parsing returned with an ordinal encoding sorted by frequency (ord3)
  - or12 / or14 are comparable to or11 / or13 but include an additional supplemental 
  transform of string parsing for numerical entries with nmrc followed by a z-score normalization 
  of returned numbers via nmbr
  - or15 / or16 / or17 / or18 comparable to or11 / or12 / or13 / or14 but incorporate an
  UPCS transform upstream and make use of spl9/sp10 insteadf of spl2/spl5 for assumption that
  set of unique values in test set is same or subset of train set for more efficient postmunge
  - or19 / or20 comparable to or16 / or18 but replace the 'nmrc' string parsing for numeric entries
  with nmc8 which allows comma characters in numbers and makes use of consistent assumption to
  spl9/sp10 that set of unique values in test set is same or subset of train for efficient psotmunge
  - assignparam parameters accepted: 'minsplit': indicating lowest character length for recognized overlaps 
  (note that parameter has to be assigned to specific categories such as spl2/spl5 etc)


And here are the series of family trees currently built into the internal library.

```
    transform_dict.update({'nmbr' : {'parents' : ['nmbr'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : [bint]}})
    
    transform_dict.update({'dxdt' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['dxdt'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'d2dt' : {'parents' : ['d2dt'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : ['dxdt']}})
    
    transform_dict.update({'d3dt' : {'parents' : ['d3dt'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : ['d2dt'], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'dxd2' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['dxd2'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'d2d2' : {'parents' : ['d2d2'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : ['dxd2']}})
    
    transform_dict.update({'d3d2' : {'parents' : ['d3d2'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : ['d2d2'], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'nmdx' : {'parents' : ['nmdx'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : ['dxdt']}})
    
    transform_dict.update({'nmd2' : {'parents' : ['nmd2'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : ['d2dt'], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'nmd3' : {'parents' : ['nmd3'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : ['d3dt'], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mmdx' : {'parents' : ['mnmx'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : ['dxdt']}})
    
    transform_dict.update({'mmd2' : {'parents' : ['mmd2'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : ['d2dt'], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mmd3' : {'parents' : ['mmd3'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : ['d3dt'], \
                                     'coworkers' : [], \
                                     'friends' : []}})

    transform_dict.update({'bnry' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bnry'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})

    transform_dict.update({'text' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['text'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'txt2' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['text'], \
                                     'cousins' : [NArw, 'splt'], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'txt3' : {'parents' : ['txt3'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['text'], \
                                     'friends' : []}})

    transform_dict.update({'UPCS' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['UPCS'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
  
    transform_dict.update({'Utxt' : {'parents' : ['Utxt'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['text'], \
                                     'friends' : []}})
    
    transform_dict.update({'Utx2' : {'parents' : ['Utx2'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['text'], \
                                     'friends' : ['splt']}})

    transform_dict.update({'Utx3' : {'parents' : ['Utx3'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['txt3'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'Uord' : {'parents' : ['Uord'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['ordl'], \
                                     'friends' : []}})
        
    transform_dict.update({'Uor2' : {'parents' : ['Uor2'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['ord2'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'Uor3' : {'parents' : ['Uor3'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['ord3'], \
                                     'friends' : []}})
    
    transform_dict.update({'Uor6' : {'parents' : ['Uor6'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['spl6'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['ord3'], \
                                     'friends' : []}})
    
    transform_dict.update({'U101' : {'parents' : ['U101'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['1010'], \
                                     'friends' : []}})
    
    transform_dict.update({'splt' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['splt'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'spl2' : {'parents' : ['spl2'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['ordl'], \
                                     'friends' : []}})
    
    transform_dict.update({'spl3' : {'parents' : ['spl2'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['ord3'], \
                                     'friends' : []}})
    
    transform_dict.update({'spl4' : {'parents' : ['spl4'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['spl3'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'spl5' : {'parents' : ['spl5'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['ord3'], \
                                     'friends' : []}})
    
    transform_dict.update({'spl6' : {'parents' : ['spl6'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['splt'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : ['ord3']}})
    
    transform_dict.update({'spl7' : {'parents' : ['spl7'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['ord3'], \
                                     'friends' : []}})

    transform_dict.update({'spl8' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['spl8'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})

    transform_dict.update({'spl9' : {'parents' : ['spl9'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['ordl'], \
                                     'friends' : []}})

    transform_dict.update({'sp10' : {'parents' : ['sp10'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['ord3'], \
                                     'friends' : []}})
    
    transform_dict.update({'nmrc' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['nmrc'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
  
    transform_dict.update({'nmr2' : {'parents' : ['nmr2'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'nmr3' : {'parents' : ['nmr3'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['mnmx'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})

    transform_dict.update({'nmr4' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['nmr4'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
  
    transform_dict.update({'nmr5' : {'parents' : ['nmr5'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'nmr6' : {'parents' : ['nmr6'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['mnmx'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'nmr7' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['nmr7'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
  
    transform_dict.update({'nmr8' : {'parents' : ['nmr8'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'nmr9' : {'parents' : ['nmr9'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['mnmx'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'nmcm' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['nmcm'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
  
    transform_dict.update({'nmc2' : {'parents' : ['nmc2'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'nmc3' : {'parents' : ['nmc3'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['mnmx'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})

    transform_dict.update({'nmc4' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['nmc4'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
  
    transform_dict.update({'nmc5' : {'parents' : ['nmc5'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'nmc6' : {'parents' : ['nmc6'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['mnmx'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})

    transform_dict.update({'nmc7' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['nmc7'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
  
    transform_dict.update({'nmc8' : {'parents' : ['nmc8'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'nmc9' : {'parents' : ['nmc9'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['mnmx'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'ors7' : {'parents' : ['spl6', 'nmr2'], \
                                     'siblings': [], \
                                     'auntsuncles' : ['ord3'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'ors5' : {'parents' : ['spl5'], \
                                     'siblings': [], \
                                     'auntsuncles' : ['ord3'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'ors6' : {'parents' : ['spl6'], \
                                     'siblings': [], \
                                     'auntsuncles' : ['ord3'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'ordl' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['ordl'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
        
    transform_dict.update({'ord2' : {'parents' : ['ord2'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['mnmx'], \
                                     'friends' : []}})
    
    transform_dict.update({'ord3' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['ord3'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
        
    transform_dict.update({'ord4' : {'parents' : ['ord4'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['mnmx'], \
                                     'friends' : []}})
    
    transform_dict.update({'ors2' : {'parents' : ['spl3'], \
                                     'siblings': [], \
                                     'auntsuncles' : ['ord3'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'or10' : {'parents' : ['ord4'], \
                                     'siblings': [], \
                                     'auntsuncles' : ['1010'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['mnmx'], \
                                     'friends' : []}})
    
    transform_dict.update({'or11' : {'parents' : ['sp11'], \
                                     'siblings': [], \
                                     'auntsuncles' : ['1010'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
  
    transform_dict.update({'or12' : {'parents' : ['nmr2'], \
                                     'siblings': ['sp11'], \
                                     'auntsuncles' : ['1010'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'or13' : {'parents' : ['sp12'], \
                                     'siblings': [], \
                                     'auntsuncles' : ['1010'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'or14' : {'parents' : ['nmr2'], \
                                     'siblings': ['sp12'], \
                                     'auntsuncles' : ['1010'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})

    transform_dict.update({'or15' : {'parents' : ['or15'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['sp13'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['1010'], \
                                     'friends' : []}})
  
    transform_dict.update({'or16' : {'parents' : ['or16'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['nmr2'], \
                                     'niecesnephews' : ['sp13'], \
                                     'coworkers' : ['1010'], \
                                     'friends' : []}})
    
    transform_dict.update({'or17' : {'parents' : ['or17'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['sp14'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['1010'], \
                                     'friends' : []}})
    
    transform_dict.update({'or18' : {'parents' : ['or18'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['nmr2'], \
                                     'niecesnephews' : ['sp14'], \
                                     'coworkers' : ['1010'], \
                                     'friends' : []}})
    
    transform_dict.update({'sp13' : {'parents' : ['sp13'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : ['sp10'], \
                                     'coworkers' : ['ord3'], \
                                     'friends' : []}})
    
    transform_dict.update({'sp14' : {'parents' : ['sp14'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : ['sp13'], \
                                     'coworkers' : ['ord3'], \
                                     'friends' : []}})

    transform_dict.update({'or19' : {'parents' : ['or19'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['nmc8'], \
                                     'niecesnephews' : ['sp13'], \
                                     'coworkers' : ['1010'], \
                                     'friends' : []}})
    
    transform_dict.update({'or20' : {'parents' : ['or20'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['nmc8'], \
                                     'niecesnephews' : ['sp14'], \
                                     'coworkers' : ['1010'], \
                                     'friends' : []}})
    
    transform_dict.update({'sp11' : {'parents' : ['sp11'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : ['spl5'], \
                                     'coworkers' : ['ord3'], \
                                     'friends' : []}})
    
    transform_dict.update({'sp12' : {'parents' : ['sp12'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : ['sp11'], \
                                     'coworkers' : ['ord3'], \
                                     'friends' : []}})
    
    transform_dict.update({'om10' : {'parents' : ['ord4'], \
                                     'siblings': [], \
                                     'auntsuncles' : ['1010', 'mnmx'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['mnmx'], \
                                     'friends' : []}})

    transform_dict.update({'mmor' : {'parents' : ['ord4'], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mnmx'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'1010' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['1010'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})

    transform_dict.update({'null' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['null'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'NArw' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['NArw'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'NAr2' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['NArw'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'NAr3' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['NArw'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'NAr4' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['NArw'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'nbr2' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['nmbr'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'nbr3' : {'parents' : ['nbr3'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : ['bint']}})
    
    transform_dict.update({'MADn' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['MADn'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'MAD2' : {'parents' : ['MAD2'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'MAD3' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['MAD3'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mnmx' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mnmx'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mnm2' : {'parents' : ['nmbr'], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mnmx'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mnm3' : {'parents' : ['nmbr'], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mnm3'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mnm4' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mnm3'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mnm5' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mnmx'], \
                                     'cousins' : ['nmbr', NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mnm6' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mnm6'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mnm7' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mnmx', 'bins'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})

    transform_dict.update({'mean' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mean'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mea2' : {'parents' : ['nmbr'], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mean'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mea3' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mean', 'bins'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})

    transform_dict.update({'date' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['year', 'mnth', 'days', 'hour', 'mint', 'scnd'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
  
    transform_dict.update({'dat2' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bshr', 'wkdy', 'hldy'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'dat3' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['year', 'mnsn', 'mncs', 'dysn', 'dycs', 'hrsn', 'hrcs', 'misn', 'mics', 'scsn', 'sccs'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'dat4' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['year', 'mdsn', 'mdcs', 'hmss', 'hmsc'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'dat5' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['year', 'mdsn', 'mdcs', 'dysn', 'dycs', 'hmss', 'hmsc'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'dat6' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['year', 'mdsn', 'mdcs', 'hmss', 'hmsc', 'bshr', 'wkdy', 'hldy'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'year' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['year'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
  
    transform_dict.update({'yea2' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['year', 'mdsn', 'mdcs'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mnth' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mnth'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
  
    transform_dict.update({'mnt2' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mnsn', 'mncs'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mnt3' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mnsn', 'mncs', 'dysn', 'dycs', 'hrsn', 'hrcs', 'misn', 'mics', 'scsn', 'sccs'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mnt4' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mdsn', 'mdcs'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mnt5' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mdsn', 'mdcs', 'hmss', 'hmsc'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mnt6' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mdsn', 'mdcs', 'dysn', 'dycs', 'hmss', 'hmsc'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mnsn' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mnsn'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mncs' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mncs'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mdsn' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mdsn'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mdcs' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mdcs'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})

    transform_dict.update({'days' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['days'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
  
    transform_dict.update({'day2' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['dysn', 'dycs'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'day3' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['dysn', 'dycs', 'hrsn', 'hrcs', 'misn', 'mics', 'scsn', 'sccs'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'day4' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['dhms', 'dhmc'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'day5' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['dhms', 'dhmc', 'hmss', 'hmsc'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'dysn' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['dysn'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'dycs' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['dycs'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'dhms' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['dhms'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'dhmc' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['dhmc'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'hour' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['hour'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
  
    transform_dict.update({'hrs2' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['hrsn', 'hrcs'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'hrs3' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['hrsn', 'hrcs', 'misn', 'mics', 'scsn', 'sccs'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'hrs4' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['hmss', 'hmsc'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'hrsn' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['hrsn'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'hrcs' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['hrcs'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'hmss' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['hmss'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'hmsc' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['hmsc'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mint' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mint'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
  
    transform_dict.update({'min2' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['misn', 'mics'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'min3' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['misn', 'mics', 'scsn', 'sccs'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'min4' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mssn', 'mscs'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'misn' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['misn'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mics' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mics'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mssn' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mssn'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'mscs' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['mscs'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})

    transform_dict.update({'scnd' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['scnd'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
  
    transform_dict.update({'scn2' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['scsn', 'sccs'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'scsn' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['scsn'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'sccs' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['sccs'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'bxcx' : {'parents' : ['bxcx'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['nmbr'], \
                                     'friends' : []}})
    
    transform_dict.update({'bxc2' : {'parents' : ['bxc2'], \
                                     'siblings': ['nmbr'], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'bxc3' : {'parents' : ['bxc3'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : ['nmbr'], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'bxc4' : {'parents' : ['bxc4'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : ['nbr2'], \
                                     'friends' : []}})
    
    transform_dict.update({'pwrs' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['pwrs'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'pwr2' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['pwr2'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'log0' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['log0'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'log1' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['log0', 'pwr2'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'sqrt' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['sqrt'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'wkdy' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['wkdy'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'bshr' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bshr'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'hldy' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['hldy'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'bins' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bins'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'bint' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bint'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'bsor' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bsor'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'bnwd' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bnwd'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'bnwK' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bnwK'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
  
    transform_dict.update({'bnwM' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bnwM'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'bnwo' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bnwo'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
  
    transform_dict.update({'bnKo' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bnKo'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'bnMo' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bnMo'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})    
    
    transform_dict.update({'bnep' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bnep'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'bne7' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bne7'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'bne9' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bne9'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'bneo' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bneo'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'bn7o' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bn7o'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})

    transform_dict.update({'bn9o' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['bn9o'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'pwor' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['pwor'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'por2' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['por2'], \
                                     'cousins' : [NArw], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'copy' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['copy'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'excl' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['excl'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'exc2' : {'parents' : ['exc2'], \
                                     'siblings': [], \
                                     'auntsuncles' : [], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : []}})
    
    transform_dict.update({'exc3' : {'parents' : [], \
                                     'siblings': [], \
                                     'auntsuncles' : ['exc2'], \
                                     'cousins' : [], \
                                     'children' : [], \
                                     'niecesnephews' : [], \
                                     'coworkers' : [], \
                                     'friends' : ['bins']}})
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
and incorproate a handful of simple data structures, which we'll demonstrate below.

Let's say we want to recreate the mm3 category which caps outliers at 0.01 and 0.99
quantiles, but instead make it the 0.001 and 0.999 quantiles. Well we'll call this 
cateogry mnm8. So in order to pass a custom transformation function, first we'll need 
to define a new root category transformdict and a corresponding processdict.

```
#Let's creat a really simple family tree for the new root category mnmn8 which
#simply creates a column identifying any rows subject to infill (NArw), performs 
#a z-score normalization, and seperately performs a version of the new transform
#mnm8 which we'll define below.

transformdict = {'mnm8' : {'parents' : [], \
                           'siblings': [], \
                           'auntsuncles' : ['mnm8', 'nmbr'], \
                           'cousins' : ['NArw'], \
                           'children' : [], \
                           'niecesnephews' : [], \
                           'coworkers' : [], \
                           'friends' : []}, \

#Note that since this mnm8 requires passing normalization parameters derived
#from the train set to process the test set, we'll need to create two seperate 
#trasnformation functions, the first a "dualprocess" function that processes
#both the train and if available a test set simultaneously, and the second
#a "postprocess" that only processes the test set on it's own.

#So what's being demonstrated here is that we're passing the functions under
#dualprocess and postprocess that we'll define below.

processdict = {'mnm8' : {'dualprocess' : process_mnm8_class, \
                         'singleprocess' : None, \
                         'postprocess' : postprocess_mnm8_class, \
                         'NArowtype' : 'numeric', \
                         'MLinfilltype' : 'numeric', \
                         'labelctgy' : 'mnm8'}}

#Now we have to define the custom processing functions which we are passing through
#the processdict to automunge.

#Here we'll define a "dualprocess" function intended to process both a train and
#test set simulateously. We'll also need to create a seperate "postprocess"
#function intended to just process the test set.

#define the function
def process_mnm8_class(mdf_train, mdf_test, column, category, \
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

  #create thee new column, using the category key as a suffix identifier
  
  #copy source column into new column
  mdf_train[column + '_mnm8'] = mdf_train[column].copy()
  mdf_test[column + '_mnm8'] = mdf_test[column].copy()
  
  
  #perform an initial infill method, here we use mean as a plug, automunge
  #will seperately perform a infill method per user specifications elsewhere
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
  
  #replace values < quantile10 with quantilemin for both train and test data
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
  # 'origcolumn' : column, \ -> ID of original column in train set
  # 'columnslist' : nmbrcolumns, \ -> a list of columns created in this transform, 
  #                                  later fleshed out to include all columns derived from same source column
  # 'categorylist' : [nc], \ -> a list of columns created in this transform
  # 'infillmodel' : False, \ -> populated elsewhere, for now enter False
  # 'infillcomplete' : False, \ -> populated elsewhere, for now enter False
  # 'deletecolumn' : False}} -> populated elsewhere, for now enter False
  
  #for column in nmbrcolumns
  for nc in nmbrcolumns:

    if nc[-5:] == '_mnm8':

      column_dict = { nc : {'category' : 'mnm8', \
                           'origcategory' : category, \
                           'normalization_dict' : nmbrnormalization_dict, \
                           'origcolumn' : column, \
                           'columnslist' : nmbrcolumns, \
                           'categorylist' : nmbrcolumns, \
                           'infillmodel' : False, \
                           'infillcomplete' : False, \
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

def postprocess_mnm3_class(mdf_test, column, postprocess_dict, columnkey, params={}):
  #where mdf_test is a dataframe of the test set
  #column is the string of the column header
  #postprocess_dict is how we carry packets of data between the 
  #functions in automunge and postmunge
  #columnkey is a key used to access stuff in postprocess_dict if needed
  #(columnkey is only valid for initial root categories, if you want to use function
  #as a downstream category we have to recreate a columnkey such as follows for normkey)
  #and params are any column specific parameters to be passed by user in assignparam

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
#and in that case we wouldn't need to define seperate functions for 
#dualprocess and postprocess, we could just define what we call a singleprocess 
#function incorproating similar data strucures but without only a single dataframe 
#passed

#Such as:
def process_mnm8_class(df, column, category, postprocess_dict):
  
  #etc
  
  return return df, column_dict_list

#For a full demonstration check out my essay 
"Automunge 1.79: An Open Source Platform for Feature Engineering"


```

And there you have it, you now have all you need to wrangle data on the 
Automunge platform. Feedback is welcome.


...

As a citation, please note that the Automunge package makes use of 
the Pandas, Scikit-learn, and NumPy libraries.

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

Sorry I don't know what paper to cite, but Numpy website at:
https://www.numpy.org/

...

Have fun munging!!

...

You can read more about the tool through the blog posts documenting the
development on Medium [here](https://medium.com/automunge) or for more
writing I recently completed my first collection of essays titled "From
the Diaries of John Henry" which is also available on Medium
[turingsquared.com](https://turingsquared.com).

The Automunge website is helpfully located at URL
[automunge.com](https://automunge.com).

...

Patent Pending, application 16552857
