Changelog

v2.11
fixed process flaw in feature importance evaluation

v2.12
corrected family tree derivations for primitives with offspring

2.13
a few clarifications associated with the new functionality for passing lists of ID column strings such as to trainID_column from version 2.13

v2.14
apply ravel to returned labels numpy array if appropriate - converts e.g. [[1,2,3]] to [1,2,3]

v2.15
added comparable treatment from version 2.14 updates for labels set to both validation labels sets returned from automunge(). (Move fast and fix things, that's our motto.)

v2.16
1) User can now pass ML_cmnd to automunge function with 'PCA_cmnd':{'bool_PCA_excl':True}
Which excludes returned boolean columns from any PCA dimensionality reduction. This may 
result in memory performance improvements for large datasets.

2) Corrected bug for labels processing in postmunge function

3) Corrected derivation of LabelFrequencyLevelizer, number of appended sets reduced by 1

4) Corrected bug for LabelFrequencyLevelizer of "singlect" (binary) category labels

v2.17
User can now pass dataframes with inclusion of non-index-range column as index or with multi column index (requires named index columns). Such index types are added to the returned "ID" sets which are consistently shuffled and partitioned as the train and test sets.

v2.18
New infill methods mean and median available for numeric sets.
User can now pass partial list of infill methods to automunge in assigninfill.
A fed clean ups in the code.

v2.19
2.19
Fixed an issue with passing non-contiguous integer indexes introduced in the 2.17 update.

v2.20
Small corrections to the insertinfill function to address found bug associated with data type of inserted values.

v2.21
reverted one of the two edits from 2.20, much better now

v2.22
- fixed bug with 'null' processing (for deletion of set)
- removed all instances of SciKit's Labelencoder / OneHotEncoder functions, replaced with method making use of pandas
- set default feature importance evaluation to off since it requires inclusion of labels column in training set
- new processing functions for log transform and bins groupings by powers of 10, 
processing categories log0, log1, pwrs

v2.23
- ordinal processing for categorical data via 'ordl'
- numbercategoryheuristic repurposed for ordinal evaluation in evalcategory
- option to exclude ordinal columns from PCA transform with PCA_cmnd bool_ordl_PCAexcl
- "verbose" option via printstatus
- new method to process labels for validation sets, now incorporated into postmunge call
- Feature importance evaluation now uses validation set ratios passed to automunge
- a few improvements to labelfrequencylevelizer, especially for single column labels
- labelfrequencylevelizer works with ordinal labels

v2.24
- added check for assigncat and assigninfill to return error message if redundant column assignments
- checked Null determination in evalcategory from >= 85% NaN to >80% NaN. 
- unique value evaluation in evalcategory now only performed once and saved
- added a note at header for livening /. intellectual property
- cleanup inn postprocess_bxcx
- fixed bug with shuffle train recently introduced as df_labels was referenced before assignment
- silenced evaluation for train category and test category comparison for cases where no df_test was passed
- changed the use of numbercategoryheuristic now it is only used for determination of ordinal processing
- fixed bug with process_mnm3
- changed transformdict definition of MAD2
- fixed bxc4 trasnformdict definition
- fixed scenario where test column only has two categories (bury) and train has more for either text or ordl
(necessary when deferring to evalcategory for category)
- bug fix for postprocess_date applicable to dat2
- added new argument to insertinfill function to handle different use in adjinfill when only inserting infill
to one column at a time for a multi column set
- performed audit of processing function root categories

v2.25
- noted that evaluation of numerical data now includes method to apply transformations based on inferred distribution properties
- updated the family tree primitives based on removal of grandparents, great-grandparents to address discrepancy in documentation (equivalent functionality was possible with fewer primitives)
- added description for MAD3 processing category added to library with 2.25

v2.26
found a typo in evalcategory

v2.27
additional printstatus printout before category evaluation to aid troubleshooting in operation

v2.28
- made powertransform evaluation optional to be more consistent with documentaiton
- changed order of operations for label column to before infill to be consistent with documentaiton
- fixed infill loop to be consistent with documentation
- updated zeroinfill to work on multicolumn sets
- added new printouts to signal completion

v2.29
- added a test for nan for binarymissingplug in process binary class
- updated logic tests in evalcategory to address potential assignment scenarios to bnry or text class
- fixed error in evalcategory (an elif was used instead of an if)
- fixed another error in evalcateogry, changed mc2 from mc2[0][0] to mc2[1][0]
- small correction in PCA methods archiving parameters
- printouts for feature importance and pCA dimensionality reduction
- final printouts

v2.30
- replaced use of Scikit labelbinarizer in bnry category with pandas method
- changed a few default parameters for automunge function (now defaults to validation ratios at 0 and no binstransform)

v2.31
- fixed binary processing bug for another outlier scenario
- new automunge parameter NArw_marker to elect whether to include marker column for rows subject ot infill (default to True)

v2.32
fixed issue for binary processing of columns that were already boolean encoded

v2.33
updated bxcx family tree definition

v2.34
bug fix in postprocess pwrs

v2.35
- new infill type oneinfill
- lower powertransform normality threshold for z-score
- printouts now include type of PCA model applied in automunge
- edge case for standard infill in boolean sets with only one value, plug value instead of most common
- fixed powertrasnform to only apply box-cox to all-positive sets
- fixed feature importance evaluation to include any columns from user passed values for NArw_marker

v2.36
- Frequency levelizer option added to postmunge.
- fixed nbr3 definition
- Printout returned label set before Automunge complete
- assign powertransform to specific columns in assigncat via 'eval'
- validate passed transformdict to determine if a root defined without a replacement column
- validate passed ML_cmnd and fixed format if missing entries

v2.37
fixed bug introduced in 2.36 associated with printouts for postmunge

v2.38
- user passed trasnformdict without a replacement primitive now assigned a passthrough category (excl) to auntsuncles
- revisited a few of the logic steps performed in evalcategory
- removed a few steps of validation comparing train and test data column configurations for simplicity and run time. Method now makes assumption without verification that test data is consistently formatted as train data.
- cleanup of a few variable naming conventions

v2.39
- a few more printouts for feature importance evaluation for clarity
- update method for calculating bins for powers of 10 
- added two new entries to postprocess_dict listing returned columns for labels and ID sets for cases where user elected return of numpy arrays and later needs to identify column. You know, logistics.

v2.40
- Added support for passed numpy arrays. Input datasets can now be either pandas or numpy objects.

v2.41
- incorporated feature importance evaluation for postmunge, available by passing featureeval=True

v2.42
quick typo correction for postmunge feature importance evaluation
quick fix missing comma

v2.43
- removed use of get_values() as was kind of amateurish

v2.44
found bug originating from conversion to boolean when running on alternate platform

v2.45
a few small bug fixes

v2.46
- cleaned up one hot encoding function
- cleaned up normalization dictionary for categorical function
- overhaul of labels processing, now with generalized implementation
- now integer array indexes can be passed as integers as well as strings
- no more use of sklearn preprocessing library

v2.47
- new binary encoding for categorical sets: '1010'
- overhaul of date time processing
- new sin & cos transformations for time series date at period of time scale
- ML infill now available for time series data

v2.48
- changed default automated time series processing to 'dat6'
- added special cases for all standard deviation and mean abs dev normalizations if ==0 then set = 1 to avoid div by 0
- changed all float to float32, all bool int to int8, all ordl left alone
(the thinking is that since the data is normalized we don't need as many significant figures, note that we're keeping this conversion inside the processing functions so we have option to later create alternate processing functions with different precisions)
- new processding function ord3, ordinal processing in order of frequency of occurance
- and ord4 (ord3 with minmax scaling)
- for logarithmic and powers transofrm, replaced missing data with 0 instead of mean (makes more sense)
- new featuremethod option 'report' to only return results of feature importance evaluation with no further processing
- fixed bug for assembly of ID set in postmunge

v2.49
fixed small bug with standard deviation bins introduced in last update

v2.50
- new infill option modeinfill, inserts most common value for a set
- modeinfill also works for one-hot encoded sets
- changed default option for exc2 category to mode

v2.51
- new automunge parameter floatprecision
- new root categories or10, om10, mnm0

v2.52
- changed dtype evaluation method for floatprecision
- replaced mnm0 category with mmor

v2.53
- fixed improperly named variable

v2.54
- new method for retrieving normalization parameters in text category, now one hot encoding can be performed downstream of prior transformations
- prior method reframed as support function for other functions' use

v2.55
restructuring imports to pip install Automunge etc. I think this will work fingers crossed.

v2.56
trying to get imports set up, forgot to include init duh

v2.57
revision to new import procedure

v2.58
update version number

v2.59
new transformation categories to convert continuous numerical sets into bounded range:
dxdt, d2dt, d3dt, dxd2, d2t2, d3d2, nmdx, nmd2, nmd3, mmdx, mmd2, mmd3
where e.g. dxdt is rate of change (velocity), d2dt (acceleration), d3dt (jerk)
assumes unshuffled data with consistent time steps
dxd2 comparable but evaluates delta of average of two rows with average of two preceding rows
such as to denoise the data
nmdx is comparable but a z-score normalization is performed prior
mmd is comparable but a min-max scaling is performed prior

v2.60
small updates:
- added support for edge case of user passing empty dictionaries to assigncat and assigninfill
- cleaned up a few errant comments

v2.61
- simplified logic tests to evaluate default transformations categories

v2.62
- new postmunge option driftreport
- performs evaluation of data set drift for comparison between columns of the original training data passed to automunge(.) and corresponding columns of subsequent data passed to postmunge(.), prints report.

v2.63
- new postprocess_dict entries to support versioning: 
'application_number', 'application_timestamp', 'version_combined'
- improved printouts for driftreport
- new driftreport metric:
'pct_NArw' tracks percent of cells subject to infill for NArw columns

v2.64
- added "versioning serial stamp" to automunge printouts
- new processing function 'splt':
compare unique values of categorical set strings, if portions of the strings have overlap, 
create a new column indicator 
    #for example, if a categoical set consisted of unique values ['west', 'north', 'northeast']
    #then a new column would be created idenitifying cells which included 'north' in their entries
    #(here for north and northeast)

v2.65
- new feature engineering transformation category 'spl2'
- similar to splt rolled out in 2.64, but instead of creating new column identifier
it replaces categorical entries with the abbreviated string overlap
for example, the set of unique values in a column {'square', 'circle', 'spongebobsquare', 'turingsquared'}
would be transformed such that column only containd the set {'square', 'circle'}
- also defined a suite of sets of feature engineering functions that make use of this method
including 'txt3', 'spl2', 'spl3', 'spl4', 'ors2'
entry to assigncat in default automunge call for completeness

v2.66
- small correction to 'spl2' processing function for populating data structures

v2.67
- new automunge parameter evalcat modularizes the automated evaluation of column properties for assignment of root transformation categories, allowing user to pass custom functions for this purpose (as we're little stretched thin I think there is much room for innovation on this point and don't want to hold anyone back with my version if they have a better mousetrap)
formal documentation for this option is pending, for now if you'd like to experiment copy the evalcategory function from master file
- new driftreport metrics for numerical sets such as standard deviation for min-max scaled columns, min and max for z-score normalized columns, activation ratios for binary encoded sets, and column specific activation ratios for one-hot-encoded sets
- fixed bug associated with driftreport assembly for one-hot encoded columns

v2.68
- added dependancies to setup file
- new root category spl5, comparable to spl2 in which strings are parsed for overlap and new column created with overlap entries consolidated, but in this version entries without overlap are set to 0
- new root category ors5, uses spl5, in which a copy of column is ordinal encoded with ord3 and a second copy of column has spl5 transforms applied then ordinal encoded with ord3

v2.69
- New processing function spl6, ors6
(comparable to spl5, ors6, but with an additional splt transform applied downstream of the spl5 for identification of a second tier of text overlaps within the text overlaps)
- New driftreport metric for activation ratios of categorical entries boolean encoded in 1010
- Corrected column id suffix appender for spl5 transform

2.70
Updates associated with quality control audit to ensure application matches documentation.
- extensive update to READ ME documentation
- new design philosophy, any processing function can be applied to any type of data, 
although if it is not suited for that data (such as applying numerical transform to a 
categorical set) it will return all 0's
- updated all processing functions to achieve this functionality
- updated NArw function and evalcategory function to perform evaluations on copy of 
column instead of source column to avoid potential for data corruption
- new NArw root categories based on different distinct NArowtypes NArw (justNaN), NAr2 
(numeric) and NAr3 (positivenumeric)
- updated log transforms and bxcx transform functions for postitivenumeric NArwtype
- exc2 now forces all data to numeric and applies modeinfill
- exc3 now includes the bins trasnform such as if user wants to prepare for class 
imbalance oversampling a numerical set while leaving the format of source label column 
intact (previously we had these in exc2, I think since exc2 is the base transform is less
confusing to include bins in exc3)
- corrected mnm3 transform, found an inconsistency in infill methods between dualprocess 
and postprocess functions
- updated postmunge label processing to be consistent with automunge in that rows 
corresponding to missing label values are dropped.
- found and corrected bug associated with infill (was saving and accessing infill values
incorrectly resulting in unintentional overwrite in some instances)
- fixed bug with ors6
- fixed bug with dhmc
- fixed bug with sccs
- a few code comment cleanups in suite of time-series processing functions
- corrected derivation of driftreport metrics for a few of the numerical transforms to 
take place prior to transformations
- corrected family tree for nmdx
- addressed outlier scenario for splt, spl2, spl6 when cateogrical entries include 
numeric values, converted all numbers to strings for this transform

2.71 updates
- a few code comment cleanups
- fixed default MLinfill parameter

2.72 updates
- updated splt categories to base search of max string length in unique values
- updated methods for splt, spl2, spl5 to account for bug when overlaps exceed hard coded search length. Now method bases string overlap search on max length of unique values.
- updated length of min evaluated overlap length to match READ ME

2.73 updates
- updated TrainLabelFreqLevel (class imbalance levelizer) function for support of numerical data levelized basis on powers of 10 for floats <1 (eg 10^-1, 10^-2, etc)

2.75 updates:
- found and fixed error in dxd2 transform (number of rows offset was inconsistent with documentation)
- new transform 'sqrt' applies square root transform
- new NArowtype 'nonnegativenumeric', similar to 'positivenumeric' but allows 0 values
- new NArw root category NAr4 based on nonnegativenumeric

2.78 updates:
- simplified bulk exclusion from processing with new powertransform option
- if user passes powertransform = 'excl', columns not explicitly assigned to a root category in assigncat will be left untouched
- or if user passes powertransform = 'exc2', columns not explicitly assigned to a root category in assigncat will be forced to numeric and subject to default modeinfill
- made a special case for 'excl' category in general in that columns processed with this (pass-through) function are now returned without a suffix appender to the column header
- found and fixed missing application of floatprecision to labels set if included in test data

2.79 updates:
- replaced the default trasnforms for powers of ten encoding with 'pwr2' (one-hot) and 'por2' (ordinal)
- powers of ten now encodes both positive and negative values as 'column_10^#' and 'column_-10^#'
- powers of ten one-hot encoding now defaults infill as no activations
- powers of ten ordinal encoding now matches test set potential set of values to train set
- powers of ten ordinal encoding now defgaults infill as distinct encoding (0)

2.80 updates:
- new processing functions nmrc/nmr2/nmr3/nmcm/nmc2/nmc3/ors7/spl7
- nmrc: parses strings to return any numeric entries, entrisd without number suject to infill
- nmrc: if multiple numbers found in entry returns the number with most characters
- nmrc: does not recognize numbers with commas, for instance '1,200.20 ft' returns as 200.20
- nmr2: nmrc with z-score normalization applied to returned numbers
- nmr3: nmrc with min-max scaling applied to returned numbers
- nmcm: similar to nmrc, but recognizes commas, for instance '1,200.20 ft' returns as 1200.20
- nmc2: nmcm with z-score normalization applied to returned numbers
- nmc3: nmcm with min-max scaling applied to returned numbers
- ors7: similar to ors6 family of derivatoins as demonstrated in last essay and incorporates nmr2 
- spl7: same as spl5 but recognizes string character overlaps do0wn to single character instead of minimum 5
- new NArowtype 'parsenumeric' identifies rows without parsed number entries as subjhec tot infill
- new NArowtype 'parsenumeric_commas' identifies rows without parsed number entries with commas as subject to infill
- new NArowtype 'datetime' identifies rows without datetime entries as subject to infill
- changed name of NArows(.) function to getNArows(.)
- new support functions for these methods is_number(.), is_number_commas(.), parsenumeric(.), parsedate(.)

2.81 updates
- new processing functions nmr4/nmr5/nmr6/nmc4/nmc5/nmc6/spl8/spl9/sp10 (spelled sp"ten")
- comparable to functions nmrc/nmr2/nmr3/nmcm/nmc2/nmc3/splt/spl2/spl5
- but make use of new assumption that set of unique values in test set is same or a subset of those values from the train set, which allows for a more efficient application (no more string parsing of test sets)

2.82 updates
- new processing functions nmr7/nmr8/nmr9/nmc7/nmc8/nmc9
- comparable to functions nmrc/nmr2/nmr3/nmcm/nmc2/nmc3
- but make use of new method whereby test set entries are string parsed only in cases where those unique entries weren't found in the train set

2.83 updates
- new processing functions for mean normalization => (x - mean) / (max - min)
- mean: basic mean normalization
- mea2: mean normalization coupled with a z-score normalization
- mea3: mean normalization coupled with standard deviation bins

2.84 updates
- new processing function 'UPCS' converts categorical string sets to uppercase strings
- such as for consistent encodings if same entry included with upper and lowercase characters
- (e.g. 'USA' and 'usa' would be consistently encoded)
- new suite of categorical processing functions incorporating UPCS upstream of encodings:
- Utxt / Utx2 / Utx3 / Uord / Uor2 / Uor3 / Uor6 / U101
- comparable to text / txt2 / txt3 / ordl / ord2 / ord3 / ors6 / 1010

2.85 updates
- replaced postmunge returned object finalcolumns_test with postreports_dict
- postreports_dict contains results of optional feature importance evaluation, drift report, and list of final columns, may be used for future extensions as well
- see postreports_dict['featureimportance'] / postreports_dict['driftreport'] / postreports_dict['finalcolumns_test']
- updated printouts for automunge and postmunge associated with feature importance and postmunge drift report, postmunge printouts now tied to printstatus parameter
- fixed bug in automunge and postmunge for processing of dataframes with non-range integer index

2.86 updates
- new processing root cateogry family trees: or11 / or12 / or13 / or14
- or11 / or13 intended for categorical sets that may include multiple tiers of overlaps and include base binary encoding via 1010 suppplemented by tiers of string parsing for overlaps using spl2 and spl5, or11 has two tiers of overlap string parsing, or13 has three, each parsing returned with an ordinal encoding sorted by frequency (ord3)
- or12 / or14 are comparable to or11 / or13 but include an additional supplemental transform of string parsing for numerical entries with nmrc followed by a z-score normalization of returned numbers via nmbr 

2.87 updates
- new processing root category family trees: or15 / or16 / or17 / or18
- comparable to or11 / or12 / or13 / or14
- but incorporate an UPCS transform upstream of encodings
- for consistent encodings in case of string case discrepencies
- and make use of spl9 / sp10 instead of spl2 / spl5
- for assumption that set of unique values in test set is same or subset of train set 
- (for more efficient postmunge)

2.88
- new processing root category family trees: or19 / or20 
- comparable to or16 / or18
- but for numeric string parsing make use of nmc8 
- which allows comma characters in numbers
- and make use of consistent assumption to spl9/sp10 
- that set of unique values in test set is same or subset of train set 
- (for more efficient postmunge)

2.89 updates
- ML infill now available for '1010' binary encoded categorical sets
- Feature importance evaluation now available for '1010' binary encoded categorical label sets
- new MLinfilltype '1010' available for assignment in processdict
- thinking about making '1010' the new default for categorical sets instead of one-hot encoding, first will put some thought into specifics, to be continued...

2.90 updates
- quick bug fix

2.92 updates
- changed default categorical processing to binary encoding via '1010' instead of one-hot encoding (labels will remain 'text')
- changed default numbercategoryheuristic to 63 (<=63 number unique values a column will be binary encoded via '1010', above ordinal encoded via 'ord3')
- moved some edge cases for '1010' MLinfill from automunge(.) into the support functions
- fixed bug for '1010' MLinfill support functions
- changed default infill for '1010' encoding to all zeros
- simplified the evalcategory function to support customizations with evalcat
- fixed a bug in evalcategory function for ordinal encoding assignment

2.94 updates
- new automunge parameter assignparam, for passing column-specific parameters to transformation functions
- thus allowing more tailored processing methods without having to redefine processing functions
- Automunge now a 'universal' function programming language
- updated various functions to support the method
- processing functions now have a new optional parameter which defaults to params={}
- logistics further detailed in READ ME
- 'splt' family of transforms now accept 'minsplit' parameter to customize the minimum character length threshold for overlap detection
- fixed bug for NArowtype's 'parsenumeric', 'parsecommanumeric', and 'exclude'
- fixed bug for driftreport when set includes a category 'null'

2.95 updates
- new option for user to overwrite transformation function parameters on all columns
- without need to assign them individually
- by passing entries in assignparam under 'default_assignparam'
- further details of logistics provided in READ ME
- note that thisz method only overwrites the default parameters for those not otherwise specified

2.96 updates
- fixed a small bug with functionality rolled out in 2.95 (found an edge case)
- fixed a small bug for assignparam associated with feature importance evaluation

2.97 updates
- new automunge parameters LabelSmoothing_train / LabelSmoothing_test / LabelSmoothing_val
- new postmunge parameter LabelSmoothing
- note that Label Smoothing as implemented still supports oversampling preparation via TrainLabelFreqLevelizer
- Label Smoothing refers to the regularization tactic of transforming boolean encoded labels from 1/0 designations to some mix of reduced/increased threshold - for example passing the float 0.9 would result in the conversion from the set 1/0 to 0.9/#, where # is a function of the number of cateogries in the label set - for example for a boolean label it would convert 1/0 to 0.9/0.1, or for the one-hot encoding of a three label set it would be convert 1/0 to 0.9/0.05. Hat tip for the concept to "Rethinking the Inception Architecture for Computer Vision" by Szegedy et al.

2.98 updates
- new edge case for label smoothing
- if labels category evaluated as 'bnry' and label smoothing desired
- reset labels category to 'text' (one hot encoding)
- a reminder 'bnry' is the single column encoding of binary variables

2.99 updates
- a user can now pass True to automunge(.) parameters LabelSmoothing_test and LabelSmoothing_val to consistently encode as parameter LabelSmoothing_train
- a user can now pass True to postmunge(.) parameter LabelSmoothing to consistently encode as automunge(.) parameter LabelSmoothing_train accessed from the postprocess_dict

3.00 updates
- new automunge and postmunge parameter LSfit
- LSfit removes assumption of equal distribution of labels for smoothing parameter K in label smoothing to a fitted K tailored to activation ratios associated with each label category
- Thus LSfit introduces a little more intelligence into the Label Smoothing equation by way of creating a parameterized smoothing factor K as a function of the activation column and the target column associated with each cell
- LSfit defaults to False for prior assumption of even distribution of label classes, conducts fitting operation when passed as True
- I'll have to put some thought into it but am currently undeciuded if LSfit has benefit in cases when conducting oversampling of training data for class imbalance in labels via the TrainLabelFreqLevel option, it might still.

3.1 updates
- Label Smoothing with LSfit now carries through fit properties to different segments of data (e.g. labels for train, test, or validaiton sets) for consistent processing based on properteis derived from distribution of label categories in the train set
- To carry through Label Smoothing parameters from train set to other segments pass LabelSmoothing parameters as True
- User now has ability to pass single column label sets to automunge(.) and postmunge(.) such as for processing labels independent of corresponding train sets
- Note accepts single column numpy arrays or pandas dataframes but (for now) not pandas series as input, so can just convert a series to dataframe with e.g. df = pd.DataFrame(df[column])
- fixed silly bug associated with passing empty assignparam to automunge(.)

3.2 updates
- Label Smoothing now supports application to multiple one-hot encoded sets originating from the same label source column
- (such as may be a product of our family tree primitives for applying sets of feature engineering transformations which may include generations and branches)

3.3 updates
- transformation category 'dxdt' family now accepts parameter 'periods' for number of time steps for evaluation
- such as may be useful for calculating velocity / acceleration / jerk over custom time steps
- useful for cumulative data streams that may challenge current paradigms of deep learning

3.4 updates
- transformation 'dxd2' is similar to 'dxdt', but instead of comparing singular cells accross a period time step, it compares the average of the sets of cells at 1 and 2 periods preceeding current time step
- such as to smooth / denoise data
- transformation category 'dxd2' family (dxd2/d2d2/d3d2) now accepts parameter 'periods' for number of time steps for evaluation
- such as may be useful for calculating velocity / acceleration / jerk over custom time steps
- useful for cumulative data streams

3.5 updates
- new dimensionality reduction technique, bulk transform of all boolean columns via binary encoding, available by passing Binary = True
- a tradeoff between positive aspects of memory efficiency / energy efficiency / number of weights vs perhaps some small impairment for ability of model to handle outliers, as now any single configuration of boolean sets not seen in training data will trigger an infill for the entire set
- I suspect this might take longer to train, as less redundancy in training data, but again energy efficiency in inference etc.
- fixed bug for overwriting default transform categories under automation

3.6 updates
- fixed outlier bug for PCA application involving inconsistent index numbers
- fixed bug which ommitted NArw columns (infill markers) from Binary dimensionality reduction
- added more detail to Binary dimensionality reduction printouts
- removed a superfluous copy operation that was possibly costing some memory overhead

3.7 updates
- new processing functions for numerical set graining to fixed width bins
- bnwd/bnwK/bnwM for one-hot encoded bins (columns without activations in train set excluded in train and test data)
- bnwo/bnKo/bnMo for ordinal encoded bins (integers without train set activations still included in test set)
- first and last bins are unconstrained (ie -inf / inf)
- bins default to width of 1/1000/1000000 eg for bnwd/bnwK/bnwM
- bin width can also be user specified to distinct columns by passing parameter 'width' to assignparam e.g. assignparam = {'bnwM' : {'column1' : {'width' : 200}}}
- found and fixed edge case bug in MLinfill associated with columns with entire set subject to infill

3.9 updates
- new transformation function 'copy' which does simply that
- useful when applying a transformation to the same column more than once with different parameters
- accepts parameter 'suffix' for the suffix appender strng added to the column header

3.10 updates
- code efficiency audit identified a few transformation categories with room for improvement
- revised categories: text/bxcx/pwrs/pwor/pwr2/por2/bins/bint/bsor/bnwd/bnwK/bnwM/bnwo/bnKo/bnMo
- every little bit helps 

3.11 updates
- new processing function family for numerical set graining to equal population bins
- i.e. based on a set's distribution the bin width in high density portion of set will be narrower than low density portion
- bnep/bne7/bne9 for one-hot encoded bins
- bneo/bn7o/bn9o for ordinal encoded bins
- first and last bins are unconstrained (ie -inf / inf)
- bins default to width of 5/7/9 eg for bnep/bne7/bne9
- bin count can also be user specified to distinct columns by passing parameter 'bincount' to assignparam e.g. assignparam = {'bnep' : {'column1' : {'bincount' : 11}}}

3.12 updates
- new postmunge driftreport metrics assmebled for numerical set binned transformation categories.
- including one-hot encoded categories' column activation ratios:
- pwrs / pwr2 / bins / bint / bnwd / bnwK / bnwM / bnep / bne7 / bne9
- and ordinal encoded categories encoding activation ratios:
- pwor/ por2 / bnwo / bnKo / bnMo / bneo / bn7o / bn9o
- (driftreport metrics track distribution property drift between original training data prepared with automunge(.) and inference data prepared with postmunge(.) when postmunge passed with driftreport=True)

3.13 updates
- extension of driftreport methods to evaluate distribution property drift between sets
- now in addition to a set of transformation category specific metrics measured for each derived column, a generic set of metrics is also evaluated for each source column, based on the root category that was either assigned or based on an evaluation of set properties (specifically determined by the NArowtytpe entry for the root category's process_dict entry)
- as an example, for a vanillla numeric set, metrics are evaluated for: max / quantile_99 / quantile_90 / quantile_66 / median / quantile_33 / quantile_10 / quantile_01 / min / mean / std / MAD / nan_ratio
- a user can now pass driftreport to a postmunge call as any of {False, True, 'efficient', 'report_effic', 'report_full'}
  - False means no postmunge drift assessment is performed
  - True means a full assessment is performed for both the source column and derived column stats
  - 'efficient' means that a postmunge drift assessment is only performed on the source columns (less information but much more energy efficient)
  - 'report_effic' means that the efficient assessment is performed and returned with no processing of data
  - 'report_full' means that the full assessment is performed and returned with no processing of data
- the results of a postmunge driftreport assessment are returned in the postreports_dict object returned from a postmunge call, as follows
postreports_dict = \
'featureimportance':{(not shown here)},
'finalcolumns_test':[columns],
'driftreport': {(column) : {'origreturnedcolumns_list':[(columns)], 
                           'newreturnedcolumns_list':[(columns)],
                           'drift_category':(category),
                           'orignotinnew': {(column):{'orignormparam':{(stats)}},
                           'newreturnedcolumn':{(column):{'orignormparam':{(stats)},
                                                          'newnormparam':{(stats)}}}},
'sourcecolumn_drift': {'orig_driftstats': {(column) : (stats)}, 
                      'new_driftstats' : {(column) : (stats)}}}

- new printouts included in postmunge for source column stats when processing each source column
- also fixed bug in populating driftreport entries for postreports_dict
- also fixed silly bug triggered by passing df_test=False associated with a recent update


3.14 updates
- resolved edge case bug associated with postmunge one-hot encoding ('text' category) for cases where corresponding automunge column was all infill

3.15 updates
- resolved edge case bug associated with postmunge binary encoding ('bnry' and '1010' categories) for cases where corresponding automunge column was all infill. Tested full library for this edge scenario and passed.
- added entry to postreports_dict driftreport entries, 'newnotinorig', capturing returned columns that were not returned in the original set.

3.16 updates
- added a few more statistic evlauations to source column Data Distribution Drift assessment
- source column assessment for numeric sets now includes Shapiro and Skew stats such as to test for normality and tail characteristics
- source columns for root categories where e.g. 0 value subject to infill now includes an assessment of 0 value ratio in the set (similarly for nonpositive, etc)
- here is an example of source column drift assessment statistics for a positive numeric root category:
```
postreports_dict['sourcecolumn_drift']['new_driftstats'] = \
{(column) : {'max'         : (stat),
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

3.17 updates
- new methods for hyperparameter tuning in ML infill predictive algorithms via column specific grid search or randomized search of user passed sets of hyperparameters
- user can pass predictive algorithm parameters to ML_cmnd dicitonary in an automunge call either as distinct parameters to overwrite the defaults or as lists or distributions of parameters for tuning, as an example:
ML_cmnd = {'MLinfill_type':'default',
           'hyperparam_tuner':'gridCV',
           'MLinfill_cmnd':{'RandomForestClassifier':{'max_depth':range(4,6)},
                            'RandomForestRegressor':{'max_depth':[3,6,12]}}}
- hyperparameter tuning defaults to grid search for cases where user passes parameters as lists or ranges
- to perform randomized search user can pass via ML_cmnd the entry {'hyperparam_tuner':'randomCV'}, in which default number of iterations is 10, user can also assign number of iterations via ML_cmnd such as via {'hyperparam_tuner':'randomCV', 'randomCV_n_iter':20}
- in context of a randomized search for hyperparameter tuning user can also pass parameters as distributions via scipy stats module
- (also fixed bug for predictive model initializer associated with passing parameters)

3.18 updates
- found and fixed efficiency issue for ML infill application in automunge(.) and postmunge(.) originating from infilliterate methods
- ML infill runs faster now

3.19 updates
- Added support for passing single entry column headers to assigncat and assigninfill without list brackets.

3.20 updates
- built out transformation categories for numerical set transformations
- new transformation categories addd/sbtr/mltp/divd/rais/absl
- (for addition, subtraction, multiplication, division, raise to power, absolute
- accept respective parameters add/subtract/multiply/divide/raiser/(na)

- also built out transformation categories for numerical set bins and grainings
- new transformation categories bkt1/bkt2/bkt3/bkt4
- each accept parameter of 'buckets' for list of boundaries of user defined bins
- bkt1/bkt3 have first and last bin unbounded, bkt2/bkt4 bounded
- bkt1/bkt2 create new columns with binned activations
- bkt3/bkt4 are ordinal encoded activations


3.21 updates
- new postmunge parameter returnedsets 
- returnedsets can be passed as {True, False, 'test_ID', 'test_labels', 'test_ID_labels'}
- defaults to True for default returtned set composition
- other options only return single test set from a postmunge call
- where False is just returned as the df_test, and other options append ID, labels, or both sets onto the returned df_test
- removed extranious copy operation in postmunge for more efficient operation

3.22 updates
- new transform categories nmbd/101d/ordd/texd/bnrd/datd/nuld
- serving as alternate to default transformation categories nmbr/1010/ord3/text/bnry/dat6/null
- so user can assign original transforms through assigncat in cases where default trasnforms for automation were overwritten to another category
- a few small code comment cleanups

3.23 updates
- fixed printouts for infilliterate so as not to print status unless user activates infilliterate parameter
- new transform category 'shfl' to shuffle a column based on the passed randomseed, defaults to adjacent cell infill, non-numeric entries allowed

3.24 updates
- changed default for NArowmarker parameter from True to False
- new transform category 'bnr2', comparable to 'bnry' for single column binary encoding, but default infill is least common value instead of most common value
- fixed family tree for exc2 & exc3
- new transform category 'exc4', offers pass-through force to numeric in addition to powers of ten bins, such as to support oversampling preparation of numeric sets via powers of ten bins
- found and fixed bug for trainfreqlevelizer for numerical set levelizing based on binses

3.25 updates
- removed extraneous copy operation in automunge(.) function for improved efficiency
- new transform category bxc5 comparable to first multitier output demonstration from Automunge Explained video
- fixed postprocess processdict entries for exc3 and exc4 
- fixed labelctgy processdict entries for bxc2, bxc3, bxc4

3.26 updates
- new transform categories: lngt, lnlg
- lngt calculates string length of categorical entries, followed by a min/max scaling
- lnlg calculates string length of categorical entries, followed by a log transform
- think of this as a rough heuristic for information content of a categorical string entry

3.27 updates
- revisited family of transforms for sequential data (eg time series or cumulative data streams)
- created new normalization method "retain" as 'retn', somewhat similar to a min/max scaling but retains the +/- values of original set, such that if max>0 and min<0, x = x/(max-min), else just defaults to traditional min/max scaling x=(x-min)/(max-min)
- incorproated 'retn' normalizaiton into all outputs of previously configured sequential trasnforms 'dxdt', 'd2dt', 'd3dt', 'dxd2', 'd2d2', 'd3d2', 'nmdx', 'nmd2', 'nmd3'
- (as a reminder dxdt/d2dt/d3dt are approximations of velocity/acceleration/jerk based on deltas for a user designated time step, dxd2/d2d2/d3d2 are comparable but for smoothed values based on an averaged range of user designated time step, nmdx/nmd2/nmd3 are comparable to dxdt but performed downstream of a string parsing for numeric entries)
- repurposed the transforms mmdx/mmd2/mmd3 to comparable to dxdt set but with use of z-score normalziation instead of retn (comparable to video demonstration)
- also created new varient of dxdt trasnforms as dddt/ddd2/ddd3 and dedt/ded2/ded3 where the dddt set are comparable to dxdt but with no normalizations performed, and the dedt set are comparable to dxd2 set with no normalziations performed
- also new parameters allowed for mnm3 transform (which is a min-max scaling with cap and floor based on quantiles), user can now pass qmin and qmax through assignparam to designate the quantile values for a given column
- also a few new driftreport metrics for the lngt transform rolled out in 3.26

3.28 updates
- revised date-time processing functions mdsn and mdcs for tailored monthly periodicity based on number of days in the month instead of average days in a month (including account for leap year)
- updated default label column processing category for numeric sets to exc2 (to match documentation)
- updated default of shuffletrain parameter from False to True, meaning automunge(.) returned train sets are now by default returned shuffled, and validation sets are randomly selected (taking this step was inspired by review of a few papers that had noted shuffling tabular training data was generally beneficial to training, including "fastai: A Layered API for Deep Learning" by Howard and Gugger and I believe I also saw this point in "Entity Embeddings of Categorical Variables" by Guo and Berkhahn). Note that returned "test" sets are not shuffled with this parameter.
- Family of sequential data transformations were all extended from three tier options to six tiers (e.g. dxdt/d2dt/d3dt now have comparable extensions to d4dt/d5dt/d6dt, and comparable extensions for series dxd2/nmdx/mmdx/dddt/dedt)
- Corrected auntsuncles primitive entry for mmdx transform_dict from mmmx to nbr2 to be consistent with other downstream normalizations
- added a few clarifications to READ ME documentation
- slight tweek to printouts for processing label columns

3.29 update
- a code quality audit revealed that the "process_dict" data structure for "labelctgy" entry was poorly documented and used kind of awkwardly in two ways - one as a category identifier and one as a suffix identifier, which obviously doesn't scale well
- so revised the use of process_dict entry for labelctgy to only use as a category identifier, via updates to feature selection methods and label frequency levelizer for oversampling methods 
- updated READ ME for improved documentation of this item
- also decided to update another case where we were relying on column suffix appenders for transformations, which was in the label frequency levelizer, updated methods for better universality, in the process now have support accross all categories offered in libary for numerical set one-hot encoded bins
- in the process realized that many of the transformation categories processing functions had this empty relic from earlier methods which inspected column strings, scrubbed to avoid confusion
- and well went this far and just occured to me there's really no need to base any methods at all on the column header strings, as turned out had a few more straggler instances from earlier methods. So did a full audit for cases of basing methods on column header string suffix appenders and replaced all instances with methods making use of postprocess_dict data store.
- collectively these improvements are expected to make the tool much more robust to edge case bug scenarios such as from passed column headers with coincident strings.
- found and fixed edge case bug for bxcx transform (design philosophy is that every transform supports any kind of passed data)
- found and fixed two edge case bugs for bkt4 (ordinal encoding of custom buckets)
- found and fixed edge case bug for drift property collection when nunique <3
- updated feature selection methods to default to shuffling training data for the evaluation model training
- a few small cleanups for formatting to printouts
- improved documentation for powertransform options in default transforms section of read me

3.30 updates:
- added 'traintest' option for automunge(.) shuffletrain parameter to shuffle both train and test sets
- (previously shuffletrain only shuffled training data, occured to me there is a workflow scenario in which the "test" set is applied for training as well, so now have option to shuffle both train and test sets within automunge(.))
- added shuffletrain parameter to postmunge(.) to allow shuffling returned test set 
- (for alternate workflow scenario where the postmunge "test" data is used in a training operation)

3.31 updates
- added new derived column to the returned ID sets, automatic population with a column titled 'Automunge_index_#' which contains integers corresponding to the rows in the original order that they were presented in the corresponding train or test sets
- (where # is a 12 digit number associated with the specific automunge(.) call, kind of a serial stamp associated with each function call)
- this 'Automunge_index_#' column is included in ID sets returned for train, validaiton, and test data from automunge(.) as well as the test data returned from postmunge(.)
- (note that dataframe index columns or any other desired columns may also be designated for segregation and returned in the ID sets, consistently partitioned and shuffled but otherwise unaltered, by passing column header strings to automunge(.) parameters trainID_column and testID_column)
- (note also that any non-range index columns passed in a dataframe are already automatically placed into the ID sets)
- oh and found and fixed this edge case bug for feature selection methods causing under-represented columns to be dropped (basically for very small datasets the feature importance evaluation sometimes may only see a subset of values in a categorical column, causing analysis to not return results for those category entries not seen)
- changed default feature importance evaluation validation set ratio from 0.33 to 0.2 to lower the risk of this edge case

3.32 updates
- new process_dict MLinfilltype entry option of 'binary' to distinguish between single column categorical ordinal entries (singlct) and single column categorical boolean entries (binary)
- updated methods that test if a column is boolean by basing on the process_dict entry for MLinfilltype instead of inspecting unique values in column (for improved operating efficiency)
- (except retained label smoothing functions boolean column testing in original method in case user wants to call these methods outside of an automunge(.) call)
- corrected process_dict MLinfilltype entries for bkt1, bkt2 to singlct
- fixed PCA bug associated with checking for excl categories
- removed normalize_components parameter from scikit SparsePCA application option as will be depreciated
- fixed postmunge feature importance bug associated with PCA

3.33 updates
- added support for "modeinfill" (infill with most common value) to '1010' binary encoded sets
- reintroduced "modeinfill" for one-hot encoded sets
- improved implementation of modeinfill (moved all scenarios into single function for simplicity and clarity)

3.34 updates
- found and fixed edge case bug for assigned infill options meaninfill / medianinfill / modeinfill associated with cases where an entire set is subject to infill
- found that infill assignments were in some cases reseting the dtypes of sets returned from transform functions (eg in some cases changing a column from integers to floats). Updated infill functions to ensure returned data types are consistent with those types returned from transform functions

3.35 updates
- added support for hyperparameter tuning of predictive models to feature importance evaluation in both automunge(.) and postmunge(.) (previously was only available for ML infill)

3.36 updates
- new infill type 'lcinfill' available for assignment in assigninfill, comparable to modeinfill but applies least common value of a set instead of most common value
- new validation check run on passed assigncat dictionary to ensure passed keys have corresponding entries of family tree assignments available in process_dict
- fixed issue with ML infill postmunge application (turned out to be originating from edge case where one of columns had an entire set subject to infill)
- fixed second issue with ML infill associated with reseting a marker after infill
- added note to READ ME about infilliterate
- updates to feature importance associated with edge cases when model not trained
- updated process_dict MLinfilltype entries for exc2, exc3, exc4 from 'label' to 'numeric' ('label' MLinfilltype for now discontinued)
- removed convention of dropping rows corresponding to label column NaN, new convention is label columns are subject to default infill associated with transformation category

3.37 updates
- fixed bug for scenario of dataframes passed with non-range index introduced in 3.31

3.38 updates
- new datetime binned aggregations wkds, wkdo, mnts, mnto
- wkds and mnts are one-hot encoded days of week and months of year
- wkdo and mnto are comparable ordinal encoded sets
- new start/end parameters accepted for bshr (boolean marker for whether a time falls within business hours, which default to 9-5)
- new cap or floor parameters accepted for mnmx (min-max scaling)
- where default of False means no cap/floor, True means cap/floor set based on training data max/min, otherwise passed values put limit to the scaling range
- noting that if cap < max then max reset to cap, and if floor > min then min reset to floor
- please note that the wkds transform was inspired by a comment in a prerelease chapter for the "Deep Learning for Coders with fastai and PyTorch" book being written by Jeremy Howard and Sylvian Gugger
- removed amateurish relic from an early draft associated with global fillna() application

3.39 updates
- found and fixed small code typo in nmr7 and nmc7 transforms
- removed a redundant shuffle operation from postmunge feature importance derivation

3.40
- Was working on some new passed parameter validations and realized had gotten mixed up in a prior validation between transformdict and processdict. No biggie, found and fixed.
- to be clear, transformdict contains the family tree entries for each root category, processdict contains the corresonding transformation functions and column properties associated with each category used as an entry in a family tree
- added a few images to the READ ME

3.41
- Expanded range of parameter validations with new validation functions check_am_miscparameters and check_pm_miscparameters
- Now passed parameters with fixed set or fixed range of values are all validated to ensure legal entries
- Note that parameter validation results for automunge(.) are returned as postprocess_dict['miscparameters_results'] and the results for postmunge(.) are returned as postreports_dict['pm_miscparameters_results']
- A few clarifications added to parameter descriptions in READ ME

3.42
- Added some new validations to identify edge cases where returned columns with suffix appenders overlap with original passed column headers (which may result in an unintentional column overwrite in some cases)
- Results of validation returned in postprocess_dict['miscparameters_results']['columnoverlap_valresults']
- (False is good)
- Note that user is not expected to encounter this scenario in common practice, just trying to ensure robust to edge case
- Added some clarification to READ ME associated with passing Numpy Arrays - specifically that recomended practice is for any passed label column in train set be positioned as the final column of the set to ensure consistent column headers for test set (since test set may be passed without label column)
- Added option to automunge(.) parameter labels_column to pass as True in order to indicate that final column of the set is intended as labels_column (previously was only available for single column sets)

3.43
- Went for a walk and realized the 3.42 validations were incomplete, needed to move to before the application of circleoflife function in order to include in validation those columns which are subsequently deleted as part of a replacement operation. Complete.

3.44
- New parameters accepted for 'nmbr' (z-score normalization function):
- 'cap' and 'floor' for setting cap and lfoor to pre-transform values, note thsi applied prior to default mean infill derivation, not can also be passed as True for setting based on min/max found in train set, or can be passed as False for off
- 'multiplier' and 'offset' for applying mnultiplier and offset to posttransform values, note multiplier is applied prior to offset, these default to 1 and 0
- New parameters accepted for 'retn' (retained sign normalization function):
- same as those shown above (cap/floor/multipliuer/offset)
- an additional retn parameter of 'divisor' to select between 'minmax' for divided by max-min or 'std' to divide by standard deviation

3.45
- Rolled out a new section of READ ME containing concise sorted list of all suffix appenders so user passing custom transform function can confirm no overlap.
- A little housekeeping, found that suffix appenders for bnep/bne7/bne9 had an extra underscore so a few small tweeks to match conventions of other transforms.
- New 'holiday_list' parameter accepted for hldy trasnform as list of dates in string enocdings (eg ['2020/03/30']) 
- (hldy is boolean indicator if a date is a US Federal Holiday)
- 'holiday_list' parameter allows user to add additional dates to list of recognized holidays.

3.46
- new feature importance report variation and printouts associated with columns sorted by metrics 'metric' and 'metric2'
- available for automunge(.) in postprocess_dict['FS_sorted'] and for postmunge(.) in postreports_dict['FS_sorted']
- added an 'origcolumn' entry to feature importance report
- added support to hldy transform for passed data with timestamp included

3.47
- corrected parameter validations for featuremethod parameter for legal entries in range 0-100
- updated feature importance dimensionality reduction methods to only retain NArw columns that correspond to those columns that remain in the set
- label smoothing is now available for one-hot encoded label sets in which some rows may be missing an activation
- new options for automunge(.) parameter TrainLabelFreqLevel 'test' and 'traintest' for oversampling preparation in test set in addition to train set (for scenario where the set designated as 'df_test' may also include labels and be intended for a training operation)

3.48
- corrected the default setting for retn transform multiplier parameter from False to 1
- updated methods for retn transform, now if all values in set are <1, scales data between -1 and 0 
- (all positive sets are given minmax scaling between 0 and 1, all negative sets are given maxmin scaling between -1 and 0, and mixed sets are given retn scaling between -1 and 1 (at min/max decimal points based on set's min/max)
- corrected MLinfilltype process_dict entries for wkdy, bshr, hldy from singlct to binary
- removed the postmunge(.) convention to drop rows corresponding to label column nan to match recent update to automunge(.)
- found and replaced two straggler methods that had relied on evaluating header string composition
- moved postmunge(.) initialization of empty label set a little earlier to fix just found bug for shuffletrain parameter for cases without passed labels

3.49
- new automunge(.) parameters defaultcategoric, defaultnumeric, defaultdatetime
- to simplify overwriting default transform categories for categoric, numeric, and datetime data under automation
- these default to defaultcategoric = '1010', defaultnumeric = 'nmbr', defaultdatetime = 'dat6'
- for example to change default categorical encoding from binary encoding to one-hot, pass defaultcategoric = 'text'
- or to change default numeric normalization from z-score to mean scaling, pass defaultnumeric = 'mean'
- note that family trees for default transformations can alternatively be overwritten with a passed trasnsformdict
- updated READ ME description of evalcat parameter for new evalcategory function arguments in support of defaultcategoric, defaultnumeric, defaultdatetime
- 'mean' normalization transform now accepts comparable parameters to 'nmbr', including cap/floor/multiplier/offset
- corrected description of mean scaling derviation for 'mean' transform in READ ME

3.50
- after rolling out 3.49, pulled up The Zen of Python
- realized that was comitting a cardinal error with the new default category parameters
- "There should be one-- and preferably only one --obvious way to do it."
- thus 3.50 removes the parameters defaultcategoric, defaultnumeric, defaultdatetime that had just rolled out in 3.49
- any updates to default categories are intended to be accomplished using existing methods by passed transformdict
- feeling much more zen about everything now

3.51
- rounding out the set of driftreport metrics collected for a few categories of transformations
- dxdt/dxd2 now collect metrics positiveratio / negativeratio / zeroratio / minimum / maximum / mean / std
- bshr / wkdy / hldy now collect metrics for activationratio
- wkds now collects metrics for mon_ratio / tue_ratio / wed_ratio / thr_ratio / fri_ratio / sat_ratio / sun_ratio / infill_ratio
- mnts now collects metrics for infill_ratio / jan_ratio / feb_ratio / mar_ratio / apr_ratio / may_ratio / jun_ratio / jul_ratio / aug_ratio / sep_ratio / oct_ratio / nov_ratio / dec_ratio

3.52
- found and fixed small bug for source column drift stats assembly associated with 'exclude' MLinfilltype
- added entries {'excl_columns_with_suffix':[], 'excl_columns_without_suffix':[]} to postprocess_dict to help identify and distinguish column header entries for 'excl' transforms, a special case pass-through transform which is returned to user without suffix appender 
- (internally the '_excl' suffix is first applied and then removed, such that postprocess_dict['column_dict'] entries for these columns include the suffix and returned column headers don't, so there might be a scenario where these new entries could be useful to a user after completion of an automunge(.) call such as when comparing returned column headers to postprocess_dict['column_dict'] entries

3.53
- fixed postmunge(.) bug corresponding to application of feature importance dimensionality reduction in automunge(.) 
- new NArowtype 'integer' available for assignment in process_dict, for sets where non-integer values are subject to infill
- new transform category 'exc5', a pass-through transform that forces values to numeric and subjects non-integer entries to mode infill as default infill
- added automunge(.) parameter excl_suffix, defaults to False, can also be passed as True, to select whether '_excl' suffix appenders will be included on pass-through column headers
- added support for inclusion of 'excl' pass-through categories in feature importance evaluation
- (feature importance labels still require minimum of exc2 for target regression or exc5 for target classification)
- changed the order of sort for feature importance metric2 results to match basis of interpretation
- performed an audit of postprocess functions for column key retrieval in transform functions with multi-column returned sets
- found potential for error in multi-generation family trees associated with accessing incorrect normalization_dict
- resolved by incorporating two new methods:
- 1) ensure unique normalization_dict entry identifier between categories for cases where entry used in column key retrieval
- 2) new entry in transform functions' returned column_dict of 'inputcolumn' identifying the column header string that was passed to function (this is now checked in some of the column key retrieval methods for transforms with multicolumn returned sets)
- 'inputcolumn' differs from 'origcolumn' in that 'origcolumn' is subsequently updated to basis of original source column, while 'inputcolumn' retains column entry as was passed to the transform function
- replaced an earlier method for column key retrieval in 'text' transform postprocess function to one potentially more efficient

3.54
- found an issue in feature importance evaluation with sorted results reported in FS_sorted  orginating from entry overwrite for cases where multiple columns achieved the same feature importance score
- resolved by collecting FS_sorted bottom tier of results in a list
- (a small amount of complexity in the implementation, nothing couldn't handle)
- added support for NArw entries to be included in feature importance evaluation 
- (the convention is that if all other columns from same source column are trimmed in a feature importance dimensionality reduction, the corresponding NArw column will be trimmed independent of score)
- new transform root category NAr5 which produces NArw activations corresponding to non-integer entries
- corrected family tree entries for NAr2/NAr3/NAr4
- remade 'excl' transform, a direct pass-through function, to only rename a column instead of copying the column
- this update is expected to improve the efficiency of operation at scale when there are columns excluded from processing 
- (which in some workflow scenarios may be a majority of columns)
- as a result a restriction is placed on 'excl' in that it may only be used as a supplement primitive in a root category's family tree (eg siblings, cousins, niecesnephews, friends)
- (a function checks for cases of excl passed as replacement primitive and if found moves to corresponding supplement primitive, this is all "under the hood")
- new validation function to ensure normalization_dict entries which require unique identifier for purposes of deriving a column key to retrieve parameters don't have overlap from user passed transformation functions (as discussed in 3.53 rollout)
- slight reorg of automunge(.) call default parameter layout in READ ME for improved readability
- slight reorg of assigncat default layout in READ ME for improved readability
- slight reorg of postmunge(.) call default parameter layout in READ ME for improved readability

3.55
- revised order of operations in processfamily / processparent functions to apply replacement primitive entries before supplement primitive entries (to accomodate an inplace operation performed to a supplement operator such as 3.54 update to excl category)
- processfamily function now applies primitive entries in order of: parents, auntsuncles, siblings, cousins
- processparent function now applies primitive entries in order of: upstream, children, coworkers, niecesnephews, friends
- improved the implementation of validation function to evaluate user passed transformdict automunge(.) parameter
- new transform category 'exc6', comparable to 'excl' (direct pass-through) but relies on a copy operation instead of an inplace operation
- added clarification to read me that 'excl' should only be applied in a user-passed transformdict as an entry to the cousins primitive, and if comparable functionality is desired for any other primitive the 'exc6' transform should be applied in it's place
- note that the transformdict validation function now checks for this eventuality
- updated convention for feature importance dimensionality reduction treatment of NArw entries to basis that NArw columns are treated just like others, including for purposes of feature importance dimensionality reduction
- trying to avoid accumulation of too many edge cases, realized there are scenarios where an NArw derivation from a column may be just as valid an information source for ML training as other columns
- NArw is still excluded from ML infill (not applicatble) and is inelligible as a target label for feature importance evaluation
- fixed bug for postmunge(.) feature importance evaluation corresponding to cases when a feature importance dimensionality reduction was performed in automunge(.)
- slight cleanup tweek for populating feature importance results in FS_sorted report

3.56
- new transform category 'ptfm', similar to 'eval'
- the intent is that the (modular) data property evaluation function for automation may have a few distinct default bases activated by the powertransform parameter:
- a regular default evaluation when powertransform = False, and a seperate (potentially more elaborate) default evaluation basis when powertransform = True
- or the evaluation may be turned off such as by passing powertransform = 'excl' (which means unassigned columns are left unaltered)
- the purpose of the 'eval' and 'ptfm' options are for cases where the automated evaluation is set to one of these bases (regular, powertransform, off), a user can still assign to distinct columns one of the other bases for evaluation
- added support for 'eval' and 'ptfm' options to label columns
- a little bit of code cleanup in automunge(.) function

3.57
- found and fixed silly bug for ML infill
- (had fixed this while back and just realized that update got set on hold before rolling out)
- found and fixed bug for postmunge(.) feature importance corresponding to application of Binary parameter activation in automunge(.)

3.58
- found and fixed small edge case for automunge(.) powertransform evaluation function under automation
- found and fixed small typo for category identifiers logged for bnwK, bnwM, bnKo, bnMo
- found and fixed small bug for driftreport assembly in postmunge(.)

3.59
- improved memory overhead of ML infill by consolidating seperately named sets in a few data conversions between pandas and numpy
- improved memory overhead of feature importance evaluation in similar fashion
- improved memory overhead of PCA dimensionality reduction in similar fashion

3.60
- ML infill now trains models even if automunge(.) passed sets have no infill points 
- for potential use in postmunge(.)
- (had meant to roll this out a while ago)

3.61
- assigninfill now supports passing column headers of dervied columns in addition to source columns
- thus enabling user to assign distinct infill methods for different columns dervied from the same source column
- (assigning source column headers are applied to all downstream columns in aggregate, and then passing a derived column header will override)
- replaced a function call associated with infill application in the postmunge(.) function for improved efficiency

3.62
- fixed bug in postmunge(.) originating from 3.61 update
- fixed material typo in READ ME associated with documenting one of data structures for 'singlct' MLinfilltype

3.63
- new validation check to detect presence of infinite loops in transformdict entries
- in other words, solved the halting problem

3.64
- new processing function "tail bins" as tlbn
- intended for use to evaluate feature importance of different segments of a numerical set's distribution
- returns equal population bins in seperate columns with activations replaced by min-max scaled values within that segment's range (between 0-1) and other values subject to an infill of -1 
- (where the -1 infill intended as a register to signal to ML training that infill applied)
- note that the bottom bin has order reversed to maintain consistent -1 register and support subsequent values out of range
- accepts parameter 'bincount' as an integer to specify number of bins returned
- when run through feature importance, the metric 'metric2', such as printed and returned in postprocess_dict['FS_sorted'], can give an indication of relative importance of different segments of the distribution
- this may be useful to evaluate influence of tail events for instance
- also corrected parameter initializations for bnep/bne7/bne9/bneo/bn7o/bn9o

3.65
- new parameter accepted for splt family transforms splt/spl2/spl5/spl7/spl8/spl9/sp10
- (splt transforms are the string parsing functions which identify character overlaps between unique entries in a categorical set)
- 'space_and_punctuation' parameter can be passed as True/False, defaults to True
- when passed as False, character overlaps are only recorded when excluding space and punctuation characters in their composition
- based on the space and punctuation characters [' ', ',', '.', '?', '!', '(', ')']
- as an example, when using spl9 function to evaluate a set with unique entries {'North Florida', Central Florida', 'South Florida', 'The Keys'}, if this parameter set as default of True, the returned set would have unique entries {'th Florida', 'Central Florida', 'The Keys'}
- if this parameter passed as False, the returned set would instead have unique entries {'Florida', 'The Keys'}

3.66
- extended the methods rolled out in 3.65, with additional parameter accepted for splt family transforms splt/spl2/spl5/spl7/spl8/spl9/sp10
- (splt transforms are the string parsing functions which identify character overlaps between unique entries in a categorical set)
- 'excluded_characters' parameter can be passed as a list of strings, defaults to `[' ', ',', '.', '?', '!', '(', ')']`
- these are the strings that are excluded from evaluation of character overlaps when 'space_and_punctuation' parameter passed as False
- thus a user can designate custom sets of characters which they wish to exclude from overlap identifications
- note that entries in this list may also include multi-character strings

3.67
- new processing function srch
- accepts parameter 'search' as a list of search strings
- parses categorical set entry strings to find character subset overlaps with search strings
- when overlaps identified returns new columns with boolean activations for identified overlaps
- new processing function src2
- comparable to srch, but assumes unique values in test set is same or subset of train set for more efficient operation
- fixed bug with postmunge(.) infill application originating from 3.61 (a real head scratcher)
- fixed edge case bug for postmunge ML infill
- corrected process_dict entry for splt transform

3.68
- added ML_cmnd support for 'n_jobs' parameter for scikit Random Forest and PCA training
- (n_jobs allows user to parallelize training accross concurent processor threads)
- added ML_cmnd support for 'criterion' parameter for scikit Random Forest training
- (the exclusion of this parameter turned out to have been result of a silly mixup)
- added new section for concise sorted list of root categories to READ ME

3.69
- fixed edge case unintentional overwrite for data object returned in postprocess_dict['postprocess_assigninfill_dict']
- now able to use this object to replace a postmunge function call for (slightly) more efficent infill operation
- scrubbed reference to the superseded 'label' MLinfilltype
- removed an unused parameter from the assembletransformdict function to avoid temptation of repurposing

3.70
- some cleanup of infill methods
- moved some steps associated with MLinfill parameter into a support function for clarity
- renamed an infill item for consistency
- replaced an infill methods test for 'NArw' category to a test of MLinfilltype for better generality
- removed an unused 'if' indentation in postmunge(.)
- removed an unused function (superseded)
- cleaned up code comments in several places

3.71
- As a relic from early implementations, there were a few cases where NArw category columns were given special treatment. This update replaces a few of these edge cases to make more generalizable treatment of NArw columns as consistent with other categories.
- 1) updated methods for convention that label sets are returned without NArw columns. Previously we derived the NArw column for labels if NArw_marker was activated and then deleted. New way is to have dedicated root category family trees for labels under automation which don't have NArw in their trees. This way a user still has option to derive an NArw column for label sets if desired by redefining the label trees in the transformdict parameter.
- 2) in the population of data structures for postprocess_dict['origcolumn'], we had given special treatment to using NArw column as an entry to ['columnkey']. This no longer required, now NArw columns treated just like any other.
- 3) in feature importance dimensionality reduction, we previously had given special treatment to NArw columns, this convention is now scrubbed.
- also corrected process_dict entry for ord3, ordd
- also removed an uneccesary logic test in postmunge infill application associated with infilliterate
- also removed a few unneccesary logic tests in automunge and postmunge infill application
- Updated convention for various infill options to add support for categories with MLinfilltype 'exclude', so although 'exclude' not included in MLinfill, it is elligible for other infill types. The only MLinfilltype exlcuded from all infill is 'boolexclude', the one used for NArw.

3.72
- found and fixed small bug in feature importance evaluation associated with numeric label sets.
- added convention that label processing under automation does not consider the powertransform parameter.
- (user can still apply powertransform option by passing label column to ptfm in assigncat)

3.73
- performed a walkthrough of various uses of the process_dict entry for 'labelctgy'
- found there was a redundant step performed in shuffleaccuracy function
- removed the redundancy, also cleaned up a few code comment relics

3.74
- Found a small piece that I had missed in the 3.71 scrubbing to generalize NArw, resolved

3.75
- A few small revisions to feature importance evaluations in automunge(.) and postmunge(.)
- To allow continuation of processing even when feature importance halted due to edge cases (like empty label sets)

3.76
- performed a walkthough of the various methods based on inspection of MLinfilltype
- found a redundant inspection of MLinfilltype exterior and interior to the levelizer function calls
- consolidated to only inspect internal to function for clarity
- also cleaned up a few code comment relics in the levelizer function

3.77
- a tweak to convention for user specified infill with respect to interplay between MLinfill parameter and specified columns in assigninfill['stdrdinfill']
- now when MLinfill passed as True, user can designate distinct source columns or derived columns to default infills by passing to 'stdrdinfill'
- (comparable functionality was available previously by making use of assigninfill['MLinfill'], this way seemed a little more intuitive and inline with the other infill types)
- a small cleanup in imports

3.78
- moved infill application into a support function for improved modularization / generality
- found an errant data structure relic convention in postmunge(.) that was no longer needed
- so removed all instances of preFSpostprocess_dict and replaced with postprocess_dict
- (the original inclusion was kind of a hack for edge case bug that is now resolved in much much cleaner fashion)
- performed an audit of shuffle applications
- found an inconsistency between automunge(.) and postmunge(.) that was impacting validation checks
- (automunge was shuffling prior to validation split and then again after levelizer, while postmunge was only shuffling after levelizer)
- so revised automunge(.) validation split to perform any random seeding as part of scikit train_test_split instead of with seperate shuffle operation
- so now automunge(.) application of shuffletrain parameter is fully consistent with postmunge(.)
- which makes validation a little easier since now have consistent order of rows between automunge and postmunge returned sets when shuffletrain applied in conjunction with levelizer
- also removed a redundant shuffle operation for test data in automunge(.) at levelizer application
- implemented new expanded set of parameter validations for each update rollout going forward

3.79
- remade function assemblepostprocess_assigninfill
- which serves purposes of converting user passed entries from assigninfill to final assignments to apply infill
- (conversion required because user has option to pass both source column headers and derived column headers, plus any updates based on MLinfill parameter)
- resulting function is much cleaner than prior, 100 fewer lines of code, and I think much more transparent of what is being accomplished

3.80
- added new convention that np.inf values are recognized as NaN for infill by default
- by making use of use_inf_as_na setting in Pandas
- impacts the getNArows function used in automunge(.) and postmunge(.)
- as well as any isna calls in transformation funcitons

3.81
- corrected the conversion from np.inf to np.nan from 3.80
- so to be clear, by default Automunge does not recognize np.inf values, they are treated as np.nan for purposes of infill
- added new 'retain' option for Binary parameter
- which can now be passed as one of {True, False, 'retain'}
- as prior, False does no conversion, True collectively applies a Binary transform to all boolean encoded sets as a replacement (such as for improved memory bandwidth)
- in the new 'retain' option, the returned collective Binary encoding acts as a supplement instead of a replacement to the columns serving as basis (such as a means of presenting boolean sets collectively in alternate configuration)
- I suspect this may prove a very useful option
- found and fixed edge case for spl9 and sp10 transforms preivously missed in testing
- associated with string conversion of numerical entries to test data
- performed a walkthrough of postmunge(.) labelscolumn parameter
- found a code snippet that had been carried over from automunge(.) that was inconsistent with documentation, now struck
- moved the postmunge(.) initialization of empty label sets a little earlier for clarity
- added a marker to returned dictionary noting cases when df_train is a single column designated as labels, just in case that might come in handy
- new transformation category ucct
- in same neighborhood as ord3 which is an ordinal integer categorical encoding sorted by frequency
- ucct counts in train set the unique class count for each categopry class and returns that count divided by total row count in place of the category
- e.g. for a train set with 10 rows, if we have two cases of category "circle", those instances would be returned with the value 0.2
- and then test set conversion would be same value independant of test set row count
- ucct inspired by review of the ICLR paper "Weakly Supervised Clustering by Exploiting Unique Class Count" by Mustafa Umit Oner, Hwee Kuan Lee, Wing-Kin Sung
- additional new transform category Ucct, performs an uppercase character conversion prior to encoding (e.g. the strings usa, Usa, and USA treated consistently)
- followed by a downstream pair of offspring ucct and ord3

3.82
- added new validation for user-passed transformdict
- checking for redundant entries accross upstream or downstream primitives
- corrected printouts in a validation function for user-passed assigncat

3.83
- realized there was potential bug when passing 0/1 integer column identifiers
- from overlap with option to pass values as boolean
- (such as for parameters labels_column, trainID_column, testID_column)
- easy fix, just performed global conversion from {== True : is True, == False : is False, != True : is not True, != False : is not False}

3.84
- cleanup of string parsing for numeric entry support functions
- fixed edge case from casting partitioned string "inf" as float

3.85
- remade the search function 'srch'
- now expected more efficient for unbounded sets
- by making use of a pandas.Series.str.contains method
- srch accepts parameters of 'search' as a list of search terms
- and 'case', a boolean signal for case sensitivity of search
- returns a new column for each search term containing activations corresponding to search terms identified as substring portions of categorical entries
- also updated the application of floatprecision data type adjustments for consistency between train set and labels
- (any data type conversions from floatprecision take place after processing functions and then again after infill)
- also a few small code comment cleanups

3.86
- new transform logn, comparable to log0 but natural log instead of base 10
- new normalization lgnm intended for lognormal distributed sets
- where lgnm is achieved by a logn performed upstream of a nmbr normalization
- reintroduced the original srch configuration as src3
- where expectation is srch is preferred for unbounded range of unique values
- including for scaling with increasing number of search terms
- src2 preferred when have bounded range of unique values for both train & test
- and (speculating) that src3 may be beneficial when have a bounded range of unique values and high number of search terms but still want capacity to handle values in test set not found in train set
- (leaving validation of this point for future inquiry)

3.87
- replaced all instances of scikit train_test_split and shuffle functions with streamlined pandas methods
- trying to reduce the number of imports for simplicity, pandas is prefered for consistency of infrastrucure

3.88
- found a few small efficiency improvement opportunities in bnry and bnr2 transforms
- replaced pandas isin calls with np.where
- (this item was a relic from some early experiments)

3.89
- new postmunge parameter inversion
- to recover formatting of source columns
- such as for example to invert predictions to original formatting of labels
- this method is intended as an improvement upon the labelsencoding_dict returned label set normalization dictionaries which were intended for this purpose
- note that for cases where columns were originally returned in multiple configurations
- inversion selects a path of inversion transformations based on heuristic of shortest depth
- giving priority to those paths of full information retention
- method supported by new optional processdict entries inverseprocess and info_retention
- inversion transformations now available for transformation categories: nmbr, nbr2, nbr3, mean, mea2, mea3, MADn, MAD2, MAD3, mnmx, mnm2, retn, text, txt2, ordl, ord2, ord3, bnry, bnr2, 1010, pwrs, pwr2, pwor, por2, bins, bint, boor, bnwd, bnwK, bnwM, bnwo, bnKo, bnMo, bene, bne7, bne9, bneo, bn7o, bn9o, log0, log1, logn, lgnm, sqrt, addd, sbtr, mltp, divd, rais, absl, bkt1, bkt2, bkt3, bkt4
- inversion can be passed to postmunge as one of {False, 'test', 'labels'}, where default is False
- where 'test' accepts a test set consistent in form to the train set returned from automunge(.)
- and 'labels' accepts a label set consistent in form to the label set returned from automunge(.)
- note that inversion operation also inspects parameters LabelSmoothing, pandasoutput and printstatus
- note that inversion operation is pending support for train sets upon which dimensionality reduction techniques were performed (such as PCA, feature importance, or Binary). 
- recovery from smoothed labels is supported.
- note that categorical encodings (bnry, 1010, text, etc) had a few diferent conventions of plug values for various scenarios requiring them, now standardizing on the arbitrary string 'zzzinfill' as a plug value in categorical set encodings for cases where required for infill.
- so to be clear, 'zzzinfill' is now a reserved special character set for categorical encodings
- (this item becomes more customer facing with inversion)

3.90
- corrected typo in datatype assignment for inversion from smoothed labels (corrected "int8" to "np.int8")
- small cleanup for floatprecision methods to only inspect for 64bit precision in single location for clarity
- populated a new data structure in postprocess_dict as postprocess_dict['inputcolumn_dict']
- don't have a particular use for this right now, but thought it might be of potential benefit for downstream use

3.91
- updated spl2 family tree for downstream ord3 instead of ordl
- updated ors6 processdict for use of spl2 transformation function instead of spl5
- fixed "returned column" printouts to show columns in consistent order as they are returned in data sets
- new transform src4, comparable to srch but activations returned in an ordinal encoded column
- note that for cases for multiple search activations, encoding priority is given to entries at end of search parameter list over those at beginning
- new processing function strn, which extracts the longest length string set from categoric entries whose values do not contain any numeric entries
- note that since strn returns non-numeric entries, it's family tree is populated with a downstream ord3 encoding
- found and fixed edge case bug associated with our reserved special character set 'zzzinfill'
- so now when raw data is passed which contains that entry, it is just treated as infill instead of halting operation (as was the case for a few categoric transforms prior to this update)
- new option for srch function to pass search parameter, previously supported as a list of search terms, to now embed lists within that list of search terms
- these embedded lists are aggregated into a common activation
- for example, if parsing a list of names, could pass search parameter to aggregate female surnames as as ['Mr', ['Miss', 'Ms', 'Mrs']] for a common activation to the embeded list terms
- From a nuts and bolts standpoint the convention to name returned column of the aggregated activations is taken by the final entry in the embedded list, so here the aggregated list would be returned as column_srch_Mr and column_srch_Mrs
- Note that the intent is to carry this search parameter option to aggregate activations to the other srch functions (src2,src3,src4), saving that for another update
- Added inversion function for UPCS transform (simply a pass-through function, original lower cases not retained)

3.92
- found potential source of noise for encodings based on value counts
- relevant to ord3, bnry, bnr2, ucct, and lcinfill
- this was not impacting consistency between automunge and postmunge
- but it did have potential to demonstrate inconsistency between seperate automunge calls
- the issue was when grabbing value counts those entries with same number of counts did not have a sort method
- so now have incorporated a second basis of sort into value counts (first by count, then by alphabetical)

3.93
- new 'concurrent_activations' parameter accepted for splt and spl8 string parsing transforms
- allows the returned boolean columns to have simultaneous activations for overlaps detected accross entries
- note that activating this parameter may result in high dimensionality of returned data

3.94
- removed a redundant intermediate floatprecision transform as rounding for 16 bit float scenario was interfering with validation
- now floatprecision transforms take place only at conclusion of operation
- realized the new splt/spl8 parameters for concurrent_activations neccesitates different MLinfilltype due to returned columns with multiple simultaneous activations
- so scratched the parameter option and incorporated that functionality into new transformation categories sp13/sp14
- also found an efficiency improvement opportunity for splt family transforms by incorporating a break in one of the for loops

3.95
- 3.94 rollout contained a silly mistake
- I had somewhat lazily overwritten the sp13 and sp14 and replaced them with new transforms
- and just renamed the original to sp15 and sp16
- this of course broke compatibility with prior family trees that had incorporated sp13 and sp14
- which I had forgotten about
- so this update reverts the original sp13 and sp14 to prior configuration and remade the new trasnforms as sp15 and sp16
- details matter

3.96
- added a data type retention check to postmunge ML infill to match automunge
- found a sort of redundant operation in sp10 that was interfering with drift stat population intent, removed the redundancy
- inversion now supported with partial recovery for splt / spl2 / spl3 / spl4 / spl5 / spl6 / spl7 / spl8 / spl9 / sp10 / sp11 / sp12 / sp13 / sp14 / sp15 / sp16 / txt3 / ors7 / ors5 / ors6 / ors2 / or11 / or12 / or13 / or14 / or15 / or16 / or17 / or18 / or19 / or20 / srch / src2 / src3 / src4 / nmrc / nmr2 / nmr3 / nmcm / nmc2 / nmc3
- decided to simplify the search options to just focus on srch and src4 so as to avoid distraction
- option to aggregate search terms into common activation rolled out for srch in 3.91 now available in 3.94
- new MLinfilltype 'concurrent_nmbr'
- intended for returned multi-column numerical sets
- such as may be returned from tlbn for instance
- previously these sets were excluded from MLinfill
- in this new type each column in the returned set has a distinct trained model
- updated MLinfilltype for tlbn to concurrent_nmbr
- new MLinfilltype 'concurrent_act'
- intended for multi-column boolean sets which may contain concurrent activations
- such as may be returned from sp15 for instance
- previously these sets were given comparable ML infill treatment to 1010
- such that a single model was trained for the collection of activations
- in this new type each column in the returned set has a distinct trained model
- updated MLinfilltype for sp15 ,sp16, srch to concurrent_act

3.97
- A few improvements to simplify trainID_column and testID_column specification requirements
- For scenario when passing both train and test data to automunge(.)
- Now only have to specify trainID_column if ID columns are the same between train and test sets
- Changed conviention for postmunge(.) inspection of ID columns from postprocess_dict
- To inspect the train ID columns instead of test ID columns
- To be consistent with convention that postmunge(.) is based on properties of the automunge(.) train set

3.98
- Populated a new data structure final_assigncat available in postprocess_dict
- Comparable to assigncat with added results for those column categories that were derived under automation
- This is primarily intended as an informational resource
- Although there are some workflow scenarios where could be beneficial with data set composition permutations
- Also, new methods to mitigate the overhead caused by automunge(.) evaluation functions
- A new heuristic bases evaluations under automation on a subset of randomly sampled rows
- Based on configurable heuristic defaulting to 50% with the new automunge(.) eval_ratio parameter
- Which can be passed as a float 0-1 for a ratio of rows or as integer >1 for number of rows
- evalcat parameter functions now have two new positional arguements for eval_ratio and randomseed
- This heuristic makes automunge(.) run much much faster on large data sets without loss of functionality

3.99
- An audit of the string parsing functions identified a few more places where break operations could be applied to for loops
- This won't have huge impact on efficiency, but every little bit helps
- Also updated spl9 and spl3 family trees from ordl to ord3 to be consistent with rest of family

4.00
- 3.96 had noted intent to narrow focus on search options to srch and src4
- However just realized that src2 had been eluded to in the string theory paper
- So went ahead and added src2 to READ ME assigncat demonstrations
- Also added support for aggregated activations to src2 to be consistent with srch and src4
- The case parameter from srch is a little more tricky for this variant, so recomend if you need case neutrality for src2 just perform an upstream UPCS transform

4.10
- cleanup of formatting in function populating family trees
- motivated by inclusion of this code in the READ ME
- also corrected populated category key for mnts transform

4.11
- rewrite of data structure maintenance function
- to correct potential inconsistency in output
- issue resolved
- this update should also slightly improve efficiency of automunge(.)

4.12
- slight improvement to data structure population in por2 transform
- to accomodate disparity in data points between train and test sets
- also new aggregation transform aggt
- intended for categorical sets in which the may be redundant entries with different spellings
- relies on user passed parameter 'aggregate' as a list or as a list of lists
- in which are entered sets of string entries to aggregate
- with consolidated value based on final entry in each list
- note that aggt does not return numerically encoded sets
- so default family tree has a downstream ord3 encoding
- note that we already had similar functionality built into the search functions
- still hat tip to Mark Ryan for noting this type of operation in "Deep Learning with Structured Data"
- which sparked the idea of a dedicated version of aggregation outside of search operation

4.13
- small (immaterial) update to string parsing functions for code clarity

4.14
- new root categories or21, or22
- comparable to or19, or20 but make use of spl2/spl5 instead of spl9/sp10
- which allows string parsing to handle test set entries not found in the train set
- which is a trade-off vs efficiency
- also new UPCS parameter 'activate'
- can pass as boolean, defaults to True
- when False the UPCS character conversion is not performed, just pass-through
- such as may be useful in context of an or19 call
- note that to assign UPCS parameter with assignparam
- the parameter should be passed to the associated transformation category
- eg UPCS is processdict entry for the or19 category entry which is entry to the or19 root category family tree
- so could pass an UPCS 'activate' parameter to the UPCS function application in or19 root category through assignparam using the or19 transformation category entry as:
- assignparam = {'or19' : {'column1' : {'activate' : False}}}
- (this clarification intended for advanced users to avoid ambiguity)

4.15
- revised collection of source column drift stats to only collect the range of unique entries when number of unique entries is below an arbitrary threshold of 500
- this is to ensure don't run into postprocess_dict file size issues with larger data sets, such as for sets with all-unique entries

4.16
- found and fixed bug in postmunge application of nmc7
- (was not stripping commas comparable to automunge nmc7)
- found and fixed incorrect processdict NArowtype entries for nmc4-nmc9
- rolling out new numeric string parsing family nmEU
- including nmEU/nmE2/nmE3/nmE4/nmE5/nmE6/nmE7/nmE8/nmE9
- similar to nmcm, which strips commas before testing extracts for numeric validity
- nmEU strips spaces and periods, then converts commas to periods
- such as to recognize numbers of international format embedded in strings
- and return a dedicated column of the extracts
- where nmE2/nmE5/nmE8 are folled by z-score normalization
- nmE3/nmE6/nmE9 are followed by min-max scaling
- and where 1-3, 4-6, and 7-9 are distinguished by assumptions of test set composition in relation to the train set
- i.e. 1-3 parse all entries in test set, 4-6 don't parse test set, and 7-9 only parse test set entries not found in train set
- inversion currently supported with full info recovery for nmEU/nmE2/nmE3
- nmEU family supported by new NArowtype parsenumeric_EU

4.17
- corrected support of nmEU family transforms for parsing numeric extracts for formats containing space as thousands deliminator
- added inversion support for nmr4/nmc4/nmE4 with full recovery, 
- added inversion support for nmr5/nm6/nmr7/nmr8/nmr9/nmc5/nmc6/nmc7/nmc8/nmc9/nmE5/nmE6/nmE7/nmE8/nmE9 with partial recovery
- added source column drift stat assembly for nmrc/nmcm/nmEU families of transforms

4.18
- new inversion option to pass list of columns to inversion parameter for partial recovery
- where list may include headers of source columns and/or returned columns with suffix appenders
- as an example, if inversion only desired for two columns 'col1' and 'col2'
- can pass the inversion parameter as ['col1', 'col2']
- note that columns not inverted are retained in the returned set
- inversion support added for excl family transforms excl/exc2/exc3/exc4/exc5/exc6
- found and fixed bug in inversion operation for labels column with excl transform

4.19
- found and fixed edge case bug with partial inversion for cases where user includes an excl column with default excl_suffix parameter
- code cleanup throughout, all line breaks capped at single line break
- primarily motivated by reducing number of lines to make code editor rendering a little faster

4.20
- updated UPCS transform to allow retention of nan values for infill with downstream transform
- new transform category 'onht' for one-hot encoding
- similar to 'text', but different convention for returned column headers
- i.e. text returns columns as column + '_entry'
- and onht instead returns columns as column + '_onht_#'
- where # is integer associated with entry
- new transform category 'Unht' similar to onht with an upstream UPCS
- added inversion support with partial recovery for onht, Unht, Utxt, Utx2, Utx3, Ucct, Uord, Uor2, Uor3, Uor6, U101
- also slight tweak to 'text' transforms to replace an array operation with list for clarity

4.21
- rewrite of pwrs transform for binning by powers of 10
- now closer in composition to pwr2
- also pwrs now accepts boolean 'negvalues' parameter
- defaults to False, when True bins for values <0 are aggregated
- similar rewrite of pwor transform for ordinal binning by powers of 10
- now closer in composition to por2
- pwor also accepts negvalues parameter
- inversion now supported with full recovery for: texd, lb10, lbte, bnrd, lbbn, ord4, ordd, lbor, 101d

4.22
- a small update
- conformed the normkey retrieval of pwrs and pwr2 postmunge functions to conventions for other returned multicolumn sets with unknown headers
- just trying to make this more uniform, now have single method as standard

4.23
- new option for Binary parameter
- (Binary parameter consolidates boolean columns with a macro binary conversion)
- can now pass as a list of source columns and/or returned columns for partial consolidation
- such as to only consolidate a subset of returned boolean columns
- passing a source column header will include all boolean columns returned from that source column, and passing returned column headers allow inclusion of only portion of boolean columns returned from a source column
- note that Binary already has option to either replace or retain the boolean columns inputted to a consolidation
- for this partial consolidation, the default is to replace the target columns
- for option to retain the target columns, can pass the first entry of list as False
- e.g. Binary = [False, 'target_column_1', 'target_column_2']

4.24
- inversion now supported for sets with Binary dimensionality reduction
- (as may be applied to consolidation some or all of the boolean integer encoded sets)

4.25
- aggregated postmunge inversion operations into a support function
- inversion now supported for sets with feature importance dimensionality reduction

4.26
- tweak to feature importance dimensionality reduction
- now binary encoded columns, such as via '1010' transform, are retained as a full set in returned data, even when a subset would otherwise have been part of dimensionality reduction
- (based on inspection of MLinfilltype for the transformation category)

4.27
- found and fixed bug for feature importance dimensionality reduction that in some cases was interfering with postmunge
- fixed feature importance dimensionality reduction printouts to retain order of columns in returned set
- added new printout to Binary dimensionality reduction with list of consolidatecd boolean columns
- tweak to code from 4.26 to run more efficiently

4.28
- new data privacy option for string parsing functions via 'int_headers' parameter
- 'int_headers' is boolean, defaults to False
- when passed as True string partitions are not included in returned column headers
- such as may be appropriate for like healthcare applications or such
- also improved inversion for string parsing with concurrent activations, now with more information retention

4.29
- removed a redundant adjacent row infill application in dxdt and dxd2
- corrected NArowtype processdict entry for shuffle transform to exclude

4.30
- found and fixed silly bug with the oversampling method
- removed an unused function relic
- found an opportunity to consolidate two of the MLinfilltypes to a single entry
- which is much cleaner / less confusing now
- in summary, MLinfilltypes multirt and multisp are both now aggregated as multirt
- multisp no more

4.31
- revision of the oversampling method
- for cases where numeric target labels are aggregated into bins
- now this method is generalized to support binning by custom transformation functions

4.32
- revision of the oversampling method
- to accomodate edge case when ordinal labels have supplemental columns such as when NArw included

4.33
- new 'shft' family of transforms for sequential data
- inspired by ICML Autonomous Driving workshop discussions by Sanjiban Choudhury and Arun Venkatraman
- shft is similar to dxdt family of transforms, but instead of taking relative difference between time steps it simply shifts a prior time step forward
- accepts parameter 'periods' for number of time steps
- and parameter 'suffix' for returned column suffix appender (as may be useful if applying the transform multiple times such as to distinguish)
- offered in library in six varients: shft, shf2, shf3, shf4, shf5, shf6
- where shft/shf2/shf3 are comparable with default periods of 1/2/3 respectively
- and shf4/shf5/shf6 return two seperate column: one derived from source column with a retn transform applied, and the second derived from the source column with either a shft/shf2/shf3 followed by a downstream retn normalziation
- where retn normalizaiton is similar to a min/max scaling but retains the sign of the source data (ie +/-)
- again this is sort of like a simpler version of the various dxdt family of sequential transformations already in library
- also inversion now supported for shft, shf2, shf3, shf4, shf5, shf6, lbnm, nmbd, copy, mnm3, mnm4, mnm5, mnm6, year

4.34
- new shft family entries for shf7 and shf8
- where shf7 applies upstream retn, shf4, shf5 and a downstream retn on shf4 and shf5
- and shf8 applies upstream retn, shf4, shf5, shf6 and a downstream retn on shf4, shf5, and shf6
- these are intended to simplify application for cases where user desires to apply mulitple shift operations at different time step periods on same data stream
- new infill type 'naninfill' to return data with inputation of NaN for infill points
- as some external libraries may prefer data without inputation
- added support for dimensionality reduction via Binary and PCA when excl_suffix passed as False
- corrected info_retention on mnm5 to True

4.35
- some soul searching for shft family transforms
- realized the infill options are innappropriate given row shifts
- so special case for shft family transforms to just default to adjacent cell infill
- without NArw aggregation
- this applies to shft family transforms, 
- the dxdt family sequential transforms, as retain a basis for target row, is left in place

4.36
- rethinking of defaults for binary classification labels transform
- recast from 'bnry' (single column boolean integer) to 'text' (one-hot encoding)
- found further validation that label interpretability benefits from seperate predictions for each class, ie plausability of each does not have to sum to unity as would be case for bnry encoding
- partly inspired by discussions in paper by Freund, Y., Schapire, R. E., et al. "Experiments with a new boosting algorithm"

4.37
- added validations for column header overlap detection associated with Binary and PCA dimensionality reductions
- removed convention of different default categoric encoding when applying Binary transform
- (now that Binary transform can be applied on partial sets this made less sense)

4.38
- found and fixed small bug for feature importance evaluation with binary classification
- originating from the 4.36 update

4.39
- material update to transforms applying one-hot encoding and power of ten binning: text, onht, pwrs, pwr2
- this update breaks backward compatibility for these specific transforms
- corrected the order of returned columns to maintain consistency with order of received columns
- note that prior method still had consistent order between automunge and postmunge so passed validation
- but this way just makes more sense, especially to maintain order grouping when one-hot is part of a multi-output transform set

4.40
- new validation to confirm feature selection model successfully trained, returned as check_FSmodel_result
- new validation to confirm passed numpy array is tabular (eg 1 or 2D), returned as check_np_shape_train_result, check_np_shape_test_result
- new one-hot encoding varient onh2, similar to onht but includes a NArw column by default even when NArw_marker parameter not activated
- (as one-hot by default otherwise returns rows without activation for infill)
- the thought, and just a hunch, but I expect there may be some libraries where labels may require all rows to have an activation, so yeah this is now available as an option for label encoding for this scenario
- some tweaks to onht/text/pwrs/pwr2 to remove use of a second (temporary) column suffix - one less opportunity for overlap error
- a little cleanup to postprocess_text and postprocess_onht
- add some clarification of model basis to READ ME for ML infill

4.41
- a few small tweaks to evalcategory function for efficency

4.42
- quality control audit:
- visual inspection of transformation outputs of entire library (passed)
- visual inspection of inversion operation of entire library (passed)
- visual inspection of complete data structure initializations for transformdict and processdict (see below)
- visual inspection of complete infill library operation (passed)
- quick walkthrough of various operations that inspect MLinfilltype (passed)
- reduced scope of import from collections library to just the counter module
- found and fixed incorrect processdict entry for spl2 labelctgy (this only comes into play when spl2 applied as label category during feature importance)
- removed spl3 and spl4 from library as spl3 was redundant with spl2 and spl4 was not very useful
- changed ors2 use of spl3 to spl2
- updated bxcx family tree to allow bins assembly downstream of nmbr if binstransform selected (equivalent to prior configuration without bins assembly available as bxc4)

4.43
- new transformation family ntgr / ntg2 / ntg3
- intended for integer sets of unknown interpretation
- such as may be any one of continuous variables, discrete relational variables, or categoric
- (similar to those sets included in the IEEE competition)
- ntgr addresses this interpretation problem by simply encoding in multiple formats appropriate for each
- such as to let the ML determine through training which is the most useful
- ntgr includes ord3_mnmx / retn / 1010 / ordl / NArw 
- where ord3_mnmx encodes info about frequency, retn normalizes for continuous, 1010 encodes as categoric, ordl is ordinal encoding which retains integer order information, and NArw for identification of infill points
- ntg2 same as ntgr but adds a pwr2 power of ten binning
- ntg3 reduces number of columns from ntg2 by using ordinal alternates 1010->ordl, pwr2->por2
- thoughts around integer set considerations partly inspired by a passage in chapter 4 of "Deep Learning with Pytorch" by Eli Stevens, Luca Antiga, and Thomas Viehmann
- also corrected the naninfill option to correct a potential bug originating from inconsistent data types between infill and other data

4.44
- new DP transformation family for differential privacy purposes
- such as may enable noise injection to training data (but not test data)
- includes root categories DPnm for numerical data, DPbn for boolean/binary data, DPod for ordinal encodings, DP10 for binary encodings, and DPoh for one-hot encodings
- where DPnm injects Gaussian noise and accepts parameters for mu and sigma (defaults to 0, 0.06), as may be suitable for application to z-score normalized data
- DPbn injects Bernoulli distributted activation flips to bnry encodings, accepting parameter flip_prob for probability of flip (defaults to 0.03)
- DPod / DPoh / DP10 changes categoric activation per a Bernoulli distribution for whether to change and when changed per an equal probability within the set of activations, accepting parameter flip_prob for probability of flip (defaults to 0.03)
- note that when passing parameters to the DP functions, be sure to use the transformation category in the family trees associated with the transformation function (which in some cases may be different than you'd expect)
- DP operation in postmunge(.) supported by a new "traindata" parameter to distinguish whether the df_test is to be treated as train or test data (defaults to False for test data)
- note that methods make use of numpy.random library, which doesn't accept a random seed for repeatability, so for now DP family is randomized between applications (prob better this way, just making note)
- idea for a DP noise injection transform partly inspired by the differential privacy podcast series from TWiML documented in the From the Diaries of John Henry essay "Differential Privacy"
- added code comment about potential deep copy operation for postprocess_dict in postmunge so as not to edit exterior object
- some cleanup to normalization_dict entries for various ordinal encoding transforms such as to standardize on a few of the activation encoding dictionary formats

4.45
- recasting of family tree definitions for DP family of transforms to address issue found with inversion
- bug originated from transformation function saving transformation category key which was associated with another transformation function due to the way we structured family trees, now resolved
- also new DP transform DPnb for numerical data
- similar to DPnm to inject gaussian noise intended for z-score normalized data
- but only injects to a subset of the data based on flip_prob parameter defaulting to 0.03
- also accepts parameters from DPnm of mu and sigma defaulting to 0 and 0.06
- added traindata parameter to postmunge parameter validations

4.46
- new DP transform DPmm intended to inject noise to min-max scaled numeric sets
- injects Gaussian noise per parameters mu, sigma, to ratio of data based on flip_prob
- with parameter defaults of 0., 0.03, 1.0 respectively
- noise capped at -0.5 / +0.5
- DPmm scales noise based on recieved minmax value to ensure output remains in range 0-1
- for example if recieved input is 0.1, scales any negative noise by 0.2 multiplier
- updated default flip_prob for DPnb to 1.0, which makes it equivalent to DPnm
- so just making DPnb the default numeric DP transform to save space etc
- user can still elect sampled injection by decreasing this value

4.47
- new DP transform DPrt intended to inject noise to retn scaled data
- (retn is numerical set normalization similar to min-max that retains +/- sign of recieved data for interpretability purposes)
- note that retn doesn't have a pre-defined mean or range so noise injection instead incorporated directly into retn transform for a common transformation function
- (as opposed to other DP transforms which instead break noise injection into a seperate transformation passed to family tree primitives)
- DPrt includes parameters available in retn: divisor / offset / multiplier / cap / floor defaulting to 'minmax'/0/1/False/False
- as well as parameters available in other numeric DP transorms: mu / sigma / flip_prob defaulting to 0./0.03/1.

4.48
- performed an additional code review of today's rollout and found a small bug missed in testing
- (since noise injection prevents us from comparing similar sets validation methods are impacted)
- so quick fix for small bug from 4.47 in DPrt postprocess function

4.49
- new automunge(.) assignment parameter assignnan
- for use to designate data set entries that will be targets for infill
- such as may be entries not covered by NArowtype definitions from processdict
- for example, we have general convention that NaN is a target for infill,
- but a data set may be passed with a custom string signal for infill, such as 'unknown'
- this assignment operator saves the step of manual munging prior to passing data to functions
- assignnan accepts following form:
- assignnan = {'categories':{}, 'columns':{}, 'global':[]}
- populated in first tier with any of 'categories'/'columns'/'global'
- note that global takes entry as a list, while categories and columns take entries as a dictionary with values of the target assignments and corresponding lists of terms
- which could be populated with entries as e.g.:
- assignnan = {'categories':{'cat1':['unknown1']}, 'columns':{'col1':['unknown2']}, 'global':['unknown3']}
- where 'cat1' is example of root category
- and 'col1' is example of source column
- and 'unknown1'/2/3 is example of entries intended for infill
- in cases of redundant specification, global takes precendence over columns which takes precedence over categories
- note that lists of terms can also be passed as single values such as string / number for internal conversion to list
- note that validations on this data structure are returned in postprocess_dict['miscparameters_results'] validations as check_assignnan_result

4.50
- an iteration on the validations for new assignnan parameter to break previously consolidated results into seperate statements
- also realized MLinfilltype assigned to excl root category was innapropriate, so created new one
- MLinfilltype 'totalexclude' now availalbe for assignment in processdict
- 'totalexclude' is intended for complete pass-through columns
- now applied to excl and exc6 which are the root categories for total passthrough
- some small details added for assignnan, now global option for infill designated entries excludes application to complete passthrough columns without infill (e.g. excl, exc6)
- assignnan treatment still availbable for these root categories but must be assigned explicitly in categories or columns assignnan entries
- making note of this special treatment in read me under library of transformations entries for excl and exc6

4.51
- small tweak to exclude pass through columns from inf infill to be consistent with philosophy for passthrough columns
- (we have passthrough columns excl/exc6 with no transformations or infill because they may still be desired to serve a purpose of providing part of basis for ML infill to adjoining columns even though no preparation is performed. We have similar pass through columns with added infill support for transforms exc2-exc5)

4.52
- corrected bug from 4.51

4.53
- some spot testing of validation methods revealed that the tests for suffix overlap error
- were not being applied in a few cases when transformations include a temporary support column
- (suffix overlap error refers to cases where a new column is created with header that was already present in the dataframe)
- an example of suffix overlap could be applying transform mnmx to column 'column1' when the column header 'column1_mnmx' was already present in the dataframe
- so 4.53 introduces a new method for detecting suffix overlap error, replacing prior
- by performing a validation at each time of new column creation within transformation functions
- with support functions df_copy_train to perform a copy operation in parallel to validation or df_check_suffixoverlap to just check for overlaps
- note these validations only performed on train sets in automunge(.) transformation functions to avoid unneccesary redundancy
- the results of the validations are returned in postprocess_dict['miscparameters_results']['suffixoverlap_results']
- in addition to printouts in cases of identified overlap

4.54
- standardized on suffix overlap error detection methods between transformation functions and PCA/Binary dimensionality reductions
- so now they all use common approach for simplicity and consistency and you know just good practice
- PCA suffix overlap results now returned in postprocess_dict['miscparameters_results']['PCA_suffixoverlap_results']
- Binary suffix overlap results now returned in postprocess_dict['miscparameters_results']['Binary_suffixoverlap_results']
- added this validation for excl column processing, returned in postprocess_dict['miscparameters_results']['excl_suffixoverlap_results']
- consolidated final printouts for suffix overlap detection into a support function
- a few small tweaks to some of column header lists to run faster
- found an unused relic data structure entry in a few of the time series transforms now struck
- finally cleaned up the use of one hot encoding support function postprocess_textsupport_class used in several places 
- now can be called in much cleaner fashion without initialized supporting data structure

4.55
- rewrote the insertinfill function for simplicty / clarity, also to segregate supporting columns from attachment to df_train, one more edge case failure mode eliminated
- this is a pretty central support function and was one of the first that had written, part of reason it was a little sloppy, much cleaner now
- new validation infill_suffixoverlap_results out of abundance of caution
- also a few more tweaks to dataframe column inspections to use more efficient .columns instead of list()
- similar tweaks to a few dictionary key inspections

4.56
- added a validation and printout for ML infill edge case scenario
- a little clean up to infill validation aggregations

4.57
- found and fixed small bug in postmunge feature importance evaluation

4.58
- improved printouts to support postmunge(.) troubleshooting
- for cases where df_train columns passed to automunge(.) inconsistent with df_test passed to postmunge(.)
- removed a validation test for bug scenario that was eliminated as part of 4.55 update
- and also simplified the reporting for ML infill validation test added in 4.56 since only needs to be run once instead of for every column

4.59
- added returned_PCA_columns to postprocess_dict
- added a memory clear operation to PCA transform to reduce overhead
- added returned_Binary_columns to postprocess_dict
- improved printouts for PCA and Binary dimensionality reductions
- removed ordinal columns from Binary dimensionality reduction, just made more sense
- improved read me writeup for ord4, intended as a scaled metric ranking redundant entries by frequency of occurance
- new report classifying returned columns by data type now available in postprocess_dict as postprocess_dict['columntype_report']
- includes aggregated lists of columns per types: continuous, boolean, ordinal, onehot, onehot_sets, binary, binary_sets, passthrough
- where onehot captures all one-hot encoded columns, and onehot_sets is redundant except that it subaggregates by those from same transform as a list of lists
- similarily with binary and binary_sets
- these aggregations should be helpful for training downstream models in libraries that accept specification of column types, such as eg for entity embeddings

4.60
- created companion columntype_report for columns returned from label set
- available in postprocess_dict as postprocess_dict['label_columntype_report']
- new label processing category 'lbos' that applies ordinal encoding followed by conversion to string
- such as to support downstream machine learning libraries that may consider data types of labels for determination of application of regression vs classification
- (ie some libraries may treat integer sets as targets for regression when user intends classification)
- note that inversion is supported to recover original form eg to convert predictions back to form

4.61
- a few small cleanups:
- updated returned dtype from bsor transform to be integer instead of categoric (to be consistent with the other ordinal encodings)
- replaced transformation category designator 'DP06' with 'DPo6' to follow convention of other transforms from the series
- updated family trees for wkdo and mnto to downstream ordinal encoding by ordl instead of ord3 such as to maintain order of weekday / months from calendar in encoding
- added ID columns to printouts at conclusion of automunge(.) and postmunge(.)

4.62
- new transform por3, for power of ten bins aggregated into a binary encoding
- such as may be useful with high variability of received data
- also found an edge case scenario for inversion operation in which a downstream encoding interferes with inversion operation on an upstream ordinal encoding by returning floats instead of int
- so added an int conversion operation into inversion transforms associated with ordinal enocodings
- also corrected the MLinfilltype processdict entries for spl5 and sp10 transforms from singlct to exclude
- (sort of an immaterial update just trying to make everything uniform)

4.63
- new numeric binning variants for binary encoding
- as oposed to one-hot or ordinal encoding of bins
- available as root categories for binary bkb3, bkb4, bsbn, bnwb, bnKb, bnMb, bneb, bn7b, bn9b, pwbn
- which are extensions of ordinal variants bkt3, bkt4, bsor, bnwo, bnKo, bnMo, bneo, bn7o, bn9o, pwor
- which are extensions of one-hot variants bkt1, bkt2, bins, bnwd, bnwK, bnwM, bnep, bne7, bne9, pwrs  

4.64
- postmunge(.) now accepts numpy arrays without column headers when original data passed to automunge(.) was a dataframe
- inversion now accepts numpy arrays without column headers for both test set and label inversion

4.65
- default infill for bkt1 and bk2 (one-hot buckets) updated from mean insertion to no activation
- default infill for bkt3 and bk4 (ordinal buckets) updated from mean insertion to unique activation
- where bkb3 and bkb4 (binary buckets) inherit infill from bkt3 and bkt4
- new string parse transform function sbst, standing for "subset"
- sbst is similar to sp15, but only parses to identify subsets of entries that overlap with a complete unique value string set
- (as opposed to comparing subsets of unique values to subsets of other unique values)
- sbst allows test set entries that weren't present in train set
- also new transform sbs2, comparable to sbst but incorporates assumption that test set entries are subset of train set entries for more efficient application
- note that sbst transforms accept minsplit parameter to set floor to overlap detection length and int_headers parameter for privacy preserving encodings

4.66
- new transforms sp19 and sp20
- which are extensions of sp15 and sp16 for string parsed encodings with concurrent activations
- but with activations aggregated into a binary encoding for reduced column count
- new transforms sbs3 and sbs4
- which are extensions of sbst and sbs2 for string parsed subset encodings
- but with activations aggregated into a binary encoding for reduced column count
- note that ML infill is expected to perform better on concurrent activations configuration
- but there may be scenarios with high dimensionality where binary is of benefit
- corrected 1010 binary inversion for edge case when one of categoric entries had overlap with one of binary encodings
- corrected 1010 binary inversion edge case for distinguishing floats and inters
- added fix for remote edge case scenario in 1010 transform associated with concurrent overlaps between recieved categoric entries, binary encodings, and categoric entry with suffix to address first overlap
- found and fixed a bug or two associated with inversion operation in conjunction with Binary dimensionality reduction
- slight cleanup to sbst and sbs2 parsing algorithms to remove a redundant step
- new automunge parameter privacy_encode
- privacy_encode converts the returned column headers to integers to preserve privacy for downstream applications, e.g. for cases where pandasoutput is True
- conversion dictionaries matching original and encoded column headers are returned in postprocess_dict for reference
- note that inversion is supported with privacy_encode

4.67
- found and fixed a small bug in assignparam_str_convert
- removed an unused code block in postcircleoflife
- small code comment cleanup

4.68
- found and fixed a typo bug in recently rolled out transforms sp19, sp20, sbs3, sbs4
- added support for unnamed non-range index in dataframes passed to df_train and df_test
- new root categories similar to demonstrations from recent paper "A Numbers Game"
- rtbn (retain normalization with ordinal encoded standard deviation bins) 
- rtb2 (retain normalization with one-hot encoded standard deviation bins)

4.69
- a few cleanups to the ID column extractions in postmunge
- new 'mad' divisor parameter option for retain normalziation via retn and DPrt
- 'mad' applies median absolute deviation divisor instead of max-min
- mad divisor may be appropriate when range of values unconstrained, to avoid outliers interfering with in distribution range of normalizated set
- (e.g. if most of values fall in range 0-100, a train set outlier of 10,000,000,000 would interfere with normalization)
- in some distributions median absolute deviation may be more tractable than standard deviation

4.70
- new differential privacy series for numerical data
- featuring transforms DLnb, DLmm, and DLrt
- comparable to DPnb, DPmm, and DPrt
- but apply laplace distributed noise (i.e. double exponential) instead of gaussian
- where DLnb applies to z-score normalized data, DLmm to min-max normalized data, and DLrt to retain normalized data
- uses same parameters as the DP versions, where scale is passed as sigma, and loc as mu, and ratio of application as flip_prob
- inspired by a NIST post just saw on Hacker News
- also hat tip to Numpy for their numpy.random which serves as noise source

4.71
- performed another code review of 4.70 and found a small snafu populating processdict for new transforms
- so quick fix to processdict entries for DLnb, DLmm, and DLrt

4.72
- new 'defaultparams' option for the processdict data structures for defining transformation category properties
- can now define a transformation category to accept custom default parameters for passing to transformation functions
- such as may be useful if you want to distinguish between versions of transformation categories that apply the same transformation functions but with different default parameters
- note that manually defined parameters passed to assign_param will still overwrite these defaults
- this new convention allows us to scrub the recently defined 'DL' differential privacy functions with laplace distribution
- now laplace is available as a parameter to the corresponding 'DP' differential privacy functions
- saving about 1,000 lines of code in the process
- also a little cleanup to the processfamily functions
- with consistent transformation function calls independant of parameter assignments
- which just makes more sense

4.73
- a few transfomation category consolidations to make use of common transformation functions
- by making use of the new processdict defaultparams option
- consolidated transformation categories bnwd/bnwK/bnwM to use common transfomation functions
- consolidated transformation categories bnwo/bnKo/bnMo to use common transfomation functions
- consolidated transformation categories bnep/bne7/bne9 to use common transfomation functions
- consolidated transformation categories bneo/bn7o/bn9o to use common transfomation functions
- consolidated transformation categories pwrs/pwr2 to use common transfomation functions
- consolidated transformation categories pwor/por2 to use common transfomation functions
- altogether saved about 2500 lines of code

4.74
- a few transfomation category consolidations to make use of common transformation functions
- by making use of the new processdict defaultparams option
- consolidated transformation categories splt/spl8/sp15/sp16 to use common transformation functions
- consolidated transformation categories sp19/sp20 to use common transformation functions
- consolidated transformation categories spl2/spl5/spl7/spl9/sp10 to use common transformation functions
- consolidated transformation categories sbst/sbs2 to use common transformation functions
- consolidated transformation categories sbs3/sbs4 to use common transformation functions
- altogether saved about 5000 lines of code

4.75
- consolidated time-series transforms into two master function sets
- consolidated numeric partition string parsing transforms into two master function sets
- altogether saved about 4400 lines of code

4.76
- new transforms for aggregating standard deviation bins with custom bincount
- (prior versions are based on 6 bins for values <2, 2-1, 1-0, 0-1, 1-2, >2)
- binz for onehot-encoded, bzor for ordinal, bzbn for binary
- each accepts parameter bincount for integer number of bins
- bincount defaults to 5
- where if bincount is odd the center bin straddles the mean
- and if bincount is even the center two bins are on different sides of the mean
- also updated inversion transform for standard deviation bins to return bucket midpoint instead of left point

4.77
- new report populated and returned in postprocess_dict as a direct map from input columns to returned columns
- i.e. as a dictionary with keys of input column headers and entries of a list of returned column headers derived from that input column
- available as postproces_dict['column_map']
- excludes returned columns consolidated as part of a dimensionality reduction
- includes label columns
- this information was already available but now is more directly accessible
- also corrected MLinfilltype of datetime meta transforms from exclude to numeric

4.78
- a backward compatibility breaking rollout for the bins and bsor transforms
- wanted to get this out quickly before get too many users
- replacement of the standard deviation bins with the new versions rolled out in 4.76
- which is improved as accepts a parameter for bincount
- only slightly different in column header suffix encodings with integers which generalizes better

4.79
- reintroduced a varient of the standard deviation bins that assume input data is already normalized
- by incorporating boolean parameter normalizedinput into bins and bsor
- such as may reduce the computational overhead associated with the binstransform parameter

4.80
- rewrite of the dxd2 transform
- a review found that I had kind of bungled the implementation
- when had performed the conversion to accepting periods parameter
- current form is consistent with original intent

4.81
- rethinking the PCA heuristic
- which was in place to apply dimensionality reductions automatically when # features > 0.5 number of rows
- I was finding that this heuristic was kind of cumbersome when working with small data sets to run experiments
- in which case the heuristic had to be turned off with a ML_cmnd
- and decided from a usability standpoint would be better to make this heuristic optional instead of default
- so new default value for PCAn_components is False to completely turn off PCA
- if you want to run the heuristic and only apply PCA for that scenario, can pass PCAn_components = None

4.82
- another accomodation for small data sets, this time with the eval_ratio parameter
- eval_ratio is meant to improve sampling efficiency for evaluating data properties
- by only inspecting subset of data
- new convention is that eval_ratio is only applied when a training set has > 2,000 rows

4.83
- improved transformation category specification in user passed processdict data structure
- incorporated 'functionpointer' option
- so user can specify the set of transfomation functions associated with a transformation category without having to dig into code-base to identify naming conventions for existing transformation functions
- now processdict entry can instead just identify a transformation category in functionpointer whose transformation functions the new category would like to match
- such as to automaticaly populate with entries for dualprocess, singleprocess, postprocess, inverseprocess, and info_retention
- defaultparam entries are also accessed, and if the new category specification contains any redundant defaultparam entries with the pointer category the new category entries will take precedence
- as an example, if we previously wanted to define a processdict entry for a new transformation category that reused transformation functions of the mnmx category, it would have looked something like this:
{'newt' : {'dualprocess' : am.process_mnmx_class, \
          'singleprocess' : None, \
          'postprocess' : am.postprocess_mnmx_class, \
          'inverseprocess' : am.inverseprocess_mnmx, \
          'info_retention' : True, \
          'NArowtype' : 'positivenumeric', \
          'MLinfilltype' : 'numeric', \
          'labelctgy' : 'mnmx'}}
- in the new convention with functionpointer entry, we can now more easily consistently specify as:
{'newt' : {'functionpointer':'mnmx', \
          'NArowtype' : 'positivenumeric', \
          'MLinfilltype' : 'numeric', \
          'labelctgy' : 'mnmx'}}
- also perormed a few misc code comment cleanups

4.84
- corrected one of the support functions from 4.83
- to ensure that if we are following chains of functionpointers
- the defaultparams entries from intermediate links are incorporated into the final result
- also added a few code comments for clarity

4.85
- fixed inverison transform for bnwo for cases where a downstream categoric transform is performed
- found and fixed small bug in DPnb postprocess function
- new validation function to confirm that root categories in transformdict have corresponding processdict entries
- corrected an error message that had eroneously noted a scenario where a transformdict entry does not require a coresponding processdict entry
- so to be clear, every transformation category requires a processdict entry
- only in cases where a transformdict entry is being passed to overwrite an existing category already defined internal to the library is a corresponding processdict entry not required. 
- transformation categories that are to be used as family tree primitive entries with offspring require a root category entry in transformdict for family tree definition
- I recomend defaulting to defining a family tree for all transformation categories used in transformdict

4.86
- found and fixed a small bug in src4 (ordinal search function)
- where was incorrectly producing a printout

4.87
- slight tweak to the box-cox power law transform function 'bxcx'
- no change in output, just resolved an inconsequential edge case printout
- associated with scipy stats implementation
- to avoid confusion

4.88
- added transformation function parameter 'adjinfill' into a few normalizations nmbr, mnmx, retn, DPrt
- such as to allow changing default numeric infill from mean imputation to adjacent cell
- adjinfill parameter can be passed as True/False, defaults to False
- this parameter is redundant with what can be specified with assigninfill to return sets with adjacent infill
- but thought it might be beneficial to have option for customizing default infills applied prior to ML infill
- I have a hunch that adjacent infill may be better suited as a default imputation
- because of inherent stochasticity
- but don't have evidence so for now primary numeric normalization default remains mean imputation
- also added a few code comments for clarity for assignparam parameters in these normalization transforms

4.89
- consolidated mnm6 and mnmx transformation functions by use of mnmx for mnm6 with defaultparams as floor = True
- added parameter support for the other numeric normalizations to accept default adjacent cell infill comparable to those updates from 4.88
- ie added adjinfill parameter support to MADn, MAD3, mnm3, mean

4.90
- performed a micro-audit on a specific segment of the processing functions
- associated with populating column_dict data structures
- found a few opportunities for cleanups for purposes of clarity
- that were mostly relics from awkward code reuse issues
- so column_dict specifications are a little more uniform now

4.91
- found an opportunity to eliminate some redundancy in data stored in postprocess_dict
- associated with the normalization_dict entries
- this redundancy was a relic of some earlier methods for accessing entries in postmunge
- which have since been replaced with a more straightforward approach
- this update has potential to materially reduce the memory footprint of postprocess_dict in some cases
- also found a few support functions that had a parameter not used by functions
- associated with the labelsencoding_dict report
- so removed that entry as a parameter to those functions
- in interest of reducing complexity

4.92
- extended transformation function parameter support for adjinfill to include base categoric options
- including bnry, bnr2, text, onht, ordl, ord3, 1010
- similar to updates from 4.88 and 4.89
- don't worry, I have a plan

4.93
- rewrite of a few support functions associated with ML infill
- improving code and comment clarity
- and for purposes of enabling better modularity
- such as to facilitate plug and play of additional architecture / ensemble options
- also replaced a parameter key in ML_cmnd for clarity
- replaced {'MLinfill_type':'default'} with {'autoML_type':'randomforest'}
- where MLinfill_type as previously used was only a placeholder
- so this update won't impact backward compatibility
- more to come

4.94
- found and fixed bug in printouts for model training functions rolled out in 4.94
- incorporated modular aspects of model inference to feature importance shuffle permutation support function
- so feature importance now has potential to support alternate autoML architectures like ML infill
- revised the default feature importance regression accuracy metric from mean_squared_log_error to mean_squared_error
- (I think is more generalizable since allows negative values)

4.95
- the last rollout included an update to the accuracy metric used for regression in feature importance
- after running some more tests came to conclusion that was in error
- so this update reverts to default regression accuracy metric of mean_squared_log_error
- move fast and fix things as is our motto
- sorry to be a flake

4.96
- increased granularity of options for types of autoML applied in the autoMLer specifications
- such as to distinguish between classification options for boolean, ordinal, and onehot encoded labels
- even though for random forest they all use the same models
- this was partly motivated by a vague impression that autosklearn treats two class and multiclass classification differently

4.97
- a review of the testID_column parameter usage in automunge and postmunge found a few opportunities for improvement
- added automunge test_ID_column parameter support to pass as True for cases where test set contains consistent ID columns as train set
- which may also be passed as the default of False for this case
- but just trying to make this intuitive, since postmunge allowed passing test_ID_column as True, now can do the same in automunge
- as part of that update realized there was a small redundancy where ID column extraction had gotten integrtated in with index column extraction, so broke the ID column extraction out of index column extraction and consolidated the redundancy
- then in postmunge(.) found that could offer comparable testID_column support to automunge as well, where if a trainID_column was passed to automunge(.), now there is no need to specify a corresponding testID_column in postmunge, it is detected automatically 
- (previously this required passing testID_column as either True or as the string or list of columns)
- so to be clear, parameter support is now completely equivalent for testID_column between usage in automunge and postmunge, where if trainID columns are present in the test set can either pass as True or leave as default of False and will automatically detect
- also added overdue support for postmunge to automatically detect if a labels column is present and treat accordingly
- (there is a tradeoff in this approach in that if sets passed as numpy arrays if ID columns between train and test are different the label column may have different header, this is a bit of an edge case and not an issue when prefered input of pandas dataframes)
- also started experimenting with ML infill support for an alternative to random forest
- specifically AutoGluon (an autoML library)
- requires installing and then importing AutoGLuon external to the function call as:
import autogluon.core as ag
from autogluon.tabular import TabularPrediction as task
- and then can be activated with ML_cmnd = {'autoML_type':'autogluon'}, and MLinfill=True
- parameter support for autogluon pending
- so to be clear, I don't consider the current autogluon implementation highly audited
- it needs more testing
- please consider this autoML option as experimental for now

4.98
- added a printout to feature importance reporting base accuracy of feature importance model
- added an entry to FS_sorted report available in postprocess_dict with the base accuracy metric as 'baseaccuracy'

4.99
- found bug associated with new ML infill autoML option for AutoGluon library rolled out in 4.97
- reconsidered imports approach
- now imports are conducted internal to ML infill support functions instead of by user
- which fixes the bug
- also found and fixed bug associated with ML infill targets of binary encoded sets for this autoML type
- so yeah I still consider this implementation somewhat experimental
- any edge scenarios where a model doesn't train can be addressed by assigning a different infill type for a column in assigninfill
- still a step in the right direction
- also new default transform category applied to all unique entry categoric sets
- ord5 which is comparable to ordl but excludes application of ML infill to that column

5.00
- paying down a little technical debt
- renamed a parameter in a few of the ML infill support functions to be consistent with usage / avoid confusion
- which in some cases was labeled as a "columnslist" but was actually passed to the function as a "categorylist"
- so yeah these parameters are a little more clearly labeled now
- also a fix for shuffleaccuracy function to support labels prepared in multiple configurations
- also created a label set version of ord5 trasnfomation category rolled out yesterday as lbo5
- which is fully consistent just creating a version intended for labels to follow convention
- such as may be useful if you want to overwrite a family tree for labels but not for training data or visa versa

5.1
- a small cleanup to feature importance to support edge case for labels column parameters as updated in 4.99

5.2
- cleaned up edge case in featrue importance associated with newly reported baseaccuracy from 4.98
- allowed the reversion of labels column update in 5.1 which in hindsight was better served in prior configuration
- aded base accuracy printout to postmunge feature importance
- also bug fix in assembly of columntype_report associated with populating onehot sets and binary sets

5.3
- one more cleanup to columntype_report populating support function
- (fixed scenario when Binary transform applied)
- columntype_report now in good shape
- a few code comment cleanups
- added 'ordered_overide' parameter to ordinal encodings ordl and ord3
- which are the integer encodings sorted by alphabetic and frequency respectively
- ordered_overide is boolean defaults to True
- when activated target columns are inspected for if they are Pandas categorical with ordered = True
- in which case pandas may have already recieved input from user of an ordered sequence of categoric entries
- in which case the ordinal integer encoding order defers to the recieved designation

5.4
- a revision to support functions associatec with edge cases for infill such as inf values and those infill points assigned in assignnan
- to use .loc instead of np.where in order to retain pandas column properties
- such as to ensure ordered categoric method rolled out in 5.3 works properly

5.5
- revision to ordinal encoding functions ordl and ord3
- to facilitate distinguished encodings between numerical and equivalent string entries
- for example, previously entries of 2 and '2' would have been consistently encoded
- now these each will return a distinct encoding
- consistent support to one-hot and binary encodings pending

5.6
- revision to one hot encoding function onht and binary encoding function 1010
- to facilitate distinguished encodings between numerical and equivalent string entries
- for example, previously entries of 2 and '2' would have been consistently encoded
- now these each will return a distinct encoding
- note that due to column labeling convention, 'text' version of one hot encoding retains treatment of numbers as strings
- so if you want to disginguish numbers from strings in one hot encoding use onht instead of text
- also found a remote edge case for ordl associated with dtype shift between object and int after a replace operation, cleaned it up

5.7
- cleaned up the implementation to address edge case identified in 5.6
- now dtype conversion is conditional which may improve efficiency
- applicable to categoric transforms with sequences of replace operations

5.8
- corrected printouts in assignnan
- one more revision to categoric transforms associated with recent updates to allow distinct encodings between numbers and string equivalents
- added str_convert parameter to ordl, ord3, 1010, onht such as revert to to consistently encoding between strings and numbers
- str_convert defaults to False e.g. 2 != '2', when passed as True e.g. 2 == '2'
- thus allowing allowing user to have consistent convention betweehn these transforms and text transform if desired
- again where text is one-hot encoding that requires a string convert operation due to convention for returned column headers
- this is really diving into the weeds. Details matter.

5.9
- ok I think this is final update needed for uniformity in categoric trasnforms with respect to options for distinct encodings between numbers and string equivalent
- added str_convert parameter support to bnry and bnr2 transforms similar to updates in 5.8
- defaults to False for distinct encodings e.g. 2 != '2'
- so yeah everything looks good

5.10
- major architecture revision, backward compatibility impacted
- now transformation functions may automatically apply an inplace operation when available
- as opposed to a column copy operation which was default convention prior
- with the expectation that this may benefit memory overhead
- inplace transforms if available are applied to the final replacement transformation category entry to family tree primitives
- (where parents are applied prior to auntsuncles, and children prior to coworkers)
- transformation functions which have inplace operation available are designated by boolean inplace_option entry to processdict
- such that transformation categories can optionally be initialized as processdict entries with inplace turned off if desired such as to maintain column grouping correspondance
- where the designation for inplace is passed to the transformation function by way of an inplace parameter which defaults to False
- such that transfomation functions inspect this parameter for determination of whether to rename or copy input column
- the convention is that a user can turn off inplace by passing parameter to a columns as {'inplace' : False}, but cannot turn on inplace manually (to avoid a channel for error)
- it was a design decision not to update order of columns to retain grouping of columns returned from same family tree, as this approach expected to be lower latency
- grouping is accessible by returned column_map, all that matters is order returned from postmunge(.) is same as automunge(.)
- as part of this update replaced the convention for excl transform to utilize this method
- previously excl was an exception to family tree operations in order to apply inplace by different method, now excl is fully consistent with other categories
- removed the exc6 varient from library as no longer needed
- also revised the order of family tree applications in processfamily and processparent as:
- parents, auntsuncles, siblings, cousins => siblings, cousins, parents, auntsuncles
- children, coworkers, niecesnephews, friends => niecesnephews, friends, children, coworkers
- inplace option now supported for: nmbr, nbr2, nbr3, DPn3, DPnb, DLn3, nmdb, mnmx, mnm2, mnm3, mnm4, mnm5, mnm6, DPm2, DLm2, retn, rtbn, rtb2, excl, exc2, exc3, exc4, exc5, dxdt, d2dt, d3dt, d4dt, d5dt, d6dt, dxd2, d2d2, d3d2, d4d2, d5d2, d6d2, mmdx, mmd2, mmd3, mmd4, mmd5, mmd6, dddt, ddd2, ddd3, ddd4, ddd5, ddd6, dedt, ded2, ded3, ded4, ded5, ded6, shft, shf2, shf3, shf4, shf5, shf6, shf7, shf8, MADn, MAD2, MAD3, mean, mea2, mea3, bnry, bnr2, DPb2, log0, log1, logn, lgnm, sqrt, addd, sbtr, mltp, divd, rais, absl, pwor, por2, por3, bsor, btor, bsbn, bnwo, bnKo, bnMo, bnwb, bnKb, bnMb, bneo, bn7o, bn9o, bneb, bn7b, bn9b, bkt3, bkt4, bkb3, bkb4, shfl
- also added inversion support for nmdx, nmd2, nmd3, nmd4, nmd5, nmd6
- also found and fixed an edge case for 1010 transform associated with 5.6

5.11
- added option for assignparam to pass the same parameter to all transformations applied to all columns
- e.g.:
- assignparam = {'global_assignparam'  : {'globalparameter' : 42}}
- this may be useful if for any reason one wants to turn off inplace trasnforms, such as to retain return column grouping correspondance
- e.g.:
- assignparam = {'global_assignparam'  : {'inplace' : False}}
- note that if a specific transformation function does not accept a particular parameter it will just be ignored
- note that, in order of precendence, parameters assigned to distinct category/column configurations take precedence to default_assignparam assigned to categories which takes precendence to global_assignparam assigned to all transformations

5.12
- ran some validations on the assignparam order of precedence noted in last rollout and found they were inconsistent with expectations
- so rewrote the assignparam support functions to meet system requirements
- assignparam implementation is much much cleaner now
- also created new validation function for assignparam

5.13
- some further validations found a material oversight with 5.10 rollout
- associated with data sctructure maintennce for inplace operations
- that was interfering with ML infill
- problem solved

5.14
- added inplace support for ordinal trasnforms
- inplace now supported for ordl, ord2, ord3, ord4, ord5, ntgr, ntg2, ntg3, DPo4, DPo5, DPo6, ordd, lbo5, lbor, lbos

5.15
- a few tweaks to the family processing functions to allow passing processdict entries without entries for (dualprocess / postprocess) or (singleprocess) for cases when not applicable
- so the new convention is only at least one of these two sets of entries for a processdict entry is required 
- instead of both sets with "None" for those without use as was convention prior
- for cases where user specifies entries for both sets (dualprocess / postprocess) and (singleprocess), the (dualprocess / postprocess) takes precedence
- instead of checking != None now the entry is tested for validatity as a python function type which makes much more sense
- update to support function associated with functionpointer to accomodate these revisions
- also update to functionpointer support function to carry through additional designations to pointer recipient iff not previously specified, including inplace, NArowtype, MLinfilltype, and labelctgy
- thus new convention is that when defining a custom processdict entry with functionpointer, no additional entries are required if intent is to just carry through all designations from functionpointer entry
- similarly for chains of function pointers, the closest link for these items with entry takes precendence
- these updates all make things considerably simpler for a user, the goal after all

5.16
- some additional testing determined that the 5.15 update impacted the scenario when user externally defines custom transformation functions passed as entries to processdict
- so quick fix to accomodate this scenario
- (basically just means an additional type test for externally defined functions)
- also found and fixed an edge case for ML infill associated with inplace operations

5.17
- a revision to the NArw transform to reduce memory overhead (comparable functionality)
- added some printouts for remote edge case associated with inaccurate processdict lagelctgy entries to support troubleshooting

5.18
- removed an edge case printout associated with direct pass-through columns
- printout was there to support troubleshooting
- no longer applicable after new inplace convention for excl transform
- added a filler entry for exc6 back into default transformdict and processdict
- to avoid troubleshooting printout activated with prior code demosntrtations which incldued exc6

5.19
- a slight reorg of some of MLinfill support functions
- which were originally built around scikit-learn which accepts numpy arrays as default
- as am adding autoML library options finding that some prefer pandas dataframes as input
- so moving the numpy conversion for scikit into the scikit specific support functions
- which in practice means also moving the ravel conversion and also reworking some of support functions from numpy to pandas basis, including some feature importance support functions
- weirdly finding that there is some kind of distinction between renaming pandas headers to integers vs conversion to numpy and back to pandas
- which currently comes up in autogluon support functions
- tabling that inquiry for now
- what's in place works for whatever that's worth
- also found and fixed an edge case for numeric extraction transforms addressed by additional conversion to numeric

5.20
- another oportunity found to reduce memory overhead
- this time associated with the getNArows function associated with recording infill entries
- such that now the evaluation is applied directly to a received column instead of a copy
- which has a small tradeoff in that projections to numeric types (such as force to numeric or force to positive numeric) applied in the function are carreid through the received column
- which means that for multi-transform sets categories with different NArowtypes will have a master NArowtype cast from root category
- this was deemed an acceptable tradeoff and is consistent with the assigninfill approach

5.21
- some new options incorporated into assignnan
- which is parameter to designate received entries that will be targets for infill by nan conversion
- assignnan now supports stochastic and range-based nan injections
- such as to inject infill points into specific segments of a set's distribution
- further documented in read me
- this isn't expected to come up often in mainstream use
- primarily intended to support some experiments on missing data infill

5.22
- a small cleanup to some errant printouts associated with code demonstrations for assignparam parameter
- now printouts muted for read me demonstration plug value '(category)'

5.23
- updated ML infill hyperparameter tuning metric for classificaion from accuracy to weighted F1 score
- which we understand does a better job of balancing bias variance tradeoff and takes into account class imbalance
- updated the default parameter values of 'flip_prob' ratio for DPnb, DPmm, DPrt from 1.0 to 0.03
- which means for noise injection the noise is injected to only 3% of entries instead of full set
- which based on our experiments we believe makes for a better default for data augmentation
- updated the default numbercategoryheuristic automunge(.) parameter from 63 to 127
- which is the threshold for number of unique values where default categoric encoding under automation changes from '1010' binary to ordinal
- also updated the default ordinal encoding under automation from 'ord3' to 'ord5', which applies the ordl transform and is excluded from ML infill
- this update was based on some experiments with very high cardinality sets and finding that ML infill models were impacted

5.24
- new 'hash' transform intended for high cardinality categoric sets
- applies what is known "the hashing trick"
- works by segregating entries into a list of words based on space seperator
- stripping any special characters
- and hashing each word with hashlib md5 hashing algorithm
- which is converted to integer and taken remainder from a division by vocab_size
- where vocab_size is passed parameter intended to align with vocabulary size
- note that if vocab_size is not large enough some of words may be returned with encoding overlap
- returns set of columns containing integer word representations 
- with suffix appenders '_hash_#' where # is integer
- note that entries with fewer words than max word count are padded out with 0
- also accepts parameter for excluded_characters, space
- uppercase conversion if desired is performed externally by the UPCS transform
- ('hash' root category doesn't includes UPCS, 'Uhsh' root category does)
- hash transform was inspired by some discussions in "Machine Learning Design Patterns" by Valliappa Lakshmanan, Sara Robsinson, and Michael Munn
- also added inplace support for UPCS transforms

5.25
- extension of hash transform rolled out in 5.24
- new root categories hsh2, Uhs2
- hsh2 differs from hash in that space seperator for seperately encoding distinct words found in entries is discarded
- in other words encodings are returned in a single column with singe encoding for each entry
- hsh2 also differs in that special characters are not scrubbed
- so hsh2 closer resembles traditional categoric encodings like text, 1010, etc
- Uhs2 performs an upstream UPCS uppercase conversion for consistent encoding between different case configurations
- also small bug fix for space parameter in hash transform

5.26
- extension of hash transforms rolled out in 5.24 and 5.25
- new root categories hs10, Uh10
- hs10 differs from hsh2 in that returned encodings are translated from integers to a binary encoding
- with a column count as determined by the vocab_size parameter defaulting to 128 for 7 returned columns
- in other words encodings are returned in a set of boolean integer columns with activations encoded as zero, one, or more simultaneous activations
- so hs10 differs from 1010 in that no conversion dictionary is recorded
- which is a tradeoff in that inversion is not supported for hs10
- also as with hash there is a possibility of redundant activation sets for different entries
- as the range of supported activations is a function of vocab_size parameter
- Uh10 performs an upstream UPCS uppercase conversion for consistent encoding between different case configurations

5.27
- quick fix to hs10 transform rolled out yesterday
- associated with dtype of returned set
- i.e. converting strings to integers
- details matter

5.28
- extension of hash transforms rolled out in 5.24-5.26
- new root categories hsh3, Uhs3, hs11, Uh11
- they are similar to hsh2, Uhs2, hs10, Uh10
- but instead of accepting parameter for vocab_size, the vocab_size is determined automatically based on a heuristic
- more specifically, they accept parameters heuristic_multiplier and heuristic_cap
- where heuristic_multiplier defaults to 2 and heuristic_cap defaults to 1024
- the vocab_size is derived based on number of unique entries found in train set times the multipler
- where if that result is greater than the cap then the heuristic reverts to the cap as vocab_size
- since requires passing parameters between train and test sets the implementation is dualprocess instead of singleprocess convention

5.29
- reorganizing hash transforms
- realized the the various accumulated permutations were starting to overcomplicate
- so consolidated to two master functions, hash and hs10
- with the other permutations available by variations on parameters to those functions
- permutations are as follows:
- hash: parsed words extracted from entries, returned in multiple columns, accepts parameter for excluded_characters and space
- hsh2: no word extraction, just hashing of unique entries, returned in one column
- hs10: comparable to hsh2 but hashings are binary encoded instead of integers, returned in multiple columns
- With each of these having an additional permutation with upstream uppercase conversion:
- hash/hsh2/hs10 -> Uhsh/Uhs2/Uh10
- in each of these cases vocab_size derived based on heuristic noted in 5.28 with parameters heuristic_multiplier and heuristic_cap to configure heuristic (defaulting to 2 and 1024)
- the heuristic derives vocab_size based on number of unique entries found in train set times the multipler
- where if that result is greater than the cap then the heuristic reverts to the cap as vocab_size
- and where for hash the number of unique entries is calculated after extracting words from entries
- in each of these cases can also pass parameter for vocab_size to override heuristic and manually specifiy a specific vocab_size

5.30
- quick small fix to hs10 transform
- fixed suffix appender assembly
- quick small fix to hash transform
- fixed edge case where first character in string is space

5.31
- new inversion option intended to support dense labels scenario
- i.e. when labels are served to ML model in multiple configurations for simultaneous predictions
- inversion postmunge parameter can now be passed as 'denselabels'
- which recovers label form derived from each path for comparison
- instead of relying on heuristic for single recovery from shortest transformation path
- each version of label recovery is returned with column header 'A***_B***'
- where A*** is labels column header originally passed to automunge
- and B*** is header of the transformed column that is basis for the inversion recovery
- currently denselabels inversion option only supported for label sets

5.32
- new dupl_rows parameter available for automunge(.) and postmunge(.)
- consolidates duplicate rows in a dataframe
- in other words if duplicate rows present only returns one of duplicates
- can be passed to automunge(.) as one of {True, False, 'test', 'traintest'}
- where True consolidates only rows found in train set
- 'test' consolidates only rows found in test set
- and 'traintest' sperately consolidates rows found within train set and rows found within test set
- defaults to False for not activated
- can be passed to postmunge(.) as one of {True, False}
- where True conolidates rows in test set
- and defaults to False for not activated
- note that ID and label sets if included are consistently consolidated

5.33
- added parameter support for AutoGluon
- can be passed to fit command in ML_cmnd parameter
- further documented in read me

5.34
- new autoML option for ML infill using CatBoost library
- requires installing CatBoost with 
pip install catboost
- available by passing ML_cmnd as 
ML_cmnd = {'autoML_type':'catboost'}
- uses early stopping by default for regression and no early stopping by default for classifier
- (to avoid classifier channel for error when all label samples are included in validation set)
- can turn on early stopping for classifier by passing 
ML_cmnd = {'autoML_type':'catboost', 'MLinfill_cmnd' : {'catboost_classifier_fit' : {'eval_ratio' : # }}}
- where # is float between 0-1 to designate validation ratio (defaults to 0.15 for regressor)
- in general can pass parameters to model initialization and fit operation as
ML_cmnd = {'autoML_type':'catboost', 
           'MLinfill_cmnd' : {'catboost_classifier_model' : {'parameter1' : 'value' },
                              'catboost_classifier_fit'   : {'parameter2' : 'value' },
                              'catboost_regressor_model'  : {'parameter3' : 'value' },
                              'catboost_regressor_fit'    : {'parameter4' : 'value' }}}
- in general, accuracy performance of autoML options are expected as AutoGluon > CatBoost > Random Forest
- in general, latency performance of autoML options are expected as Random Forest > CatBoost > AutoGluon
- in general, memory performance of autoML options are expected as Random Forest > CatBoost > AutoGluon
- and where Random Forest and Catboost are more portable than AutoGluon since don't require a local model repository saved to hard drive
- for now retaining random forest as the default

5.35
- new parameter accepted for hash family of transforms
- 'salt' can be passed as arbitrary string, defaulting to empty string ''
- salt perturbs the hashing to ensure privacy of encoding basis
- which is consistently applied between train and test data for internal consistency
- quick fix to suffix appender assembly for hash transform from 'column_hash#' to 'column_hash_#'
- added edge case support for catboost associated with very small data sets

5.36
- ran some additional tests on hashing algorithm speed 
- and came to convincing conclusion that md5 wasn't best default for hash transforms
- so new default is the native python hash function
- md5 hash is still available with new 'hash_alg' parameter for hash transforms
- which defaults to 'hash' but can be passed as 'md5' to revert to original basis
- note that salt is still supported for both cases

5.37
- found and fixed small bug from 5.36 missed in initial testing

5.38
- new max_column_count parameter accepted for hash transform
- since number of returned columns is based on longest entry wordcount
- there may scenarios with extreme outliers that can result in excessive dimensionality
- max_column_count caps the number of returned columns
- defaulting to False for no cap
- when word extraction reaches thresshold remainer of string treated as single word
- e.g. for string entry "one two three four" and max_column_count = 3
- hashing would be based on extracted words ['one', 'two', 'three four']
- also updatred defaults under automation for high cardinality categoric sets
- now when number of unique entries exceeds numbercategoryheuristic parameter (which currently defaults to 127)
- it is treated with hsh2 which is a hashing similar to ordinal
- unless unique entry count exceeds 75% of train set row count
- in which case it is treated with hash which extracts the words from within entries

5.39
- corrected processdict entry for DPnb
- which had incorrect True classification for inplace support
- resulting in intermediate column retention from upstream nmbr application
- now upstream nmbr is subject to replacement as intended

5.40
- parameter support added to binning transforms with user specified buckets (bkt1,bkt2,bkt3,bkt4,bkb3,bkb4)
- now buckets parameter can be passed as percentages of range instead of set specific values
- eg with bucket boundaries in range 0-1
- in order to signal this option bcuket boundaries should be passed as a set instead of list
- eg for a set with range -5:15, buckets with exact boundaries could look something like [-5,0,5,10,15]
- and buckets as percentages could look like eg {0,0.25,0.50,0.75,1}
- which would give a consistent output
- partly inspired by a comment in book "Spark The Definitive Guide" by Bill Chambers and Matei Zaharia

5.41
- improved convention for time transforms tmsn and time, which segregate time series data by time scale
- now if a particular time scale is not present in the training data that segment ommitted in returned data
- e.g. if a time series records calendar days but not clock times, hour/minute/seconds ommitted in retuned data
- we expect these type of scenarios are not uncommon in real world data sets
- also added inplace support for datetime transforms built on top of tmsc and time transformation functions
- also added inplace support for datetime transforms built on top of wkds and mnts transformation functions

5.42
- new datetime transform tmzn applied upstream of time stamp transform aggregations date/dat2/dat3/dat4/dat5/dat6
- defaults as a pass-through
- when timezone parameter passed to the aggeragate transform category, can designate desired time zone
- this doens't really impact sin/cos scalings
- primary benefit is for the business hours bin aggregator
- note pandas accepts kind of non-intuitive abbreviations for time-zones consistent with pytz.all_timezones

5.43
- new mxab normalization option for max absolute scaling
- which quite simply divides data by max absolute value found within training set
- which returns a range not to exceed -1:1
- just trying to be comprehensive in normalziaiton options
- have seen this normalizaiton procedure referenced in a few places
- including Spark by Chambers Zaharia and ML Design Patterns by Lakshmanan et al
- just finally got around to it
- still recomend z-score normalization as default under automation for unknown distributions

5.44
- found an opportunity for edge case troubleshooting printout
- associated with cases where a transformation function records a transfomation category in column_dict
- which doesn't have corresponding entry in process_dict
- now printout provides explanation
- note that in general Automunge includes prinouts for all identified potential error channels to support troubleshooting

5.45
- found a small oversight with auto ML options for ML infill
- realized the autogluon and catboost implementations were missing the random seeding
- from randomseed parameter passed to automunge
- so went ahead and incorproated random seeding into fit / model initialization
- in general we try to use a consistent random seed for all methods based on this parameter

5.46
- found and fixed a small issue where postmunge inversion was overwriting one of entries in postprocess_dict
- associated with entry for postprocess_dict['finalcolumns_train']
- which was creating an edge case associated with numpy inversion in cases where Binary transform applied
- also two other mostly immaterial revisions to prevent postprocess_dict overwrites during postmunge
- associated with entries for traindata and infilliterate

5.47
- a review of the noise injection transforms identified a point inconsistent with code comments / readme description
- specifically in DPmm and DPrt the scaled noise is subject to a cap on outliers at +/- midpoint of range
- to ensure returned range consistent with scaled input
- realized the implementation had instead of capping noise subjected outliers to infill
- so reverted to capped outliers to be consistent with documentation
- note that this is not expected to have any material impact on experiment results from Numbers Game paper
- since the noise profile of those experiments had standard deviations well below the cap threshold

5.48
- a review of the noise injection transform DPod identified an opportunity for cleaner code
- by replacing a for loop through activations to a single operation performed in parallel 
- much cleaner this way, should be more efficient

5.49
- a small improvement
- added an entry to the postreports_dict reports returned from postmunge
- now includes details of row counts that served as basis for drift stats
- including row counts from automunge train set and postmunge test set
- may be a little helpful for quickly running sanity check on validaty of drift stats

5.50
- a quality control audit performed on returned data types from ordinal transforms
- turns out we had a few inconsistent approaches
- where base ordinal transforms ordl and ord3 set the returned data type as a function of size of encoding space
- (ranging from uint8 / uint16 / uint32)
- and a few of the other ordinal transforms did not include these conditional types
- so went ahead and updated returned data types for transformation functions pwor / bnwo / bneo / bkt3 / bkt4 to be conditional based on size of encoding space
- also updated wkds transform to set data type as int8
- also small cleanup to correct a few MLinfill types associated with spl2 transform (from singlct to exclude)
- note this isn't expected to change operation for any current family tree configurations, just trying to keep everything consistent

5.51
- was doing some research and apparently pandas to_numpy() is newer / more consistent approach to converting pandas to numpy as opposed to .values
- so went ahead and updated the treatment to returned sets to be based on .to_numpy()
- this is relevant to cases where pandasoutput=False, which has been the default
- and yeah something I've been mulling over for a very long time is whether defaulting to returning numpy arrays is best approach
- scikit likes numpy arrays, but as far as I can tell almost all other frameworks prefer pandas dataframes for tabular
- so made the executive decision to change default for pandasoutput parameter from False to True
- which means returned sets are now pandas dataframes by default
- and to otherwise return numpy arrays can designate pandasoutput=False
- also updated hash transform to make returned data types condiitional based on size of encoding space

5.52
- new root transformation category or23
- or23 is inspired by the experiments conducted in String Theory paper
- and is an alternative to or19 that makes use of sp19 instead of chains of spl2
- with an upstream UPCS and sp19 supplemented by nmcm and ord3

5.53
- a slight cleanup to the featureimportance report returned from automunge
- now the returned featureimportance report includes both the sorted results as well as the raw data serving as basis
- (where prevously had the sorted results only saved in postprocess_dict which was admittedly kind of not user friendly)
- so yeah feature importance results now all aggregated in single location for ease of reference

5.54
- a comprehensive audit of use of .values
- which was used in several places to convert dataframes to arrays
- in some cases it turns out more appropriately than others
- so yeah was able to strike a significant number of instances
- and replaced remainder with .to_numpy()
- now much more inline with pandas recomended practice
- also updated logic tests for conditional data types to be a little more precise

5.55
- found a sort of unneccesary for loop that was part of central flow
- replaced with extracted data structure negating the need for for loop
- should make everything run a little quicker
- especially with high density category assignments

5.56
- found and fixed a few small bugs in catboost wrapper

5.57
- for assignnan missing data injections, we were basing on a random seed
- which was causing concurrent appplications to duplicate row configurations
- which was unintentional
- so struck the random seedings from these injections
- resulting in a random random seed between columns
- (missing data injections are supporting some experiments and demonstrations)
- also found the onh2 root category was redundant with onht
- so struck onh2 from library and read me

5.58
- a small bug in assignnan missing data injections found and fixed
- (realized injections were being redundantly applied instead of once per target)
- also found a poorly executed edge case disparity between automunge and postmunge
- associated with null category
- which is for columns that are simply deleted
- such as is default for training data sets containing all nan values
- so conformed treatment to be consistent with other transformation categories
- now null columns are deleted in circle of life function instead of special treatment
- also updated the column_map report returned in postprocess_dict
- so that it now contains entries for source columns that did not return any sets
- also small tweak to align flow for master infill functions for automunge and postmunge

5.59
- fixed imports for AutoGluon ML infill option
- as it appears they recently revised some of their imports for tabular

5.60
- new default for automunge(.) randomseed parameter
- now set as False signalling application of a random random seed
- otherwise can still pass as an integer for specific desired random seed

5.61
- a small tweak to last rollout
- further testing demonstrated that randomseed range limited to 0:2**32-1
- (we had implemented range of 0:10**12)
- causing a bug when seed sampled in upper range
- found and fixed
- also moved the randomseed initialization to a slightly more approrpiate location in flow

5.62
- new convention for user passed transformdict
- now user is able to pass partially populated family trees
- and any missing primitives are automatically added internally
- just requires a minimum of one populated primitive for each root category
- this makes user specification much cleaner / less typing
- also added a new validation for format of transformdict
- ensuring valid primitive spelling, data types, etc

5.63
- another iterative improvement to transformdict specification
- now primitives with single entry can have entry passed as a string
- instead of string embedded in list
- (in other words can omit list brackets for primitives with single entry)
- this is just a little more intuitive, can already do this in assigncat and assigninfill
- also another validation added for data types of primitive entries

5.64
- found and fixed bug associated with trainID_column parameter
- associated with cases when passed as a list of columns
- used as an opportunity for some cleanups to ID / index column processing
- in both automunge and postmunge
- including some code comments
- and reductions of redundancy
- ID code portion is a little more legible now I think

5.65
- found and fixed a bug introduced in 5.61 associated with feature selection
- just needed to move the randomseed initialization a few lines up
- also found a new edge case for parsenumeric NArowtype
- fixed by converting entries to strings for parsing operation
- and in the process realized that could consolidate three of NArowtype categories into a single type
- so now NArowtypes 'parsenumeric', 'parsenumeric_commas', and 'parsenumeric_EU'
- are all grouped together as NArowtype 'parsenumeric'
- which registers NArw activations when numeric characters aren't present in an entry

5.66
- a simplification of the index column populated in ID sets
- now index column header string defaults to 'Automunge_index'
- with exception of reverting to the previous convention 'Automunge_index_###' 
- (where ### is the 12 digit random integer associated with the application)
- for remote edge case with a column with header 'Automunge_index'
- is already found in carved out ID columns
- this should make the ID sets a little more user friendly 
- with a index header now known in advance

5.67
- revisiting defaults for cv scoring in grid search hyperparameter tuning
- for ML infill hyperparameter tuning in cases of classification
- reverting performance metric from f1 score back to accuracy
- (the problem with f1 is if folds split doesn't have fully represented activations triggers printouts)
- iterate, iterate, and then iterate some more

5.68
- new Q Notation family of transforms available as qbt1 / qbt2 / qbt3 / qbt4
- where encoding is to binary with seperate registers for integers, fractionals, and sign
- transforms accept parameters suffix / integer_bits / fractional_bits / sign_bit
- parameters designate the qubit capacity of each register, 
- defaulting for qbt1 to {'sign_bit' : True, 'integer_bits' : 3, 'fractional_bits' : 12}
- (the defaults are arbitrary representing a compact register for range +/- 8.0000)
- qbt2 is for signed integers defaulting to {'sign_bit' : True, 'integer_bits' : 15, 'fractional_bits' : 0}
- qbt3 is for unsigned floats defaulting to {'sign_bit' : False, 'integer_bits' : 3, 'fractional_bits' : 12}
- qbt4 is for unsigned integers defaulting to {'sign_bit' : False, 'integer_bits' : 15, 'fractional_bits' : 0}
- and with suffix corresponding to category key for each
- the expectation is in many workflows users may wish to deviate from default register counts, these are just starting points
- register sizes were selected to accomodate z-score normalized data with +/-6 standard deviations from mean and approx 4 sig figures in decimals
- requiring 16 qubits in base qbt1 configuration for signed floats
- missing data and overflows default to zero infill
- if markers are needed for missing data can turn on NArw_marker parameter
- (NArw won't pick up overflow cases, so care should be taken for adequate register size)
- for example, with default parameters an input column 'floats' will return columns:
- ['floats_qbt1_sign', 'floats_qbt1_2^2', 'floats_qbt1_2^1', 'floats_qbt1_2^0', 'floats_qbt1_2^-1', 'floats_qbt1_2^-2', 'floats_qbt1_2^-3', 'floats_qbt1_2^-4', 'floats_qbt1_2^-5', 'floats_qbt1_2^-6', 'floats_qbt1_2^-7', 'floats_qbt1_2^-8', 'floats_qbt1_2^-9', 'floats_qbt1_2^-10', 'floats_qbt1_2^-11', 'floats_qbt1_2^-12']
- inversion also supported
- excluded from ML infill for now
- Q notation was inspired by discussions in "Programming Quantum Computers: Essential Algorithms and Code Samples" by Eric R. Johnston, Nic Harrigan, and Mercedes Gimeno-Segovia

5.69
- an extension of the Q Notation transforms rolled out in 5.68
- now with new root categories nmqb, nmq2, mmqb, mmq2
- which incorproate an upstream noramlization before the binarization
- where nmqb has upstream z score to qbt1 and z score not retained
- nmq2 has upstream z score to qbt1 and z score is retained
- mmqb has upstream min max to qbt3 and min max not retained
- mmq3 has upstream min max to qbt3 and min max is retained

5.70
- a little quality control on Q Notation transform
- updated convention for overflow entries
- now instead of replacing with 0 replace with max/min capacity based on register count
- also updated the overflow capacity calculation to include fractionals register
- added inplace support (sort of a compromise of inplace, results in one fewer copy operation)
- small consolidation to single variable for new column header representation
- small code comment cleanups

5.71
- revisiting the hashing family of transforms
- reworked some of the pandas code resulting in about 20% speedup
- every little bit helps

5.72
- improvements from hash functions carried through to a few more misc snippets
- should make everything run a little quicker here and there
- also added new parameter support for hs10
- parameter excluded_characters defaults to [] as empty list
- can pass a list of strings to scrub from entries (e.g. punctuations etc)

5.73
- performed an audit of the insertinfill function
- found and fixed a small but impactful bug that was interfering with ML infill for multi-column categoric sets
- also replaced a somewhat inellegant case of replace operation embeded in a where operation, now only replace is called
- also found and fixed a bug interfering with applicaiton of mode infill to binarized sets

5.74
- found an opportunity to reduce the memory overhead of postprocess_dict
- by eliminating a redundancy in stored trained models for ML infill
- now for multi-column sets the trained model is only saved once
- instead of once for each column
- the reduction in memory overhead will vary by auto ML library
- in some cases savings may be substantial
- also eliminated a few legacy imports (no longer used or redundant)

5.75
- new MLinfilltype option 'ordlexclude'
- intended for hashed encodings
- 'ordlexclude' is excluded from infill
- purpose is to include hashed sets in ordinal set of columntype_report
- also updated MLinfilltype for hs10 (binary encoded hashed sets) to 'boolexclude'

5.76
- important update
- new defaults for two automunge(.) parameters
- MLinfill = True, NArw_marker = True
- based on findings from paper Missing Data Infill with Automunge 
- (new paper revision following later today)
- resulting in ML infill on by default 
- supplemented by NArw support columns with boolean integers signalling presence of infill
- settings are expected to improve downstream model performance in presence of missing data

5.77
- small efficiency improvement to postmunge
- now columns are excluded from assembling NArows support columns
- when not needed for assigned infill types
- (the aggregated internal masterNArows dataframe is seperate from support columns appended to train set)
- this will speed up postmunge a little when not running ML infill / reduce memory overhead

5.78
- found an opportunity to speed things up a tad
- many instances of searching in list
- now replaced with searching in set
- which runs much faster
- relatively speaking

5.79
- for oversampling, small update to levelizer function
- added support for oversampling in numeric labels
- based on supplemented ordinal encoded bins
- (previously this was only supported for supplemented one-hot encoded bins)

5.80
- update to __init__ file to simplify import procedure
- now can import AutoMunge directly instead of having to access from Automunger.py
- now imports recomended as
from Automunge import AutoMunge
am = AutoMunge()
- previous version of imports still work as well

5.81
- another small tweak to imports in __init__ 
- to avoid potential for overlaps with external naming space

5.82
- revert changes from 5.81

5.83
- apparently missed the memo that scikit started accepting pandas input way back in 0.20
- so removed a numpy conversion in ML infill for random forest scenario
- in my defense this project started back in 2018
- all good

5.84
- new root category lgnr
- for logarithmic number representation of numeric sets
- including seperate column bitwise registers for 1 sign, 4 log integer, and 3 log fractional
- can increase or decrease register counts by overwriting processdict defaultparam entry for qbt5
- I'm not sure how useful this will be, just a new way to represent numbers trying to be thorough
- inspired by William Dally's talk at MLSys Chips and Compilers Symposium

5.85
- put some additional thought into lgnr transform set
- realized an additional sign register is needed
- one for before ln transform and one for after
- so lgnr now returns 9 bitwise registers instead of 8
- (actually 10 including the NArw aggregation for missing data markers)
- also added inversion support for lgnr, with partial recovery
- (input sign is not recovered for esoteric reasons)
- in other words, under inversion negative values are returned as positive

5.86
- lgnr now has full inversion support (including sign retention)
- lgnr now registers NArw missing data marker for zero entries

5.87
- Now when passing a processdict entry to overwrite an internally defined processdict entry, you can pass the functionpointer to point to itself, and then only have to populate the entries you are overwriting.

5.88
- a housekeeping cleanup to processing function naming conventions
- had included the suffic '\_class' dating back to very earliest experiments
- in hindsight this may have potential to be a point of confusion
- so scrubbed that suffix
- processing function naming now follows convention process_#### / postprocess_#### / inverseprocess_####
- where #### is the transformation category returned in column_dict
- much cleaner this way
- also found a potential edge case channel for inconsistent processing between train and test associated with NArw aggregation for infill
- originating from NArow assessment overwriting entries for NArowtypes positivenumeric, nonzeronumeric, nonnegativenumeric
- cleaned that up, issue resolved

5.89
- update to convention for returned sets
- now single column pandas sets are returned as series instead of dataframes
- this decision was based on conventions of some downstream libraries for receiving labels
- kind of like how numpy arrays need to be flattened with ravel
- also small tweak to NArw update from last rollout to reduce memory overhead

5.90
- a big simplification to label set encodings under automation
- realized had accumulated too many scenarios, this way much clearer
- now quite simply, numeric data is given pass-through (no normalization), categoric data is given ordinal encoding (alphabetical sorted encodings)
- other label encoding options documented in new section in library of transformations in read me
- also small bug fix in feature selection originating from new convention of single column sets returned as series

5.91
- inspired by the success of 5.90, a further simplificaiton to categoric defaults under automation
- now removed a kind of weird singluar scenario for training data sets with 3 unique entries which were treated with one-hot encoding
- and instead treated them to binarization consistent with other categoric sets
- also increased defaults for numbercategoryheuristic from 127 to 255
- (numbercategoryheuristic is the size of unique value counts beyond which sets are treated to hashing instead of binarization under automation)
- 255 unique values returns an 8 column binarized set (1 activation set is reserved for missing data)
- this update does not impact backward compatibility

5.92
- two new categoric encoding options incorporated into library
- with transformation categories 'smth' and 'fsmh'
- these borrow from the label smoothing options previously available for label sets
- and allow them to be applied to training data categoric encodings in addition to labels
- accepts parameter 'activation' to designate the value for activations
- as float between 0.5-1, defaults to 0.9
- smth applies a one-hot encoding followed by label smoothing operation
- fsmh applies a on-hot encoding followed by a fitted label smoothing operation
- where fitted smoothing refers to fitting the null values to activation frequency in relation to current activation
- more info on label smoothing and fitted smoothing noted in essay "A New Kind of ML"
- (we still recomend the prior label smoothing parameters for target categoric labels in order to distinguish between smoothing as applied to train / test / validation sets)
- inversion supported with full recovery
- also found and fixed a small bug in fitted label smoothing

5.93
- with the new smth family of transforms rolled out in 5.92, found opportunity to decrease the number of parameters for simplicity
- so automunge(.) parameters LabelSmoothing_train / LabelSmoothing_test / LabelSmoothing_val / LSfit are now deprecated
- as are postmunge(.) parameters LabelSmoothing / LSfit
- replaced by the transformation categories smth and fsmh
- (where smth is vanilla label smoothing and fsmh is fitted label smoothing)
- the new convention for smth and fsmh is that smoothing is only applied to training data
- so in automunge(.) valiadation sets and test sets are not smoothed
- and in postmunge(.) smoothing can optionally be applied by activating the traindata parameter
- the only tradeoff is that oversampling no longer supported for smoothed labels on their own, requires supplementing smoothing transform with a categoric like one-hot or ordinal
- this update results in a material simplification of code base surrounding label processing
- much cleaner this way
- also new root categories lbsm and lbfs equivalent to smth and fsmh but without the NArw aggregation intended for label sets

5.94
- inspired by the reduction in parameters of 5.93
- took a look at other parameters and found another opportunity to consolidate
- for simplicity
- so automunge(.) parameters featureselection / featurepct / featuremetric / featuremethod
- are now replaced and consolidated to featureselection / featurethreshold
- with equivalent functionality
- featureselection defaults to False, accepts {False, True, 'pct', 'metric', 'report'}
- where False turns off feature importance eval, 
- True turns on
- 'pct' applies a feature importance dimensionality reduction to retain a % of features
- 'metric' applies a feature importance dimensionality reduction to retain features above a threshold metric
- and 'report' returns a feature importance report with no further processing of data
- featurethreshold only inspected for use with pct and metric
- accepts a float between 0-1
- eg retain 0.95 of columns with pct or eg retain features with metric > 0.03
- so to be clear, automunge(.) parameters featurepct / featuremetric / featuremethod are now deprecated
- replaced consolidated to parameters featureselection / featurethreshold

5.95
- corrected a small typo from 5.94 missed in testing associated with backwards compatibility for deprecated parameters
- (the updated feature selection parameters still inspect the deprecated versions for now, although have stricken reference from documentation)

5.96
- added support for passing df_train and df_test as pandas Series instead of DataFrame
- new parameter for smth family of transforms (label smoothing transforms)
- boolean 'testsmooth', defaults to False, when True smoothing is applied to test data in both automunge and postmunge
- also updated family tree for label smoothing root categories lbsm and lbfs
- now when passing parameter through assignparam can pass directly to root category

5.97
- inspired by the new label smoothing parameter from 5.96
- added a new parameter to DP family of noise injection transforms
- DP family by default injects noise to train data but not to test data
- previously noise could optionally be injected to test data in postmunge by the traindata parameter
- now with new 'testnoise' parameter, noise can be injected to test data by default in both automunge and postmunge
- testnoise defaults to False for no noise injected to test data, True to activate

5.98
- new family of transforms for categoric encodings
- maxb (ordinal), matx (one-hot), ma10 (binary)
- for scenario where user wishes to put a cap on the number of activations
- such that any of following assignparam parameters may be passed
- maxbincount: set a maximum number of activations (integer)
- minentrycount: set a minimum number of entries in train set to register an activation (integer)
- minentryratio: set a minimum ratio of entries in train set to register an activation (float between 0-1)
- parameters default to False for inactive
- parameters may be passed in combination for redundant specifications if desired
- in each case consolidated entries are grouped in the top activation
- maxb transforms are each performaed downstream of an ord3 (ordinal sorted by frequency)
- and matx performed upstream of a onht (one hot encoding) and ma10 performed upstream of a 1010 (binary encoding)

5.99
- new parameter inplace available for both automunge(.) and postmunge(.)
- defaults to False
- when True the df_train and df_test passed to automunge(.) or postmunge(.)
- are overwritten with the returned train and test sets
- inplace reduces memory overhead since don't have to redundantly hold data set in ram

6.0
- small iteration associated with the new inplace parameter
- incorporated inplace into parameter validations
- also revised inplace to default to off when incorrect format passed

6.1
- new auto ML option for ML infill by FLAML library
- available by setting ML_cmnd = {'autoML_type':'flaml'}
- parameters can be passed to fit operation by e.g.
```
ML_cmnd = {'autoML_type':'flaml', \
           'flaml_classifier_fit'  :{'time_budget' : 5}, \
           'flaml_regressor_fit'   :{'time_budget' : 5}}
```
- where time_budget specifies max seconds training each feature
- parameters default to verbose = 0
- FLAML library courtesy of Microsoft team
- Chi Wang, Qingyun Wu, Markus Weimer, Erkang Zhu, "FLAML: A Fast and Lightweight AutoML Library"

6.2
- important: backward compatibility breaking update
- reduced the number of sets returned from automunge(.) and postmunge(.)
- automunge returned sets reduced from 17 to 10
- postmunge returned sets reduced from 5 to 4
- function calls now look like:
```
train, train_ID, labels, \
val, val_ID, val_labels, \
test, test_ID, test_labels, \
postprocess_dict = \
am.automunge(df_train)


test, test_ID, test_labels, \
postreports_dict = \
am.postmunge(postprocess_dict, df_test)
```
- also deprecated automunge(.) parameters valpercent1, valpercent2
- replaced with / consolidated to valpercent
- so functions now only return a single validation set when elected
- (Had been so focused on retaining compatibility of tutorials published years ago lost sight of design principles. Came to my senses.)

6.3
- small bug fix for inversion applied to subset of columns
- originating from parameter validation function
- for when passing postmunge inversion parameter as a list of columns

6.4
- a few small cleanups associated with validation sets
- removing code relics relevant to deprecated valpercent2

6.5
- an update to infill application in automunge and postmunge
- now order of infill application is based on a reverse sorting of columns
- sorted by a count of a column's missing entries found in the train set
- this convention should be beneficial for ML infill
- as columns with most missing will have improved coherence for serving as basis of other columns
- postmunge order of infill consistent with order from automunge
- this update inspired by a similar convention applied in the MissForest R package

6.6
- sort of a housekeeping cleanup
- read somewhere that is good python practice to use underscore in function naming
- where leading underscore indicates functions intended for internal use
- and no leading underscore for external use
- so performed a global update to all support functions to include leading underscore
- only functions without underscore for external use are automunge(.) and postmunge(.)
- also small cleanup to remove unused variable in infill application

6.7
- new 'infill' option for powertransform parameter
- intended for cases where data is already numerically encoded
- and user just wants to apply infill functions
- follows convention of deleting feature sets with no numeric entries in train set
- applying 'exc2' if any floats are present or all integer numeric entries and unique ratio > 0.75
- else applying 'exc5'
- where exc2 is passthrough numeric with infill for nonnumeric entries
- and exc5 is passthrough integer with infill for non integer entries
- also updated exc2 and exc5 family trees to include NArw when NArw_marker parameter activated
- created backup trees exc6 and exc7 for exc2 and exc5 without NArw
- also a few small cleanups to evalcategory support function

6.8
- new 'integer' mlinfilltype available for assignment in processdict entries
- integer is for transforms that return single column integer sets
- and differs from 'singlct' in that singlct is for ordinal encodings subject to ml infill classificaiton
- while integer is for continuous integer sets subject to ml infill regression
- in ml infill the imputation predictions after regression inference are subject to a rounding
- to conform to the integer form of other entries
- where rounding applies up or down to nearest integer based on 0.5 decimal midpoint
- recast transformation categories lngt and lnlg as integer mlinfilltype
- new exc8 and exc9 for passthrough integer sets with integer mlinfilltype (exc8 includes NArw option)
- update to the powertransform 'infill' option in scenario of integer sets above heuristic of 0.75 ratio of unique entries
- which are now applied to exc8 instead of exc2

6.9
- added new classification to columntype_report
- as 'integer' for continuous integer sets
- previously these were grouped into 'continuous'
- figured worth a distinction for clarity
- differs from ordinal which is a categoric representation
- small cleanup had some deprecated parameters in automunge(.) definition for backward compatibility
- associated with feature importance and label smoothing
- went ahead and struck for cleanliness purposes
- a very small cleanup replaced instance or two of searching within list to searching within set
- found a small opportunity for improved clarity of code
- by adding exclude mlinfilltype to the infill functions
- a few cleanups to read me:
- settled on single convention for '=' placement in function call demonstrations
- a few corrections and clarifications to mlinfilltype descriptions
- added a description of or23 to library of transformations writeup 
- finally, a big cleanup of a bunch of transformation category parameters
- for parameter adjinfill, which was to change default imputation for standard infill for a transformation category
- this was only included on a hunch and for running a few experiments
- which I think the ml infill validation experiments sufficiently demonsrtated adj not a good default convention
- so yeah, also one of those zen of python things, should be one and only one way to do it

6.10
- important update
- found and fixed a single character bug 
- in support function _convert_1010_to_onehot
- which I now know has been interfering with ML infill on binary encoded sets
- I expect this may result in an improvement to benchmarking experiments for performance of ML infill to categoric targets
- going to re-run the missing data infill paper benchmarks
- updates to follow after re-training

6.11
- added some more detail to function description of _postprocess_textsupport
- regarding format of textcolumns entries needed for operation
- (this convention was source of bug fixed in 6.10)
- new parameter supported for tlbn transform
- parameter 'buckets' can pass as list of bucket boundaries
- otherwise defaults to False
- leave out -/+ inf for first and last bins those will be added
- buckets is an alternative to bincount for number of equal population bins
- buckets, when specified, takes precendence over bincount
- allowing user to perform influence evaluation on custom segements of feature set distribution
- also changed mlinfilltype for tlbn from concurrent_nmbr to exclude
- after realizing that ML infill was messing up -1 out of range convention
- much cleaner this way, for tlbn missing data is just grouped with out of range bucket for entire set 

6.12
- fixed a feature importance bug introduced in 6.9
- associated with removal of some deprecated parameters
- reverted tlbn mlinfilltype from exclude to concurrent_nmbr
- which is needed for use in feature selection
- added some comments to predictinfill of potential extension for tlbn
- fixed a tlbn postmunge bug introduced in 6.11
- associated with accessing normalization_dict in postprocess function
- also added inversion support for tlbn

6.13
- quick fix
- found a bug in both DPmm and DPrt that was introduced in 6.6
- all better

6.14
- added a new preset to AutoGluon ML infill option
- to use 'optimize_for_deployment' which reduces required disk space
- applied by default unless user activates best_quality
- appropriate since user doesn't need auxiliary functionality, models are just used for inference

6.15
- ok trying to get ahead of a new edge case for tlbn transform from 6.11
- for cases when transform is being performed multiple times in the same family tree
- such as might be applied redundantly with different bincounts or different distribution segments
- previously we inspected the bincount in postprocess function to confirm the correct normkey
- where normkey is the column header with suffix used to access normalization_dict entries
- however now we have optional alternate buckets parameter that may take precendence over bincount
- so went ahead and added buckets to the normkey derivation, meaning now postprocess fucntion inspects both
- which resolves the edge case
- also added both of these parameters to the _check_normalization_dict validation function
- which is a support function that confirms any required unique normalization_dict entry identifiers are so
- (we have a handful of processing functions that make use of similar methods to derive a normkey)
- also added a few code comments here and there for clarity

6.16
- ok last rollout helped me recognize kind of a flaw
- which was the requirement for reserved strings in normalization_dict
- for use to derive a normkey in a few postprocess functions
- in cases where don't know returned columns or might return an empty set
- so revisited the columnkey parameter passed to postproces functions
- and repurposed from use as a string header returned from upstream transforms
- to now constituting a list of headers returned from all transforms with same recorded category applied to an input column
- (that is same recorded category as returned in the process function column_dict)
- this greatly simplifies the derivation of a normkey in those postprocess functions where needed
- and as a side benefit eliminates the need for reserved strings in keys of normalization_dict
- thus completely eliminating a channel for error in user defined transformation functions
- to implement had to make a few tweaks to processfamily functions as well as a new required process_dict entry
- new process_dict entry 'recorded_category' details the category recorded by a transformation function in column_dict
- which may be different than the associated category populated in family tree 
- since the same transformation function may be assigned to multiple process_dict category entries
- also a new postproces_dict data structure as columnkey_dict
- which is populated for each applied transformation as
- {inputcolumn : {recorded_category : categorylist_aggregate}}
- where categorylist_aggregate is assembled as a list of all headers returned from transforms with the same recorded_category to the same inputcolumn
- (as may constitue columns returned from redundant applications like in cases where same trasnform is redundaantly applied to same input column but with different passed parameters) 
- this categorylist_aggregate is then passed to the postprocess functions as the columnkey
- which can be used to derive a normkey without need to validate required unique keys of normalization_dict
- the tweaks to the processfamily functions were twofold
- in automunge processfamily functions we now populate entries to postprocess_dict['columnkey_dict'] with each column_dict_list returned from processing function application
- making use of the new support function _populate_columnkey_dict
- and then in postmunge postprocessfamily functions prior to calling the postprocess functions access the categorylist_aggregate to pass as a columnkey from the columnkey_dict
- using the inputcolumn to be passed to postprocess function and the recorded_category derived from the category populated in family tree's process_dict entry
- updated the 13 postprocess functions in library that previously used the required unique normalization_dict entries to derive a normkey
- to the new simplified convention available by use of the categorylist_aggregate passed to the columnkey parameter
- also updates to support function that validates the process_dict to account for new recorded_category entry
- and support function to apply functionpointer entries to user passed processdict to account for new recorded_category entry

6.17
- ok last rollout was a great step in right direction
- but still fell a little short of fully standardized method to extract a normkey in postprocess functions
- realized that by revising the columnkey_dict population from basing on recorded category
- to basing on transformation category (as populated in family tree)
- then had ability to eliminate need to inspect any transformation parameters in postprocess transforms
- which were previously compared to ensure in cases of redundant transformations to same inputcolumn we were accessing version with right parameters
- now since cases of redundant transfomations to same input column won't be populated in the same categorylist_aggregate entry to a column_key_dict entry associated with a transformation category
- we've eliminated the scenario requiring inspection of parameters
- note that we already had the convention that a transformation category can only be entered once in a set of upstream or downstream parameters (although may be entered in both if desired)
- which is validated in the _check_transformdict2 support function
- this update is a great step in standardizing on form of postprocessing functions
- updated to new convention of accessing normkey in postprocess transformations for onht/text/splt/sp19/sbst/sbs3/hash/pwrs/bnwd/bnep/tlbn/bkt1/bkt2/smth/spl2/srch/src2/src3
- we intend in future updates to take advantage of this to fully standardize on postprocessing transform normkey retrieval accross the entire library
- and also to eliminate inspection of transformation parameters in postmunge which will clear up a little overhead to speed things up
- also small cleanup struck a postmunge columnkey variable initialization no longer needed after 6.16

6.18
- added inplace parameter to processing function normalization dictionaries
- which ommission was kind of a shortcut in the first place, better to have it accessible
- standardized on single convention for suffix parameter regarding inclusion of underscore
- such that transforms that accept a suffix parameter should pass just the (traditionally 4 character) string and a leading underscore will be added internally
- standardized on single form of postprocess functions with respect to accessing normkey
- in the process found a channel for error previously missed
- for cases when a transform returns an empty set and was selected for inplace
- as without a returned set the inplace operation doesn't replace the input column
- resolved by standardizing on inspection of inplace param in empty set scenario
- which means we will need to retain param inspections in postmunge after all
- no big deal is not a significant amount of overhead
- also found and fixed a small edge case for Binary dimensionality reduction when applied to an all numeric set (which is a pass through)
- a few small cleanups and code comments for support function _postprocess_textsupport
- update to the demonstration of custom postprocess function definition in read me

6.19
- new convention for the library
- now every transform supports the 'suffix' parameter
- which can be passed as a (traditionally 4 character) string
- which will be appended on the returned column(s) with a leading underscore
- previously this parameter was only supported in a selection of transforms
- where without the parameter the suffix was a hard coded property of a transformation function
- the benefit of the convention is to support use case where user wishes to redundantly perform the same transform on the same input column but with different parameters
- as an example to conduct redundant bin aggregators with different boundaries
- the only requirement to facilitate is that each of the redundant applications have the transformation function associated with a distinct transformation category and a distinct suffix parameter entry
- where the suffix parameter could be part of the category's processdict definition by use of a defaultparams entry
- such that each of the distinct transformation categories can be entered into the root category family tree
- only exception to this convention is for the text and excl transforms which have some quirks with suffix conventions
- updated the pwrs transform suffix convention to support suffix parameter
- fixed validation function printout associated with new recorded_category processdict entry introduced in 6.16
- found and fixed a small formatting snafu in splt and sbst transforms associated with int_headers option

6.20
- ok just realized that the column_map had an edge case
- associated with the excl transform
- where excl is direct pass-through, and is unique in library for suffix convention
- in that suffix is recorded in internal data structures to support operation
- and then removed in the returned dataframe so that returned column title is consistent with recieved
- which makes sense since excl is for use on direct pass-through with no infill
- (where if excl suffix retention is desired in returned data to support data structure navigation the excl_suffix parameter can still be activated)
- anyway point was that the column_map for excl columns was showing the internal representation with suffix even when returned data did not include
- so new convention is that excl column suffix convention in column_map is consistent with returned data

6.21
- sort of a quality control audit / additional walkthrough of edits from last week
- found and fixed bug in pwr2 inversion function originating from new suffix convention from 6.19
- everything else looked good
- a few code comments added here and there
- removed the '\' character in family tree definitions since shown in read me
- and since went that far went ahead and conformed the process_dict intitializer to match

6.22
- some additional quality control conducted
- performed a visual inspection on entire inversion library
- in the process identified a few small snafus
- bnep/bneo/tlbn -> found and fixed inversion edge case associated with all non-numeric data in train set
- splt/sbst -> found and fixed inversion bug originating from new suffix parameter support from 6.19
- in the process developed a new validation test for inversion library to be performed prior to rollouts
- which should help catch edge cases going forward for inversion

6.23
- struck an unneccesary numpy conversion in PCA application
- revisited a heuristic in place for PCA application associated with identifying cases when didn't have enough samples for the number of features (aka the col_row_ratio)
- since this was previously struck in documentation went ahead and initialized variable so it isn't inspected in ML_cmnd
- but left the code in place in case later decide to reintroduce
- improved a convention used in _postprocess_DPod from accessing an upstream normalization_dict to accessing own (just needed to add an additional entry to DPod normalization_dict)
- conducted audit of returned data types from entire library
- where convention is returned continuous sets have data type not addressed in transformation function but are converted based on the floatprecision parameter externally
- boolean integer sets are cast as np.int8
- and ordinal encoded integer sets are given a conditional data type based on size of encoding space as either uint8, uint16, or uint32
- found a few transforms needing update to dtype convention
- recast bsor from int8 to conditional (since bin count is customizable by parameter)
- added missing data type casting for DPbn (int8) and DPod (conditional)
- added some documentation to the read me associated with returned data types in section on automunge(.) returned sets and postmunge(.) returned sets
- then just a few opportunities to consolidate transformations to reduce lines of code:
- consolidated bnry and bnr2 to use of a single comnmon transformation function by adding parameter for infillconvention
- consolidated MADn and MAD3 to use of a single common transformation function by adding parameter for center
- consolidated transformation functions for shft, shf2, and shf3 into a single common trasnformation function (taking advantage of what is now standard accross library allowing redundant transformations to common input column applied with different parameters)

6.24
- updated returned data type convention for exc5 and exc8 which are for integer passthrough
- exc5 has mlinfilltype singlct as intended for encoded categoric sets
- exc8 has mlinfilltype integer as intended for continuous integer sets
- previously these were being returned as float data types based on floatprecision designation
- recast to set data type in the transformation, making use of new assignparam parameter integertype
- where integertype defaults to 'singlect' and can also be passed as 'integer'
- where for 'singlct' returned data type is conditional uint based on max in train set which is used as proxy for encoding space, which is the default for exc5
- and for 'integer' returned data type is int32, which is the default for exc8
- updated the logic test for powertransform = 'infill' to distinguish between assigning exc5 and exc8
- previously exc5 was default for integer sets unless for train len unique set > 75% number of rows then cast as exc8
- now new scenario added to exc8 for cases where any negative integers found in train set allowing us to infer set is a continuous integer
- new convention for floatprecision conversion to now only be applied to columns returned from transforms with MLinfilltype in {'numeric', 'concurrent_nmbr'} or columns returned from PCA
- instead of previous convention's use of a type check for float
- which in practice means that now excl passthrough columns will retain their data type from received data instead of defering to floatprecision
- moved the application of floatprecision conversion in automunge and postmunge workflow to take place prior to excl suffix extraction to accomodate the MLinfilltype check
- which as side benefit means don't have to worry about privacy encodings
- finally, an update to methods for populating the column_map report
- which now will include empty set entries for source columns that had all of their derivations consolidated in a PCA or Binary dimensionality reduction

6.25
- new set of validations performed for scenarios with model training
- including ML infill, feature selection, and PCA
- now prior to training data is validated to confirm all valid numeric entries
- which is automatic for most transforms in library, but there are a select few that may return non-numeric or NaN values
- such as excl, copy, shfl, strg
- if validation doesn't pass results in printout message
- validation results are returned with other tests in postprocess_dict['miscparameters_results'] for automunge(.) 
- and postreports_dict['pm_miscparameters_results'] for postmunge(.)
- removed some automunge(.) initializations of variables that were no longer used (multicolumntransform_dict and LSfitparams_dict)
- a few code comments added, a few tweaks to printouts
- added inplace support for copy transform

6.26
- a refinement of the conditional data type conversions for ordinal encodings
- which again convert transformation outputs to one of uint8/uint16/uint32
- based on the size of the encoding space
- found that our logic test for selection was off by a few integers
- now will maximally utilize capacity of data types

6.27
- updated postmunge featureeval methods to include support for cases where automunge privacyencode was elected
- updated scikit random forest initializer to include support for parameters ccp_alpha and max_samples (which scikit introduced in 0.22)
- update tweak to pwor transform to circumvent some kind of strange interaction between np.nan serving as key to python dictionary

6.28
- performed a comprehensive audit of edge case printouts
- to ensure validation test results are being recorded in appropriate reports
- identified a few cases where validation checks were missing recorded results
- so added new entries to postprocess_dict['miscparameters_results']
- as trainID_column_valresult, testID_column_valresult, evalcat_valresult, validate_traintest_columnnumbercompare, validate_traintest_columnlabelscompare, validate_redundantcolumnlabels, validate_traintest_columnorder, numbercategoryheuristic_valresult
- and added new entries to postreports_dict['pm_miscparameters_results']
- as testID_column_valresult, validate_traintest_columnlabelscompare, validate_traintest_columnorder, validate_labelscolumn_string
- in the process developed some documentation defining various validation checks
- which am keeping as internal for time being

6.29
- improved function description in codebase for _evalcategory
- new validation result reported in miscparameters_results
- as suffixoverlap_aggregated_result which is single boolean marker aggregating all of the suffixoverlap_results into a single result
- removed a redundant parameter inspection in _madethecut
- found and offered accomodation to kind of a remote edge case where trainID_column passed as False and testID_column passed as True
- in which case we just revert testID_column to match trainID_column
- found and fixed edge case for Binary dimensionality reduction that was interfering with application of Binary to subset of categoric columns as specified by passing parameter as list
- added entry to postprocess_dict of Binary_orig
- assembled some new documentation detailing contents of postprocess_dict (keeping this as internal for now)
- also 6.23 had struck convention for applying a PCA heuristic
- which now realize was adequately covered in documentation
- as this heuristic is only applied when PCAn_components passed as None
- so went ahead and reintroduced option for heuristic to the code base

6.30
- important update: new convention for automunge trasnformation functions
- now process and singleprocess functions have an additional parameter as treecategory
- which is passed to the functions as the transformation category populated in the family tree
- which is the category serving as the key for accessing those functions in process_dict
- treecategory now serves as the default suffix for a transformation
- unless suffix otherwise specified in assignparams or defaultparams
- treecategory is also recorded in the column_dict entry for 'category'
- replacing the hard coded transformation category recorded there previously
- benefits of the update include that we can now tailor the inversion function applied
- to each category with a process_dict entry
- as opposed to prior convention where inversion function was based on recorded category in the function
- no updates needed to inversion to support, everything already works as coded, difference is that now inversion inspects the recorded treecategory process_dict entry to access inversion function instead of the hard coded recorded category associated with a transformation function
- another benefit is that methods that inspect mlinfilltype will no longer inspect mlinfilltype of the recorded category
- they will instead inspect mlinfilltype of the treecategory
- which means can now have multiple mlinfilltypes for a given transformation function as reorded in different process_dict entries
- another benfit associated with the updated suffix convention
- is that now the returned headers will display the categories that may serve as a key for assigning parameters in assignparam
- where previously a user might have to dig into the documentaiton to identify a key based on the family trees
- so went ahead and scrubbed (almost) all cases of suffix assignment in process_dict defaultparams
- as will instead defer to matching suffix to treecategory which is much less confusing
- another benefit is that we were able to strike use of process_dict entry for 'recorded_category'
- as well as scrubbed recorded_cateory from support functions in place to accomodate such as for functionpointer
- since now the recorded category will be the treecategory serving as key for the process_dict entry
- also added validation check for PCAexcl parameter to _check_am_mioscparameters
- result returned in miscparameters_results as PCAexcl_valresult
- also updated featurethreshold validation test to allow passing integer 0 instead of float 0. if desired
- finally small updates to postprocess_bxcx, and Binary dimensionality reduction, and inverseprocess_year to accomodate new default suffix convention associated with this update

6.31
- new 'silent' option for printstatus parameter to both automunge(.) and postmunge(.)
- new convention is that printstatus can be passed as one of {True, False, 'silent'}
- where True is the default and returns all printouts
- False only returns error message printouts
- and 'silent' turns off all printouts
- the thought is that there may be preference for a silent option
- for cases where Automunge incorporated as resource in script workflows outside of context of jupyter notebooks
- note that results of validations which may generate error message prinouts are generally included in the postprocess_dict['miscparameters_results'] or postreports_dict['pm_miscparameters_results']
- so validations are available for inspection even when all printouts are silent
- also updated the suffix overlap detection method for Binary dimensionality reduction to align with rest of library

6.32
- ok so 6.31's survey of printouts helped me recognize that 6.28's audit of validation results was incomplete
- (this was because have a few different convention for returned printouts with flag so didn't survey all at the time)
- so performed an additional audit of validation results printouts
- and found a few more cases of validation tests missing entries in the returned reports
- so postprocess_dict['miscparameters_results'] now contains additional entries for {ML_cmnd_hyperparam_tuner_valresult, returned_label_set_for_featureselect_valresult, labelctgy_not_found_in_familytree_valresult, featureselect_trained_model_valresult, assignnan_actions_valresult, labels_column_for_featureselect_valresult, featureselect_automungecall_validationresults}
- and postreports_dict['pm_miscparameters_results'] now contains additional entries for {labelscolumn_for_postfeatureslect_valresult, returned_label_set_for_postfeatureselect_valresult, labelctgy_not_found_in_familytree_pm_valresult, postfeatureselect_trained_model_valresult, postfeatureselect_with_inversion_valresult, labels_column_for_postfeatureselect_valresult, postfeatureselect_automungecall_validationresults}
- updated check_haltingproblem validation so that it only runs when there is a user defined transformdict to speed things up under default
- also new convention: labelctgy is now only an optional entry to user passed processdict 
- if one is not assigned or accessed from functionpointer then an arbitrary entry is populated from family tree
- removed a validation check and if statement indentation in support function _featureselect 
- associated with checking for cases where labels_column passed as False (this was redundant with a check performed prior to calling function)
- finally found and fixed bug for postmunge inversion in conjunction with Binary dimensionality reduction

6.33
- performed an audit of internal process_dict 'labelctgy' entries
- where labelctgy again is kind of only used in a few remote scenarios
- such as when a category is applied as a root category to a label set
- and has a family tree that returns labels in multiple configurations
- such that when that label set is a target for feature selection or oversampling
- the labelctgy entry helps distinguish which of the constituent sets will serve as target
- so yeah previous convention was that labelctgy was meant to point to the recorded category
- populated in column_dict by the transformation function returning set to serve as target
- turns out the 6.30 update where we revised the recorded category convention impacted labelctgy
- such that any place where the recorded category entry to labelctgy did not match the target treecategory
- required update to new convention
- so result is several updates to recorded labelctgy entries
- also in process identified that process_dict entries for 'time' and 'tmsc'
- did not have correspoinding transform_dict family trees defined
- which was kind of intentional since those are only used as support functions in other family trees
- but to be consistent with convention of rest of library, went ahead an defined some simple family trees

6.34
- corrected a parameter validation for ML_cmnd associated with a changed parameter naming convention from a while back
- (ML_cmnd['MLinfill_type'] was replaced with ML_cmnd['autoML_type'] and default was changed from 'default' to 'randomforest'
- in the transformdict check for infinite loops found and fixed a scenario where validation result wasn't being reported correctly
- another fix for Binary inversion associated with partial inversion including Binary columns when Binary passed to automunge(.) as True
- improved printouts for clarity for not supported edge case for partial inversion including Binary columns when Binary passed to automunge(.) as 'retain'
- (this edge case not needed since Binary columns will be redundnat with what can be recoved from inversion of rest of set)
- improved documentation for transformdict and processdict parameters in read me

6.35
- updated PCAexcl default from False to empty list [] to address an edge case with prior convention
- new default convention: boolean and ordinal columns are excluded from PCA unless otherwise specified in ML_cmnd
- new scenarios added to Binary dimensionality reduction as can be specified by the Binary parameter
- which original implementaiton was to consolidate categoric features into single common binarization
- either with or without replacement of the consolidated features
- which in cases of redundancies/correlations between features may result in reducing column count of returned set
- as well as help to alleviate overfit that may result from highly redundant features
- new scenarios are to instead of consolidate into a common binarization, will instead consolidate into a common ordinal encoding
- which may be useful to feed into a entity embedding layer for instance
- Binary='ordinal' => to replace the consolidated categoric sets with a single column with ordinal encoding (via ord3)
- Binary='ordinalretain' => to supplement the categoric sets with a single consolidated column with ordinal encoding (via ord3)
- so Binary already had convention that could pass as a list of column headers to only consolidate a subset instead of all categoric features, with first entry optionally as a boolean False to trigger retain option
- so added more special data types for first entry when Binary passed as list to accomodate ordinal options
- so now if Binary passed as list, Binary[0] can be True for 'ordinalretain', Binary[0] can be False for 'retain', Binary[0] can be None for 'ordinal', otherwise when Binary[0] is a string column header Binary is just treated as the default which is 1010 encoding replacing consolidated columns
- a few cleanups to Binary and Binary inversion implementation in the process
- finally, a small tweak to default transformation category under automation applied to numeric label sets
- reverting to a prior convention from the library of treating numeric labels with z-score normalization instead of leaving data unscaled
- have seen kind of conflicting input on this matter in literature on whether there is ever any benefit to scaling numeric labels
- did a little more digging and found some valid discussions on stack overflow offering scenarios where scaling labels may be of benefit
- as well as finding domain experts following the practice such as in implementation of experiments in paper  “Revisiting Deep Learning Models for Tabular Data” by Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko

6.36
- rewrote bxcx transformation functions to closer align with conventions of rest of library and for clarity of code
- new transform available as qttf
- which is built on top of Scikit-Learn sklearn.preprocessing.QuantileTransformer
- qttf supports inversion, inplace not currently supported
- qttf defaults to returning a normal output distribution
- which differs from scikit default of uniform
- the thought is in general we expect surrounding variables will closer adhere (by degree) to normal vs. uniform
- and thus this default will better approximate i.i.d. in aggregate
- alternate category available as qtt2 which defaults to uniform output distribution
- more info on quantile transform available in scikit documentation
- currently qttf is the only transform in library built on top of a scikit-learn preprocessing implementation
- and bxcx is only transform in library built on top of a scipy stats transformation
- (we also use scikit-learn for predictive models / tuning / PCA and scipy stats for distributions / measurement)
- the incorporation of a quantile transform option to library was partly inspired by a comment in paper “Revisiting Deep Learning Models for Tabular Data” by Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko
- also a few cleanups to imports for clarity of presentation
- struck numpy import for format_float_scientific which wasn't used
- struck some redundant scipy imports for statistics measurement
- a few other cleanups and clarifications to imports

6.37
- extensive walkthrough of automunge(.) and postmunge(.) functions
- several code comment cleanups and clarifications for both
- a few reconfigurations to order of operations for clarity
- struck some unused variables
- several improvements along these lines
- new validation test reported as familytree_for_offspring_result
- performed in _check_haltingproblem, _check_offspring
- familytree_for_offspring_result triggered when a transformation category is entered as an entry to family tree primitive with offspring but doesn't have it's own family tree defined for accessing downstream offspring
- gave some added aqttention to PCA implmentaiton which was kind of inelegant 
- a few cleanups and clarificaitons to PCA workflow in automunge
- fixed PCA heuristic (when PCAn_components passed as None) so that it can be applied with other PCA types other than default
- where PCA type is based on ML_cmnd['PCA_type']
- rewrote support function _evalPCA 
- now with improved function description and much clearer implementation
- found and fixed bug for returning numpy sets when pandasoutput set as False associated with an uninitialized variable
- removed placeholders for deprecated postmunge parameters LabelSmoothing and LSfit

6.38
- ok an additional audit found that 6.37 introduced a small bug in automunge label processing
- associated with inconsistent variable name in conjunction with category assignment to one of ['eval', 'ptfm']
- aligned label processing to use a single variable name
- also removed an unused variable for train data processing
- added code comment about process_dict inspection for label processing potential for future extension
- (somewhat similar to comment that was struck in 6.37)

6.39
- reconsidered treatment of one of primary edge cases in the library
- which is for the excl passthrough category (direct passthrough with no transforms or infill)
- we have convention that for excl transform a suffix is appended to populate data structures
- and then the suffix is scrubbed from the returned columns
- unless user elects to maintain suffix with the excl_suffix parameter
- (as suffix retention makes navigating data structures much easier for excl columns)
- so we had a few places in library, particularily in inversion, where we were manipulating column header strings to accomodate this convention
- which was somewhat inelegant
- so populated two new data structures returned in postprocess_dict as excl_suffix_conversion_dict and excl_suffix_inversion_dict
- where excl_suffix_conversion_dict maps from columns without suffix to columns with suffix
- and excl_suffix_inversion_dict maps from columns with suffix to columns without
- these are now the primary means of conversions to accomodate excl suffix stuff
- coupled with a short support function _list_replace which replaces items in list based on a conversion dictionary
- used the opportunity to cleanup the initial scubbing of suffix appenders and improved code comments
- extensive cleanups to inversion stuff to simplify excl edge case accomodation
- which in the process fixed a newly identified edge case for inversion of excl columns
- rewrote _inverseprocess_excl making use of excl_suffix_inversion_dict
- and finally a cleanup (streamlined with equivalent functionality) to _df_inversion_meta

6.40
- important update, deprecated postmunge parameter labelscolumn
- which originally was used to specify when a labels column was included in df_test
- eventually evolved the convention that presence of labels were automatically detected independant of labelscolumn
- so this parameter is no longer needed
- also deprecated boolean True option for parameter testID_column for both automunge(.) and postmunge(.)
- originally testID_column=True was used to indicate that test set has same ID columns as train set
- later introduced the convention that presence of trainID_column entries in test set is automatic when testID_column=False
- so testID_column=True scenario no longer needed
- note user can still pass testID_column as list if ID columns for test set are different than for train set
- as is needed if test ID columns are a subset of train ID columns or are otherwise different
- moved the support function _list_replace in code base to location with other automunge support functions
- (in general, the code base is organized as:
  - initialize transform_dict and process_dict
  - processfamily functions
  - dualprocess / singleprocess functions
  - column evaluations via evalcategory
  - ML infill support functions
  - misc automunge support functions
  - automunge(.)
  - postprocessfamily functions
  - postprocess functions
  - postmunge ML infill support functions
  - misc postmunge support functions
  - postmunge(.)
  - inverseprocess functions
  - inversion support functions
- there is a method to the madness :)
- also, we had a convention of duplicating column_dict entries for excl columns so they could be accessed easily both with and without suffix appenders
- now that we've simplified the suffix conversion scheme with _list_replace decided to strike 
- as the redundant entries kind of muddies the waters
- as the zen of python says, there should be one and preferably only one way to do it
- so went ahead and scrubbed these redundant column_dict entries for excl without suffix
- also found one more excl suffix relic in postmunge missed in last rollout, converted to the new _list_replace convention
- struck two unused variables in postmunge labelsprocess_dict and labelstransform_dict
- also consolidated to single location accounting for parameters passed as lists to be copied into internal state
- in process found a few more parameters where was needed
- (now in addition to trainID_column and testID_column also perform for Binary and PCAexcl and inversion in postmunge)
- new validation check twofeatures_for_featureselect_valresult performed for featureselection to confirm a minimum of two features included in train set (a remote edge case)
- new automunge(.) validation check trainID_column_subset_of_df_train_valresult performed to identify cases where user specified trainID_column entries not present in df_train
- similar new automunge(.) validation check testID_column_subset_of_df_test_valresult performed to identify cases where user specified testID_column entries not present in df_test
- similar new postmunge(.) validation check pm_testID_column_subset_of_df_test_valresult to identify cases where user specified testID_column entries not present in df_test
- new postprocess_dict entry PCAn_components_orig which is the PCAn_components as passed to automunge(.) (which may be different than the recorded PCAn_components entry if was reset based on heuristic)
- finally, a few more cleanups to PCA
- eliminated a redundant call of support function _evalPCA
- in the process a few tweaks for clarity to support funcitons _PCAfunction and _postPCAfunction

6.41
- really important update
- full rework of conventions for defining custom transformation functions
- now requirements for custom transfomation functions are greatly simplified
- populating data structures, default infill, suffix appending, inplace operation, suffix overlap detection, and etc are all conducted externally
- in short, now all user has to do is define a pandas operation wrapped in a function
- where the function recieves as arguments a dataframe, a column, and a set of parameters (as may have been passed in assignparam)
- and returns the resulting transformed dataframes, a list of returned columns associated with the transform, and a dictionary ('normalization_dict') storing any properties derived from the train set needed for test set processing
- where if train set properties aren't needed to process test data the same function can be applied to test data, or otherwise user can define a corresponding test processing function
- where the test processing function recieves as argument a dataframe, column, and the normalization_dict populated from the train set
- and returns the resulting transformed dataframe
- similarly, the conventions for defining custom inversion transforms have been simplified
- where an inversion transform now recieves as arguments a dataframe, a list of the columns returned from the original transformation, the inputcolumn header to be recovered, and the associated normalization_dict
- and returns the transformed dataframe
- full demonstrations provided in the read me under section Custom Transformation Functions
- a few more details, to pass functions with these conventions to a category in processdict they should be passed as entries to 'custom_train', 'custom_test', and 'custom_inversion'
- where if 'custom_test' isn't populated then the custom_train entry will be applied to both train and test data (similar to the singleprocess convention in library)
- note that functionpointer works for these entries too
- to incorporate included updates to support functions _processcousin, _processparent, _postprocesscousin, _postprocessparent, _df_inversion, _grab_processdict_functions_support, _populate_inverse_categorytree _populate_inverse_family, and possibly a few more 
- created templates for the custom transformations shared in the read me as custom_train_template, custom_test_template, and custom_inversion_template
- created wrappers for the received custom functions as _custom_process_wrapper, _custom_postprocess_wrapper, and _custom_inverseprocess_wrapper
- in the process a few cleanups to the processfamily functions
- such as consolidating some redundancies in inplace stuff or for postprocessfamily also consolidating some redundancies in columnkey_list stuff
- a cleanup to support column _df_inversion to remove a redundant parameter derivation
- found and fixed bug in _grab_processdict_functions_support for functionpointer for accessing postprocess functions
- improved process flow for function pointer so that it only access dual/single/post process functions if they are not already populated
- lowered printout tier for unspecified labelctgy assignment from False to True
- reverted convention for _getNArows from evaluating a column to evaluating a copy of the column (helps to preserve data types)

6.42
- further simplified conventions for user defined transformation functions
- eliminated the need for a returned list of column headers from custom_train which is now automatically derived
- only exception is for support columns created but not returned, their headers should be designated by a normalization_dict entry as 'tempcolumns' for purposes of suffix overlap detection
- now user defined custom transformation functions support designation of alternate default infill conventions in processdict entry
- where with 6.41 the transforms applied adjinfill, which will remain the default when not specified
- otherwise to designate alternate default infills to a transformation category can set processdict entry for 'defaultinfill'
- where defaultinfill may be passed as one of strings {'adjinfill', 'meaninfill', 'medianinfill', 'modeinfill', 'lcinfill', 'zeroinfill', 'oneinfill', 'naninfill'}
- note this is only designating the infill performed as a precursor to any applicaiton of ML infill
- or as a precursor to other infill conventions when assigned to a column in assigninfill
- defaultinfill includes functionpointer support
- added one additional infill application for custom transformation functions as adjinfill, but this one following their application instead of preceding
- which is meant to accomodate unforeseen edge cases in user defined transforms
- in the process a few various cleanups to the custom_train support functions _custom_process_wrapper and _custom_postprocess_wrapper
- found and fixed a bug for suffix attachment in _custom_postprocess_wrapper
- finally a slight rework of the processdict functionpointer option
- originally functionpointer was just intended for processing functions and thus functionpointers weren't supposed to be entered when processing functions were already present
- then we added convention that other entries of the pointer target were also copied when not previously specified
- with pointer potentially following chains of pointer targets until reaching a stopping point based on finding processing functions
- realized it made more sense to halt functionpointer when it reaches an entry without pointer as opposed to reaching an entry with processing functions
- so settled on convention that a processdict entry may include both populated processing functions and a functionpointer target
- in other words, processing functions are now on equal footing with other processdict entries in functionpointer chains
- in the process conducted a little sanity check walkthough on functionpointer, everything looks good

6.43
- further streamlined the custom_train_template published in readme
- basically we previously had two dictionaries, a received dictionary with any passed parameters (params) and a returned dictionary we populated in the transform with normalization parameters (normalization_dict)
- realized we could just combine the two, and treat the received params dictionary as a starting point for normalization_dict that had been prepopulated with any passed parameters
- this saves steps of initializing normalization_dict and also steps for transfering any passed parameters from params to normalization_dict
- yeah a really clean solution
- updated default for inplace_option processdict entry. Now when omitted defaults to True. (In other words only need to specify when inplace_option is False.)
- (this default is better aligned with custom_train conventions)
- Added inplace_option = False specifications in process_dict library for prior omissions to match new convention.
- revisions to the readme documentation for processdict for format, clarity, and to be more comprehensive
- oh and big cleanup to the readme, moved the family tree definitions reference material into a seperate file
- available in github repo as "FamilyTrees.md"
- incorporated documentation for defaultinfill processdict option (which had forgot to include with last rollout)
- found opportunity to simplify the code associated with functionpointer by consolidating some redundancies
- removed labelctgy from functionpointer since it is intended to be specific to a root category's family tree
- renamed the functionpointer support functions for clarity (_grab_functionpointer_entries, _grab_functionpointer_entries_support)

6.44
- added post-transform data type conversion to custom_train and custom_test wrappers
- dtype conversion is based on MLinfilltype
- where numeric sets are converted elsewhere based on floatprecision parameter
- boolean integer sets 
- ordinal sets are given a conditional dtype based on size of the encoding space (determined by max entry in train set)
- now transformations passed through custom_train convention are followed by a dtype conversion conditional on the assigned MLinfilltype
- where {'numeric', 'concurrent_nmbr'} have datatype conversion performed elsewhere based on floatprecision parameter
- {'binary', 'multirt', '1010', 'concurrent_act', 'boolexclude'} are cast as np.int8 since entries are boolean integers
- ordinal sets {'singlct', 'ordlexclude'} are given a conditional (uint 8/16/32) dtype based on size of encoding space as determined by max activation in train data
- {'integer', 'exclude', 'totalexclude'} have no conversion, assumes any conversion takes place in transformation function if desired
- also new processdict option as dtype_convert which can be passed as boolean, defaults to True when not specified
- when dtype_convert == False, data returned from custom_train for the category are excluded from dtype conversion
- dtype_convert is also inspected to exclude from floatprecision conversions in both the custom_train convention and dual/singleprocess conventions
- where floatprecision refers to the automunge(.) parameter to set default returned float type
- (in general, we use lower bandwidth data types as defaults for floats than pandas because we assume data is returned normalized, I think pandas generally defaults to float64 when not otherwise designated, floatprecision devaults to 32 and can also be set to 16/64. we also try to use smallest possible integer type for integer encodings, either int8 for boolean integers or uint8/16/32 for ordinal encodings based on size of encoding space. passthrough columns via excl leave received data types intact. continous integer sets are based on whatever is applied in the transformation function.)
- small tweak to custom_train convention, now temporary columns logged as tempcolumns can have headers of other data types (like integers)
- settled on convention that integer mlinfilltype defaults to int32 data type unless otherwise applied in transformation function
- renamed the column_dict entry populated in custom_process_wrapper from defaultinfill_dict to custom_process_wrapper_dict for clarity (since now using to store properties for both infill and dtype conversion)
- rewrote function description for _assembletransformdict
- much clearer now

6.45
- a few cleanups to code comments for custom_train_template and custom_test_template
- a small correction to read me on default transform for categoric labels under automation
- (ordinal applied to all categoric sets, even if 2 unique entries)
- added traindata entry to normalization_dict passed to custom_test in postmunge
- in the process did a little rethinking on whole strtategy for traindata option
- decided to introduce a traindata automunge parameter
- similar to traindata parameter in postmunge
- for purposes of distinguishing where df_test will be treated as train or test data
- which is relevent to a handful of transforms in library like noise injection and smoothing
- added traindata support to existing transforms where relevant
- and thus of course added traindata entry to normalization_dict passed to custom_test in automunge
- note that validation data prepared in automunge uses basis of automunge(.) traindata parameter
- note traindata differs from testnoise assignparam option available for noise transforms
- as testnoise turns on for all data in automunge and postmunge
- while traindata allows user to distinguish treatment between test data in automunge and postmunge

6.46
- reverted to convention that label sets with two unique values in train set given a lbbn root category instead of lbor under automation
- reverted to convention that traindata option is specific to postmunge in dual/singleprocess convention
- (in hindsight having to align for validation data kind of made this muddy, much cleaner to keep it a postmunge option, all potential workflows supported with combination of automunge and postmunge)
- lifted requirement for reserved strings in the keys of normalization_dict accessed in custom_train convention

6.47
- updated custom_test application convention in automunge to be consistent with postmunge
- from the standpoint that if custom_train returned an empty set (including deletion of suffixcolumn)
- then suffixcolumn simply deleted from mdf_test without calling custom_test
- corrected some code comments in processfamily and processparent (and corresponding postprocess functions) regarding inplace elligibility
- updated processparent and postprocessparent to eliminate an edge case so that downstream transforms are halted when the associated upstream transform returned an empty set
- if user needs support for this scenario, need to configure upstream transform so that in the null scenario instead of returning empty set it performs passthrough
- also updated the parentcolumn derivation for passing an input column to downstream generations in processparent and postprocessparent
- to ensure consistent parentcolumn applied in both
- corrected a code comment in processparent that stated that downstream transforms require as input transforms returning a single column
- prior configuration already supported performing downstream transforms on multi-column sets with dual/single process convention
- just hadn't documented it well since don't currently have any applications in the library
- the convention is downstream transforms on received multicolumn sets will recieve as input a single column (which is now the first entry in the upstream categorylist)
- which they can then use as a key to access the upstream categorylist and normalization_dict from column_dict if needed
- updated the validation split performed in df_split to remove a redundant shuffle
- in the process found and fixed small snafu interfering with index retention in cases of validation split
- used that as a hint that needed to audit index retention, so ran everything with all options on and yeah looked good accross all returned sets
- fixed some printout categorizations for labelctgy assignment
- corrected single entry access approach to pandas.iat in a few places
- streamlined printouts at start of automunge / postmunge function calls (removed word "processing")

6.48
- new option for the postmunge inversion parameter to specify a custom inversion path
- custom inversion path can be specified by passing inversion as a single entry set
- containing a string of a returned column header with suffix appenders
- such as to recover a specific input column based on inverting from a starting point of a specific target returned representation
- (note that label inversion is also available to collectively invert each of returned representations by the 'denselabels' option)
- inversion is for recovering the form of input data from train or test data returned from an automunge(.) or postmunge(.) call
- in default configuration, inversion selects an inversion path based on heuristic of shortest path of transformations with full information retention
- in the new custom inversion path option, alternate inversion paths can be specified
- which I don't have a specific use case in mind, just seemed like a reasonable extension

6.49
- simplified index recovery for validation data
- dropped printouts for stndrdinfill
- which were there to indicate columns with no infill performed as part of ML infill or assigninfill
- in which case infill defers to default infill perfomed as part of transformation function
- realized the inclusion was kind of cluttering the printouts
- and by omitting printouts just for this case there is no loss of information content
- moved validation that populated processing functions are callable
- from within the process family functions to distinct validation function _check_processdict4
- this validation now limited to entries in user passed processdict (after any functionpointer entries are populated)
- and validates that entries to any of the processing function slots (dual/single/post/inverseprocess or custom_train/custom_test/custom_inverison)
- are either callable functions or passed as None
- validation results returned in printouts and as check_processdict4_valresult
- a few improvements to functionpointer support function _grab_functionpointer_entries_support
- added support for edge scenario of self-referential pointers linked from a chain
- which were previously treated as infinite loops
- also removed a check for processing functions that was sort of superfluous
- a few small cleanups for clarity
- change evalcat format check from type to callable to be consistent with processing functions
- removed a comment in read me about adding assigninfill support for label sets
- if alternate infill conventions are desired for labels they can be applied with defaultinfill processdict entry in custom_train convention

6.50
- added NArw_marker support to a few datetime family trees in whose omission had been an oversight
- fixed a bug with root category 'time'
- which was as a result of 'time' being entered in the 'time' family tree as a tree category
- where the 'time' process_dict entry was not populated with processing functions
- which is ok as long as a category is primarily intended to be applied as a root category but not a tree category
- otherwise when applied as a tree category no transforms are performed and downstream offspring not inspected when applicable
- updated the processfamily functions so that this scenario no longer produces error, just no transforms applied with printout for clarity
- oh and started to update process_dict entries in general for root categories lacking processing functions so they could be used as tree categories and midway decided some categories it actually makes more sense to leave them without, now that this scenario no longer halts operation won't be an issue
- changed default infill for datetime transforms (excluding timezone) to adjinfill
- reconsidered default infill for passthrough transforms with defaultinfill support (e.g. exc2 and exc5)
- previously we had applied mode infill based on neutrality towards numeric or categoric features
- decided mode is too computationally expensive for a passthrough transform, so reverting to adjinfill as default for categories built on top of exc2 / exc5
- updated default infill for shfl from adjinfill to naninfill
- added defaultinfill processdict specification support to dual/single/post process convention
- (a handful of transforms still pending support for esoteric reasons, those without process_dict defaultinfill specification in familytrees file)
- new option for defaultinfill as negzeroinfill, which is imputation by the float negative 0 (-0.)
- negzeroinfill is the new default infill for nmbr (z-score normalization) and qbt1
- as benefit the convention allows user to eliminate the NArw aggregation without loss of information content
- note that nmbr is the default transform for numeric sets under automation
- and previously applied meaninfill as precursor to ML infill which since the data is centered was equivalent to zero infill
- we anticipate there may be potential for downstream libraries to build capabilities to selectively distinguish between zero and negative zero based on the use case, otherwise we believe negative zero will be neutral towards model performance
- as a bonus the convention benefits interpretibility by visual inspection as user can distinguish between imputation points without NArw when ML infill not applied
- negzeroinfill also available for assignment via assigninfill

6.51
- new assignparam option for nmbr transform as abs_zero
- abs_zero accepts booleans and defaulting to True
- when activated, abs_zero converts any negative zeros to positive zero prior to imputation
- as may desired to ensure the negzeroinfill imputation maintains a unique encoding in the data
- new transformation category nbr4
- similar to z-score normalization configuration prior to 6.50 update (with new abs_zero parameter deactivated)
- except changed defaultinfill for nbr4 from meaninfill to zeroinfill to solve rounding issue sometimes causing mean imputation to return as tiny decimal instead of zero

6.52
- found and fixed a snafu in qbt1 transform originating from 6.50
- everything now aligned with original intent for 6.50 and back working as it should
- added abs_zero assignparam support to qbt1, defaulting to True
- abs_zero is boolean defaulting to True which converts received negative zeros to positive zero
- updated qbt1 family of transforms for cases that don't default to a returned sign column (qbt3 and qbt4) to defaultinfill of zeroinfill instead of negzeroinfill
- corrected a typo in read me library of transforms, mmq3 now corrected to read mmq2

6.53
- a walkthrough of the evalcategory function identified a snafu for numeric sets passed as pandas categoric type
- it looks like had accidentially inserted an intermediate if statement in between a prior paired if/else combo and resulted in categoric sets with integer or float entries getting treated as numeric under automation
- (that I believe is the most likely explanation, although it is also possible I just poorly implemented the first if statement if I am remembering the order of these code edits incorrectly, this function has gradually evolved over a long period)
- the intent was that received columns that are pandas type 'category' get treated to default categoric encoding (bnry or 1010), even in cases where their entries were numeric
- the intermediate if statement I just struck, I think from a while back I was trying to get just a little too creative and was trying to treat numeric sets with 3 unique entries as a one hot encoding instead of normalization (for reasons that I'm now having a hard time trying to identify, in other words I don't think there was a good reason), to make matters worse along the way the 3 state one-hot got converted to a binarization and yeah long story short (too late) etc
- so now the corrected (and simplified) convention is that all majority numeric sets (with integers or floats) under automation are normalized, unless the set is received as a pandas 'categoric' data type, and then they are treated to a binarizaiton by bnry or 1010 based on their unique entry count
- also removed an unused code snippet in same function

6.54
- added a clarification to read me that under automation numeric data received as pandas categoric data type will be treated as categoric for binarization instead of normalizations
- new temporary postproccess_dict entry temp_miscparameters_results added at initialization
- temp_miscparameters_results is for storing validaiton results recieved in various support functions that might not have access to miscparameters_results
- which is then consolidated with miscparameters_results and the temporary entry struck
- new validation result reported in miscparameters_results as treecategory_with_empty_processing_functions_valresult
- which is for the tree category without processing functions populated that we noted in 6.50
- validation result is populated with unique entry for each occurance recording the tree category without processing functions and the associated root category whose generations included the tree category
- populated with key as integer i incremented with each occurance as:
treecategory_with_empty_processing_functions_valresult = \
{i : {'treecategory' : treecategory,
      'rootcategory' : rootcategory}}
- identified a superfluous copy operation within the transformation functions associated with populating returned column_dict data structure
- so globally struck this copy operation
- created a new support function for one hot encoding as _onehot_support
- this function returns a comparable order of columns as would pd.get_dummies
- the index retention convention is also comparable
- part of the reason for creating this function is so will be able to experiment with variations
- for potential use in different transformation function scenarios
- new trigometric family of transforms sint/cost/tant/arsn/arcs/artn
- built on top of numpy trigometric operations np.sin, np.cos, np.tan, np.arcsin, np.arccos, np.arctan
- inversion supported with partial information recovery
- defaults to defaultinfill of adjinfill (since these are for periodic sets a static imputation makes less sense)
- the inspiration for incorporporating trigometric functions came from looking at a calculator
- audited family trees for cases with omitted NArw_marker support, found a few entries whose omission I believe was accidental
- in the process identified by memory I think a case where I may not have provided sufficient citation at the time of rollout
- for clarity, our use of adjacent sin and cos transformations for periodic datetime data, as well as binarization as an alternative to onehot encoding
- were direclty inspired by github issue suggestions submitted by github user "solalatus"
- (I had noted his input in acknowledgements of one of papers, just thought might be worth reiterating)
- this user provided links to blog posts associated with both of these concepts which were rolled out in 2.47
- the datetime blog post was an article by Ian London titled "Encoding cyclical continuous features - 24-hour time"
- the binarization blog was one of a few articles, not sure which, I think it was a blog post by Rakesh Ravi titled "One-Hot Encoding is making your Tree-Based Ensembles worse, here’s why?"

6.55
- added inplace support to search functions srch/src2/src3/src4
- added defaultinfill support to search functions srch/src4
- consolidated a redundant defaultinfill application to search functions src2/src3
- added case assignparam support to search functions src2/src3 (previously included with srch/src4)
- added support for aggregated search terms to src3 (previously included with srch/src2/src4)
- found and fixed edge case in srch/src2/src4 for cases where an aggregated search term returns an empty set 
- please note that our search functions are patent pending
- added boolean entry support to the one-hot encoding support function rolled out in 6.54
- boolean entries sorted differently than pd.get_dummies, which means there will be a small impact to backward compatibility associated with transforms that previously applied the pandas version corresponding to boolean entries
- added defaultinfill support for received columns of pandas category dtype
- consolidated a redundancy in defaultinfill code so as to use single common support function in both dual/single/post process and custom_train conventions

6.56
- ok trying to formalize conventions for data returned from inversion
- we had previously noted in read me that general convention is that data not successfully recovered was recorded the the reserved string 'zzzinfill'
- a survey of the various transforms found that there are actually multiple conventions
- for transforms that had some kind of imputation in the forward pass, the convention is data returned from inversion corresponds to that imputation
- for transforms without imputation, such as one-hot encoding when ML infill not imputed, the revised convention is entries not recovered is now recored as NaN, which we believe is more inline with common practice
- there is a still a scenario where returned data from inversion records entries without recovery as the reserved string 'zzzinfill', specifically for transforms that return from inversion a column of pandas object dtype and were empty entries may not correlate with missing data (such as string parsing and search functions), which is applied since pandas object dtype would otherwise convert NaN to the string 'nan'

6.57
- new gradient boosting option for ML infill via XGBoost
- available by ML_cmnd = {'autoML_type':'xgboost'}
- defaults to verbosity 0
- can pass model initiliazation and fit parameters by e.g.
ML_cmnd = {'autoML_type'  :'xgboost',
           'MLinfill_cmnd':{'xgboost_classifier_model': {'verbosity'    : 0},
                            'xgboost_classifier_fit'  : {'learning_rate': 0.1},
                            'xgboost_regressor_model' : {'verbosity'    : 0},
                            'xgboost_regressor_fit'   : {'learning_rate': 0.1}}}
- hyperparameter tuning available by grid search or random search, tuned for each feature
- can activate grid search tuning by passing any of the fit parameters as a list or range of values
- received static parameters (e.g. as a string, integer, or float) are untuned, e.g.
ML_cmnd = {'autoML_type'  :'xgboost',
           'MLinfill_cmnd':{'xgboost_classifier_fit'  : {'learning_rate': [0.1, 0.2],
                                                         'max_depth'    : 12},
                            'xgboost_regressor_fit'   : {'learning_rate': [0.1, 0.2],
                                                         'max_depth'    : 12}}}
- random search also accepts scipy stats distributions to fit parameters
- to activate random search set the hyperparam_tuner to 'randomCV'
- and set the desired number of iterations to randomCV_n_iter (defaults to 100)
from scipy import stats
ML_cmnd = {'autoML_type'     :'xgboost',
           'hyperparam_tuner':'randomCV', 
           'randomCV_n_iter' : 5,
           'MLinfill_cmnd':{'xgboost_classifier_fit'  : {'learning_rate': [0.1, 0.2],
                                                         'max_depth'    : stats.randint(12,15)},
                            'xgboost_regressor_fit'   : {'learning_rate': [0.1, 0.2],
                                                         'max_depth'    : stats.randint(12,15)}}}
- also revisited random forest tuning and consolidated a redundant training operation

6.58
- new early stopping criteria available for iterations of ML infill applied under infilliterate
- ML infill still defaults to 1 iteration with increased specification available by the infilliterate parameter
- the infilliterate parameter will serve as the maximum number of iterations when early stopping criteria not reached
- user can activate early stopping criteria by passing an ML_cmnd entry as ML_cmnd['halt_iterate'] = True
- which will evaluate imputation deirvations in comparison to the previous iteration and halt iterations when sufficiently low tolerance is reached
- the evaluation considers seperately categoric features in aggregate and numeric features in aggregate
- the categoric halting criteria is based on comparing the ratio of number of inequal imputations between iterations to the total number of imputations accross categoric features to a categoric tolerance value
- the numeric halting criteria is based on comparing for each numeric feature the ratio of max(abs(delta)) between imputation iterations to the mean(abs(entries)) of the current iteration, which are then weighted between features by the quantity of imputations associated with each feature and compared to a numeric tolerance value
- the tolerance values default to categoric_tol = 0.05 and numeric_tol = 0.01, each as can be user specificied by ML_cmnd entries of floats to ML_cmnd['categoric_tol'] and ML_cmnd['numeric_tol']
- early stopping is applied when both the numeric featuers in aggregate and categoric features in aggregate are below threshold of their associated tolerances as evaluated for the current iteration in comparison to the preceding iteration
- and the resulting number of iterations is then the infilliterate value applied in postmunge
- please note that our numeric early stopping criteria was informed by review of the scikit-learn iterativeimputer approach, and our categoric early stopping criteria was informed by review of the MissForest early stopping criteria, however there are some fundamental differences with our approach in both cases
- such as for instance the formula of our numeric criteria is unique, and the tolerance evaluation approach of our categoric criteria is unique

6.59
- introduced default convention of applying a stochasticly derived random seed to model training for each ML infill model, including each model accross features and each model accross iterations
- to deactivate for deterministic training based on automunge randomseed parameter can pass ML_cmnd['stochastic_training_seed'] = False
- please note that currently a randomseed is only inspected in randomforest, catboost, and xgboost autoML options
- new option for incorporating stochasticity into imputations derived through ML infill
- can activate for numeric and/or categoric features by passing ML_cmnd['stochastic_impute_categoric'] = True and/or ML_cmnd['stochastic_impute_numeric'] = True
- stochastic imputations bear some similarity to DP family of noise injection transforms
- in that sampled noise with numpy.random is injected into the imputations prior to insertion
- for numeric stochastic imputation we applied a similar method to our DPmm transform
- here we convert the imputation set into a min/max scaled reprentation to ensure noise distribution parameters aligned with range of data, with the scaling based on min/max found in the training data
- we sampled noise from a gaussian distribution or optionally from a laplace distribution
- with noise scaling defaulting to mu=0, sigma=0.03, and flip_prob=0.06 (where flip_prob is ratio of a feature set's imputations receiving injections)
- note that in order to insure range of resulting imputations is consistent with range in df_train we cap outlier noise entries at +/- 0.5 and scale negative noise when min/max representation is below midpoint and scale positive noise when minmax representation is above the midpoint, resulting in a consistent returned range independant of noise sampling
- after noise injection the imputation set is converted back by an inversion of the min/max representation
- noise injection to categoric features are based on parameter defaulting as flip_prob=0.03 (where flip_prob is ratio of a feature set's imputations receiving injections)
- injections are conducted randomly flipping the entries in a target row to a random draw from the set of unique activation sets (as may include 1 or more columns) based on draw from a uniform distribution
- please note that this includes the possibility that an injection entry will retain the original represtntation based on the random draw
- please note that the associated parameters can be configured by ML_cmnd entries to 'stochastic_impute_categoric_flip_prob', 'stochastic_impute_numeric_mu', 'stochastic_impute_numeric_sigma', 'stochastic_impute_numeric_flip_prob', 'stochastic_impute_numeric_noisedistribution'
- (where these entries all accept floats except 'stochastic_impute_numeric_noisedistribution' accepting one of {'normal', 'laplace'}
- please note that we suspect stochastic imputations may have potential to interfere with infilliterate early stopping criteria as rolled out in 6.58 based on the scale of injections

6.60
- reverting the updates associated with 6.57
- upon some reflection we do not feel we have sufficient comfort in our hyperparameter tuning implementation to justify gradient boosting from an autoML standpoint
- and don't want to distract our users with an option that has tendency to overfit when not tuned

6.61
- a few cleanups to support functions _createMLinfillsets and _createpostMLinfillsets
- including replacement of a few kind of hacky concatinate and drops with a much cleaner pandas.iloc
- (this funcion was first implemented very early in development)
- new ML_cmnd option as can be passed to ML_cmnd['leakage_sets']
- leakage_sets can be passed as either a list of input columns or a list of lists of input columns
- user can also pass returned column headers if a subset of features derived from common input feature are the be included in a leakage set
- where each list of input columns is for specification of features that are to be excluded from each other's ML infill basis
- in other words, features with known data cross leakage issues can now be specified by user for accomodation in each other's ML infill basis
- new ML_cmnd option as can be passed to ML_cmnd['leakage_tolerance']
- leakage tolerance is associated with a new automated evaluation for a potential source of data leakage accross features in their respective imputation model basis
- compares aggregated NArw activations from a target feature in a train set to the surrounding features in a train set and for cases where separate features share a high correlation of missing data based on the shown formula we exclude those surrounding features from the imputation model basis for the target feature. 
- ((Narw1 + Narw2) == 2).sum() / NArw1.sum() > leakage_tolerance
- where target features are those input columns with some returned coumn serving as target for ML infill
- leakage_tolerance defaults to 0.85 when not specified, and can be set as 1 or False to deactivate the assessment
- to perform the operation, set ML_cmnd['leakage_tolerance'] to a float between 0-1
- where the lower the value, the more likely for sets to be excluded between each other's basis
- leakage_tolerance is implemented by adding results of evaluation to any user specified leakage_sets
- where sets are collected in three forms in returned postprocess_dict['ML_cmnd'], as 'leakage_sets_orig' (user passed sets prior to derivations) 'leakage_sets_derived' (derived sets) and 'leakage_sets' (combination of orig and derived)
- please note that there is a small latency penalty associated with this operation in automunge(.) and no meaningful penalty in postmunge(.)
- new postprocess_dict entry ['ML_cmnd_orig'] which is a dictionary recording (for informational purposes) the original form of ML_cmnd as passed to automunge(.) prior to any initializations and updates such as based on leakage_tolerance or _check_ML_cmnd

6.62
- new suite of parameter validations associated with the ML_cmnd parameter
- which is data structure to set options and pass parameters to predictive models associated with ML infill, feature importance, or PCA
- as well as various other ML infill options like early stopping thorugh iterations, stochastic noise injections, hyperparpameter tuning, leakage assessment, etc
- also moved all of the default ML_cmnd initializations and validations into a single location for code clarity, now all performed in _check_ML_cmnd
- should make future developments on this data structure much easier to maintain
- parallel improved some corresponding internal documentation

6.63
- corrected function name of validate_allvalidnumeric to include leading underscore for consistent convention for internal functions applied for rest of library
- reconfigured "leakage_dict" data structure population associated with leakage sets to be based on set aggregation
- instead of prior configuration of searching within lists and etc
- which will benefit automunge(.) latency associated with this operation
- significant rewrite for speed and clarity of _evalcategory
- revised the derivations of most common data types to a much more efficient method
- resulting in a material improvement to automunge(.) latency
- the rewrite also resulted in a much cleaner code presentation, we believe will make this function easier to understand now
- (this function was one of the first ones that we wrote :)

6.64
- removed collections.Counter import that is no longer used after 6.63 rewrite of _evalcategory
- performed an audit of leakage_sets rolled out in 6.61
- identified an opportunity for improved specification granularity
- specifically, leakage_sets as implemented were for specifying bidirectional ML infill basis exclusions
- i.e. for a list of features, every feature in list was excluded from basis of every other feature in list
- realized there may be scenarios where a unidirectional exclusion is prefered
- such as to exclude feature2 from feature1 basis but include feature1 in feature2 basis
- so now in addition to ML_cmnd['leakage_sets'] for bidirectional specification, user can also specify unidirectional exclusions in ML_cmnd['leakage_dict'] which accepts dictionaries in form of column header key with value of a set of column headers
leakage_dict = {feature1 : {feature2}}
- where headers can be specified in input or returned header conventions or combinations thereof (where returned headers include suffix appenders)
- note that as part of this update, ML infill exclusions derived as a result of leakage_tolerance specification are now captured in a unidirecitonal capacity as opposed to bidirectional
- which is more in line with our description provided as part of conference review
- note that in returned postprocess_dict['ML_cmnd'], the prior returned entries of leakage_sets_orig and leakage_sets_derived are now replaced with leakage_dict_orig and leakage_dict_derived

6.65
- a big cleanup to both text and onht transforms for one-hot encoding
- in some portions may be considered a full rewrite
- note that in addition to the suffix convention, onht has a few subtle distinctions vs. text
- in text numbers are converted to strings prior to encoding, so 2 == '2' for instance (needed for suffix convention)
- whereas in onht numbers and strings recieve a distinct activation (unless str_convert parameter activated)
- also in text missing data is represented prior to ML infill as no activation, whereas prior onht missing data was given distinct activation
- new convention for onht, missing data is returned without activation to be consistent with text
- new set of parameters accepted for both text and onht as 'null_activation', 'all_activations', 'add_activations', 'less_activations', and 'consolidated_activations'
- null_activation defaults to False, when True missing data is returned with distinct activation as per prior convention for onht
- all_activations defaults to False, user can pass as a list of all entries that will be targets for activations (which may have fewer or more entries than the set of unique values found in the train set, including entries not found in the train set)
- add_activations defaults to False, user can pass as a list of entries that will be added as targets for activations (resulting in extra returned columns if those entries aren't present in the train set)
- less_activations defaults to False, user can pass as a list of entries that won't be treated as targets for activation (these entries will instead recieve no activation)
- consolidated_activations defaults to False, user can pass a list of entries (or a list of lists of entries) that will have their activations consolidated to a single common activation
- the returned activation reported as the first entry in each consolidation list
- also found and fixed edge case with pickle download operation to save a populated postprocess_dict associated with internal processdict manipulations editing exterior object (appeared to manifest when passing the same processdict to multiple automunge(.) calls without reinitializing)

6.67
- new validation test performed for categoric encodings with reserved string 'zzzinfill' 
- where string is reserved with respect to usage among entries of recieved column for a few categoric transforms
- note that this does not cause error, entries are simply treated consistent with NaN for the transform
- now when string found present a printout is returned and a validation result logged as
zzzinfill_valresult = {i : {'column' : column,
                            'traintest' : traintest}}
- where i is an integer incremented with each occurance
- traintest is one of {'train', 'test'} indicating where entry was found i.e. df_train vs df_test (singleprocess functions log both as 'train')
- recorded in postprocess_dict['temp_miscparameters_results']['zzzinfill_valresult']
- or for postmunge recorded in postreports_dict['pm_miscparameters_results']['zzzinfill_valresult']
- this validation test perfomed in relevant transforms text/onht/smth/strn/ordl/ord3/ucct/1010
- note that to support added a temporary entry to postprocess_dict in posmtunge to log support function validation results as 'temp_pm_miscparameters_results' which is subsequently consolidated with the returned validation results in postreports_dict and the temporary entry struck
- found and fixed a small defaultinfill process flow snafu in strn
- added inplace support for 1010

6.68
- added activation parameter support to ordinal and binarization categoric encodings via ordl and 1010
- specifically referring to some of activaiton parameters rolled out for one-hot in 6.65: 'all_activations', 'add_activations', 'less_activations', and 'consolidated_activations'
- all_activations defaults to False, user can pass as a list of all entries that will be targets for activations (which may have fewer or more entries than the set of unique values found in the train set, including entries not found in the train set)
- add_activations defaults to False, user can pass as a list of entries that will be added as targets for activations (resulting in extra returned columns if those entries aren't present in the train set)
- less_activations defaults to False, user can pass as a list of entries that won't be treated as targets for activation (these entries will instead recieve no activation)
- consolidated_activations defaults to False, user can pass a list of entries (or a list of lists of entries) that will have their activations consolidated to a single common activation
- note that 'null_activation' not supported because ordinal and 1010 by definition require a distinct enocding for missing data (in one hot encoding the missing data has option of being represented by no activation)
- note that for ordinal encoding activation parameter support limited to ordl transform (as opposed to other ordinal transform ord3)
- reason is that ord3 bases integer sorting on a different method from frequency of entries and would make this a little more complex, saving that for when I have a spare weekend, I don't consider it high priority since we have ordl supported for an ordinal option with activation parameter support
- note that this effort has helped us to identify possibly another opportunity for improved latency associated with these categoric encodings, is a little complex but this is an intended direction to further refine going forward

6.69
- ok just realized committed a non-trivial error in both 6.65 and 6.68
- with respect to the new parameter support in text/onht/ordl/1010
- specifically, the new configuration interfered with backward compatibility
- in scenario where a user populated a postprocess_dict in earlier version
- and tried to use with postmunge in new version
- tbh had just forgotten about this error channel
- going to give it some thought how to formalize this scenario in testing to avoid the risk going forward
- additionally, new policy to help mitigate this risk
- new policy: significant transformation function deviations are now by policy conducted by defining new transfomation function to eliminate this backward compatibility scenario

6.70
- rewrite of 1010 binarization transform, which is the default categoric encoding under automation
- ported to the custom_train convention
- with new functions _custom_train_1010 / _custom_test_1010 / _custom_inversion_1010
- resulting in much cleaner code presentation, easier to understand
- while retaining consistent functionality and accepted parameters
- the primary updated convention is that now the missing data representation prior to ML infill defaults to all 0's
- as opposed to prior where missing data may have had different activation sets per application
- in the process trimmed a little fat
- such as removed the 'zzzinfill' reserved string requirement
- as well as eliminated what we refered to as an overlap replace operation
- (which was there to accomodate an assumed edge case with pd.replace which I now am unable to duplicate)
- expecting this revision will benefit latency for this transform
- which since is the default under automation we thought was a worthy goal
- in the process I think identified the edge case that was original rationale for the 'zzzinfill' convention
- which turns out can be accomodated in a much simpler fashion by simply resetting dtype to 'object'
- note the prior defined 1010 functions are retained in library for backward compatibility purposes
- which are also used for Binary dimensionality reduction
- also small tweak to the wrapper function for custom_train
- now the recorded categorylist is guaranteed as consistent order of entries as found in returned data
- intend going forward to continuing porting a few more foundational transforms to custom_train convention and lift reserved string requirements where possible

6.71
- new form of input accepted for automunge(.) valpercent parameter
- previously valpercent was accepted as float in range 0-1
- used to specify the ratio of the validation split for partitioning from the training data
- where validation set was either partitioned based on a random sampling of rows when shuffletrain was activated
- or otherwise partitioned from bottom sequential rows of the training set when shuffletrain was deactivated
- new convention is that valpercent can optionally be specified as a tuple in the form valpercent=(start, end)
- where start is a float in the range 0<=start<1
- and end is a float in the range 0<end<=1
- and where start < end
- the tuple option allows user to designate specific portions of the training set for partitioning
- for example, if specified as valpercent=(0.2, 0.4), the returned training data would consist of the first 20% of rows and the last 60% of rows, while the validation set would consist of the remaining rows
- note that if shuffletrain activated (as either True or as 'traintest'), the returned train set and validation set rows will subsequent to partitioning be individually shuffled
- please note that automunge(.) already had support for simultaneous preparations of training and validation data, where validation data was partitioned from the received training data and prepared seperately on the train set basis
- however in prior configuration user only had options for validation partitioning from random sampled rows or bottom sequential
- with the new configuration user can now specify specific partitions of the training data to segregate for validation sets
- the purpose of this new valpercent tuple option is to support integration into a cross validation operation
- also revised the prior function for partitioning validation sets which should result in reduced memory overhead
- also, further validation identified a scenario where the new porting of 1010 to custom_train (from 6.70) had an edge case. It’s a very remote edge case, but an edge case nonetheless. Going to revert 1010 to the prior transform convention until get this resolved. (Edge case only manifested when 1010 was performed downstream of a string operation on a particular testing feature, why it was missed in testing with last rollout, we didn’t think to validate application of 1010 as a downstream transform. Lesson learned to adhere to comprehensive validations with each rollout.)

6.72
- reintroduced the custom_train convention for 1010 (default categoric encoding under automation)
- identified the source of the edge case for ported 1010 to the custom_train convention noted in 6.71
- it was associated with our removal of the 'zzzinfill' reserved string for NaN missing data representation
- found that NaN had potential to interfere witih operations in a few edge cases related to both set operations and pandas column dtype drift
- for set operations, NaN demonstrated at times inconsistent behavior, for example with 'in' detection or use of '|' 
- also in some cases set operations resulted in duplicate inclusions of a NaN entry to a set
- additionally, NaN inclusions in replace operations had potential to result in pandas column dtype drift
- which could in edge case replace our string representations of binarization sets to integers, stripping the leading zeros
- settled on a convention that every pandas.replace operation is now applied along with a .astype('object'), which helps mitigate dtype drift
- also for set management surrounding NaN inclusions, we've applied a different method to remove NaN entry, now relying on a set comprehension taking account for NaN != NaN
- we still like using set operations to manage the unique entries and encodings as much more efficient than list methods, now that we can accomodate these further identified NaN entry edge cases we can lift the 'zzzinfill' reserved string requirement
- as noted in 6.70 intent going forward is to continuing porting a few more foundational transforms to custom_train convention and lift reserved string requirements where possible

6.73
- some clarifications provided to README valpercent parameter writeup associated with use of automunge(.) in context of a cross-validation operation
- a few small code comment cleanups and a small tweak associated with activation parameters to _custom_train_1010
- ported the ordl transform for ordinal encoding to the custom_train convention
- similar to 1010, we believe this will benefit latency
- new form has consistent functionality and parameter support
- lifted 'zzzinfill' reserved string requirement
- primary deviations from user standpoint is that missing data prior to ML infill now by default is encoded as integer 0
- also trimmed a few stored entries carrying redundant information in the normalization_dict saved in postprocess_dict['column_dict'] to reduce memory overhead, as there are scenarios where ordl sets may have a large number of unique entries
- parallel small update to the ordered_overide parameter convention for ordinal encodings ordl and ord3 as implemented in dualprocess convention, now ordered treatment is based only on train set instead of both train and test (consistent with the custom_train version)
- found an assigninfill scenario where we were piggy backing off of normalization_dict populated in trasnformation functions to store 'infillvalue' as a derived infill value (associated with mean, median, mode, and lc infill from assigninfill)
- realized this was in effect resulting in a reserved string for normalization_dict populated in user defined transformation functions
- so simple simple solution, moved the infill value to first tier of column_dict associated with the column instead of populating in normalization_dict, and renamed to 'assigninfill_infillvalue'
- which in the process also resolved an issue I think originating from saved normalization_dict's accross columns in a categorylist sharing same address in memory, so when we were saving infill to one column's normalization_dict was overwriting entry in other columns from categorylist
- small tweak to Binary dimensionality reduction, now when aggregating activations from boolean integer columns, the activations are recast as integers, which addresses an edge case when negzeroinfill is applied with assigninfill to a boolean integer column resulting in dtype drift to float
- as a note, with pending various portings of transformation functions to custom_train convention, it will result in some bloat to lines of code, intent is sometime (not too far down the road) to consolidate any redundancies to just the custom_train form, which will impact backwards compatibility, so saving this step for once have a bulk of consolidations ready so can roll out in one fell swoop, like ripping a bandaid off

6.74
- rewrite of onht transform for one hot encoding, ported to the custom_train convention
- similar simplifications as with 6.73, should benefit latency
- added ordered_overide parameter support, similar to use with ordl, for setting order of activation columns when a column is received as pandas ordered categoric, accepts boolean to deactivate
- onht missing data by default returned with no activation, new convention when null_activation parameter activated missing data now encoded in final column in onht set (as opposed to based on alphabetic sorting)
- struck some troubleshooting printouts had missed with last update
- rewrote the evaluations performed for powertransform == 'infill', including fixed a few derivation edge cases and consolidated to pandas operations
- identified an edge case for np.where, which we were using extensively
- edge case was associated with dtype drift
- for example, if target column starts with a uniform dtype, including NaN, int, float, when np.where inserts a string it converts all other dtypes to str, although does not do so when target column starts with mixed dtypes
- so created an alternative to np.where as "autowhere", which is built on top of pandas.loc
- it is kind of a compromise between np.where and pd.where with a few distinctions verses each
- global replacement of np.where throughout codebase with autowhere 
- a small udpate to the onehot support function in how we access index
- found and fixed a bug with None entry conversion to NaN
- decided to make stochastic_impute_categoric and stochastic_impute_numeric on by default
- this is a nontrivial update, the justification is comments from conference review on potential for deterministic imputations to contribute to bias
- hope to perform some form of validation down the road, will take some creativity for experiment design
- stochastic impute for ML infill can be deactivated in ML_cmnd if desired, i.e.
ML_cmnd = {'stochastic_impute_numeric': False,
           'stochastic_impute_categoric':False},
- found and fixed an edge case for postmunge drift report associated with excl suffix convention

6.75
- consolidated to a single NaN representation making use of np.nan to be consistent with other data types
- (we had used float("NaN") in a few places, turns out there are some scenarios where the two forms aren't equivalent)
- now all received NaN values in df_train and df_test are consolidated to the form of np.nan
- streamlined suffix convention for custom_train transforms, the optional suffix parameters are no longer accepted, suffix is always cast as the tree category (if you need a different suffix can define a new processdict entry)
- thus transforms ported to custom_train in recent updates (1010, ordl, and now onht) no longer have suffix parameter support
- added a clarification to custom_train_template in read me that "(automunge(.) may externally consider normalization_dict keys of 'inplace' or 'newcolumns_list')"
- consolidated one-hot encoding transforms text and onht into a single transform now distinguishable by parameter
- using same function just rolled out for onht in custom_train convention as the basis
- by adding to onht trasnform parameter suffix_convention as one of {'text', 'onht'}, defaulting to text, which distinguishes between suffix conventions
- note that when suffix_convention cast as text, str_convert is reset to True and null_activation is reset to False
- resulted in a few improvements to text, now with support for ordered_override parameter, lifted reserved string, 
- new parameter accepted to one hot encodings onht and text as frequency_sort, boolean defaults to True
- when True the order of returned columns is sorted by the frequency of entries as found in the train set, False is for alphabetic sorting
- note that when ordered_override is activated if received as a pandas ordered categoric set that order takes precedence over frequency_sort
- also added support for ordered encodings in conjunction with the activation parameters (all_activations / add_activations / less_activations / consolidated_activations)
- similar updates to ordinal encodings
- now our two primary variants (ordl, ord3) are consolidated into a single function differentiated by parameter frequency_sort
- where frequency_sort is boolean defaulting to True indicating that the order of integer encodings will be based on the entries sorted by frequency as found in the training set (consistent with ord3)
- and frequency_sort=False is integer encodings sorted by an alphabetic sort of entries, which is default for ordl, lbor, lbos
- (noting that lbor is our default categoric label set encoding under automation)
- and other similar conventions updated as just discussed for onht/text, including support for ordered encodings in conjunction with the activation parameters 
- rewrite of the label smoothing trasnform smth
- split the operation into seperate trasnformation categtories, now smth is applied downstream of a seperate one-hot encoding
- variants available with root categories smth/smt0/fsmh/fsm0/lbsm/lbfs
- please consider this implementation a demonstration of the long existing funcitonality for specifying family trees with transforms applied downstream of transforms that return multicolumn sets
- this new configuration enables full parameter support for the upstream one-hot encoding consistent with onht
- and also means label smoothing and fitted label smoothing can be applied downstream of any multirt ML infilltype trasnform by specifying a family tree
- this update impacts backward compatibility for label smoothing trasnforms
- please note we intend for our next update to have several additional impacts to backward compatibility, as we intend to consolidate our full range of categoric encodings to the custom_train convention in order to mitigate some recent bloat in lines of code from porting transforms to custom_train
- and after this next update we intend to be much more intentional about future impacts to backward compatibility
- we have a new validation test rolled which we are comfortable will help us identify future backward compatibility impacts prior to rollout

6.76
- significant backward compatibility impacting update
- after this update we intend to be much more intentional about future impacts to backward compatibility
- we have a new validation test rolled which we are comfortable will help us identify future backward compatibility impacts prior to rollout
- struck prior dual/single/post convention implementations of suite of categoric encodings
- including text/onht/ordl/ord3/1010
- in order to reduce code bloat
- this impacts backward compatibility for most categoric encodings, as well as Binary option
- also backward compatibility impacts to power of ten binning via pwrs and pwor
- tweaked bnry transform resulting in lifted reserved string
- tweaked ucct trasnform resulting in lifted reserved string
- tweaked strn transform resulting in lifted reserved string
- rewrote Binary dimensionality reduction to be based on custom_train versions of 1010 or ordl
- replaced all use of support function _postprocess_textsupport
- now consolidated to much cleaner support function _onehot_support
- struck support function _check_for_zzzinfill
- a big cleanup to the pwrs transform for power of ten binning
- added pwrs parameter support 'zeroset' (boolean defaults as False) to include an activation column for received zero entries, returned with the suffix column_suffix_zero
- similar cleanup to pwor for ordinal power of ten binning, also added zeroset parameter support
- these updates carry thorugh to other categories built on top of pwrs or pwor transformation functions
- this update has eliminated all cases of reserved string for entries of an input column to categoric encodings
- refering to our use of 'zzzinfill' string as a placeholder for infill in categoric encodings
- we still use the string 'zzzinfill' as a placeholder in a few places for string parsing functions, intent is to audit these instances for possible future update
- this update results in a 5% macro reduction in lines of code

6.77
- added parameter support to ordl/ord3 
- now accepting parameter 'null_activation', boolean defaults to True
- when False, missing data does not receive a distinct activation
- the purpose of this parameter is to support encoding categoric sets where a missing data representation is not desired, as may be the case when an ordinal encoding is applied to a label set for instance
- and when missing data is found, it is grouped into the encoding with the 0 bucket
- which in base configuration (when frequency_sort = True) will be encoding associated with the most frequent entry found in train set
- or when frequency_sort = False, will fall in first entry in alphabetic sorting
- note that when ordered_overide parameter is in default True state, if a column is received as pandas ordered category dtype, the first entry will be per that sorting
- note that when receiving data as pandas category dtype, we have a known range of potential values, and can thus be assured that missing data won't be present at least in the train data
- (although subsequent test data may not be consistent pandas category dtype, since this parameter is for label sets shouldn't be an issue)
- null_activation parameter now reset as False for root category lbor, which is our default categoric encoding to label sets under automation
- lbor also changed sorting basis from alphabetic (like ordl) to frequency (like ord3)

6.78
- added null_activation parameter support to 1010 trasnform
- null_activation accepts boolean, defaulting to True for a dinstinct missing data encoding of all zeros
- when passed as False missing data grouped with the otherwise all zero encoding, which will be the first unique entry in an alphabetic sorting
- found and fixed a bug originating from 6.76 associated with replacing the support function _postprocess_textsupport
- it turns out the prior convention was not directly equivalent to the new support function, resulting in one-hot aggregations with no activations
- it appears I did not test this aspect sufficiently prior to rollout
- easy fix, added a new scenario to the alternative support function _onehot_support
- this impacted transformation categories bins, bnwd, bnep, tlbn, bkt1, bkt2, and also impacted ML infill to binarized categoric encodings via 1010
- issue resolved

6.79
- found and fixed implementation snafu with tlbn transform associated with the top bucket in edge case when data does not have enough diversity to populate full range of specified bincount buckets
- updated some data structure maintenance taking place in processparent that was interfering with ML infill in conjunction with transforms performed downstream of a transform returning a multi-column set
- new transformation category available for population in family trees as mlti
- mlti is intended for use to apply normalizations downstream of a concurrent_nmbr MLinfilltype which may have returned a multi-column set of independant continuous numeric sets
- and thus mlti normalizes each of the received columns on an independant basis
- mlti defaults to applying z-score normalizaiton by the nmbr trasnform, but alternative normalizations may be specified by passing an alternate trasnformation category to parameter norm_category, such as e.g. assignparam = {'mlti' : {'(targetcolumn)' : {'norm_category' : 'mnmx'}}}
- where specified transforms are accessed by inspecting that category's process_dict entries
- where targetcolumn needs to be specified as either the input column received in df_train or the first column in the upstream categorylist with suffix appenders
- note that parameters may be passed to the normalization trasnform by passing to mlti through parameter norm_params, e.g. assignparam = {'mlti' : {'(targetcolumn)' : {'norm_params' : {(parameter) : (value)}}}}
- inplace, inversion, and ML infill are all supported
- note that if an alternate treatment is desired where to apply a family tree of transforms to each column user should instead structure upstream trasnform as a set of numeric mlinfilltype transforms

6.80
- fixed a variable assignment in evalcategory that was interfering with the hash scenario for all-unique under automation
- which was the scenario associated with providing multicolumn hashing to unstructured text under automation
- update to the mlti transform rolled out in 6.79
- new parameter supported as 'dtype', accepts one of {'float', 'conditionalinteger'}, defaults to float
- found and fixed an edge case associated with memory sharing in data structure causing an overwrite scenario
- also now any defaultparam entries stored in process_dict entry for norm_category are taken into account
- which is more consistent with intent that mlti can apply any transformation category
- the only caveat is that since mlti is defined as concurrent_nmbr MLinfilltype, norm_category should be based on a process_dict entry with numeric MLinfilltype
- new MLinfilltype now supported as 'concurrent_ordl'
- concurrent_ordl is for transforms that return multiple ordinal encoded columns (nonnegative integer classification)
- note that for ML infill, each ordinal column is predicted seperately
- new process_dict entry available as mlto, which is built on top of the mlti transform but has concurrent_ordl MLinfilltype so allows norm_category specification with singlct MLinfilltype categories
- mlto defaults to a norm_category of ord3 and conditionalinteger dtype

6.81
- new transformation category available as GPS1
- GPS1 is for converting sets of GPS coordinates to normalized lattitude and longitude
- accepts parameter GPS_convention, which currently only supports the base configuration of 'default'
- which in future extensions may allow selection between alternate GPS reporting conventions
- 'default' is based on structure of the "$GPGGA message" which was output from an RTK GPS receiver
- which follows NMEA conventions, and has lattitude in between commas 2-3, and longitude between 4-5
- reference description available at https://www.gpsworld.com/what-exactly-is-gps-nmea-data/
- allows for variations in precisions of reported coordinates (i.e. number of significant figures)
- or variations in degree magnitude, such as between DDMM. or DDDMM.
- relies on comma seperated inputs
- accepts parameter comma_addresses as a list of four integers to designate locations for lattitude/direction/longitude/direction
- which consistent with the demonstration defaults to [2,3,4,5]
- i.e. lattitude is after comma 2, direction after comma 3, longitude after 4, direction after 5
- assumes the lattitude will precede the longitude in reporting, which appears to be a general convention
- also accepts parameter comma_count, defaulting to 14, which is used for inversion to pad out to format convention
- returns lattitude and longitude coordinates as +/- floats in units of arc minutes
- in the base root category definition GPS1, this transform is followed by a mlti transform for independent normalization of the lattitude and longitude sets
- in the alternate root category GPS2, the two columns are returned in units of arc minutes
- GPS1 returns two columns with suffix as column_GPS1_latt_mlti_nmbr and column_GPS1_long_mlti_nmbr
- also, moved naninfill application to following ML infill iterations to avoid interference
- new parameters accepts for power of ten binning transforms such as pwrs/pwr2/pwor/por2 as cap and floor
- cap and floor default to False, when passed as integer or float they cap or set floor on values in set
- for example if feature distribution is mostly is in range 0-100, you may not want a dinstinct bin encoding for outlier values over 1000
- found a flaw in our backward compatibility validation test, working now as intended

6.82
- small rewrite of GPS1 transforms, superseding the version rolled out yesterday, impacting backward compatibility with 6.81
- GPS1 now accepts additional GPS_convention parameter scenario of 'nonunique'
- 'nonunique' encodes comparably to 'default', but instead of parsing each row individually, only parses unique values, as may benefit latency when the data set contains a lot of redundant entries
- note that we expect most GPS applications will have primarily all unique measurements making them suitable for the default applied with GPS1, so nonunique really just here to support a particular alternate use case
- new root categories GPS3 and GPS4, comparable to GPS1 (i.e. with downstrema normalization via mlti), but applies GPS_convention = 'nonunique' as the default. 
- GPS3 differs from GPS4 in that GPS3 parses unique entries both in the train and test data, while GPS4 only parses entries in the train data, relying on the assumption that the test data entries will be the same or a subset of train data entries (as may benefit latency for this scenario)
- we recommend defaulting to GPS1 unless latency is an important factor, and otherwise experimenting based on the penetration of unique entries in your data to compare between GPS1/3/4
- new ML_cmnd specification now supported as ML_cmnd['full_exclude']
- full_exclude accepts a list of columns in input or returned column header convention, which are to be excluded from all model training, including for ML infill, feature importance, and PCA
- full_exclude may be useful when transforms may return non-numeric data or data without infill and you still want to apply ML infill on the other features
- new postprocess_dict entry 'PCA_transformed_columns' logging columns serving as input to PCA
- a tweak to PCA printouts, now displaying PCA_transformed_columns instead of column exclusions

6.83
- we had noted with 6.76 an intent to audit use of reserved string in string parsing functions
- turns out the bulk of the usage was in binarized string parsing functions sp19 and sbs3
- and then also used in inversion functions
- and then in a few otther places as an arbitrary string
- the sp19 and sbs3 usages are now replaced with the np.nan convention to be consistent with rest of library
- the inversion usages are also replaced with np.nan
- noting that we had a mistaken code comment on the matter, we were attributing a data type drift issue to pandas that we now believe had originated from numpy
- now resolved with the autowhere function rolled out in 6.74
- otherwise struck the reserved string throughout just out of spite for all of the trauma it put me through
- although left it in one place as a memorial for all of the times that we shared together
- reserved string now fully lifted, automunge accepts all the data, all the time, anywhere, anyhow

6.84
- transformation categories of MLinfilltype 'totalexclude' are now excluded from ML infill basis for other features
- by appending onto user specifications recorded in ML_cmnd['full_exclude']
- this serves purpose of allowing returned non-numeric data to automatically be excluded from ML infill basis
- which saves user the hassle of manually specifying exclusions
- 'totalexclude' is MLinfilltype for 'excl' transform (full passthorugh) and is now updated be registered for all other categories that may return non-numeric data
- if a passthrough transformed column is desired to be included in ML infill basis, can instead assign to one of other excl variants such as exc2, exc5, exc8
- clarifications added to read me library of transformations section on excl variants
- new NArowtype option as 'totalexclude', which is comparable to 'exclude' but source columns are also excluded from assignnan global option and nan conversions for missing data
- 'totalexclude' is now assigned as NArowtype for excl and null trasnforms
- updates to the processdict specifications associated with MLinfilltype, NArowtype, and concluding clarifications
- added clarifications of which between MLinfilltypes exclude/boolexclude/ordlexclude/totalexclude are included in ML infill basis
- added desription of new NArowtype totalexclude
- rephrased the concluding clarifications to processdict specifications for improved clarity, we believe this rephrasing makes the distinction between MLinfilltype and NArowtype scope more clear
- updates to feature selection in both automunge and postmunge to accomodate non-numeric sets, using basis exclusion specified under ML_cmnd['full_exclude'] as well as new convention of MLinfilltype 'totalexclude' exclusions
- updates to inspect NArowtype instead of MLinfilltype in _convert_to_nan and _assignnan_convert
- update to the wrapper functions for custom_train so that final infill exclusions are based on MLinfilltype instead of NArowtype
- identified a few cases of redundant sort operations applied in sp19 and sbs3 as well as un-needed search in list, simplified
- moved the validations of all valid numeric entries preceding ML infill model training to be inspected for each trained model instead of once for all models (made sense since the feature exclusions may be tailored to each infill model)

6.85
- found and fixed a snafu originating from 6.63 rewrite of evalcategory associated with the null scenario
- which is associated with received training feature sets of all missing data, which are struck
- it appears the all nan case was incorectly being assigned to the default transform for numeric, now resolved
- added a new powertransform option as 'infill2'
- comparable to powertransform = 'infill' in that it is intended for application towards data that is already numerically encoded and just ML infill is desired without normalizations and etc
- the difference is that with infill2 user is now allowed to include non-numeric sets in the data, which are given an excl pass through transform and excluded from ML infill basis
- one way to think about it is that infill will ensure returned sets are all valid numeric, and infill2 will only ensure received majority numeric sets are returned as all valid numeric.

6.86
- adjustment to label column treatment under automation in context of alternate powertransform scenarios
- added support for comparable label column treatment in powertransform scenarios {'excl', 'exc2', 'infill', 'infill2'}
- label column treatment for powertransform scenarios {False, True} remains comparable
- found and fixed bug associated with label columns applied to category excl
- ran an audit of the two support functions associated with 1010 conversions for ML infill
- which convert 1010 to onehot and vice versa
- revised support function for 1010 to onehot with comparable functionality to eliminate edge case
- found and fixed snafu in onehot to 1010 associated with derivation of binary form
- we anticipate these revisions will benefit performance of ML infill towards 1010 encodings which is the default categoric encoding under automation

6.87
- found and fixed a snafu originating from the autowhere rollout
- associated with populating a column for assignnan missing data injections
- was impacting assignnan['injections'] option, now resolved

6.88
- fixed the exc8 family tree to apply exc8 instead of exc5
- which impacts powertransform = 'infill'
- a few clarifications to read me regarding exc6/exc7/exc9
- a survey of family trees for cases where single category in family tree doesn't match root category
- there are tradeoffs, a more common category populated as a tree category might be easier to understand when viewing transformdict without inspecting processdict, decided would be more intuitive for this use to match the tree category to the root category, as would be less hassle to assign parameters in assignparam
- (we had several root categories populated in both conventions, decided it would be better to align to a single common convention, however limiting this update to root categories with a single tree category other than NArw and without offspring, root categories with more elaborate family trees we'll keep the convention of populating the more common tree category)
- updated nbr2/mnm4/101d/ordd/texd/bnrd/nuld/exc7/exc9/lbnm/lbnb/lb10/lbos/lbte/lbbn for this purpose with comparable functionality
- update to lbos so defaultparams is consistent with lbor
- corrected lbbn MLinfilltype to binary
- new NArowtype option as 'binary'
- binary now populated for transforms in the bnry family, such as bnry/bnr2/DPbn/bnrd/lbbn
- and treats entries other than the two most frequent as targets for infill for purposes of NArw aggregation
- identified and mitigated a potential edge case for noise injection transforms associated with index matching
- found and fixed bug in pwor associated with scenario negvalues=False, originating from recent rewrite
- revised our rollout validation tests, there were a few root categories we were previously omitting for esoteric reasons, rollout validations now cover all root categories in library (this was benefited by the new approach for MLinfilltype totalexclude)
- found and fixed bug in exc8 postprocess

6.89
- updated the read me processdict writeup to align with last rollout
- note that we have decided not to make further revisions to the medium version, that version is frozen
- audited automunge parameters to identify cases where internal operations could result in editing exterior objects from dictionary memory sharing, identified a few cases not yet mitigated, now resolved
- surveyed ML_cmnd usage for feature selection, consolidated to use of single version (now using only version returned from automunge call which includes any validation preparations)
- added function description to support function _trainFSmodel
- corrected code comments to align with what is drafted in code, in regards to MLinfillytpe being based on a tree category and not the root category in support function _LabelFrequencyLevelizer 
- (specifically the levelizer takes into account the MLinfilltype of the labelctgy entry (which is a tree category) associated with the root catregory)
- the struck comments were added in a recent code review in which I believe I had a small amount of confusion in interpretation between function variables labelscategory (the labelctgy tree category associated with the root category) vs origcategory (the label root category), the code implementation long preceded the addition of the struck comments
- struck scenario for concurrent_ordl MLinfilltype in TrainLabelFreqLevel
- added TrainLabelFreqLevel support for scenario of labels with integer MLinfilltype labelctgy supplemented by multirt
- recent convention is that any added code associated with backward compatibility accomodation includes the phrase "backward compatibility" in the code comment, the intent is that if we were ever to decide on any kind of backward compatibility breaking update we could then consolidate all of these scenarios at once (as used here backward compatibility refers to passing a postproces_dict to postmunge that was populated in a prior version than the one running postmunge)
- a few code comments added here and there for clarity
- struck some redundant MLinfilltype exclusions for boolexclude in support function _apply_am_infill and _apply_pm_infill
- struck concurrent_nmbr MLinfilltype exclusion from modeinfill and lcinfill in support function _apply_am_infill and _apply_pm_infill
- added stochastic_impute support for integer MLinfilltype
- exclude MLinfilltype is now subject to any data type conversion based on floatprecision parameter
- updated qbt1 family of transforms from exclude MLinfilltype to boolexclude to ensure exclusion from PCA dimensionality reduction
- this also results in qbt1 returned sets now being included in Binary dimensionality reduction and included in the boolean set in the returned columntype_report
- updated columntype_report so that MLinfilltype exclude is reported as numeric
- audited MLinfilltype and NArowtype inspections throughout codebase

6.90
- lngt family tree revised to omit downstream mnmx scaling
- new root category lngm, comparable to prior configuration of lngt
- lnlg now has downstream logn instead of log0
- (lngt returns string length of categoric entries)
- new root category GPS5, comparable to GPS3 (with GPS_convention of nonunique and assumption of test entries same or subset of train entries), but with downstream ordinal enocding instead of numeric scaling,  with lattitude and longitude seperately encoded
- GPS5 may be appropriate when there are a fixed range of GPS coordinates and they are wished to be treated as categoric
- note that alternate categoric enocdings may be applied by passing norm_category partameter to the downstream mlto
- note that if a single categoric encoding is desired representing the combined lattitude and longitude, the string representation can be passed directly to a categoric transform without a GPS1 parsing
- new root category GPS6, comparable to GPS5 but performs both a downstream normalization and a downstream ordinal encoding, allowing lattitude and longitude to be evaluated both as categoric and continuous numeric features. This is probably a better default than GPS3 or GPS5 for sets with a fixed range of entries.
- updated validation tests so that there is a category assignment representing each of the MLinfilltype options and a correpsonding inversion excluding PCA
- fixed the feature selection carveouts associated with totalexclude MLinfilltype that had been incorporated in 6.84 (we had missed a few edge cases resulting from the exclusions)
- new transform bnst, intended for use downstream of multicolumn categoric encodings, such as with 1010 or multirt MLinfilltype
- bnst aggregates multicolumn representations into a single column categoric string representation
- accepts parameter upstreaminteger defaulting to True to indicate that the upstream encodings are integers
- new root categories bnst and bnso, where bnst returns the string representation, bnso performs a downstream ordinal encoding
- inversion supported
- bnst or bnso may be useful in scenario where a multicolumn categoric transform is desired for label encoding targeting a downstream library that doesn't accept multicolumn representations for labels
- update to mlti transform to take into account dtype_convert processdict entry associated with the normcategory
- (so that if normcategory is in custom_train convention and dtype is conditionalinteger dtype conversion only applied when dtype_convert is not False, consistent with basis for custom_train otherwise)
- audited defaultinfill and dtype_convert, identified missing dtype_convert acomodation associated with floatprecision application to label sets, resolved
- updated tutorials to include link to data sets
- found and fixed bug in Binary inversion
- slight tweak to the returned column header conventions associated with Binary and PCA dimensionality reduction. Added an extra underscore to align with convention that received column headers that omit the underscore character are ensured of no suffix overlap edge cases
- now Binary returned with form 'Binary__#' and PCA returned with form 'PCA__#'
- conducted a walkthough of openning automunge code surrounding initial dataframe preparations, repositioned a few snippets for clarity
- moved a few spots up for automunge inplace parameter inspection so that it is performed prior to column header conversions
- moved assign_param variable initialization to a more reasonale spot
- moved list copying to internal state next where we do same for dictionaries
- moved application of _check_assignnan
- this walkthrough was partly motivated by ensuring inplace parameter inspection performed prior to any header conversions, and turned out to result in a much cleaner layout
- a few similar repositionings at openning of postmunge
- new validation results reported as check_df_train_type_result, check_df_test_type_result
- these results are from a validation that df_train is received as one of np.array, pd.Series, or pd.DataFrame and df_test is received as one of same or False
- similar validation results reported in postmunge for check_df_test_type_result
- the omission of this validation prior was an oversight

6.91
- with last rollout had put some thought into accomodating all potential edge cases for column header overlaps
- including a clarification added to the documentation in section for other reserved strings
- realized the Binary and PCA dimensionality reductions deserve some additional treatment due to their unique convention
- referring to how they apply new column headers by means other than suffix appention which is case for rest of library
- so have added convention that in cases where header overlaps are identified for PCA or Binary, the new column header is adjusted by addition of a string of the application number, which is a 12 digit random integer sampled for each automunge(.) call
- e.g. for PCA, 'PCA__#' would be replaced by 'PCA_############_#'
- this is similar to what has previously been done for the Automunge_index column returned in the ID sets
- additionally, to ensure we are being comprehensive, in very remote edge case where even the adjusted header has overlap, we add comma characters until resolved
- a validaiton result is logged for both Binary and PCA for cases where header is adjusted as set_Binary_column_valresult and PCA_columns_valresult
- added a note to the Binary writeup associated with how stochastic_impute_categoric may reduce the extent of contraction
- found an oversight in the Binary dimensionality reduction
- most of our rollout validations are conducted to ensure prepared train data matches prepared consistent test data
- in Binary dimensionality reduction, there is also a complication where test data activation sets may not match activation sets found in train data
- which since we have to accomodate inversion needs to be handled a little differently than the 1010 transform it is built on top of
- so accomodated by new null_activation scenario to 1010 transform as null_activation = 'Binary'
- which results in test data activation sets not matching train data activation sets being both returned from Binary as all zero activations as well as being returned from inversion as all zero activations
- returning from inversion as all zeros ensures upstream transforms will be able to handle inversion since 1010 transform encodings start from the all zero case, and in multirt the all zeros are treated as missing data
- similarly, new null_activation scenario to ordl and onht transform as null_activation = 'Binary' to support Binary options
- new convention for specifying list of Binary target by passing column headers as list
- previously we had a few reserved entries such as boolean False/True, None, etc which were reserved for this use in these lists, realized it is much cleaner to use alternate convention that uses same specification entries as the master Binary parameter, so now when specifying a subset of columns for Binary, can pass the first entry in the list as a Binary parameter value embedded in set brackets, for example if you want to consolidate any boolean integer sets derived from columns 'c1', 'c2', 'c3', and you want them aggregated into a ordinal Binary encoding, you can pass the automunge(.) Binary parameter as Binary = [{'ordinal'}, 'c1', 'c2', 'c3']
- new options for Binary dimensionality reduction as Binary = 'onehot' or 'onehotretain'.
- these are comparable to 'ordinal' and 'ordinalretain' except the returned consolidation is onehot encoded instead of ordinal encoded as you would expect
- the reason for this extension was to align with general convention in the library that all of the fundamental categoric transforms are available in corresponding forms differing as ordinal, one hot, or binarized representations.
- and even when a transform is only available in one of these forms can apply a downstream transform to translate
- such as can make use of an intermediate bnst to translate multicolumn forms or can otherwise apply directly

6.92
- updated Binary onehot scenario so that test activation sets not found in train are returned with all zeros to be consistent with other Binary options by way of tweaks to the null_activation='Binary' scenario
- aligned Binary postmunge(.) printouts with automunge(.)
- added Binary support for specifying multiple consolidation subsets by passing Binary as list of lists
- where first entry in each sublist can optionally serve to pass specification for that sublist by embedding specificaiton in set
- thus Binary can now be applied to seperately consolidate multiple non overlapping categoric sets
- corrected paraemter list copying at start of automunge(.) and postmunge(.)
- formalized the convention that we follow version numbers using float equivalent strings
- to support backward compatibility checks
- going forward, when reaching a round integer, the next version will be selected as int + 0.10 instead of 0.01

6.93
- postmunge(.) inversion now records and returns validation results in inversion_info_dict

6.94
- quick fix
- found a bug introduced in 6.88 with the new 'binary' NArowtype
- I don't know why this didn't show up in testing, will put some thought into that
- bug originated from the new NArw aggregations returning integers instead of boolean
- which was inconsistent with other NArowtypes
- resolved

6.95
- Binary now also accepts ordinal encoded categoric columns as targets for consolidation
- new validation test performed for labels_column parameter

6.96
- moved the validation for labelctgy processdict entry
- now takes place after identifying the root label category
- and limits validations to that entry
- which reduces overhead and moves an unnecessarily prominent printout for custom processdict
- a tweak to the labelctgy writeup (struck reference to functionpointer) to align with the functionpointer writeup

6.97
- moved the conversion of passed dictionaries and lists to internal state a little earlier
- updated the support function parameter_str_convert so that if first entry is enclosed in set brackets it is exclude form string conversion (relvant to new labels_column scenario)
- added parameter_str_convert to Binary specifications to be consistent with convention that received numeric column headers are converted to string
- restated for clarity: automunge(.) converts all column headers to string to align with suffix appention convention
- added a clarifying word to df_train description in read me that column headers should be unique strings
- labels_column automunge parameter now accepts specification of list of label columns, resulting in multiple labels in returned label sets
- each of the list entries may have their own root category assigned in assigncat if desired
- labels_column automunge parameter, when passed as a list of label columns, accepts a first entry set bracket specification to activate a consolidation of categoric labels, resulting in a single classifciation target even in cases of multiple categoric labels, where the form of multiple labels can automatically be recovered in an inversion operation for data returned from an inference
- such that a single classification model can then be trained for use with multiple classification targets
- set bracket specification options are consistent with those options supported for Binary, with exception that binarized with replacement requires specification as {True} 
- (instead of automatic when ommitted as is case with Binary)
- we recommend defaulting to the ordinal option, e.g. to consolidate three labels with headers 'categoriclabel_#':
- labels_column = [{'ordinal', 'categoriclabel_1', 'categoriclabel_2', 'categoriclabel_3']
- which can then be recovered back to the form of three seperate labels in postmunge with inversion='labels'
- updated form of postprocess_dict['labelsencoding_dict'], now with extra tiers of entries as well as entries associated with categoric consolidations

6.98
- a cleanup to strike some of the intermediate column forms from Binary returned from inverson (inversion was working, this just results in a cleaner returned set)
- similarly, a cleanup to strike in set returned from inversion Binary columns produced from a retain consolidation in cases where inversion was originally passed as 'test' or 'labels (to be consistent with the original form)
- added TrainLabelFreqLevel support for consolidated labels (supporting ordinal and onehot consolidations)
- added validation in automunge for column header string conversion to confirm did not result in redundant headers (could happen if e.g. received headers included 1 and '1')
- added validation to postmunge inversion denselabels case to confirm single label_column entry
- fixed a snafu in privacy_encode (turned out was recording length of wrong list as part of a derivation)

6.99
- a walkthrough of the process_dict data structure
- process_dict is initialized from assembleprocessdict and then after some preparations and validations are performed on user passed processdict the two are consolidated
- process_dict is then populated in the postprocess_dict upon initialization
- integration of process_dict into postprocess_dict was a deviation to support inspection of process_dict in custom_train postmunge wrapper function, which doesn't see process_dict but does see postprocess_dict
- but had the side effect of redundant variable being passed through family processing functions as process_dict and postprocess_dict['process_dict']
- realized it was a potential point of confusion to have the same data structure inspected in two different ways
- which was compounded when moved the labelctgy intialization from automunge start to label processing, as labelctgy initialization in some cases edits entries to the process_dict
- very simple solution, standardized on a single version of process_dict inspected in automunge(.) after postprocess_dict initialization as postprocess_dict['process_dict']
- the exception is for feature importance application, which takes place prior to postprocess_dict initialization, so this still sees as process_dict 

7.0
- extensions to privacy_encode option
- now an alternate columntype_report is returned as postprocess_dict['private_columntype_report']
- populated with the alternate privacy encoded column headers
- and when privacy_encode activated this replaces the original columntype_report
- also now when privacy_encode is activated, the order of columns is shuffled prior to assigning alternate headers
- postmunge printouts other than bug reports are now silenced in privacy_encode scenario
- struck a redundancy in postmunge inversion associated with renaming columns which interfered now that privacy_encode shuffles order of columns
- consolidated a redundant parameter in support functions processfamily, postprocessfamily, circleoflife, postcircleoflife
- a little fleshing out of the returned data structure postprocess_dict['labelsencoding_dict']
- which is intended as an alternative resource for label inversion for cases where user doesn't wish to share the entire postprocess_dict, such as for file size or privacy reasons
- now labelsencoding_dict records any trasnform_dict and process_dict entries that were inspected as part of label processing
- as well as recording normalization_dict entries associated with derived columns that were subject to replacement

7.10
- performed a walkthrough of ML infill support functions
- validated various translations applied as part of data prep for 1010 MLinfilltype and model training for libraries that that only accept single column classification labels
- found and fixed a ML infill bug for catboost associated with accessing Binary columns in the support function to populate columntype_report
- (catboost calls this function prior to applying Binary was reason for bug, I think this might have been associated with 6.89 updates)
- found and fixed edge case for autogluon ML infill associated with infill targeting single column categoric feature
- (autogluon likes categoric labels as strings to recognize the classification application)
- replaced a validation split with shuffling performed for ML infill with catboost, now using support function _df_split
- fixed a small bug in one of support functions rolled out in last update via negative (there was a scenario where recursion would halt too early)
- another privacy_encode extension, now when activated the column_map is erased in returned postprocess_dict
- found and fixed a process flaw for ML infill targeting a concurrent MLinfilltype

7.11
- major backward compatibility impacting update
- meaning postprocess_dict's populated in prior versions will require re-fitting to the train set using this or a later version or running an earlier version for postmunge(.) to prepare additional data
- this update was to align with the intent that all operations are to be channeled through the interface of two master functions: automunge(.) and postmunge(.)
- all internal support functions other than automunge(.) and postmunge(.) are now private functions, not accessible outside of the class
- took this backward compatibility impact as an opportunity to clean up all postmunge and postprocess operations that had dual configurations to accomodate backward compatibility 
- also, an audit of the insertinfill function identified opportunity for a more efficient application
- now with what was a kind of hacky pandas replace application replaced with an operation built on top of loc
- we expect this will benefit latency of this function which is used throughout ML infill and other assigninfill options
- struck some unused variables initialized in inversion
- new convention: returned postprocesss_dict omits entries for transformdict and processdict which were copies of user passed parameters
- new convention: returned postprocess_dict entries for transform_dict and process_dict only record transformation categories that were inspected as part of the automunge(.) call
- the thought was that this will benefit privacy in scenario where user has developed their own library of custom transformations, such that if they want to publish a populated postprocess_dict publicly, it will only reveal those portions of their library that were applied in derivation
- as further clarification on last update
- the concurrent ML infill process flaw was associated with the categorylist passed to inference
- although did not show up in testing since categorylist isn't inspected for default random forest implementation
- it is inspected for other learning libraries in inference
- was resolved by reframing categorylist passed to inference in concurrent scenario

7.12
- 7.11 introduced a bug for downloading and uploading postprocess_dict’s through pickle, which we didn’t catch in our testing since we didn’t run a backward compatibility check since it was a backward compatibility breaking update
- resolved by recasting functions directly stored in postprocess_dict as public functions (by removing a leading underscore from function name)
- which includes transformation functions and wrappers for training and inference
- fully resolved
- also, put some thought into privacy preservation associated with inversion operation
- inversion passed as list or set specification now halts when applied in conjunction with privacy_encode
- when privacy_encde is activated, inversion can be passed to postmunge(.) as one of {'test', 'labels', 'denselabels'}
- new convention is that the dataframe returned from inversion only includes recovered features
- also in process improved inversion so that the order of recovered features in the returned dataframe matches order of features as passed to automunge(.) through df_train
- found and fixed bug for inversion passed as list or set in privacy_encode scenario
- inversion denselabels option now preserves transformation privacy in returned headers
- now with privacy_encode inversion the returned inversion_info_dict masks the recovery path, replacing with boolean True

7.13
- new option for ML infill, user can now define and integrate custom functions for model training and inference
- documented in read me in section Custom ML Infill Functions (final section before conclusion)
- trying to clean up postprocess_dict a little bit
- new convention is postprocess_dict['autoMLer'] only returns entries that will be inspected in postmunge
- meaning if custom ML infill functions are passed they only need to be reinitialized prior to uploading postprocess_dict when they were applied
- postprocess_dict['orig_noinplace'] recast from a list to a set which should slightly benefit postmunge latency
- found and mitigated a remote Binary edge case associated with improper specification
- replaced an operation to reset column headers from use of numpy conversion to a pandas method in a few of autoML training functions
- new entries returned in postprocess_dict['columntype_report'] as postprocess_dict['columntype_report']['all_categoric'] and postprocess_dict['columntype_report']['all_numeric']
- these are list aggregations of all returned numeric features and all returned categoric features
- (columntype_report already included more granular detail such specific feature types and groupings)

7.14
- updated the ML_cmnd address to pass parameters to customML training and inference from ML_cmnd['MLinfill_cmnd']['customClassifier'] to ML_cmnd['MLinfill_cmnd']['customML_Classifier']
and from ML_cmnd['MLinfill_cmnd']['customRegressor'] to ML_cmnd['MLinfill_cmnd']['customML_Regressor']
- this was to better align on terminology by referring to custom ML operations as "customML"
- new validation result returned with Binary application as postprocess_dict['miscparameters_results']['Binary_columnspresent_valresult']
- Binary_columnspresent_valresult activates when a Binary specification includes a column header not found in the input or returned sets
- added a mitigation to leakage_dict specification for cases where was specified with key not found in set

7.15
- new option for privacy_encode parameter as 'private'
- previously privacy_encode accepted boolean defaulting to False, when True the column names and order of columns are anonymized, as well as some of returned data structures like columntype_report
- in the new 'private' option, these measures are supplemented by also activating that all datasets in automunge and postmunge will have their rows shuffled, consistent with what is otherwise available with the shuffletrain parameter
- additionially, as measures to further anonymize, inversion not supported for privacy_encode=='private', dataframe indexes are reset, and index columns retruned in ID sets are reset
- we recommend privacy_encode=='private' primarily as a resource for unsupervised learning applications
- or otherwise for scenarios of allowing model training / hyperparameter experiments by external party without needing to pair resulting inferences to specific input rows
- note that if you want to match the privacy_encode form in a seperate automunge(.) call with correpsonding data, you can do so by matching the automunge(.) randomseed
- and if you want to retain a row identifier without sharing with external party you can populate your own ID set
- we thought about some additional measures, like having postmunge require a minimum number of unique rows in order to prepare additional data, but for now since running postmunge means user has access to postprocess_dict there is no added benefit

7.16
- new encrypt_key parameter now available for automunge(.) and postmunge(.)
- automunge(.) accepts encrypt_key as one of {False, 16, 24, 32, bytes}
- where bytes means a bytes type object with length of 16, 24, or 32
- encrypt_key defaults to False, other scenarios all result in an encryption of the returned postprocess_dict
- 16, 24, and 32 refer to the block size, where block size of 16 aligns with 128 bit encryption, 32 aligns with 256 bit
- when encrypt_key is passed as an integer, a returned encrypt_key is derived and returned in the closing printouts
- this returned printout should be copied and saved for use with the postmunge(.) encrypt_key parameter
- in other words, without this encryption key, user will not be able to prepare additional data in postmunge(.) with the returned postprocess_dict
- when encrypt_key is passed as a bytes object (of length 16, 24, or 32), it is treated as a user specified encryption key and not returned in printouts
- when data is encrypted, the postprocess_dict returned from automunge(.) is still a diciotnary that can be downloaded and uploaded with pickle
- and based on which scenario was selected by the privacy_encode parameter, the returned postprocess_dict may still contain some public entries that are not encrypted, such as ['columntype_report', 'label_columntype_report', 'privacy_encode', 'automungeversion', 'labelsencoding_dict', 'FS_sorted']
- where FS_sorted is ommitted when privacy_encode is not False
- and all public entries are omitted when privacy_encode = 'private'
- the encryption key, as either returned in printouts or basecd on user specification, can then be passed to the postmunge(.) encrypt_key parameter to prepare additional data
- thus privacy_encode may now be fully private, and a user with access to the returned postprocess_dict will not be able to invert training data without the encryption key
- small deviation for privacy_encode == 'private' scenario
- we are keeping convention that train and test data have their rows shuffled and dataframe index reset
- decided that would be better to have some channel to recover index position in private scneario if needed
- so in the private scenario, the Automunge_index column returned in the ID sets is retained
- since ID sets are returned as a seperate datframe, if user wishes data to remain fully row wise anonymous they can share just the train/test/labels data but keep the ID sets private
- found a oversight in the privacy encoded versions of columntype_report, had thought columntype_report contained both training features and labels, forgot that labels are broken out into seperate label_columntype_report, now both are anonymized for privacy_encode
- found an oversight for privacy_encode associated with information channels in postmunge
- postmunge returned postreports_dict now omits entries ['featureimportance', 'FS_sorted', 'driftreport',  'rowcount_basis', 'sourcecolumn_drift']
- postmunge no longer supports non default entries for following parameters when privacy_encode was activated in postmunge: [featureeval, driftreport]

7.17
- quality control audit / walkthrough of: ML infill, stochastic impute, halting criteria, leakage tolerance, parameter assignment precedences, infill assignments
- found a process flaw for noise injections to ML infill imputations for categoric features, resolved by replacing a passed dataframe df_train with df_train_filllabel
- and also some consolidation to a single convention for infill and label headers returned from predictinfill
- so basically the new convention is derived infill, in all scenarios and all infill types, now is passed to insertinfill with a consistent column header(s) to the target feature
- added a copy operation to df_train_filllabel received by a few of the autoML wrappers to ensure column header retention outside of function
- updated column header convention for infill returned from predictinfill and predictpostinfill, resulting in now headers of returned infill match headers of labels which are matched to the categorylist (or adjusted categorylist for concurrent)
- this resolves a newly identified issue with recently revised insertinfill being applied to inferred onehot categoric imputations
- updated derived infill header for meaninfill and medianinfill
- fixed a variable address bug for naninfill in postmunge
- found a few cases of running for loops in lists that were editing, replaced with a deep copy, better python practice
- (deepcopy and .copy() for lists are a little different, is kind of grey area when list is a list derived from dictionary keys, went ahead with a deepcopy as a precautionary measure)
- new noise distribution options supported for DP family of transforms targeting numeric sets, including DPnb, DPmm, DPrt, DLnb, DLmm, DLrt, available by passing parameter noisedistribution as one of {'abs_normal', 'negabs_normal', 'abs_laplace', 'negabs_laplace'}, where the prefix 'abs' refers to injecting only positive noise by taking absolute value of sampled noise, and the prefix negabs refers to injecting only negative noise by taking the negative absolute value of sampled noise
- this may be suitable for some applications
- comparable noise distribution options also added for stochastic_impute, available by specification to ML_cmnd['stochastic_impute_numeric_noisedistribution']
- lifted restriction that inversion not supported with privacy_encode='private', now that we have encryption available user has avenue to restrict inversion as needed
- updated the convention for privacy_encode, now returned ID sets do not receive anonymization, only row shuffling and index reset is applied to match other sets with privacy scenario
- removing anonymization for ID sets which makes sense for a few reasons. First because inversion is not available for ID sets. Second is that it allows a channel for recovering row information that wouldn't otherwise have ability to recover even with inversion, with information in a seperate set from the returned features and labels. So if row anonymization is desired user can withhold the ID sets. It also makes sense since ID columns specified for automunge may be different than ID columns specified for postmunge. I like this convention.
- update to convention for privacy_encode, now label anonymization is only performed for the privacy_encode='private' scenario
- basically distinction between privacy_encode=True and 'private' is with respect to private has row shuffling, index reset, and label anonymization, and when encryption is performed, the True scenario returns a public resource for label inversion while the private scenario only allows inversion with the encryption key
- gradually honing in on a cohesive strategy for privacy_encode
- a correction to automunge ML infill targeting test data to have infil be conditional on test NArw instead of train NArw (this was a relic from earlier iterations where we didn't train a ML infill model when train set didn't have missing data, our current convention is when ML infill activated we train an imputation model for all features with supported MLinfilltypes
- a tweak to postmunge ML infill stochastic impute flow to better align with test data in automunge
- updated the writeup for encryption options rolled out in last update for clarity that it is built on top of the pycrypto library

7.18
- public labels inversion now supported for encrypted postprocess_dict without encryption key when privacy_encode != 'private'
- (basically that just means the entries needed for label inversion are not encrypted)
- this aligns with convention that labels are only anonymized for privacy_encode = 'private'
- in the process sort of a reorg of labelsencoding_dict returned in postprocess_dict, previous keys for 'transforms' and 'consolidations' are now omitted with information stored elsewhere
- reordered a few of the postmunge early operations to result in fewer postprocess_dict inspections prior to inversion and for clarity
- struck an unused postmunge variable testID_column_orig
- found a snafu with feature selection originating from 7.13 autoMLer cleanup in returned set (because feature selection calls automunge without ML infill it resulted in a returned postprocess_dict without autoMLer entries, updated convention so that an autoMLer entry is recored aligned with the autoMLtype (defaulting to random forest when not specified) even for cases where no ML infill was performed.)
- fixed a bug for postmunge feature selection when naninfill was performed in automunge (just turned off naninfill for the postmunge postmunge call)
- added a few code comments here and there for clarity

7.19
- struck the entry for 'privacy_headers_labels' from labelsencoding_dict since it reveals information about how many features are in train set
- fixed an edge case for public label inversion associated with excl suffix convention
- updated printouts for randomseed parameter validation
- new postmunge parameter randomseed, defaults to False, accepts integers within 0:2**32-1
- postmunge randomseed is now used for postmunge seeds that don't need to match automunge
- including row shuffling for privacy encode, which otherwise could have served as a channel for information recovery
- we still have order of column shuffling maintained between automunge and postmunge, is needed for ML

7.20
- new parameter accepted to DPod transform for categoric noise injection as 'weighted'
- weighted accepts boolean defaulting to False
- default False is consistent with prior configuration where noise injections are by a uniform random draw from the set of unique entries
- in weighted=True scneario, the draw is weighted based on frequency of unique entries as found in the training data
- this operation is built on top of the np.random.choice p parameter after calculating weights based on the training data
- we're leaving the unform sampling as our default since per numpy documentation it runs more efficently
- a small cleanup to processfmaily funciton reseulting in fewer lines of code with comparable functionality
- (basically some of the if/else scenarios were associated with derived columns as per a similar operation in processparent, column as passed to processfamily is always an input column, so just eliminated to irrelevant scenarios)
- new validation result reported as check_processdict4_valresult2 which checks for a chhanel of postmugne error when user specified processdict contains a prioritized dualprocess convention callable transformation function without a coresponding callable singleprocess trasnfomation function
- (prioritized just means that a callable custom_train wasn't also entered which would otherwise take precedence)
- note that if a common function is desired for both train and test data user can instead pass a function in the 'custom_train' or 'singleprocess' convention
- a few code comments related to potential future extensions

7.21
- updated defaults for categoric noise injections applied in DP family of transforms
- the new default is that noise injections are weighted by distribution of activations found in train data
- (making use of the weighted parameter from last update)
- based on numpy.random.choice documentation we expect there may be a small tradeoff with respect to latency for weighted sampling, we expect benefit to model performance may offset
- prior configuration still available by setting weighted parameter as False in assignparam
- comparable update made to categoric noise injections for ML infill, which now default to weighted injections per distribution of activations found in the train data
- weighted sampling can be deactivated for ML infill by setting ML_cmnd['stochastic_impute_categoric_weighted'] as False
- slight reconfiguration for halting criteria associated with ML_cmnd['halt_iterate']
- replaced an aggregation of tuples with a pandas operation
- (expect will benefit latency associated with summing entries by using pandas instead of iterating through tuples)
- running some empiracal inspections of halting criteria for infilliterate via ML_cmnd['halt_iterate']
- finding that particularly for the numeric tolerance criteria, the use of max delta in the halting formula is not a stable convergence criteria, as can fluctuate with single imputation entry
- so decided to revise the formula from to replace max(abs(delta)) with mean(abs(delta))
- now: "comparing for each numeric feature the ratio of mean(abs(delta)) between imputation iterations to the mean(abs(entries)) of the current iteration, which are then weighted between features by the quantity of imputations associated with each feature and compared to a numeric tolerance value"
- also based on similar empiracle inspections decided to raise the tolerance threshold for numeric halting criteria from 0.01 to 0.03
- can be reverted to prior value with ML_cmnd['numeric_tol']
- will update the associated appendix in paper Missing Data Infill with Automunge

7.22
- found and fixed a bug for multi-generation label sets originating from data structure populating (for public label inversion with encryption) rolled out in 7.18
- labels back to supporting full library including multi generations applied for noise injection
- some new powertransform scenarios now supported to support defaulting to noise injection under automation
- when powertrasnform = 'DP1', default numerical replaced with DPnb, categoric with DP10, and binary with DPbn
- when powertrasnform = 'DP2', default numerical replaced with DPrt, categoric with DPod, and binary with DPbn
- struck an unused validation result (check_transformdict_result2)

7.23
- added upstream primitive entry support for mlti
- corrected empty set scenario inplace accomodation for mlti postprocess 
- corrected mlti norm_category inplace_option inspection to align with convention that unspecified inplace_option in process_dict is interpreted as True
- found and fixed an edge case for mlti transform associated with norm_category without inplace support
- new noise injection transforms for intended use as downstream tree categories
  mlhs: for categoric noise injection targeting multicolumn sets with concurrent MLinfilltype (e.g. for use downstream of concurrent_act or concurrent_ordl)
  DPmc: for categoric noise injection targeting multicolumn sets (e.g. for use downstream of multirt and 1010)
- note that DPmc differs from DPod in that it doesn't require an ordinal encoded feature as input
- new root categories for noise injection to hashing transforms
  DPhs: hash with downstream noise injection (with support for multicolumn hashing with word extraction)
  DPh2: hsh2 with dowstream noise injection (single column case comparable to hsh2)
  DPh1: hs10 with downstream noise injection
- update to conventions for powertransform scenarios of 'DP1' and 'DP2', replacing hash and hsh2 scenarios with DPhs and DPh2 respectively
- new DPod parameter upstream_hsh2 for use when DPod is applied downstream of a hashing transform
- new mlti dtype parameter scenario mlhs for use when mlti applied downstream of a multicolumn hashing
- update to customML convention, now only the inference functions are stored in postprocess_dict for access in postmunge
- benefit of this convention is that if user downloads postprocess_dict with pickle and wants to upload in a new notebook, now the only need to reinitialize the inference functions
- which especially makes sense for QML
- which benefits privacy for special training configurations when postprocess_dict is shared publicly
- only tradeoff is lose access to cusotmML for training postmunge feature importance, compromise is we appy the default autoML_type instead
- small tweak to stochastic_impute_categoric for a better temp support column convention as integers to ensure no overlap with otherwise string column headers
- updated convention for DPbn root category, now noise is injected with DPod function instead of DPbn function
- previously they would have been equivalent, but now that DPod supported weighted samples is a better resource 
- in other words, DPbn now supports weighted sampling for noise injection

7.24
- alternate autoML libraries are now given a conditional import instead applying import by default
- since imports are conducted internal to their support function, the import requires reinitialization with each ML infill model training etc. By conducting a conditional import, user now has option to perform associated imports external to automunge(.) or postmunge(.) (instead of an automated import with each function call), which in some cases will benefit latency. Or when external import omitted support functions conduct internal exports as prior.
- update to evalcategory, now majority numeric data with nunique == 2 defaults to bnry instead of nmbr root category (usefule in scenario where numeric labels may be a classificaiotn target which is not uncommon)
- added note to the customML writeup in read me for customML_train_classifier template: "label entries will be non-negative str(int) with possible exception for the string '-1'"
- realized last update claim of "only need to reinitialize the inference functions" was not sufficiently validated, had missed that customML funcitons are also saved in ML_cmnd postprocess_dict entry, now resolved
- update to customML convention, now library has a suite of internally defined inference functions for a range of libraries, including {'tensorflow', 'xgboost', 'catboost', 'flaml', 'autogluon', 'randomforest'}
- this was inspired by realization that since we only needed to reinitialize customML inferance functions in new notebook prior to pickle upload of postprocess_dict, and since most libraries inference operations are fairly simple and common, by giving option of an internally defined inference function user can now apply customML and share the postprocess_dict publicly without a need to parallel distribute the custom inference function
- user defined customML inference functions are specified through ML_cmnd['customML']['customML_Classifier_predict'] and ML_cmnd['customML']['customML_Regressor_predict']
- user can now alternatively populate these entries as a string of one of {'tensorflow', 'xgboost', 'catboost', 'flaml', 'autogluon', 'randomforest'} to apply the default inference function associated with that library
- Please note we do not yet consider these default inference functions fully audited - pending further validations. As implemented is intended as a proof of concept.

7.25
- today's theme was all about improved clarity of code
- we performed a full codebase walkthrough for purposes of aggregating navigation support
- we've introduced convention that each set of function defintions are grouped by theme
- which in most cases was already in place, in a handful of cases we moved a few functions around to better align with this grouping coherence
- we've introduced kind of a table of contents at the start of the AutoMunge class definition
- listing  what we're referring to as "FunctionBlock" entries
- which are basically a title for the theme of a set of function definitions, each including in the table the list of associated functions
- code base navigation can now more easily be performed by using these FunctionBlock titles as a key for a control F search
- similarly, we performed a more detailed walkthrough of the two master functions for interface: automunge(.) and postmunge(.)
- and for each introduced a kind of table of contents with what we're referring to as "WorkflowBlock" entries
- the WorkflowBlocks are for cataloging segments of the lines of code by key themes
- which also can be navigated by a control F search
- and their entry in the code include codee comments of high level summary of key operations for the block
- we expect this update will significantly benefit code navigation
- which since we group everything in a single file was probably long overdue

7.26
- a few code comment cleanups to evalcategory associated with application of bnry to numeric sets with 2 unique values, added this clarifiaiton that also applies to numeric labels
- updated to convention for temporary columns initialized as part of transfomrations
- now non-returned temporary columns are named with integer column headers, which is an improvement as it eliminates a suffix overlap channel since other columns will be strings
- relevant to transforms tmsc, pwrs, pwor, DPod
- added code comment to custom_train_template in read me as:
#we recommend naming non-returned temporary columns with integer headers since other headers will be strings
- converted support function _df_split to a private function (I think I may have been using in an experiment was why it was not previously private, not positive)
- made imports conditional for encryption support functions
- added printouts to final suffix overlap results associated with cases where PCA, Binary, or index columns were adjusted to avoid overlap (not an error channel, just an FYI)
- new PCA option to retain columns in returned data that served as basis
- similar to the Binary retain options
- can be activated by passing ML_cmnd['PCA_retain'] = True
- an update to support function associated with ML infill data type conversions to eliminate editing dataframe serving as input for __convert_onehot_to_singlecolumn
- in 7.24 had added note to the customML writeup in read me for customML_train_classifier template: "label entries will be non-negative str(int) with possible exception for the string '-1'"
- this convention is now updated to label entries will be non-negative str(int)
- I think this also fixes a snafu in our flaml autoML_type implementation since was using integer labels for classification instead of strings
- updated automunge and postmunge WorkflowBlock addresses to be based on a unique string (previously there were some redundant string addresses between automunge and postmunge)
- realized I wasn't applying ravel flattening to test_labels returned from automunge in pandasoutput = False scenario, which was an oversight
- used as an opportunity to rethink aspects of single column conventions for other sets
- we had in place that all single column pandas sets are copnverted to series already
- but only single column numpy arrays were flattened
- which was kind of not aligned
- decided that for benefit of common pandas operations independant of single or multi column case to features and ID sets it made sense to limit series conversions to label sets
- now updated and now aligned convention is with pandas output only single column label sets are converted to series, and with numpy output only single column label sets are flattened with ravel
- updated the empty set scenario for numpy output in postmunge for ID and label sets to be returned as numpy arrays consistent with form returned from automunge (previously were returned as empty list)
- found a simplification opportunity for a support function associated with populating a data structure as __populate_columnkey_dict
- updated convention for normalization_dict entries found in postprocess_dict['column_dict']
- previsouly we redundantly saved this dictionary for each column returned from a transform
- since we only inspect it for one column for use in postmunge, decided to eliminate the redundancy to reduce memory overhead
- (as in some cases, as with high cardinality categoric sets, these may have material size)
- now only saved for column that is first entry in categorylist
- updated postmunge driftreport to only report for first column in a categorylist
- (this was for compatibility for new normalziation_dict convention but had side effect of making for cleaner reporting by eliminating redundancy in output)
- updated a variable naming in infill application support function for clarity (from "boolcolumn" to "incompatible_MLinfilltype")
- formalized convention that any previously reported drift stats only reported in a single normalizaiton_dict out of a multi column set are now aggregated to a single reported form
- ML_cmnd returned in postprocess_dict now records the version number of application for use in postmunge

7.27
- reverted conditional imports for ML infill and encryption
- it appears this functionality did not work as expected for cases of redundantly called functions
- (it would successfully import on the first call, but then on susbsequent application the module was present in sys.modules even though function didn't have access to it, resulting in no import)
- appears to be another case of insufficient validation prior to rollout
