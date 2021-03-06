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
