# AutoMunge Package

AutoMunge is a tool for automating the final steps of data wrangling prior to the 
application of machine learning. The automunge(.) function processes structured training 
data and if available test data, which when fed pandas dataframes for these sets returns
transformed numpy arrays numerically encoded, with transformations such as z score 
normalization for numerical sets, one hot encoding for categorical sets, and more (full
documentation pending). Missing data points in the set are also addressed by predicting
infill using machine learning models trained ont he rest of the set in a generalized and
automated fashion. automunge(.) also returns a python dictionary which can be used as an
input along with a subsequent test data set to the function postmunge(.) for subsequent
consistent processing of test data which wasn't available for the initial address.

Patent Pending

Although full documentation is pending you can read more about the tool through the
blog posts documenting the development on medium [here](https://medium.com/@_NicT_).
