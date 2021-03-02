Contains information about the folder and its contents

This folder contains the raw data, and should not be changed in any way. You should use this to make new data files.

The name of the files state how many years into the future a company went bankrupt. 
- 1year means that the company went bankrupt one year ahead from when the data was collected.
- 5year means that the company went bankrupt five years ahead from when the data was collected.

The raw data from the website is in .arff format, which is a comma delimited file with that starts with some variable declarations. These cannot normally be read by most delimiter imports.
The data is therefore imported into Stata, and reformated into .txt files, that are comma delimited and should be read by most programs without issues.