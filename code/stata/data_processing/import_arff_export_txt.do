/* This do file imports the raw .arff files, and do some minimal cleaning
and exports them as .txt files. These .txt can be used without extra work. */


clear all
* Local path to main dir
global dir = "/Users/karl/Documents/01_personlig/04_Koding/GitHub/2021_ml_seminar"

* Path to data within main dir
global raw_data = "data/raw_do_not_touch/"
cd $dir/$raw_data

* Get filenames to all .arff files
local arff_files: dir . files "*.arff"

* Loop over all .arff files, remove the first rows that contain useless 
* information. Missing are stored as "?", I replace those with "", and then
* convert all strings to numerics. I then save them as .txt files.
foreach _file in `arff_files' {
	* Import .arff file and save its name
	local _file_name = substr("`_file'", 1, 5)
	import delimited `_file', delimiters(",") clear
	
	* The first 67 rows are variable declarations. I also check that v3 is empty
	* just as an extra check, to make sure that no mistakes are made.
	drop if _n < 68
	ds, not(type numeric)

	* Loop over string variables, and replace "?" with "" (missing)
	foreach var in `r(varlist)' {
		replace `var' = "" if `var' == "?"
	}
	
	* Destring all string variables into numerics.
	destring *, replace
	
	compress
	export delimited using `_file_name'.txt, replace
}
