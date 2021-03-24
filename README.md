# HFIR_BG_Analysis
scripts for background analysis


## installation
There is a c function used to write .spe format files. If needed you must compile this:

`cd src/utilities`

`cc -fPIC -shared -o write_spe.so write_spe.c`
