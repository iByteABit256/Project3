                   ---------------
                   == Project 3 ==
                   ---------------
            Παύλος Σμιθ   -   Μάριος Γκότζαϊ
             sdi1800181         sdi1800031


Instructions
-------------

Forecast: $python forecast.py -d <dataset> -n <number of time series used>
Detect: $python detect.py -d <dataset> -n <number of time series used> (-m <max absolute error>)
Reduce: $python reduce.py -d <dataset> -q <queryset> -D <output_dataset> -Q <output_queryset>



Examples
---------

Forecast: $python forecast.py -d data/nasdaq2007_17.csv -n 10
Detect: $python detect.py -d data/nasdaq2007_17.csv -n 10 -m 0.65
Reduce: $python reduce.py -d data/nasd_input.csv -q data/nasd_query.csv -D dataset_output.csv -Q queryset_output.csv



Project Structure
------------------

.: python source files
data: input files
