Credit to Jan-Lucas Uslu, Steven Tran, and Chaitrali Duse for much of the original code base.

The code in this repository is used for fitting moirés and atomic lattices as measured using 
torsional force microscopy (TFM) scans taken on a Bruker Icon AFM. Can be applied to other
SPM techniques for probing moirés or atomic lattices quite easily. Code needs to be adjusted
for reading in scans from other instruments.

Notebook file is fairly well commented. 

To do:
Currently it can only process square scans. There is some mix-up in the x and y for the scan_size and 
scan_shape parameters (one is x, y and the other is y, x). I have to go through manually and figure out 
where the issues are, but I haven't done it yet. 
