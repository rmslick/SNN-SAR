
MSTAR (Public) Targets T-72, BMP-2, BTR-70, SLICY


This data is organized in directories that can be 
accessed using an Internet Web browser.  The START.HTM file may 
be opened with a Web browser (using 'open file') to peruse the 
contents.  The binary data files may be read using 
appropriate software.

FILE FORMAT

Each file is constructed with a prepended, variable-length, 
Phoenix formatted (ASCII) header which contains detailed ground 
truth and sensor information for the specific chip.  Following 
the Phoenix header is the data block.  The data block is written 
in Sun floating point format and is divided into two blocks, a 
magnitude block followed by a phase block.  Byte swapping may be 
required for certain host platforms.  Tools for reading and 
manipulating the header information may be found at 
https://www.sdms.afrl.af.mil .



For more information, please send an email message to:

sdms_help@mbvlab.wpafb.af.mil

or Check out the following URL:

https://www.sdms.afrl.af.mil
