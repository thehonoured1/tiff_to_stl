## tiff_to_stl
The objective is to parse a set of scans from a .tif file into an .stl file.
This program is built around tooth models, and contains diagram outputs to demonstrate where the borders are drawn.

## As of right now:
There are two components:
### converter.py
work in progress

### oct_processing_pipeline.py
It takes a .tif file and does the following:
- apply a median filter on the raw volume
- for each image: produces a cost matrix and compute the top boundary with it. Maximum kernels are applied here.
  - create a binary mask as well, used in constructing a collection of binary masks modelled after the original .tif scans.
- if 'demo_mode' is toggled true, diagrams detailing the process on the 25,50,75 percentile images will be plotted out.
