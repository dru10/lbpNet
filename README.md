# Local Binary Pattern Network

## CV2 Project

Cristina Cirstea, Horia Rusan, Octavian Geoarsa

## Local Binary Pattern descriptor calculation

Found in `lbp.py` under the function `lbp_descriptor`.

Computes LBP descriptor for central area crop 100x170 of original image

## Dataset

Using LFW dataset, can be downloaded from:
https://drive.google.com/file/d/1p1wjaqpTh_5RHfJu4vUh8JJCdKwYMHCp/view

LFW references:

1. https://talhassner.github.io/home/projects/lfwa/index.html
2. http://vis-www.cs.umass.edu/lfw/

## Running the scripts

1. Create a virtual environment
`python3 -m venv venv`

2. Activate virtual environment
`. venv/bin/activate`

3. Install requirements file
`pip install -r requirements.txt`

4. Run files. Example:
`python lbp.py`

