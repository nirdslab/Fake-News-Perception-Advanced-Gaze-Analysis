# Analyze FakeNewsPerception Dataset Using Advanced Gaze Measures

FakeNewsPerception: An eye movement dataset on the perceived believability of news stories

This repository contains the scripts that are used to analyze the FakeNewsPerception dataset.

## Dataset

* FakeNewsPerception Dataset:
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/C1UD2A

## Instructions

### Setup

* **Python Version: 3.9**

### Install packages

* Install requirements 
```commandline
pip install -r requirements.txt
```

### Process Raw Data

Here, we read the raw data and process them.

1. Filter out mouse data
2. Remove invalid data
3. Replace comma with decimal point
4. Process invalid eye data
5. Split data by stimulus
6. Remove unwanted columns

* Run `process_raw_data.py`

### Add AOI labels

In order to analyze ambient/focal coefficient k and gaze transition entropy, we need aoi information for each eye
movement data.

1. We obtain the x, y, width, height, region (text, image, subheading) information for each stimulus and version from
   D4-Stimuli-ROI/annotations.csv in FakeNewsPerception dataset.
2. We obtain each version (fake or true) of stimulus read by participant from D2-Processed-features.csv in
   FakeNewsPerception dataset.
3. We calculate the mean x and mean y from the gaze left x and gaze right x, and gaze left y and gaze right y in raw
   processed data.
4. We check the region each eye movement record belongs to by comparing against the region start x, end x, start y, and
   end y information.
5. Any eye movement record inside the stimulus but not comes under any of the regions, we label them as "outside".

* Run `add_aoi_labels.py`

### Normalize Data

Prior to sending the data to Gaze Analytics Pipeline, we

1. Normalize the data using resolution
2. Generate mean x, mean y using Gaze point x and y values for left and right eye.
3. Generate mean pupil diameter using left and right pupil diameters

* Run `normalize_eye_movements.py`

### Update Timestamp

In the dataset, Eye tracking timestamp and Recording timestamp are given milliseconds. Similarly, after processing
data (removing rows with invalid data we have modified the raw dataset), data recorded timestamp does not reflect the
eye tracker's frequency (600Hz). Hence, we update the timestamp in normalized data in the previous section, to start
from 0 seconds and increase at 1/600 seconds rate.

* Run `update_timestamp.py`

### Generate Advanced Gaze Measures

We calculate the following advanced gaze measures using the Advanced Eye Movements Analysis Pipeline (Gaze Analytics
Pipeline) [1].

* Ambient/Focal Attention with Coefficient K
* Gaze Transition Entropy
* LHIPA

The Gaze Analytics Pipeline is available at: http://andrewd.ces.clemson.edu/etra24/src/emip.zip

### Add Class Labels

We aggregate the calculated advanced gaze measures with news version and believability ratings obtained from
D2-Processed-features.csv in FakeNewsPerception dataset.

* Run `add_class_labels.py`

### Statistical Analysis

1. To analyze the behavioral performance in identifying fake and real news and generate the rated believability
   distribution figure, run `rated-believability-analysis.py`.

2. To generate the analysis of following advanced gaze measure, run respective code.
   1. Ambient/Focal Attention with Coefficient K - `analyze_cofficient_k.py`
   2. Gaze Transition Entropy - `analyze_gaze_transition_entropy.py`.
   3. LHIPA - `analyze_lhipa.py`

## References

[1] Duchowski, A. T. (2017). The gaze analytics pipeline. In Eye Tracking Methodology, pages 175–191.504
Springer Publishing, New York, NY, USA.