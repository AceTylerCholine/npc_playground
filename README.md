# All of Tyler's Notebooks & Data

 This repository holds all the scripts (in python) and data that were used for data analysis and figure creation for the the Reward Competition Extension project, as well as various other useful things.
 Full disclosure: this repository was reorganized with no double checking about dependencies in terms of calling data or other code; so be aware that there could be an issue while running things. 

 ## For a Tutorial of How Most of the EPhys Data was Processed, go to `General_ephys_notebooks\Tutorial_for_multirecording_spikeanalysis.ipynb`
  I recommend downloading the whole repo and running the notebook step by step in the base directory, so you can see how the `multirecording_spikeanalysis.py` script works.

 ## Directory Structure

- Base Directory:
    - `multirecording_spikeanalysis.py`: Current version of Padilla-Coreano Lab ephys script
    - `multirecording_spikeanalysis_old.py`: Original ephys script I based the edits off of
    - `multirecording_spikeanalysis_edit.py`: Lots of minor spacing edits and added 'create_spiketrain_df' function
    - `multirecording_spikeanalysis_edit2.py`: Same as `multirecording_spikeanalysis_old.py`, but added 'create_spiketrain_df' function
    - `multirecording_spikeanalysis_edit4.py`: Same as `multirecording_spikeanalysis_edit2.py`, but 1 line edit to 'create_spiketrain_df' function
    - `spikeanal.py`: Similar to `multirecording_spikeanalysis_old.py` but also:
        - spacing edits
        - w_assessment function edited from `except TypeError: 'NaN'` to `else: 'not significant`
        - smoothing_window defaults to 250 instead of None
    - `rce_pilot_2_per_video_trial_labels.xlsx`: Spreadsheet of event outcomes (e.g.: win/lose/rewarded) and timestamps for Cohort 2
    - `rce_pilot_3_alone_comp_per_video_trial_labels.xlsx`: Spreadsheet of event outcomes (e.g.: win/lose/rewarded) and timestamps for Cohort 3
    - `combined_excel_file.xlsx`: Python merge of 2 behavior spreadsheets (from `General_ephys_notebooks\Merge_spreadsheets.ipynb`)
    - `ms conversion.txt`: Notes on how various timestamps relate to each other
    - I'm not sure which versions of the spikeanal scripts are best to keep, but I'll try to get to that later
- `Behavioral_clustering`: Leo created an unsupervised ML clustering from 30s windows (10s before, 10s during, 10s after competitions) of SLEAP data (velocity, position, direction of both mice). The project notebooks under this folder tried to analyze how many/which units were responsive to each cluster. Some of the clusters are characterized by both mice being at the port, and some clusters are characterized by one mouse at the reward port and the other in the corner facing away. There should probably be mPFC units that are responsive to specific states like those.
    - `Transition Matrix Example.xlsx`: An example of how `Transition_prob_matrix.ipynb` works
    - `rce_pilot_3_alone_comp_cluster_ranges.pkl`: A pickled df of the timestamps of each cluster. Each row is a subject's recording, and there are 5 columns of timestamp dictionaries. I mainly used 'cluster_timestamps_ranges_dict'
        - *cluster_index_ranges_dict*: I believe is video frame, not really useful
        - *cluster_times_ranges_dict*, & *trial_cluster_times_ranges_dict*: I don't understand what these dicts are
        - *cluster_timestamps_ranges_dict*, & *trial_cluster_timestamps_ranges_dict*: These are the ephys 20 KHz timestamps, so to get to ms you divide by 20 (I use floor division). The difference between these 2 dicts/columns is that *cluster...* gives timestamps for each behavioral cluster, but *trial_cluster...* gives further divides the clusters between win/lose/tie, because for example, if 'cluster 7' was typcially characterized by 1 mouse near the port and 1 mouse in the back corner, the cluster is defined by the scene, not the subject, but the neuronal activity would be expected to be completely different between the 2 mice, so because the rows are subject specific, the 2nd dict tells you which mouse your current subject was in that cluster. Ex: 'win_7' & 'lose_7' instead of just '7'
- `General_ephys_notebooks`:
- `Move_edit_data_notebooks`:
- `Neuronal_classifying`:
- `Newest_UMAP`:
- `leo_poster`: All of the scripts to make the single-unit analysis figures for Leo's 2024 GRC poster, as well as the completed figures (although the labels/titles are edited on BioRender)
    - `Cohort2+3_Alone_Comp_Venn.ipynb`, `Cohort2+3_LinePlots.ipynb`, `Cohort2+3_PiePlots.ipynb`: These notebooks use all of Cohort 2 + Alone Comp from Cohort 3 to make the Venn Diagram, Line Plots, & Pie Plots
        - Example of a 3rd bullet point for template
- `recordings`:
- `rubbish`:
