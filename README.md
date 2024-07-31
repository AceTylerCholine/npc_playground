# All of Tyler's Notebooks & Data

 This repository holds all the scripts (in python) and data that were used for data analysis and figure creation for the the Reward Competition Extension project, as well as various other useful things.
 Full disclosure: this repository was reorganized with no double checking about dependencies in terms of calling data or other code; so be aware that there could be an issue while running things. 

 ## For a Tutorial of How Most of the EPhys Data was Processed, go to `General_ephys_notebooks\Tutorial_for_multirecording_spikeanalysis.ipynb`
  I recommend downloading the whole repo and running the notebook step by step in the base directory, so you can see how the `multirecording_spikeanalysis.py` script works.

 ## Directory Structure

- Base Directory:
    - `multirecording_spikeanalysis.py`: Current version of Padilla-Coreano Lab ephys script
    - `multirecording_spikeanalysis_old.py`: Original ephys script I based the edits off of
    - `multirecording_spikeanalysis_edit.py`: Lots of minor spacing edits and added '*create_spiketrain_df*' function
    - `multirecording_spikeanalysis_edit2.py`: Same as `multirecording_spikeanalysis_old.py`, but added '*create_spiketrain_df*' function
    - `multirecording_spikeanalysis_edit4.py`: Same as `multirecording_spikeanalysis_edit2.py`, but 1 line edit to '*create_spiketrain_df*' function
    - `spikeanal.py`: Similar to `multirecording_spikeanalysis_old.py` but also:
        - spacing edits
        - **w_assessment** function edited from `except TypeError: 'NaN'` to `else: 'not significant`
        - **smoothing_window** defaults to 250 instead of None
    - `rce_pilot_2_per_video_trial_labels.xlsx`: Spreadsheet of event outcomes (e.g.: win/lose/rewarded) and timestamps for Cohort 2
    - `rce_pilot_3_alone_comp_per_video_trial_labels.xlsx`: Spreadsheet of event outcomes (e.g.: win/lose/rewarded) and timestamps for Cohort 3
    - `combined_excel_file.xlsx`: Python merge of 2 behavior spreadsheets (from `General_ephys_notebooks\Merge_spreadsheets.ipynb`)
    - `ms conversion.txt`: Notes on how various timestamps relate to each other
    - I'm not sure which versions of the *spikeanal* scripts are best to keep, but I'll try to get to that later
- `Behavioral_clustering`: Leo created an unsupervised ML clustering from 30s windows (10s before, 10s during, 10s after competitions) of SLEAP data (velocity, position, direction of both mice). The project notebooks under this folder tried to analyze how many/which units were responsive to each cluster. Some of the clusters are characterized by both mice being at the port, and some clusters are characterized by one mouse at the reward port and the other in the corner facing away. There should probably be mPFC units that are responsive to specific states like those.
    - The only analysis completed in this project so far is the transition proability matrix, but that doesn't use epys data, just timestamps of clusters. I was still trying to figure out the best way to work with the data and how exactly to ask the question of what units change their firing rate in response to each cluster, partially because each occurrence of each cluster is a variable length, and should the comparitive baseline be the 30s window in question, all 30s windows, or the whole recording, and with or without including that cluster. You could potentially make one long array of each cluster and compare them to each other? but they would also be different lengths. I think the best baseline would be during the 30s windows excluding the cluster in question.
    - The other issue that was raised in this analysis was the duration of each cluster and the duration between clusters. Because we know the windows are 30s long, I attempted to create windows by finding the earliest timestamp, adding 30,100 ms to that, consider that the first window, then start the next window at the next timestamp. I believe I was pretty successful in this. Also, because some clusters were so short there is a function ***process_timestamps_nested*** which uses 2 other functions ***combine_intervals*** (if cluster 1 occurs and less than 250 ms later it occurs again, I assumed that would be considered noise and merged those 2 timestamp ranges into 1 long timestamp range) & ***remove_short_intervals*** (if a cluster has a duration of <250 ms, it is considered noise and is dropped). 250 ms was an arbitrary but agreed upon duration by Tyler, Meghan, & Nancy.
    - After the noise has been reduced and the windows have been created, there are still gaps in between cluster timestamp ranges (see *valid_differences_array* in the notebook). The most common duration between clusters by far is 69-70 ms. I believe the clustering was done on every 3rd video frame to make the process quicker, and if the video frames were ~34 ms, then the 69-70 ms gaps are just the 2 frames that were dropped between cluster end and start times. There are also plenty of 200-500 ms gaps, that I don't know how to explain. Then the 20-80 s gaps between clusters are just between windows.
    - `Transition Matrix Example.xlsx`: An example of how `Transition_prob_matrix.ipynb` works
    - `rce_pilot_3_alone_comp_cluster_ranges.pkl`: A pickled df of the timestamps of each cluster. Each row is a subject's recording, and there are 5 columns of timestamp dictionaries. I mainly used 'cluster_timestamps_ranges_dict'
        - ***cluster_index_ranges_dict***: I believe is video frame, not really useful
        - ***cluster_times_ranges_dict***, & ***trial_cluster_times_ranges_dict***: I don't understand what these dicts are
        - ***cluster_timestamps_ranges_dict***, & ***trial_cluster_timestamps_ranges_dict***: These are the ephys 20 KHz timestamps, so to get to ms you divide by 20 (I use floor division). The difference between these 2 dicts/columns is that *cluster...* gives timestamps for each behavioral cluster, but *trial_cluster...* gives further divides the clusters between win/lose/tie, because for example, if '*cluster 7*' was typcially characterized by 1 mouse near the port and 1 mouse in the back corner, the cluster is defined by the scene, not the subject, but the neuronal activity would be expected to be completely different between the 2 mice, so because the rows are subject specific, the 2nd dict tells you which mouse your current subject was in that cluster. Ex: '*win_7*' & '*lose_7*' instead of just '*7*'
    - `Transition_prob_matrix.ipynb`: Creates a **Transition Probability Matrix** where row is origin cluster and column is transitioned cluster. The idea here was, do some clusters typically transition to other clusters? Are some clusters simply transition states between 2 other clusters? Because the occurrence of each cluster isn't equal (Cluster 1 might occur 40 times in a recording while Cluster 2 might only occur 5 times), if we just plotted the observed transitions into a heatmap, it wouldn't tell us much, because rows/columns of the most common clusters would appear hotter than the rest. So, in this notebook, an '*expected probability*' matrix is created, then an '*observed count*' matrix is created, then the '*observed count*' is converted to an '*observed probability*', then an '*observed-expected*' matrix is created by subtracting the 2. The next step I'm less convinced about, but I still think is the right way; if a probability goes from 0.05 to 0.15, that is a lot more important than if a probability goes from 0.45 to 0.55, so the '*observed-expected*' matrix is converted to a '*proportional observed-expected* matrix. **This is the only completed notebook in this directory**
    - 
- `General_ephys_notebooks`:
- `Move_edit_data_notebooks`:
- `Neuronal_classifying`:
- `Newest_UMAP`:
- `leo_poster`: All of the scripts to make the single-unit analysis figures for Leo's 2024 GRC poster, as well as the completed figures (although the labels/titles are edited on BioRender)
    - `Cohort2+3_Alone_Comp_Venn.ipynb`, `Cohort2+3_LinePlots.ipynb`, `Cohort2+3_PiePlots.ipynb`: These notebooks use all of Cohort 2 + Alone Comp from Cohort 3 to make the Venn Diagram, Line Plots, & Pie Plots
        - Example of a 3rd bullet point for template
- `recordings`:
- `rubbish`:
