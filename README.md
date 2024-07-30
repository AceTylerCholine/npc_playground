# All of Tyler's Notebooks & Data

 This repository holds all the scripts (in python) and data that were used for data analysis and figure creation for the the Reward Competition Extension project, as well as various other useful things.
 Full disclosure: this repository was reorganized with no double checking about dependencies in terms of calling data or other code; so be aware that there could be an issue while running things. 

 ## For a Tutorial of How Most of the EPhys Data was Processed, go to `General_ephys_notebooks\Tutorial_for_multirecording_spikeanalysis.ipynb`.
  I recommend downloading the whole repo and running the notebook step by step in the base directory, so you can see how the `multirecording_spikeanalysis.py` script works.

 ## Directory Structure

`Behavioral_clustering`
`General_ephys_notebooks`
`Move_edit_data_notebooks`
`Neuronal_classifying`
`Newest_UMAP`
`leo_poster`
`recordings`
`rubbish`


- `data/`: Contains raw data (reward comp has been updated)
- `mpc_scripts/`: Contains med pc scripts used during the reward competition and training (written in med pc code) author: mixed
- `results/`: contains most of the code
    - `2023_05_12_pilot_consolidation`: author RI
    - `data`: should be empty, i put the excel sheets i need for figures 3-5 in there
    - `figure 3`: contains an old boris analysis script in python, it calculated the percentage of trials across behaviors, does not include stats or figures,  
                    author: MIC 
    - `figure 4`: contains 3 scripts for 
        - `contested_vs_uncont.ipynb`: fig4. B constested vs uncontested figure creation and some data carpentry (no stats) author: AH
        - `rc_diff.ipynb`: fig4 C percent trials won across winner vs losers, data carpentry only author: AH
        - `feature_extraction_sleap.ipynb`: fig4 D-I all the sleap analysis (feature extraction, umap, gif creation, enrichment plots etc) all figure creation is in here and stats, should also exist inpose_tracking_repo; author: MC
    - `figure 5`: has some old elo score calculations and an old correlations script, author RI for elo, correation author MC
        - `david_score_calculation.py`: calculates david score, author CY
        - `david_score_plotting.py`: plots correlation matrices (fig 5 E + F, fig S3), author CY
        - `linearity_matrix_generation.ipynb`: takes in the template dat in the outer most data folder and produces an output excel sheet of matrices of wins to be used in the R markdown linearity; author: MC 
        - `tube_test_matrix.ipynb`: create s matrix for tube tes tdata, author KP
    - `old_figures`: author RI
    - `R_GLMs_DCI`: has all the R scripts for the DCI calculations and GLMs including the project and R markdowns; author: MC
        - `GLM_stats.Rproj`: the project file
        - `linearity.Rmd`: calculates DCI based on output from the py script `linearity_matrix_generation.ipynb` 
        - `lmer_tubetest.RMD`: stats for fig 3, does the mixed model on the boris data for tube test 
        - `rewardcomp.Rmd`: does stats on the reward comp data (number of trials won per subject per strain per winer vs loser)
        - `urine_marking.Rmd`: runs a GLM mixed effects model on urine spots, winner/loser, subject, stats for fig 2
    - `reward_training`: calculates training progession, latencies and port probabilities; author RI
    - `rewardcomp_sleap`: calculates distance to port and other things; author RI 
- `src/`: src code for elo scores and med pc extracting (this might throw an error as i reorganized without checking to see if the elo scores and med pc scripts would run); author: RI 

