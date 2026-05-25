README – r/kitchencels Reddit Analysis

Project:
Billions Must Cook: Food, Loneliness, and Ironic Self-Deprecation on r/kitchencels

This project analyses public Reddit posts and comments from r/kitchencels using data downloaded from Arctic Shift. The analysis includes text preprocessing, custom affective lexicons, VADER baseline sentiment, RoBERTa sentiment analysis, zero-shot theme classification, LDA topic modelling, burst term analysis, and Reddit user-reply network analysis.

Required input files:
Place the following files in the same folder as the Python script:

1. r_kitchencels_posts.jsonl
2. r_kitchencels_comments.jsonl

The Reddit data can be downloaded from the Arctic Shift download tool:
https://arctic-shift.photon-reddit.com/download-tool

Use:
Subreddit: kitchencels
Download: posts and comments

Required Python packages:
Run the following command before executing the script:

python -m pip install pandas numpy matplotlib networkx scikit-learn nltk tqdm transformers torch

How to run:
Open a terminal/PowerShell window in the project folder and run:

python kitchencels.py

Expected output:
The script will create three folders:

1. data/
   - cleaned posts, comments, and combined item datasets
   - representative sample for submission

2. outputs/
   - dataset summaries
   - sentiment results
   - custom affective lexicon results
   - zero-shot classification sample
   - LDA topic model outputs
   - burstiness terms
   - network metrics and centrality tables

3. figures/
   - all figures used in the report, including sentiment charts, topic charts, theme charts, and network graphs

Notes:
- The script anonymises Reddit usernames using hashed IDs.
- VADER is used only as a baseline sentiment method.
- RoBERTa is used as the main sentiment model.
- Zero-shot classification is run on a sample because it is computationally expensive.

To zip the results after running:

Compress-Archive -Path data,outputs,figures -DestinationPath kitchencels_analysis_results_v2.zip -Force
