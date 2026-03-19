# Automated Essay Scoring
This directory contains scripts and resources for reproducing the AES experiments reported in our paper, covering both Arabic and English.


### Reproducing Model Outputs
- BERT classifiers:
  - To train the BERT classifiers reported in the paper, run: [bert/run_aes.sh](bert/run_aes.sh). These models are trained on the combined training sets of ZAEBUC-W1 and ZAEBUC-W2.
Before training, prepare the data using: [scripts/create_data.sh](scripts/create_data.sh).
- LLMs:
  - To reproduce LLM outputs, run the scripts in: [llms/scripts](llms/scripts).
### Evaluation
  - Evaluate the trained BERT classifiers using: [bert/run_eval.sh](bert/run_eval.sh).
  - Evaluate LLM outputs using: [llms/scripts/run_eval_metrics_llms.sh].

