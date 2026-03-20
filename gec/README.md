# Grammatical Error Correction
This directory contains scripts and resources for reproducing the GEC experiments reported in our paper, covering both Arabic and English.

## Reproducing Model Outputs:
- Arabic (Seq2Seq):
  - To reproduce the outputs of the Seq2Seq Arabic GEC model, run: [ar/run_ar_gec.sh](ar/run_ar_gec.sh).

- English (GECToR):
  - To reproduce the outputs of the GECToR English GEC model, run: [en/run_gector.sh](en/run_gector.sh). Note: The English data must be segmented before running this script. See: [utils/segmentation.ipynb](utils/segmentation.ipynb)

- LLMs:
  - To reproduce LLM outputs for both Arabic and English, run the scripts in: [llms/scripts](llms/scripts).
    
## Evaluation:
- Arabic: Evaluate Seq2Seq and LLM outputs using [ar/run_ar_eval.sh](ar/run_ar_eval.sh).
- English: Evaluate GECToR and LLM outputs using: [en/run_en_eval.sh](en/run_en_eval.sh).
