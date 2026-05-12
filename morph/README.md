# Morphological Tagging
This directory contains scripts and resources for reproducing the morphological tagging experiments reported in our paper, covering Zaebuc written and spoken corpora.

## Reproducing Model Outputs:

- Zaebuc written:
  - To reproduce CAMeL Tools' output for Arabic in the written corpus, run [morph_tagging/written/run_camelTools.sh](morph_tagging/written/run_camelTools.sh)
- LLMs:
  - To reproduce LLM outputs for both Arabic and English, run the scripts in: [llms/scripts](llms/scripts).

 ## Evaluation:
 - Zaebuc written: To evaluate CAMeL Tools, Stanza, and LLMs' output, run: [eval/run_eval_written.sh](eval/run_eval_written.sh).
 - Zaebuc spoken: To evaluate CAMeL Tools, Stanza, and LLMs' output, run: [eval/run_eval_spoken.sh](eval/run_eval_spoken.sh).
