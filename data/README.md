# data

- `head-ids-*.txt`: vocabulary ids kept by the pruned LM heads (one id per line).
  - `head-ids-ml60k.txt` (65,668 ids, "H65"): Qwen3.5. Frequency-chosen from English, code and 7 Wikipedias, plus every id < 16,384 and every special token (`tools/vocab_select.py`, E17).
  - `head-ids-ml40k.txt` (H45) and `head-ids-idrank32k.txt` (the first 32K ids): earlier Qwen3.5 heads (E17).
  - `head-ids-qwen3-ml64k.txt` (61,592 ids): Qwen3 / Bonsai, same recipe through llama.cpp's own tokenizer (`tools/head_select_tok.py`, E37), padded to a multiple of 8.
- `eval/eval-en.txt` (22,031 bytes) and `eval/eval-ml.txt`: the perplexity texts. Excerpts from Wikipedia, CC BY-SA 4.0. Held out from the head selection.
- `corpus/`, `corpus-ml/` (not in git): the text the head ids were counted on. Recreate with `tools/fetch_corpus.py`.
