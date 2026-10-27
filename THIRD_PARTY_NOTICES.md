# Third-party data and dependencies

## SQuAD 1.1 data

The bundled `data/pilot/` examples and split manifest derive from the SQuAD 1.1
training set. Attribution: Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang, *SQuAD: 100,000+ Questions for Machine Comprehension of Text* (2016).
Source: https://rajpurkar.github.io/SQuAD-explorer/
Data license: Creative Commons Attribution-ShareAlike 4.0 International,
https://creativecommons.org/licenses/by-sa/4.0/

The data builder selects and concatenates two source passages, adds passage headers,
remaps answer offsets, and creates a project-specific article split. It does not
rewrite the source questions or answers. These transformations are documented in
`gstar_fullkv_pilot/gstar_prepare_pilot.py`. The bundled pilot remains subject to its
data license; any later software license does not replace that license.

The raw SQuAD training JSON is intentionally excluded. The builder downloads it from
the original sources and caches it locally. The downloaded file is ignored by Git.

## Software and model

PyTorch, Transformers, NumPy, and Qwen model files have their own licenses and terms.
Model weights are not redistributed here. See the original packages and model card:
https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

No blanket software license has been selected for the project by this cleanup.
The project owner should choose an appropriate license before describing it as
open-source. Do not apply a software license to third-party data or model weights.
