# DEBUG : Definition Extraction for Building Useful Glossaries

This is a project in which definition extraction is explored. Definition extraction is broken into 2 processes :
1. Whether sentences contain a definition
2. Tagging the tokens of a sentence containing the definition.

## Datasets
1. WCL corpus [http://lcl.uniroma1.it/wcl/]
2. DEFT corpus [https://github.com/adobe-research/deft_corpus]

The source code related to the training of the models can be found in *notebooks* and the pretrained models on datasets
can be found in *models*.

The end to end pipeline can be found in the *definition_extractor.py* script.
