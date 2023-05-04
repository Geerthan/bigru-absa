# Bi-GRU for Aspect-Based Sentiment Analysis with Auxiliary Data
This project tests a bi-directional GRU for Aspect-Based Sentiment Analysis (ABSA) using auxiliary data. 
To be specific, a bi-directional GRU is used on the Sentihood dataset, taking in additional data from word sentiment values stored in the AFINN sentiment lexicon.
Attention weights are generated from the GRU, which are used on the word sentiment values before being fused with the initial GRU result. 
The Report.pdf file contains more detailed information.

This project credits a lot to https://github.com/HSLCY/ABSA-BERT-pair. 
