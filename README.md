# Bi-GRU for Aspect-Based Sentiment Analysis with Auxiliary Data
This project tests a bi-directional GRU for Aspect-Based Sentiment Analysis (ABSA) using auxiliary data. 
To be specific, a bi-directional GRU is used on the Sentihood dataset, taking in additional data from word sentiment values stored in the AFINN sentiment lexicon.
Attention weights are generated from the GRU, which are used on the word sentiment values before being fused with the initial GRU result. 
The Report.pdf file contains more detailed information.

This project credits a lot to https://github.com/HSLCY/ABSA-BERT-pair. 

To setup the project, you will need to download the Sentihood and AFINN datasets and put them inside the data/sentihood and data/afinn folders. 
You can then run the "testSentihood.bat" file.

To independently run the tests, run "python main.py <location> <aspect>" for all subject/aspect pairs within Sentihood.
The possible locations are loc1 and loc2. The possible aspects are general, price, safety, and transit. 

Evaluation requires all 8 tests to be run. Each test generates a text file. 
After all 8 files are generated, run "python evaluation.py". This will provide the final metrics. 

You can configure batch size and epochs at the top of the main.py file. 
To switch between AbsaGRU-Original and AbsaGRU-Last, uncomment the two comment blocks located in ABSAModel.py.
The model defaults to AbsaGRU-Original. 

The program was ran with the following Python libraries: \
python=3.7.1\
tqdm=4.64.1\
pandas=1.3.5\
pytorch=1.13.0\
pytorch-cuda=11.7 (bundled with pytorch for gpu)\
torchtext=0.14.0

Other versions may cause problems, specifically with pytorch and torchtext.
