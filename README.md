# DataManagement

# Introduction
the purpose of this project is to identify relevant keyphrases to be integrated in the Data Dictionary by analysing a set of business documents (procedures, processes, contracts,..) based on YAKE! Keyword extraction from single documents using multiple local features extraction - Ricardo Campos - 2020.

# Approach

Code can read different format of document : .docx, .ppt, .xlsx, .txt.

He  automaticall clusters documents by language (EN or FR) and treats them separately.

Using Segtok rule-based sentence segmenter, its splits text into sentences, and sentences into tokens to construct a vocabulary.

It extract different features to measure the automaticall of each token.

Features are based on : Position, Word Frequency, Sentence Frequency, Casing, Acronym, Relatedness to the context.

Score of each sentence is computed using the score of their tokens.

At the end, it filters duplicated sentence using Levenshtein metric.

details are given on YAKE Ricardo-2020 paper.

# Installation

Code is tested using `python 3.7`.
to install dependancy run from root project the following command :

```
pip install -r requirement
```

# Test

to test the code run:

```
python keyphrases_extraction.py input_folder output_name
```

in input_directory you need to have a list of folder, each folder is related to a tag.
put every document in a suitabale tag.

the code return output_name.csv file where you can find 5 columns ; Lang, Tag, Filename, Keyphrase, Score.

to change other arguments run:
```
python keyphrases_extraction.py --help
```


# To improve the approach

- Add other features.
- Add historical score to the model
- Learn an embedding for each bussiness, after having enough data
- Use a supervised learning method for each business, after having enough data


# References:

<b>In-depth journal paper at Information Sciences Journal</b>

Campos, R., Mangaravite, V., Pasquali, A., Jatowt, A., Jorge, A., Nunes, C. and Jatowt, A. (2020). YAKE! Keyword Extraction from Single Documents using Multiple Local Features. In Information Sciences Journal. Elsevier, Vol 509, pp 257-289. [pdf](https://doi.org/10.1016/j.ins.2019.09.013)

<b>ECIR'18 Best Short Paper</b>

Campos R., Mangaravite V., Pasquali A., Jorge A.M., Nunes C., and Jatowt A. (2018). A Text Feature Based Automatic Keyword Extraction Method for Single Documents. In: Pasi G., Piwowarski B., Azzopardi L., Hanbury A. (eds). Advances in Information Retrieval. ECIR 2018 (Grenoble, France. March 26 – 29). Lecture Notes in Computer Science, vol 10772, pp. 684 - 691. [pdf](https://link.springer.com/chapter/10.1007/978-3-319-76941-7_63)

Campos R., Mangaravite V., Pasquali A., Jorge A.M., Nunes C., and Jatowt A. (2018). YAKE! Collection-independent Automatic Keyword Extractor. In: Pasi G., Piwowarski B., Azzopardi L., Hanbury A. (eds). Advances in Information Retrieval. ECIR 2018 (Grenoble, France. March 26 – 29). Lecture Notes in Computer Science, vol 10772, pp. 806 - 810. [pdf](https://link.springer.com/chapter/10.1007/978-3-319-76941-7_80)



