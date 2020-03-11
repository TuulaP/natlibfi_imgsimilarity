# Natlibfi_imgsimilarity



Scripts to check whether target image is similar to the given set of target images.

Adaptation of work done at: https://douglasduhaime.com/posts/identifying-similar-images-with-tensorflow.html



### Acknowledgments

* European Regional Development Fund
* Leverage From the EU 2014-2020


### How to run

Utilizes data directory as source of illustrations, just give one as sample
and see if similar images are found. Fine tune the similarity comparison score
upper or higher within code to see which fits best.

For installation-free experiment in a browser, please use e.g. Google Colab for running the short Jupyter Notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TuulaP/natlibfi_imgsimilarity/blob/master/natlibfi_imgsimilarity.ipynb)

```
python nlf_similarity.py data/search-result-preview_019.jpg
```

Output should be something like:

```
Items in file_index 41
Parsing 0 data/search-result-preview_019.jpg

Targetbase:data/search-result-preview_019
For targetimg: data/search-result-preview_019.jpg, found 3 neighbors
Neighbours are [{'filename': 'search-result-preview_020', 'similarity': 1.0}, {'filename': 'search-result-preview_015', 'similarity': 0.893}, {'filename': 'search-result-preview_003', 'similarity': 0.8817}]

```


### Running notebook locally

If you have jupyter installed, you can start it via 

```
jupyter notebook
```

and then select the notebook from the list of files.


