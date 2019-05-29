# Runway GloVe Model

<a href="http://sdk.runwayml.com" target="_blank"><img src="https://runway.nyc3.cdn.digitaloceanspaces.com/assets/github/runway-badge.png" width=100/></a>

This repository contains a Runway model for experimenting with [Stanford NLP's GloVe](https://nlp.stanford.edu/projects/glove/), an unsupervised learning algorithm for obtaining vector representations for words. Similar to Word2Vec, GloVe creates a continuous N-dimensional representation of a word that is learned from its surrounding context words in a training corpus. Trained on a large corpus of text, these co-occurance statistics (an N-dimensional vector embedding) cause semantically similar words to appear near each-other in their resulting N-dimensional embedding space (e.g. "dog" and "cat" may appear nearby a region of other pet related words in the embedding space because the context words that surround both "dog" and "cat" in the training corpus are similar).

## License and Attribution

This repo is a hard-fork of the [brannondorsey/GloVe-experiments](https://github.com/brannondorsey) repo with Runway-specific code changes and optimizations using FAISS.

All code is released under an [MIT license](LICENSE). You are free to copy, edit, share, or sell it under those terms.

### GloVe citation

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf).

```
@inproceedings{pennington2014glove,
  author = {Jeffrey Pennington and Richard Socher and Christopher D. Manning},
  booktitle = {Empirical Methods in Natural Language Processing (EMNLP)},
  title = {GloVe: Global Vectors for Word Representation},
  year = {2014},
  pages = {1532--1543},
  url = {http://www.aclweb.org/anthology/D14-1162},
}
```
