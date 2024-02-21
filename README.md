# ExtendedGAMs
Code for the paper **"Extended Graph Assessment Metrics for Regression and Weighted Graphs"**

## Content
We here implement extended metrics for graph assessment for more versatile applications, including regression tasks and continuous/weighted graphs.
More concretely, we extend two graph assessment metrics:
- Node homophily [1] and
- Cross-class neighbourhood similarity (CCNs) [2]

Based on these two metrics, we introduce the following extensions:
1. Node homophily for regression tasks
2. Node homophily for weighted graphs
3. Cross-class neighbourhood similarity for weighted graphs
4. A new metric: CCNS distance, which collapses the cumbersome CCNS matrix into a single value

## Structure of the repo:
- All implemented metrics can be found in the file ```extended_graph_metrics.py```
- The ```utils.py``` file contains util functions
- The ```main.py``` file is a notebook-style file to execute the different metrics on small example graphs


## References
[1] Pei, Hongbin, et al. "Geom-gcn: Geometric graph convolutional networks." arXiv preprint arXiv:2002.05287 (2020).
[2] Ma, Yao, et al. "Is homophily a necessity for graph neural networks?." arXiv preprint arXiv:2106.06134 (2021).


BibTex entry:
```
@article{mueller2023extended,
  title={Extended Graph Assessment Metrics for Graph Neural Networks},
  author={Mueller, Tamara T and Starck, Sophie and Feiner, Leonhard F and Bintsi, Kyriaki-Margarita and Rueckert, Daniel and Kaissis, Georgios},
  journal={arXiv preprint arXiv:2307.10112},
  year={2023}
}
```

