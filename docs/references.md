# Project References

These references are suitable for the final presentation bibliography. The
first group supports the scientific method; the second group documents the
dataset and software used to implement it.

## Core scientific and dataset references

1. **Guttag, J. (2010).** *CHB-MIT Scalp EEG Database (version 1.0.0).*
   PhysioNet. RRID:SCR_007345.
   https://doi.org/10.13026/C2K01R  
   **Use:** Primary dataset citation.

2. **Shoeb, A. H. (2009).** *Application of Machine Learning to Epileptic
   Seizure Onset Detection and Treatment.* PhD thesis, Massachusetts Institute
   of Technology. http://hdl.handle.net/1721.1/54669  
   **Use:** Foundational CHB-MIT seizure-detection work and dataset context.

3. **Karmim, Y., Lafon, M., Fournier S'Niehotta, R., & Thome, N. (2024).**
   Supra-Laplacian Encoding for Transformer on Dynamic Graphs. In *Advances in
   Neural Information Processing Systems 37*, 17215–17246.
   https://doi.org/10.52202/079017-0547  
   **Use:** SLATE architecture, supra-graph construction, and
   supra-Laplacian spectral encoding used as the basis for Model D.

4. **Cortes, C., & Vapnik, V. (1995).** Support-vector networks. *Machine
   Learning, 20*(3), 273–297. https://doi.org/10.1007/BF00994018  
   **Use:** Theoretical foundation for Model A's support vector machine.

5. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
   A. N., Kaiser, L., & Polosukhin, I. (2017).** Attention is all you need. In
   *Advances in Neural Information Processing Systems 30*.
   https://doi.org/10.48550/arXiv.1706.03762  
   **Use:** Transformer and attention mechanism used in Model D.

6. **Chung, F. R. K. (1997).** *Spectral Graph Theory.* CBMS Regional
   Conference Series in Mathematics, No. 92. American Mathematical Society.  
   **Use:** Graph Laplacians, eigenvalues, eigenvectors, and spectral graph
   theory underlying the supra-Laplacian encoding.

7. **Pearson, K. (1895).** Note on regression and inheritance in the case of
   two parents. *Proceedings of the Royal Society of London, 58*, 240–242.
   https://doi.org/10.1098/rspl.1895.0041  
   **Use:** Original reference for Pearson product-moment correlation used to
   construct EEG connectivity matrices.

## Software references

8. **Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D.,
   Brodbeck, C., Goj, R., Jas, M., Brooks, T., Parkkonen, L., & Hämäläinen,
   M. S. (2013).** MEG and EEG data analysis with MNE-Python. *Frontiers in
   Neuroscience, 7*, 267. https://doi.org/10.3389/fnins.2013.00267  
   **Use:** EDF loading and EEG preprocessing with MNE-Python.

9. **Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D.,
   Brodbeck, C., Parkkonen, L., & Hämäläinen, M. S. (2014).** MNE software for
   processing MEG and EEG data. *NeuroImage, 86*, 446–460.
   https://doi.org/10.1016/j.neuroimage.2013.10.027  
   **Use:** Additional MNE-Python software reference.

10. **Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B.,
    Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V.,
    Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., &
    Duchesnay, E. (2011).** Scikit-learn: Machine learning in Python. *Journal
    of Machine Learning Research, 12*, 2825–2830.
    https://jmlr.org/papers/v12/pedregosa11a.html  
    **Use:** SVM implementation and evaluation utilities.

11. **Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G.,
    Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., & others. (2019).**
    PyTorch: An imperative style, high-performance deep learning library. In
    *Advances in Neural Information Processing Systems 32*, 8024–8035.
    https://arxiv.org/abs/1912.01703  
    **Use:** Model D, tensor operations, optimization, and checkpointing.

## References to prioritize in a short PPT

If the slide has room for only five entries, use:

1. Guttag (2010) — CHB-MIT dataset.
2. Shoeb (2009) — seizure-detection and dataset context.
3. Karmim et al. (2024) — SLATE and supra-Laplacian encoding.
4. Cortes & Vapnik (1995) — SVM baseline.
5. Vaswani et al. (2017) — transformer architecture.

Add Gramfort et al. (2013) if the presentation requires a citation for the EEG
processing software.

## Citation boundary

The following are project design decisions rather than external claims and do
not require separate paper citations:

- the selected 10-subject compatible cohort;
- the 23-position channel contract;
- the 5-second window and 50% overlap;
- the 12-window chunk;
- the 500 µV artifact policy;
- the four-hour interictal buffer;
- the five-minute prediction horizon;
- the top-k value of five;
- the LOSO split and local leakage checks.

These choices should be described as methods decisions and validated with
experiments, not presented as conclusions taken directly from the cited papers.

## Note on PyTorch Geometric

PyTorch Geometric is listed as a project dependency, but the current Model D
path uses native PyTorch modules rather than PyTorch Geometric layers. It is
therefore not necessary to include a PyTorch Geometric citation for the current
implemented results. Add one only if the final experiment actually uses its
operators.

