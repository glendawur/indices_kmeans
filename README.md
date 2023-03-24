## Inertia-based indexes for the number of clusters in k-means: an experimental evaluation

This folder contains the code for the experiments and results of the conducted experiments. You can find here next
data:

- zip-archives with .pkl.gz files that contain indexes choice of k and related values of Normalized Mutual Index (NMI), 
Adjusted Rand Index (ARI) and Mean Absolute Relative Error (MARE) between predicted number of classes and ground truth.
([Zip files here](intermediate_files/))
- .csv files with all results of each zip-archive ([CSV files here](intermediate_files/))
- .xlsx files with summary over each dataset ([Final Tables](final_tables/))
- python notebook and python scripts of the experiments (NB! Experiments require main package 'mircl' being installed)

Original data used for the experiments (1.3 GB) and intermediate numpy matrices with partitions for each dataset (71 GB) can be provided
on the request.

Requests: a.g.rykov@student.tue.nl

#### Datasets used in the experiments:

1. Synthetic data generated with next parameters:
   1. Size (N) = 2500
   2. Dimensionality (M/V) = 15, 50
   3. Intermix parameter (a) =  0.25 (deprecated in the research), 0.5, 0.75, 0.85 
   4. Number of clusters (K) = 7, 15, 21
2. UCI datasets:
   1. Ecoli
   2. Iris
   3. Optdigits
   4. Segmentation
   5. Wisconsin Breast Cancer (Prognosis)
   6. Wisconsin Breast Cancer (Diagnosis)
   7. Wine 
   8. Zoo
   9. Ionosphere (depricated)
   10. Glass (depricated)
   11. Pima Diabetes (depricated)
3. Clustering Benchmark Datasets
   1. G2 (Only M/V = 8, 32; var = 10, 50, 90, 100)
   2. S
   3. Unbalance


#### Table of abbreviations

| Abbreviation                    |                                             Expanded |
|:--------------------------------|-----------------------------------------------------:|
| KM                              |                                              K-Means |
| RS                              |                                          Random Swap |
| Eucl                            |                                            Euclidean |
| Conv                            |                                         Conventional | 
| HR                              |                                        Hartigan Rule |
| CH                              |                              Calinski-Harabasz index |
| WB                              |                                             WB index |
| XU                              |                                             Xu index |
| SW                              |                                     Silhouette Width | 
| EL XX                           |                 Elbow, XX - left and right step size |
| ARI                             |                                  Adjusted Rand Index |
| NMI                             |                        Normalized Mutual Information |
| MARE                            | Mean Absolute Relative Error (in number of clusters) |
| M                               |                                 number of dimensions | 
| a                               |  interval of cluster width around centers (intermix) |
| var                             |                           degree of clusters overlap |
| TrueK                           |                            Actual number of clusters | 