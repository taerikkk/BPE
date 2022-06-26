## Hardware Environments   
We conducted our experiments on the machines with i9-9900K, 64GB RAM, and GTX 1070.   

## Software Environments   
As our experiments utilize many different types of baseline methods, our software environments are rather complicated.   
The selected list of important software/libraries are as follows:
> Python ver 3.8.1,   
> Scikit Learn ver 0.22.1,   
> TensorFlow ver 1.5.1,   
> PyTorch ver 1.2.0,   
> CUDA ver 10,   
> NetworkX ver 2.4,   
> Imbalanced-learn ver 0.6.1.   

## Under/oversampling Methods
We also considered the following under/oversampling methods to address the imbalanced class problem:
> Naive random oversampling and undersampling are randomly choose samples to add and drop, respectively;   
> SMOTE and its variants are a family of the most popular oversampling methods, which include 5 variations;   
> ADASYN is also popular for oversampling;   
> Tomek's link is a representative undersampling method;   
> Clustering uses centroids of clusters after dropping other cluster members;   
> NearMiss is also popular for undersampling;   
> Various nearest neighbor methods are able to undersampling;   
> Ensemble methods mean that we use both the oversampling and undersampling methods at the same time.   

We refer to a survey paper **[1]** for more detailed information.   
*[1] G. Lemaˆıtre, F. Nogueira, and C. K. Aridas, “Imbalanced-learn: A python toolbox to tackle the curse of imbalanced datasets in machine learning,” JMLR, vol. 18, no. 17, pp. 1–5, 2017.*

## Best Hyperparameters (Our Method)
* thresholds: (ths+ = 0.7, ths− = 0.7)
* embedding: DeepWalk
* similarity: Cosine similarity
