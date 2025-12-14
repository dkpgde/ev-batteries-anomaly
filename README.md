### EV Batteries: 
This is a modification of the original code from:

**EVBattery: A Large-Scale Electric Vehicle Dataset for Battery Health and Capacity Estimation**
> He et al. (2022). **arXiv:2201.12358**. [Link](https://arxiv.org/abs/2201.12358)  

This version only uses Dataset 3 due to limited compute (Dataset 3 is the smallest provided).

Added a more robust training approach to all models, using high epoch numbers and early stopping.
The transformers didn't add as much of a performance boost as I expected, and the higher computational cost is not justified.  
On the other hand, replacing the MSE with Huber loss during inference did provide a small performance boost. This loss is less likely to classify a car as faulty based on a couple outliers, which are normal during charging.  

**Baseline (original GRU DyAD)**:
AUROC: 0.698 ± 0.128
1. **Transformers instead of GRU**:  
AUROC: 0.8250 ± 0.0687
AUPRC: 0.9189 ± 0.0356  
Train time (seconds): 355.5470 ± 309.7916  
Test time (seconds): 14.1624 ± 0.7317  
2. **Original GRU with Huber loss during inference**:  
AUROC: 0.8104 ± 0.0863  
AUPRC: 0.9066 ± 0.0576  
Train time (seconds): 253.8557 ± 96.9742  
Test time (seconds): 18.1198 ± 0.9427  
3. **Original GRU with MSE during inference (improved by more robust training)**:  
AUROC: 0.7810 ± 0.0699 
AUPRC: 0.9017 ± 0.0389
Train time (seconds): 226.9936 ± 143.9765  
Test time (seconds): 18.0903 ± 0.9725  

Next steps would be to run the same on Dataset 1 and 1 and 3 combined, then to use Mamba SSM. But I don't have the compute and my curiosity is satisfied for now.   