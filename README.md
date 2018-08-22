# Principal-Component-Analysis
Principal Component Analysis (PCA) is a technique which is used for dimension reduction.

# Purpose
+ Dimension reduction, keep the most important dimenstions
+ Find principle component
+ Project data a new space

# Skill
+ Linear algebra: eigen decomposition
+ Vector projection

# Implementation Step
1. Create data matrix which contains a lot of row vectors
2. Normalize the matrix, which transforms all dimensions to the distribution of zero mean
3. Calculate the covariance matrix
4. Do eigen decomposition and get the eigen values  and eigen vectors
5. Verify whether the decomposition is right or not, ![image](http://latex.codecogs.com/gif.latex?AX=X\Lambda)
6. Project original data to the new space by the eigen vectors
7. Visualization

# Result
![image](https://github.com/ChienKangLu/Principal-Component-Analysis/blob/master/PCA/projection.png)

