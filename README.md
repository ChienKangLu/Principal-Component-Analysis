# Principal-Component-Analysis
Principal Component Analysis (PCA) is a technique which is used for dimension reduction.

## Purpose
+ Dimension reduction, keep the most important dimenstions
+ Find principle component
+ Project data a new space

## Develop tools and techniques
+ Python
+ Pycharm

## Skill
+ Linear algebra: eigen decomposition
+ Vector projection

## Implementation Step
1. Create data matrix which contains a lot of row vectors
2. Normalize the matrix, which transforms all dimensions to the distribution of zero mean
3. Calculate the covariance matrix, ![image](http://latex.codecogs.com/svg.latex?X^TX)
4. Do eigen decomposition and get the eigen values  and eigen vectors
5. Verify whether the decomposition is right or not, ![image](http://latex.codecogs.com/svg.latex?AX=X\Lambda)
6. Project original data to the new space by the eigen vectors, 
    <p align="center">
      <img src="http://latex.codecogs.com/svg.latex?proj_{\vec{v}}{\vec{x}}=\frac{\vec{x}&space;\cdot&space;\vec{v}}{\vec{v}&space;\cdot&space;\vec{v}}\vec{v}"/>
    </p>
7. Visualization

## Result
![image](https://github.com/ChienKangLu/Principal-Component-Analysis/blob/master/PCA/projection.png)

