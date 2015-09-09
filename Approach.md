# The Current Approach: 
[*Genie and LAMP*](http://www.comp.nus.edu.sg/~atung/gl/)

### Feature Engineering 
- Use **Frequent Itemsets** to find features correlated with *IsBadBuy* attribute
- Use domain knowledge 

### Genie
- Use **Locality Sensitive Hashing** to find a set of similar items 
- Use **kNN** with Euclidean Distance to find a set of k-Nearest-Neighbours

### LAMP
- Massively parallel mining of each nearest neighbours sets
- Reduce dimensions of dataset by **PCA**
