# Don't Get Kicked Kaggle Competition

:car: == :lemon:? 1 : 0

Link to competition: https://www.kaggle.com/c/DontGetKicked/

```
$sudo pip install -r requirements.txt
$python main.py
```

### Approach
1. Read 'train.csv' and 'test.csv' files 
2. Preprocess the files
- Feature Engineering found in preprocess.py
3. Perform Locality Sensitive Hashing (LSH) on the dataframe for test.csv
- Finds 2000 similar rows for a query vector 
4. Perform K Nearest Neighbours on the LSH result 
- Finds k=1000 similar neighbours 
5. Pipeline of classification learning algorithms: Neural Network -> SVM 
