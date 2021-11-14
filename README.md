# House_Price_Prediction_Analysis in 3 Models
###### The Python code is transform form ipynb in Colab!

- [x] relu model
- [x] sklearn model
- [x] CNN model

## Analyze the data in ```House_price_simple_RELU.py```
* 1. Look the all correlation in ```correlation_matrix```
* 2. Look the high correlation with price in ```scatter```
* 3. Adjust data distribution

### 1. Look the all correlation in ```correlation_matrix```

```
correlation_matrix = train.corr().round(2)
sns.set(rc={'figure.figsize':(22,22)})
# annot = True 讓我們可以把數字標進每個格子裡
sns.heatmap(data = correlation_matrix, annot = True)
```
This is to see the correlation in our data, then we can objective to inference the relationship with every data.
