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
This is to see the correlation in our data, then we can objective to inference about the relationship with every data.
![GITHUB](pic/correlation_map.png)

### 2. Look the high correlation with price in ```scatter```

```
k = 10 #number ofvariables for heatmap
cols = corrmat.nlargest(k, 'price')['price'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
yticklabels=cols.values, xticklabels=cols.values)
plt.show()
```
This is to see the high correlation with price, pick up 10 data out of this data.
![GITHUB](pic/Price_realated.png)

And sactter this 10 data with 
```
sns.set()
cols = ['price', 'grade', 'sqft_living','sqft_above', 'bathrooms', 'lat', 'view', 'bedrooms', 'sqft_basement']
sns.pairplot(train[cols], size = 2.5)
plt.show();
```
View the linear relationship in the plot points.
![GITHUB](pic/total.png)

You already know some of the main features, and this specific scatter plot gives us a reasoning idea about the relationship between variables.<br>
One, sqft_living sqft_above The picture between sqft_above is very deep.<br>
Price and other intermediate scatter plots are also worth thinking about.<br>

### 3. Adjust data distribution
###### Test the function is important in the inference.

First, look the data plot and draw the picture.
```
from scipy.stats import norm
from scipy import stats
sns.distplot(train['price'], fit = norm);
fig = plt.figure()
res = stats.probplot(train['price'], plot=plt)
```
![GITHUB](pic/分布.png) 
![GITHUB](pic/plot.png)

It can be seen that the housing price distribution is not normal, showing the peak value and positive skewness, 
but it does not follow the diagonal. You can use logarithmic transformation to solve this problem.

