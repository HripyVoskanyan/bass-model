# Bass-model
#### Description
This package was developed by Hripsime Voskanyan for the Marketing Analytics course of American University of Armenia.
Bass Diffusion Model was initially created by marketing professor Frank Bass. The model is used to 
predict new product diffusion patterns. The base model assumes that there are two main factors, which
decide the diffusion pattern.
#### Setup
Follow the example of test.py
```
from bassmodel.models.BassOLS import *
model = BassOLS('../utils/data.csv')
model.fit()
model.predict()
model.plot()
model.summary()
```
Change the directories correspondingly.
#### Models
The package includes two types of implementations of Bass Difussion Model:
- OLS: estimations are done by choosing the best coefficients of
ordinary least-squares.
- Pol: estimations are done by non-linear LS using Levenberg-Marquardt algorithm.

In both cases coefficients of innovation, imitation, and maximum number of adopters are calculated

### Data
Data used in the example is downloaded and manipulated by excel from https://www.kaggle.com/datasets/kyanyoga/sample-sales-data.
#### Visualizations
The package includes four types of visualizations:
- plot()
- plot_predict()
- plot_actual()
- plot_cdf()
The methods visualize the predicted and actual sales, as well as 
the cumulative distribution functions.
#### Applications
The Bass Diffusion model can be used to forecast new products'
sales. In our example, it is used to predict the sales of phone sales.
#### References
- Castro, Alejandro Puerto. “Alejandropuerto/Product-Market-Forecasting-Bass-Model.” GitHub, 7 July 2022, github.com/alejandropuerto/product-market-forecasting-bass-model. Accessed 14 Dec. 2022.
- Grasman, Johan, and Marcel Kornelis. “Forecasting Product Sales with a Stochastic Bass Model.” Journal of Mathematics in Industry, vol. 9, no. 1, 7 Feb. 2019, 10.1186/s13362-019-0059-6.
- Bass-model: lectures by Karen Hovhannisyan
