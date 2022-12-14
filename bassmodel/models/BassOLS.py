from bassmodel.models.Basemodel import *
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BassOLS(BaseModel):
    """
    Model for implementing Bass Diffusion with OLS: ordinary least-squares method.
    """

    def __init__(self, data):
        """
         Method to initialize the object of Base model.
         Parameters
         ----------
         data : pandas.DataFrame
             Data to fit the model.
         Returns
         -------
         None
         """
        super().__init__(data)
        self.data = data
        self.q = None
        self.p = None
        self.m = None
        self.forecast = None
        self.model = None
        self.result = None
        self.pars = None
        self.cum_sales = None
        self.cum_sales_forecast = None
        self.cum_sales_sq = None

        if data is None:
            print("Error: No filename. Please input a filename.")
        try:
            if data[-3:] == "csv":
                self.data = pd.read_csv(data)
                self.data = np.array(self.data)
            elif data[-4:] == "xlsx":
                self.data = pd.read_excel(data)
                self.data = np.array(self.data)
            else:
                self.data = np.genfromtxt(data, delimiter="\t", encoding="utf8")
        except OSError:
            print("Error: File not found. Please input a valid filename of the data.")

        try:
            self.body = self.data[:, 1]
            self.time_range = range(1, len(self.body) + 1)

        except IndexError:
            print('Error: Incorrect file format. Please provide a file with csv, xlsx or txt formats.')

    def fit(self):
        """
        Method to fit the data to the Bass Diffusion Model using OLS: ordinary least-squares method.
        Returns
        -------
        float
            The estimation of the coefficient of the OLS regression
        """

        self.cum_sales = np.cumsum(self.body)  # body is the data about sales
        self.cum_sales_sq = self.cum_sales ** 2  # for finding the squared value of cumulative sales
        tups = list(zip(self.body, self.cum_sales, self.cum_sales_sq))
        data = pd.DataFrame(tups,
                            columns=['Sales', 'Cum_Sales', 'Cum_Sales_Squared'])
        self.model = smf.ols(formula='Sales ~ Cum_Sales + Cum_Sales_Squared', data=data)
        self.result = self.model.fit()
        self.pars = self.result.params
        return self.pars

    def __bass_model__(self, p, q, t, **kwargs):
        """
        Method to get the rate of change.
        Parameters
        ----------
        p : int
            : innovation rate or coefficient of innovation
        q : int
            imitation rate or coefficient of imitation
        t: int
            time interval
        Returns
        -------
        int or float
            rate of change
        """
        return (np.exp((p + q) * t) * p * (p + q) ** 2) / ((p * np.exp((p + q) * t) + q) ** 2)

    @staticmethod
    def __max__(a, b):
        """
        Magic method for max
        Parameters
        ----------
        a : int

        b : int
        Returns
        -------
        int
            maximum value of a and b couple
        """
        return max(a, b)

    def predict(self):
        """
        Method to get the maximum number of adopters, innovation and imitation coefficients.
        Returns
        -------
        float
            imitation coefficient, innovation coefficient, maximum number of adopters
        """
        m1 = (-self.pars['Cum_Sales'] + np.sqrt(
            self.pars['Cum_Sales'] ** 2 - 4 * self.pars['Intercept'] * self.pars['Cum_Sales_Squared'])) / (
                     2 * self.pars['Cum_Sales_Squared'])  # number of adopters
        m2 = (-self.pars['Cum_Sales'] - np.sqrt(
            self.pars['Cum_Sales'] ** 2 - 4 * self.pars['Intercept'] * self.pars['Cum_Sales_Squared'])) / (
                     2 * self.pars['Cum_Sales_Squared'])  # number of adopters
        self.m = self.__max__(m1, m2)  # get the maximum
        self.p = self.pars['Intercept'] / self.m  # innovation coefficient
        self.q = self.pars['Cum_Sales_Squared'] * (-self.m)  # imitation coefficient
        self.forecast = self.__bass_model__(self.p, self.q, self.time_range) * self.m  # saving to forecasting
        return self.m, self.p, self.q

    def plot(self):
        """
        Method to visualize the actual and forecasted sales in one plot
        Returns
        -------
            scatter and line plots of actual and forecasted sales correspondingly
        """

        plt.plot(self.time_range, self.forecast, color='tomato', label='Forecasted Sales')
        plt.plot(self.time_range, self.body, 'o', color='royalblue', label='Actual Sales')
        plt.ylabel('Sales')
        plt.xlabel('Time')
        plt.title("Bass Diffusion Model for Sales")
        plt.legend(loc='best')
        plt.show()

    def plot_predict(self):
        """
        Method to visualize the forecasted sales
        Returns
        -------
            line plot of forecasted sales
        """

        plt.plot(self.time_range, self.forecast, color='tomato', label='Forecasted Sales')
        plt.ylabel('Sales')
        plt.xlabel('Time')
        plt.title("Bass Diffusion Model for Sales")
        plt.legend(loc='best')
        plt.show()

    def plot_actual(self):
        """
        Method to visualize the actual sales
        Returns
        -------
            scatter plot of actual sales
        """

        plt.plot(self.time_range, self.body, 'o', color='royalblue', label='Actual Sales')
        plt.ylabel('Sales')
        plt.xlabel('Time')
        plt.title("Bass Diffusion Model for Sales")
        plt.legend(loc='best')
        plt.show()

    def plot_cdf(self):
        """
        Method to visualize the CDF: cumulative distribution function of the forecasted sales
        Returns
        -------
            line plot of CDF of forecasted sales
        """
        self.cum_sales_forecast = np.cumsum(self.forecast)
        plt.plot(self.time_range, self.cum_sales_forecast, label='CDF probs', color='royalblue')
        plt.legend(loc='best')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Sales')
        plt.title('Cumulative Distribution Function Over Time')
        plt.show()

    def summary(self):
        """
        Method to summarize the bass model by giving important metrics.
        Returns
        -------
            metrics of the model
        """
        print(self.result.summary())
        print('=' * 92)
        print('Bass Diffusion Model Summary')
        print('{:<24}{:<10}'.format('Variable', 'Estimation'))
        print('\n{:<24}{:<10.5f}'.format('Innovation Coef', self.p))
        print('{:<24}{:<10.5f}'.format('Imitation Coef', self.q))
        print('{:<24}{:<10.2f}'.format('Maximum Adopters', self.m))
        print('{:<24}{:<10.2f}'.format('Peak Adoption Time', (np.log(self.q) - np.log(self.p) / (self.p + self.q))))
        print('=' * 92)
