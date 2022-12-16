from bassmodel.models.Basemodel import *
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np


class BassPol(BaseModel):
    """
    Model for implementing Bass Diffusion with non-linear LS: least-squares method.
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
        self.res = None
        self.popt = None
        self.cov = None
        self.sales_forecast = None
        self.cum_sales = None
        self.err = None

        if data is None:
            print('Error: No filename. Please input a filename.')
        try:
            if data[-3:] == 'csv':
                self.data = np.genfromtxt(data, delimiter=',', encoding="utf8")
            elif data[-4:] == 'xlsx':
                self.data = np.genfromtxt(data, delimiter=',', encoding="utf8")
            else:
                self.data = np.genfromtxt(data, delimiter='\t', encoding="utf8")
        except OSError:
            print('Error: file not found. Please input valid filename.')

        try:
            self.body = self.data[:, 1]
            self.time_range = range(1, len(self.body) + 1)

        except IndexError:
            print('Error: Incorrect file format. Please provide a file with csv, xlsx or txt formats.')

    def __bass_model__(self, t, p, q, k, **kwargs):
        arg1 = -(p + q) * t
        exp = np.exp(arg1)
        num = 1 - exp
        den = (1 + (q / p) * exp)
        self.res = k * (num / den)
        return self.res

    def fit(self):
        """
        Method to fit the data to the Bass Diffusion Model using non-linear LS: least-squares method.
        Returns
        -------
        float
            The estimation of the coefficient of the non-linear LS regression
        """
        self.popt, self.cov = curve_fit(self.__bass_model__, self.time_range, self.body)
        self.p, self.q, self.m = self.popt[0], self.popt[1], self.popt[2]
        return self.p, self.q, self.m

    def predict(self):
        self.forecast = self.__bass_model__(self.time_range, *self.popt)  # optimization of popt
        self.sales_forecast = self.__bass_model__(np.linspace(1, len(self.time_range), 1000), *self.popt)
        return

    def plot(self):
        """
        Method to visualize the actual and forecasted sales in one plot
        Returns
        -------
            scatter and line plots of actual and forecasted sales correspondingly
        """
        plt.plot(self.time_range, self.body, 'o', color='royalblue', label='Actual Sales')
        plt.plot(np.linspace(-1, max(self.time_range), 1000), self.sales_forecast, color='tomato', label='Forecasted '
                                                                                                         'Sales')
        plt.ylabel('Sales')
        plt.xlabel('Time')
        plt.title("Bass Diffusion Model for Sales")
        plt.legend(loc='best')
        #plt.grid()
        plt.show()

    def plot_predict(self):
        """
        Method to visualize the forecasted sales
        Returns
        -------
            line plot of forecasted sales
        """

        plt.plot(np.linspace(-1, max(self.time_range), 1000), self.sales_forecast, color='tomato', label='Forecasted '
                                                                                                         'Sales')
        plt.ylabel('Sales')
        plt.xlabel('Time')
        plt.title("Bass Diffusion Model for Sales")
        plt.legend(loc='best')
        #plt.grid()
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
        #plt.grid()
        plt.show()

    def plot_cdf(self):
        """
        Method to visualize the CDF: cumulative distribution function of the forecasted sales
        Returns
        -------
            line plot of CDF of forecasted sales
        """
        self.cum_sales = np.cumsum(self.sales_forecast)
        plt.plot(np.linspace(-1, max(self.time_range), 1000), self.cum_sales, label='CDF probs', color='royalblue')
        plt.legend(loc='best')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Sales')
        plt.title('Cumulative Distribution Function Over Time')
        #plt.grid()
        plt.show()

    def summarize(self):
        """
        Method to summarize the bass model by giving important metrics.
        Returns
        -------
            metrics of the model
        """
        print('=' * 40)
        print('Bass Diffusion Model Summary')
        print('{:<24}{:<10}'.format('Variable', 'Estimation'))
        print('{:<24}{:<10.3f}'.format('RMSE', np.sqrt(mean_squared_error(self.body, self.forecast))))
        print('{:<24}{:<10.4f}'.format('R-Squared', r2_score(self.body, self.forecast)))
        print('{:<24}{:<10.4f}'.format('Adj. R-Squared', 1 - (
                (1 - r2_score(self.body, self.forecast)) * (len(self.time_range) - 1) / (
                    len(self.time_range) - 3 - 1))))
        print('=' * 40)
        print('{:<24}{:<10.5f}'.format('Innovation Coef', self.p))
        print('{:<24}{:<10.5f}'.format('Imitation Coef', self.q))
        print('{:<24}{:<10.2f}'.format('Maximum Adopters', self.m))
        print('{:<24}{:<10.2f}'.format('Peak Adoption Time', (np.log(self.q) - np.log(self.p) / (self.p + self.q))))
        print('=' * 40)
