class BaseModel:
    """
    Base model to initialize the object, later inherited by LSE and Pol models
    """
    def __init__(self, data):
        pass

    def __bass_model__(self, t, p, q, k):
        pass

    def fit(self):
        """
        Method to fit the data to the model.
        Returns
        ----------
        None
        """
        pass

    def predict(self):
        """
        Method to get the prediction on the data.
        Returns
        -------
        None
        """
        pass

    def plot(self):
        """
        Method to plot the prediction.
        Returns
        -------
        None
        """
        pass

    def plot_cdf(self):
        """
        Method to plot the CDF:cumulative distribution function of the prediction.
        Returns
        -------
        None
        """
        pass

    def summarize(self):
        """
        Method to summarize the results of the model prediction.
        Returns
        -------
        None
        """
        pass


