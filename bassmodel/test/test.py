from bassmodel.models.BassOLS import *
model = BassOLS('../utils/data.xlsx')
model.fit()
model.predict()
model.plot()
model.summary()
