import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import pandas as pd
# Fitting Polynomial Regression to the dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('https://s3.us-west-2.amazonaws.com/public.gamelab.fun/dataset/position_salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
print(X)

# matplotlib.use("")
years_all = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
years_poly = [[2012], [2013], [2014], [2015], [2016], [2017], [2018], [2019]]
years = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
years_comming = [[2020], [2021], [2022], [2023], [2024], [2025]]
full_years = [[2012], [2013], [2014], [2015], [2016], [2017], [2018], [2019], [2020], [2021], [2022], [2023], [2024], [2025]]


cars_sweden_registered = [247, 432, 1239, 2962, 2945, 4217, 7078, 14000]
cars_sweden = [247, 679, 1918, 4880, 7825, 10770, 17848, 31848]

cars_norway_registered = [4700, 9968, 21153, 30901, 29503, 41583, 57991, 67120]
cars_norway = [4700, 14668, 35821, 66722, 96225,137808, 195799, 262919]

cars_germany_registered = [2956, 6051, 8522, 12363, 11410, 25056, 36062, 57533]
cars_germany = [2956, 9007, 17529, 29892, 41302, 66358, 102420, 159953]

cars_france_registered = [5663, 8779, 10560, 17779, 21751, 25983, 32203, 42763]
cars_france = [5663, 14442, 25002, 42781, 64532, 90515, 122718, 165481]

cars_netherlands_registered = [1910, 4161, 6825, 9368, 13105, 21115, 44984, 61966]
cars_netherlands = [1910, 6071, 12896, 22264, 35369, 56484, 101468, 163434]

# plt.scatter(years, cars)

def fit_polynomial(years_poly, cars): 
    poly = PolynomialFeatures(degree=5)
    X_poly = poly.fit_transform(years_poly)
    pol = LinearRegression()
    pol.fit(X_poly, cars)
    predicted = pol.predict(poly.fit_transform(years_comming))
    return pol, poly

# Visualizing the Polymonial Regression results
def viz_polymonial(country, cars, pol, poly):
    plt.figure()
    plt.scatter(years, cars, color='red')
    plt.plot(years_all, pol.predict(poly.fit_transform(full_years)), color='blue')
    plt.title('Number of EVs in ' + country + ' by year (Polynomial Regression)')
    plt.xlabel('years')
    plt.ylabel('EVs')
    plt.savefig(country + '_EVs.png')
    return 

sweden, swedeny = fit_polynomial(years_poly, cars_sweden)
viz_polymonial('Sweden', cars_sweden, sweden, swedeny)

norway, norwayy = fit_polynomial(years_poly, cars_norway)
viz_polymonial('Norway', cars_norway, norway, norwayy)

germany, germanyy = fit_polynomial(years_poly, cars_germany)
viz_polymonial('Germany', cars_germany, germany, germanyy)

france, francey = fit_polynomial(years_poly, cars_france)
viz_polymonial('France', cars_france, france, francey)

netherlands, netherlandsy = fit_polynomial(years_poly, cars_netherlands)
viz_polymonial('Netherlands', cars_netherlands, netherlands, netherlandsy)

average_usage = 0.17 #kWh/1km
average_yearly = 12000 #km
public_chargers = 0.1 #10% charging publicly
price_kW = 0.5 #eur/kWh in charging station
our_market_share = 0.1 #10% of all the public charging is done through us
margin = 0.1 #we take 10% of the price

earnings_per_driver_yearly = average_usage*average_yearly*public_chargers*price_kW*our_market_share*margin
print('We make eur/year per user: ', earnings_per_driver_yearly)