
# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# %matplotlib inline
# %matplotlib notebook

datac = pd.read_csv("covid_19_india.csv")           #reading covid_19_india file

datai = pd.read_csv("IndividualDetails.csv")          #reading IndividualDeatail file
   #reading states and population of india from file
indian_states_df = pd.read_csv("Indian States Population and Area.csv")

datac.head()

datac.shape

datac.isna().sum()                #cheacking missing values in the data set

datac_latest = datac[datac['Date']=="09/12/20"]       #latest data upto 09/12/2020
datac_latest.head()

datac_latest.tail()

datac_latest['Confirmed'].sum()            #calculated totaol number of comfirmed cases

datac_latest = datac_latest.sort_values(by=['Confirmed'], ascending = False)    #bar graph of 10 states with confirmed cases
plt.figure(figsize=(12,8), dpi=70)
plt.bar(datac_latest['State/UnionTerritory'][:10], datac_latest['Confirmed'][:10], align='center',color='grey')
plt.xlabel("Name of the States", size = 12)
plt.ylabel('Number of Confirmed Cases', size = 12)
plt.title('States with maximum confirmed cases', size = 16)
plt.show()

datac_latest['Deaths'].sum()        #total number of deaths

datac_latest = datac_latest.sort_values(by=['Deaths'], ascending = False)   #bar graph representing death cases in states
plt.figure(figsize=(12,8), dpi=70)
plt.bar(datac_latest['State/UnionTerritory'][:10], datac_latest['Deaths'][:10], align='center',color='grey')
plt.xlabel("Name of the States", size = 12)
plt.ylabel('Number of Deaths', size = 12)
plt.title('States with maximum deaths', size = 16)
plt.show()

#calculating ratio of death/comformed cases
datac_latest['Deaths/Confirmed Cases']=(datac_latest['Confirmed']/datac_latest['Deaths']).round()
datac_latest['Deaths/Confirmed Cases']=[np.nan if x == float("inf") else x for x in datac_latest['Deaths/Confirmed Cases']]
datac_latest = datac_latest.sort_values(by=['Deaths/Confirmed Cases'], ascending=True)
datac_latest.iloc[:]

indian_states_df.head(40)

#dropping 5 columns which are unnecessary
datac_latest = datac_latest.drop(['Sno','Date','Time','ConfirmedIndianNational','ConfirmedForeignNational'], axis = 1)
datac_latest.shape

#dropping Aadhaar assigned as of 2019 and area of a state
indian_states_df = indian_states_df[['State', 'Aadhaar assigned as of 2019']]
indian_states_df.columns = ['State/UnionTerritory', 'Population']
indian_states_df.head(40)

#merging two datasets i.e. covid_19 latest and india_states_df
datac_latest = pd.merge(datac_latest, indian_states_df, on="State/UnionTerritory")
datac_latest['Cases/10million'] = (datac_latest['Confirmed']/datac_latest['Population'])*10000000
datac_latest.head(37)

#arranging in ascending order cases/10million
datac_latest.fillna(0, inplace=True)
datac_latest.sort_values(by='Cases/10million', ascending=False)

#Visualization to display the variation in COVID 19 figures in different Indian states
data = datac_latest[(datac_latest['Confirmed']>=1000) | (datac_latest['Cases/10million']>=200)]
plt.figure(figsize=(12,8), dpi=80)
plt.scatter(datac_latest['Confirmed'], datac_latest['Cases/10million'], alpha=0.5)
plt.xlabel('Number of confirmed Cases', size=12)
plt.ylabel('Number of cases per 10 million people', size=12)
plt.scatter(data['Confirmed'], data['Cases/10million'], color="red")
for i in range(data.shape[0]):
    plt.annotate(data['State/UnionTerritory'].tolist()[i], xy=(data['Confirmed'].tolist()[i], data['Cases/10million']
            .tolist()[i]),xytext = (data['Confirmed'].tolist()[i]+1.0, data['Cases/10million'].tolist()[i]+12.0), size=11)
#plt.tight_layout()    
plt.title('Visualization to display the variation in COVID 19 figures in different Indian states', size=16)
plt.show()

plt.figure(figsize = (12,8))        #heat map of correlations
sns.heatmap(datac_latest.corr(), annot=True)

"""# Analysis of IndivudualDeatails dataset"""

datai.isna().sum()

datai.iloc[0]

#groupping the individual data in terms of district where the case was found
grouped_district = datai.groupby('detected_district')
grouped_district = grouped_district['id']
grouped_district.columns = ['count']
grouped_district.count().sort_values(ascending=False).head(37)

#confirmed cases count on the basis of gender
grouped_gender = datai.groupby('gender')
grouped_gender = pd.DataFrame(grouped_gender.size().reset_index(name = "count"))
grouped_gender.head()
plt.figure(figsize=(10,6), dpi=80)
barlist = plt.bar(grouped_gender['gender'], grouped_gender['count'], align = 'center', color='grey', alpha=0.3)
barlist[1].set_color('r')
plt.ylabel('Count', size=12)
plt.title('Count on the basis of gender', size=16)
plt.show()

#groupping the data on the basis of the diagnosed data to count of number of cases detected each day by doing a 
#cumulative sum of this feature and adding it to a new column 
grouped_date = datai.groupby('diagnosed_date')
grouped_date = pd.DataFrame(grouped_date.size().reset_index(name = "count"))
grouped_date[['Day','Month','Year']] = grouped_date.diagnosed_date.apply(lambda x: pd.Series(str(x).split("/")))
grouped_date.sort_values(by=['Year','Month','Day'], inplace = True, ascending = True)
grouped_date.reset_index(inplace = True)
grouped_date['Cumulative Count'] = grouped_date['count'].cumsum()
grouped_date = grouped_date.drop(['index', 'Day', 'Month', 'Year'], axis = 1)
grouped_date.head(53)

#Ploting graph to see the count increased
grouped_date = grouped_date.iloc[3:]
grouped_date.reset_index(inplace = True)
grouped_date.columns = ['Day Number', 'diagnosed_date', 'count', 'Cumulative Count']
grouped_date['Day Number'] = grouped_date['Day Number'] - 2
grouped_date
plt.figure(figsize=(12,8), dpi=80)
plt.plot(grouped_date['Day Number'], grouped_date['Cumulative Count'], color="grey", alpha = 0.5)
plt.xlabel('Number of Days', size = 12)
plt.ylabel('Number of Cases', size = 12)
plt.title('How the case count increased in India', size=16)
plt.show()

"""# processing the dataset to group the data in terms of different states

# Maharastra
"""

datac_maharashtra = datac[datac['State/UnionTerritory'] == "Maharashtra"]
datac_maharashtra.head()
datac_maharashtra.reset_index(inplace = True)
datac_maharashtra = datac_maharashtra.drop(['index', 'Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational',
                                            'Cured'],  axis = 1)
datac_maharashtra.reset_index(inplace = True)
datac_maharashtra.columns = ['Day Count', 'Date', 'State/UnionTerritory', 'Deaths', 'Confirmed']
datac_maharashtra['Day Count'] = datac_maharashtra['Day Count'] + 8
missing_values = pd.DataFrame({"Day Count": [x for x in range(1,8)],
                  "Date": ["0"+str(x)+"/03/20" for x in range(2,9)],
                  "State/UnionTerritory": ["Maharashtra"]*7,
                  "Deaths": [0]*7,
                  "Confirmed": [0]*7})
datac_maharashtra = datac_maharashtra.append(missing_values, ignore_index = True)
datac_maharashtra = datac_maharashtra.sort_values(by="Day Count", ascending = True)
datac_maharashtra.reset_index(drop=True, inplace=True)
print(datac_maharashtra.shape)
datac_maharashtra.head()

datac_maharashtra.tail()

"""# kerala"""

datac_kerala = datac[datac['State/UnionTerritory'] == "Kerala"]
datac_kerala = datac_kerala.iloc[32:]
datac_kerala.reset_index(inplace = True)
datac_kerala = datac_kerala.drop(['index','Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational','Cured'], 
                                 axis = 1)
datac_kerala.reset_index(inplace = True)
datac_kerala.columns = ['Day Count', 'Date', 'State/UnionTerritory', 'Deaths', 'Confirmed']
datac_kerala['Day Count'] = datac_kerala['Day Count'] + 1
print(datac_kerala.shape)
datac_kerala.head()

datac_kerala.tail()

"""# Delhi"""

datac_delhi = datac[datac['State/UnionTerritory'] == 'Delhi']
datac_delhi.reset_index(inplace = True)
datac_delhi = datac_delhi.drop(['index','Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational','Cured'], 
                               axis = 1)
datac_delhi.reset_index(inplace = True)
datac_delhi.columns = ['Day Count', 'Date', 'State/UnionTerritory', 'Deaths', 'Confirmed']
datac_delhi['Day Count'] = datac_delhi['Day Count'] + 1
print(datac_delhi.shape)
datac_delhi.head()

datac_delhi.tail()

"""# Rajasthan"""

datac_rajasthan = datac[datac['State/UnionTerritory'] == "Rajasthan"]
datac_rajasthan.reset_index(inplace = True)
datac_rajasthan = datac_rajasthan.drop(['index','Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational',
                                        'Cured'], axis = 1)
datac_rajasthan.reset_index(inplace = True)
datac_rajasthan.columns = ['Day Count', 'Date', 'State/UnionTerritory', 'Deaths', 'Confirmed']
datac_rajasthan['Day Count'] = datac_rajasthan['Day Count'] + 2
missing_values = pd.DataFrame({"Day Count": [1],
                           "Date": ["02/03/20"],
                           "State/UnionTerritory": ["Rajasthan"],
                           "Deaths": [0],
                           "Confirmed": [0]})
datac_rajasthan = datac_rajasthan.append(missing_values, ignore_index = True)
datac_rajasthan = datac_rajasthan.sort_values(by="Day Count", ascending = True)
datac_rajasthan.reset_index(drop=True, inplace=True)
print(datac_rajasthan.shape)
datac_rajasthan.head()

datac_rajasthan.tail()

"""# Gujarat"""

datac_gujarat = datac[datac['State/UnionTerritory'] == "Gujarat"]
datac_gujarat.reset_index(inplace = True)
datac_gujarat = datac_gujarat.drop(['index','Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational',
                                    'Cured'], axis = 1)
datac_gujarat.reset_index(inplace = True)
datac_gujarat.columns = ['Day Count', 'Date', 'State/UnionTerritory', 'Deaths', 'Confirmed']
datac_gujarat['Day Count'] = datac_gujarat['Day Count'] + 19
missing_values = pd.DataFrame({"Day Count": [x for x in range(1,19)],
                           "Date": [("0" + str(x) if x < 10 else str(x))+"/03/20" for x in range(2,20)],
                           "State/UnionTerritory": ["Gujarat"]*18,
                           "Deaths": [0]*18,
                           "Confirmed": [0]*18})
datac_gujarat = datac_gujarat.append(missing_values, ignore_index = True)
datac_gujarat = datac_gujarat.sort_values(by="Day Count", ascending = True)
datac_gujarat.reset_index(drop=True, inplace=True)
print(datac_gujarat.shape)
datac_gujarat.head()

datac_gujarat.tail()

"""# Karnataka"""

datac_karnataka = datac[datac['State/UnionTerritory'] == "Karnataka"]
datac_karnataka.head()
datac_karnataka.reset_index(inplace = True)
datac_karnataka = datac_karnataka.drop(['index', 'Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational',
                                        'Cured'], axis = 1)
datac_karnataka.reset_index(inplace = True)
datac_karnataka.columns = ['Day Count', 'Date', 'State/UnionTerritory', 'Deaths', 'Confirmed']
datac_karnataka['Day Count'] = datac_karnataka['Day Count'] + 8
missing_values = pd.DataFrame({"Day Count": [x for x in range(1,8)],
                  "Date": ["0"+str(x)+"/03/20" for x in range(2,9)],
                  "State/UnionTerritory": ["Karnataka"]*7,
                  "Deaths": [0]*7,
                  "Confirmed": [0]*7})
datac_karnataka = datac_karnataka.append(missing_values, ignore_index = True)
datac_karnataka = datac_karnataka.sort_values(by="Day Count", ascending = True)
datac_karnataka.reset_index(drop=True, inplace=True)
print(datac_karnataka.shape)
datac_karnataka.head()

datac_karnataka.tail()

"""# Andra Pradesh"""

datac_ap = datac[datac['State/UnionTerritory'] == "Andhra Pradesh"]
datac_ap.head()
datac_ap.reset_index(inplace = True)
datac_ap = datac_ap.drop(['index', 'Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational','Cured'], 
                         axis = 1)
datac_ap.reset_index(inplace = True)
datac_ap.columns = ['Day Count', 'Date', 'State/UnionTerritory', 'Deaths', 'Confirmed']
datac_ap['Day Count'] = datac_ap['Day Count'] + 8
missing_values = pd.DataFrame({"Day Count": [x for x in range(1,8)],
                  "Date": ["0"+str(x)+"/03/20" for x in range(2,9)],
                  "State/UnionTerritory": ["Andhra Pradesh"]*7,
                  "Deaths": [0]*7,
                  "Confirmed": [0]*7})
datac_ap = datac_ap.append(missing_values, ignore_index = True)
datac_ap = datac_ap.sort_values(by="Day Count", ascending = True)
datac_ap.reset_index(drop=True, inplace=True)
print(datac_ap.shape)
datac_ap.head()

datac_ap.tail()

"""# Tamil Nadu"""

datac_tn = datac[datac['State/UnionTerritory'] == "Tamil Nadu"]
datac_tn.head()
datac_tn.reset_index(inplace = True)
datac_tn = datac_tn.drop(['index', 'Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational','Cured'],
                         axis = 1)
datac_tn.reset_index(inplace = True)
datac_tn.columns = ['Day Count', 'Date', 'State/UnionTerritory', 'Deaths', 'Confirmed']
datac_tn['Day Count'] = datac_ap['Day Count'] + 8
missing_values = pd.DataFrame({"Day Count": [x for x in range(1,8)],
                  "Date": ["0"+str(x)+"/03/20" for x in range(2,9)],
                  "State/UnionTerritory": ["Tamil Nadu"]*7,
                  "Deaths": [0]*7,
                  "Confirmed": [0]*7})
datac_tn = datac_tn.append(missing_values, ignore_index = True)
datac_tn = datac_tn.sort_values(by="Day Count", ascending = True)
datac_tn.reset_index(drop=True, inplace=True)
print(datac_tn.shape)
datac_tn.head()

datac_tn.tail()

"""# Uttar Pradesh"""

datac_up = datac[datac['State/UnionTerritory'] == "Uttar Pradesh"]
datac_up.head()
datac_up.reset_index(inplace = True)
datac_up = datac_up.drop(['index', 'Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational','Cured'],  
                         axis = 1)
datac_up.reset_index(inplace = True)
datac_up.columns = ['Day Count', 'Date', 'State/UnionTerritory', 'Deaths', 'Confirmed']
datac_up['Day Count'] = datac_up['Day Count'] + 8
missing_values = pd.DataFrame({"Day Count": [x for x in range(1,8)],
                  "Date": ["0"+str(x)+"/03/20" for x in range(2,9)],
                  "State/UnionTerritory": ["Uttar Pradesh"]*7,
                  "Deaths": [0]*7,
                  "Confirmed": [0]*7})
datac_up = datac_up.append(missing_values, ignore_index = True)
datac_up = datac_up.sort_values(by="Day Count", ascending = True)
datac_up.reset_index(drop=True, inplace=True)
print(datac_up.shape)
datac_up.head()

datac_up.tail()

"""# West Bengal"""

datac_wb = datac[datac['State/UnionTerritory'] == "West Bengal"]
datac_wb.head()
datac_wb.reset_index(inplace = True)
datac_wb = datac_wb.drop(['index', 'Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational','Cured'], 
                         axis = 1)
datac_wb.reset_index(inplace = True)
datac_wb.columns = ['Day Count', 'Date', 'State/UnionTerritory', 'Deaths', 'Confirmed']
datac_wb['Day Count'] = datac_wb['Day Count'] + 8
missing_values = pd.DataFrame({"Day Count": [x for x in range(1,8)],
                  "Date": ["0"+str(x)+"/03/20" for x in range(2,9)],
                  "State/UnionTerritory": ["West Bengal"]*7,
                  "Deaths": [0]*7,
                  "Confirmed": [0]*7})
datac_wb = datac_wb.append(missing_values, ignore_index = True)
datac_wb = datac_wb.sort_values(by="Day Count", ascending = True)
datac_wb.reset_index(drop=True, inplace=True)
print(datac_wb.shape)
datac_wb.head()

datac_wb.tail()

"""# Plotting Graph of 10 different states"""

plt.figure(figsize=(12,8), dpi=80)
plt.plot(datac_kerala['Day Count'], datac_kerala['Confirmed'])
plt.plot(datac_maharashtra['Day Count'], datac_maharashtra['Confirmed'])
plt.plot(datac_delhi['Day Count'], datac_delhi['Confirmed'])
plt.plot(datac_rajasthan['Day Count'], datac_rajasthan['Confirmed'])
plt.plot(datac_gujarat['Day Count'], datac_gujarat['Confirmed'])
plt.plot(datac_karnataka['Day Count'], datac_karnataka['Confirmed'])
plt.plot(datac_ap['Day Count'], datac_ap['Confirmed'])
plt.plot(datac_wb['Day Count'], datac_wb['Confirmed'])
plt.plot(datac_up['Day Count'], datac_up['Confirmed'])
plt.plot(datac_tn['Day Count'], datac_tn['Confirmed'])
plt.legend(['Kerala', 'Maharashtra', 'Delhi', 'Rajasthan', 'Gujarat', 'Karnataka', 'Andra Pradesh', 
            'Tamil Nadu', 'Uttar Pradesh', 'West Bengal'], loc='upper left')
plt.xlabel('Day Count', size=12)
plt.ylabel('Confirmed Cases Count', size=12)
plt.title('Which states are flattening the curve ?', size = 16)
plt.show()

"""We get all the curves of the states which shows us the proper variations. Here Gujarat's curse is having gradual differnet inclination in the period after 200 days as seen in other curves.
But the situation in Maharashtra looks very grave indeed. The curve has had an immense steep incline and shows no signs of slowing down. In Kerela sudddenly cases increased after 150 days. 

Only way to get out of this difficuit situation by flattening the curve. All state governments need to follow the state model to flatten the curve. Gujarat is the only state which managed to flatten the curve and hence, must have done most things right. It’s time we followed the Gujarat model.
"""