# Copyright 2017, 2018 IBM. IPLA licensed Sample Materials.
# this function is called if the user selects to use the supplied census data (b_use_census_data variable)
# it reads in the census and customer data
# the census and customer data are matched based on age, marital status, education, employment status,
# income, location and gender
# the function does cleaning to align category names in each column between customer and census datasets
# returns the prepped dataset along with marriage, migration, birth and divorce probabilities from census data
import pandas as pd
import os 
import numpy as np



class census_data():
  def prep_census(self,census_data,customer_data,prepped_data,train_or_score):

      # read in the census data and the customer data
      df_census_probabilities = census_data.copy()#pd.read_csv('/project_data/data_asset/Census Migration Birth Marriage and Divorce Probabilities.csv')
      
      df_customers = customer_data.copy()#pd.read_csv('/project_data/data_asset/customer.csv')

      # to join the census data to customer data we map our customer categories to their most similar category in the census data 
  
      # age ranges in customer data: 23 to 30, 30 to 40, 40 to 55, 55 to 65, 65 and over   
      # age ranges in census data: 18-24, 25-29, 30-34, 35-39, 40-44, 45-54, 55-64, 65-74, 75+, unknown
      # update the census categories to be the same as in the customer data
      age_dict = {'18-24':'18-24', '25-29':'23 to 30', '30-34':'30 to 40', '35-39':'30 to 40', '40-44':'40 to 55', '45-54':'40 to 55', '55-64':'55 to 65', '65-74':'65 and over', '75+':'65 and over', 'Unknown':'Unknown'}
      df_census_probabilities['AGE'] = df_census_probabilities['AGE'].map(age_dict)
  
      # marital status in customer data: Married, Divorced, 'Single'
      # marrital status in census data: Married, Divorced or Separated, Single, Widoed, Unknown
      # update census 'Divorced or Separated' category to 'Divorced'
      # All other categories can remain the same
      df_census_probabilities['MARITAL_STATUS'] = df_census_probabilities['MARITAL_STATUS'].replace({'Divorced or Separated':'Divorced'})
  
      # education in customer data: High School, College, Professional, University, PhD
      # education in census data: Grade 11 or Lower, High School, University, Professional Degree, Doctorate Degree, Unknown
      # update 'Professional Degree' category in census data to 'Professional' 
      # update 'Doctorate Degree' category in census data to 'PhD'
      df_census_probabilities['EDUCATION'] = df_census_probabilities['EDUCATION'].replace({'Professional Degree':'Professional',
                                                                                    'Doctorate Degree':'PhD'})
      # update the 'College' category in customer data to 'University'
      df_customers['EDUCATION_LEVEL'] = df_customers['EDUCATION_LEVEL'].replace({'College':'University'})
  
      # employment status in customer data: Employed, Selfemployed, Homemaker, Retired, Unemployed
      # employment status in census data: Employed, 'Not in Labor Force', Unemployed, Unknown
      # update selfemployed category in customer data to employed
      # update homemaker category in customer data to Not in Labor Force
      # update retired category in customer data to Not in Labor Force
      df_customers['EMPLOYMENT_STATUS'] = df_customers['EMPLOYMENT_STATUS'].replace({'Selfemployed':'Employed'})
      df_customers['EMPLOYMENT_STATUS'] = df_customers['EMPLOYMENT_STATUS'].replace({'Homemaker':'Not in Labor Force'})
      df_customers['EMPLOYMENT_STATUS'] = df_customers['EMPLOYMENT_STATUS'].replace({'Retired':'Not in Labor Force'})
  
      # income in customer data is numerical
      # income in census data: Under 15k, 15k-35k, 35k-75k, 75k-125k, 125k-200k, 200K+, Unknown,
      # bin income up in the customer data
      bins = [0, 15000, 35000, 75000, 125000, 200000, 9999999999]
      labels = ['Under 15k', '15k-35k', '35k-75k', '75k-125k', '125k-200k', '200K+']
      df_customers['ANNUAL_INCOME'] = pd.cut(df_customers['ANNUAL_INCOME'], bins, labels=labels)
  
      # states should match between customer and census (where customer data is in USA)
      # set everything else in customer data to 'Unknown' to align with census
      df_customers['LOCATION'] = df_customers['ADDRESS_HOME_STATE']
      df_customers.loc[(~df_customers['ADDRESS_HOME_STATE'].isin(df_census_probabilities['LOCATION'].unique())), 'LOCATION'] = 'Unknown'
  
      # gender ranges in our customer data are the same as census (ex 'unknown')
  
      # because of how we grouped above, we can have duplicate records over location, age, marital status, education, employment,
      # gender and income, but with different probabilities
      # to combat this we group by these factors and take an average of the probabilities
      df_census_probabilities = df_census_probabilities.groupby(['LOCATION', 'MARITAL_STATUS', 'EDUCATION', 'GENDER', 'EMPLOYMENT',
             'INCOME', 'AGE'])[['MIGRATION_PROB', 'BIRTH_PROB', 'MARRIAGE_PROB', 'DIVORCE_PROB']].mean().reset_index()
      
      
      ##########  Code to plot census Data on training notebook 
      if train_or_score=='train':

        cols_to_plot=["LOCATION","MARITAL_STATUS","EDUCATION","GENDER","EMPLOYMENT","INCOME","AGE"]
        for col in cols_to_plot:
          df_to_plot=df_census_probabilities.groupby(col).mean().reset_index()
      
          #df_to_plot["BIRTH_PROB"]=df_to_plot["BIRTH_PROB"]*10
          #df_to_plot["MARRIAGE_PROB"]=df_to_plot["MARRIAGE_PROB"]*10
          #df_to_plot["DIVORCE_PROB"]=df_to_plot["DIVORCE_PROB"]*10
          self.plot_census(df_to_plot,col)
        
        
      # join customer and census data on all 7 fields to get the probabilities from census
      # first filter the customer data to return one record per customer
      df_customers = pd.merge(df_customers, df_customers.groupby('CUSTOMER_ID')['EFFECTIVE_DATE'].max().reset_index(), how='inner', on=['CUSTOMER_ID', 'EFFECTIVE_DATE'])
  
      # get the records that we can match on all 7 criteria
      # mapping above should ensure that all customer records will get a match so we can use an inner join
      df_census_probabilities = pd.merge(df_customers, df_census_probabilities, how='inner', left_on=['AGE_RANGE', 'MARITAL_STATUS', 
                                                                      'EDUCATION_LEVEL', 'EMPLOYMENT_STATUS',
                                                                      'LOCATION', 'ANNUAL_INCOME', 'GENDER'],
                                                          right_on=['AGE', 'MARITAL_STATUS', 
                                                                      'EDUCATION', 'EMPLOYMENT',
                                                                      'LOCATION', 'INCOME', 'GENDER'])
  
      df_census_probabilities = df_census_probabilities[['CUSTOMER_ID', 'MIGRATION_PROB', 'BIRTH_PROB', 'MARRIAGE_PROB', 'DIVORCE_PROB']]
      
      # replace any missing values with the mean for that column
      df_census_probabilities = df_census_probabilities.fillna(df_census_probabilities.mean())
      
      # loop through the dictionary of prepped data and append the probabilities to the prepped data
      for event_type, df in prepped_data.items():
          prepped_data[event_type] = pd.merge(prepped_data[event_type], df_census_probabilities, on='CUSTOMER_ID')
          prepped_data[event_type].drop('CUSTOMER_ID', axis=1, inplace=True)
  
      return prepped_data
  
  def plot_census(self,s,column):
    
    from plotly.plotly import iplot
    import plotly as py
    import plotly.graph_objs as go
    py.offline.init_notebook_mode(connected=True)

    data = [
        go.Bar(
            x=s[column],
            y=s["MIGRATION_PROB"],
            name="MIGRATION_PROB"
        ),
        go.Bar(
            x=s[column],
            y=s["BIRTH_PROB"],
            name="BIRTH_PROB"
        ),
        go.Bar(
            x=s[column],
            y=s["MARRIAGE_PROB"],
            name="MARRIAGE_PROB"
        ),
        go.Bar(
            x=s[column],
            y=s["DIVORCE_PROB"],
            name="DIVORCE_PROB"
        )

    ]
    
    layout = go.Layout(
        barmode='group',
        title='Average Census data Probabilities by '+column
    )

    fig = dict(data = data, layout = layout)
    py.offline.iplot(fig)


    
