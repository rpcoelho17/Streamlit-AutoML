from operator import index
import pandas as pd
import streamlit as st
from mitosheet.streamlit.v1 import spreadsheet
import dtale
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import *
from sklearn.model_selection import TimeSeriesSplit
import pandas_profiling
import numpy as np
import matplotlib.pyplot as plt
import bokeh.plotting as bp
from bokeh.models import RangeSlider, CrosshairTool, HoverTool
#from typing import Any, Dict, List, Tuple
from streamlit_pandas_profiling import st_profile_report
import os
dpath = os.getcwd()
from dtale.views import startup
from dtale.app import get_instance
import io
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_halving_search_cv



# def info_to_df(df):
#     buffer = io.StringIO()
#     df.info(buf=buffer)
#     s = buffer.getvalue()
#     lines = s.splitlines()[1:-1]
#     columns = lines[0].split()
#     data = [line.split() for line in lines[1:]]
#     info_df = pd.DataFrame(data, columns=columns)
#     return info_df



#if os.path.exists('/content/drive/MyDrive/Colab Notebooks/trainSS.csv'):
#    df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/trainSS.csv', index_col=None)

# Page config
st.set_page_config(page_title="Streamlit Time Series Auto-ML", layout="wide")

st.sidebar.title("Streamlit Time Series Auto-ML")
st.sidebar.title("1. Data")

# Load data
st.title("Upload Your Dataset")
if 'df' in st.session_state: # check if session state has new_table key
    df = (st.session_state.df)
else:
    file = st.file_uploader("Upload Your Dataset")
    if file:
        #df = pd.read_csv(file, index_col=None)
        st.session_state['df'] = st.cache_data(pd.read_csv)(file, index_col=None)
        df = st.session_state['df']
        st.session_state['TargetColStr'] = None
        #st.write(st.session_state["TargetColStr"], "TargetColStr defined as empty string.")

if 'df' in st.session_state:
   st.dataframe(st.session_state.df,height=350)
   st.dataframe(st.session_state.df.describe())
   cat_cols = st.session_state.df.select_dtypes(include=['object']).columns.tolist()
   st.write("Categorical columns: ",cat_cols)
   num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
   st.write("Numerical columns: ",num_cols)
   st.write("df.shape:",df.shape)
   st.write("Variable Types: ",df.dtypes)
   Target_Col = st.selectbox('Select the target column:', st.session_state.df.columns, index=None)
   if st.session_state.get("TargetColStr") == None:
      st.session_state['TargetColStr']=Target_Col
   st.write("Target variable is: "+str(st.session_state['TargetColStr']))

      

################################## Data Cleaning Expander ###########################
DataClean=st.sidebar.expander("Data Cleaing:", expanded=True)
DataClean.write("Choose cleaning routines:")

######################### Plot Line graph ###############################
PlotSeries = DataClean.checkbox("Plot Target variable line graph:")
if PlotSeries:
    if st.session_state['TargetColStr'] is not None:
        df=st.session_state.df
        x_slider = st.slider("Choose the chart range:", min_value=df.index.min(), max_value=df.index.max(),value=(df.index.min(), df.index.max()), step=1)
        fig = bp.figure(tools="pan, box_zoom, save, reset", x_range=(x_slider[0], x_slider[1]), height = 300, width_policy="max")
        fig.add_tools(CrosshairTool())
        fig.add_tools(HoverTool(tooltips=[("y","@y"),("x", "@x")]))
        fig.line(x=df.index, y=df[st.session_state['TargetColStr']])
        st.bokeh_chart(fig)


######################### Adding Lag Columns ###############################
AddLagCols = DataClean.checkbox("Create Lag Columns for Target:")
if AddLagCols:
  if st.session_state.get('AddLagExecuted') == None:
    df=st.session_state.df
    if 'GrpByField' not in st.session_state:
      st.session_state.GrpByField=None
    st.selectbox('Is your data grouped by any field (Such as StoreID)?:', st.session_state.df.columns, index=None, key='GrpByField')
    NumLags = st.number_input("Enter the number of lag columns to create (hit ENTER to execute):", min_value=0, max_value=15, value=0)
    if NumLags >0:
        for i in range(1,NumLags+1):
          if st.session_state.GrpByField != None:
            df[f"Lag-{i}"] = df.groupby(st.session_state.GrpByField)[st.session_state["TargetColStr"]].shift(-i)
          else:
            df[f"Lag-{i}"] = df[st.session_state["TargetColStr"]].shift(-i)
        df=df.dropna(axis=0, subset=[f"Lag-{j}" for j in range(1,NumLags+1)])
        st.session_state.df = df
        st.write("Here is your dataframe with the added lag columns:")
        st.dataframe(st.session_state.df, height=205)
        st.session_state['AddLagExecuted'] = True
  else:
    st.write("Here is your dataframe with the added lag columns:")
    st.dataframe(st.session_state.df, height=205)
else:
  st.session_state['AddLagExecuted'] = None



######################### Treating Missing Values ###############################
MissingValues = DataClean.checkbox("Missing Value Analysis")
if MissingValues:
    def plot_missing_values(df):
        # calculate the number and percentage of missing values for each field
        total = df.shape[0]
        missing = df.isnull().sum()
        percent = missing / total * 100

        # create a bar chart that shows the percentage of missing values for each field
        fig, ax = plt.subplots(figsize=(10, 2)) # adjust figure size
        ax.bar(df.columns, percent)
        ax.set_ylabel('% Missing')
        ax.set_xticklabels(df.columns, rotation=90)
        ax.grid(axis='y', linestyle='--') #, which='major') # added which='major' to show only horizontal gridlines

        # display the chart using streamlit
        st.pyplot(fig)

        # create a table that shows the field names, the total number of records, the number of existing records, the number of missing values, and the percentage of missing values for each field
        table = pd.DataFrame({
            'Field': df.columns,
            'Total': total,
            'Existing': total - missing,
            'Missing': missing,
            'Percent': percent
        })
        table["Delete?"]=False
        return table
    df =st.session_state.df
    MVtable = plot_missing_values(df)
    # display the table using streamlit
    MVtable = st.data_editor(MVtable.set_index('Field'), use_container_width=True) # set the index to be the field column to avoid showing it twice
    Selected_Rows = MVtable[MVtable["Delete?"]==True].index.to_list()
    # create another table that shows only fields that have percentages above 50% missing
    HighMissing = MVtable[MVtable['Percent'] > 50]
    st.write("Fields with over 50% missing:")
    st.table(HighMissing)
    st.write("Fields marked for deletion:",Selected_Rows)

    # Display a Delete button below the data editor
    if st.button("Delete marked columns"):
        # Delete the columns in df that are in Selected_Rows
        df = df.drop(Selected_Rows, axis=1)
        st.session_state['df'] = df # store the new table in session state
        # Display a success message and the updated df
        st.success(f"Deleted {len(Selected_Rows)} columns from df:")
        st.dataframe(df.head(3))
        df.to_csv('df.csv', index=None)



############################### Data Clean - Auto-impute #############################
AutoImpute = DataClean.checkbox("Auto impute Values")
if AutoImpute:
  # Define Auto Impute function:
  def impute_column(df, col, target):
    # Check if the column has any missing values
    if df[col].isnull().any():
      # Get the percentage of missing values in the column
      percent = df[col].isnull().mean() * 100
      # If the percentage is more than 50, drop the column and return the dataframe
      if percent > 50:
        df.drop(col, axis=1, inplace=True)
        return pd.DataFrame(data=None)
      # Otherwise, proceed with the imputation process
      else:
        # Get the data type of the column
        dtype = df[col].dtype
        # If the column is numeric
        if dtype in ["int", "float"]:
          # Check if there is a strong correlation between the column and the target variable
          corr = df[[col, target]].corr().iloc[0, 1]
          st.write('Correlation between '+col+' and '+target+' is '+ str(corr))
          # If the correlation coefficient is above 0.5 or below -0.5, use random forest imputation
          if abs(corr) > 0.5:
            # Import the RandomForestRegressor or RandomForestClassifier class from scikit-learn
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            imp_model = RandomForestRegressor()
            st.write("Correlation is greater than 0.5 -> Using Random Forest regressor")
            # Fit the model on the rows that have no missing values in the column, using all other columns as predictors
            imp_model.fit(df.dropna()[df.columns.drop(col)], df.dropna()[col])
            # Predict the missing values in the column using the fitted model and the other columns as inputs
            df.loc[df[col].isnull(), col] = imp_model.predict(df[df[col].isnull()][df.columns.drop(col)])
          else:
            # Get the distribution of the column
            skewness = df[col].skew()
            # If the column is normally distributed, use mean imputation
            if abs(skewness) < 0.5:
              st.write('Absolute Skewness in column '+col+' is '+ str(abs(skewness)) + ' < 0.5, imputing the mean')
              imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
              df[col] = imp_mean.fit_transform(df[[col]])
            # If the column is skewed, use median imputation
            else:
              st.write('Absolute Skewness in column '+col+' is ' + str(abs(skewness)) + ' > 0.5, imputing the median')
              imp_median = SimpleImputer(missing_values=np.nan, strategy="median")
              df[col] = imp_median.fit_transform(df[[col]])

        # If the column is categorical
        elif dtype == "object" or dtype == "category":
          # Get the frequency of each category
          mode = df[col].mode()[0]
          count = df[col].value_counts()[mode]
          # If there is a clear mode, use mode imputation
          if count > df.shape[0] * 0.5:
            st.write("There is a clear mode in column: '+col+' -> Imputing the most frequent")
            imp_mode = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
            df[col] = imp_mode.fit_transform(df[[col]])
          # If there is no clear mode, use constant value imputation
          else:
            st.write('There is no clear mode in the Categorical column: '+col+' -> imputing "Unknown"')
            imp_constant = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="Unknown")
            df[col] = imp_constant.fit_transform(df[[col]])
        # If the column is neither numeric nor categorical, raise an error
        else:
          raise ValueError(f"Unsupported data type: {dtype}")
    else:
      st.write(col + " has no missing values.")
    return df[col]

  # Loop over all columns of df and impute values
  for col in df.columns.drop(st.session_state['TargetColStr']):
    # Call the impute_column function with each column name
    df_column = impute_column(df, col, st.session_state["TargetColStr"])
    if not df_column.empty:
      df[col] = df_column
    else:
      st.write(col+ " has over 50% missing velues column DROPPED ************************************<-----")




#################################### Decompose the date ##########################
# Define a function that decomposes the date into day, month, year, and other features
DateDecomp = DataClean.checkbox("Decompose Date field")
if DateDecomp:
  df=st.session_state.df
  Date_Col = st.selectbox('Select the Date column:', df.columns, index=None)
  if Date_Col != None:
    st.session_state['DateColStr']=Date_Col
    st.write("Date Column is: "+str(st.session_state['DateColStr']))
    def decompose_date(df, date_col,format_str="%d/%m/%Y"):
      # Convert the date column to datetime type
      df[date_col] = pd.to_datetime(df[date_col], format=format_str)
      # Extract the day, month, and year from the date column
      df['day'] = df[date_col].dt.day
      df['month'] = df[date_col].dt.month
      df['year'] = df[date_col].dt.year
      # Extract the day of week, week of year, and quarter from the date column
      df['dayofweek'] = df[date_col].dt.dayofweek
      df['weekofyear'] = df[date_col].dt.isocalendar().week
      df['quarter'] = df[date_col].dt.quarter
      # Extract the season from the month column (using meteorological seasons for Italy)
      seasons = {(12, 1, 2): 'winter', (3, 4, 5): 'spring', (6, 7, 8): 'summer', (9, 10, 11): 'autumn'}
      df['season'] = df['month'].apply(lambda x: next(v for k, v in seasons.items() if x in k))
      # Drop the original date column
      df = df.drop(date_col, axis=1)
      # Return the transformed data frame
      return df

    # Call the function with the date column name
    st.session_state.df = decompose_date(st.session_state.df, str(st.session_state['DateColStr']))
    #st.session_state['df'] = df # store the new table in session state
    st.dataframe(st.session_state.df.head(3))




############################### Target Encode Columns ################
TargetEnconde = DataClean.checkbox("Target Encode Columns")
if TargetEnconde:
  df=st.session_state.df
  features = st.multiselect('Select the Categorical Columns to target encode:', df.columns)
  def target_encode(df, target, feature):
    # Convert the feature to string type
    df[feature] = df[feature].astype(str)
    # Calculate the global mean of the target variable
    global_mean = df[target].mean()
    # Calculate the number of samples and the mean of the target variable for each category
    grouped = df.groupby(feature)[target].agg(["count", "mean"])
    # Define the smoothing parameter (you can change this value)
    m = 10
    # Calculate the weight for each category
    weight = 1 / (1 + np.exp(-(grouped["count"] - m) / m))
    # Calculate the smoothed mean for each category
    smoothed_mean = weight * grouped["mean"] + (1 - weight) * global_mean
    # Replace the feature with the smoothed mean
    df[feature] = df[feature].map(smoothed_mean)
    # Return the transformed data frame
    return df[feature]

  EncodeColumnBtn = st.button("Target Encode columns")
  if EncodeColumnBtn:
    df = st.session_state['df']
    # Loop over all columns of df and encodes columns
    for col in features:
      # Call the the function with each column name
      df_column = target_encode(df,str(st.session_state['TargetColStr']),col)
      if not df_column.empty:
        df[col] = df_column
    st.dataframe(df,height=350)



#################################### Data Clean - Mitosheet ##########################
MitosheetCB = DataClean.checkbox("Mitosheet")
if MitosheetCB:
  new_dfs, code = spreadsheet(df)
  #st.write(type(new_dfs))
  st.session_state['df'] = new_dfs.get("df1")
  df = st.session_state['df']
  st.code(code)
  st.dataframe(st.session_state['df'], height=206)



#################################### Data Clean - Dtale ##########################
DtaleCB = DataClean.checkbox("Dtale")
if DtaleCB:
    df=st.session_state['df']
    startup(data_id="1", data=df)
    curr_instance = get_instance("1")
    CSS = """
    <style>
    /* Set the padding for the sidebar content */
    div.sidebar-content {
      padding: 1rem !important;
    }
    /* Set the width and font size for the sidebar content */
    div.sidebar-content {
      width: 40rem !important;
      font-size: 10px;
    }
    </style>
    """

    html = f"""
    {CSS}
    <iframe src="/dtale/main/1" style="height: 400px;width: 100%"/>
    """
    st.markdown(html, unsafe_allow_html=True)
    # Display a button to update session state manually
    if st.button('Trasfer Dtale data to streamlit'):
        #df=curr_instance.data
        st.session_state['df'] = get_instance("1").data
        #st.session_state['df'] = df
        st.write("Transferring data to st.session_state['df']")
        #df = curr_instance.data
        st.dataframe(st.session_state['df'].head(6))




PdProfiling = DataClean.checkbox("Pandas Profiling")
if PdProfiling:
    st.title("Pandas Profiling:")
    profile_df = st.session_state['df'].profile_report()
    st_profile_report(profile_df)



################################## Feature Selection Expander ###########################
st.sidebar.title("2. Feature Selection:")
FeatSel=st.sidebar.expander("Choose Feature Selection Methods:", expanded=True)

################################## Calculate Correlations ######################################
CalcCorr = FeatSel.checkbox("Calculate Column Correlations with Target:")
if CalcCorr:
  df=st.session_state['df']
  # Calculate the correlation matrix
  corr_matrix = df.corr()
  # Get the correlation values for the target column
  target_corr = corr_matrix[st.session_state['TargetColStr']]
  st.write("Correlations with "+st.session_state['TargetColStr']+":")
  st.write(target_corr)




################################## Eliminate low Correlation Columns ###########################
DropLowCorr = FeatSel.checkbox("Eliminate Low Correlation Columns")
if DropLowCorr:
  # Define the function to drop columns with low correlation
  @st.cache_data
  def drop_low_corr(df, target, min_corr=0.5):
    corr_matrix = df.corr()
    target_corr = corr_matrix[target]
    st.write(target_corr)
    # Drop the columns that have correlation lower than min_corr
    df = df.drop(target_corr[target_corr < min_corr].index, axis=1)
    # Return the modified dataframe
    return df

  # Create a slider to select the minimum correlation value
  min_corr = st.slider("Select the minimum acceptable correlation", 0.0, 1.0, 0.5)

  # Create a button to call the function
  if st.button("Eliminate Numerical Columns w/ Low Correlation"):
    # Call the function with the selected min_corr value
    df = drop_low_corr(st.session_state['df'], st.session_state['TargetColStr'], min_corr)
    st.session_state['df']=df
    # Display the modified dataframe
    st.dataframe(st.session_state['df'])




################################## Corellation Plot ############################################
CorPlot = FeatSel.checkbox("Correlation Plot")
if CorPlot:
    df=st.session_state['df']
    st.write("df.shape:",df.shape)
    def corrdot(*args, **kwargs):
        corr_r = args[0].corr(args[1], 'pearson')
        corr_text = f"{corr_r:2.2f}".replace("0.", ".")
        ax = plt.gca()
        ax.set_axis_off()
        marker_size = abs(corr_r) * 10000
        ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
                  vmin=-1, vmax=1, transform=ax.transAxes)
        font_size = abs(corr_r) * 40 + 5
        ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                    ha='center', va='center', fontsize=font_size)

    sns.set(style='white', font_scale=1.6)
    g = sns.PairGrid(df[df.columns], aspect=1.4, diag_sharey=False)
    g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'black'})
    g.map_diag(sns.histplot, kde_kws={'color': 'black'})
    g.map_upper(corrdot)
    st.pyplot(g)



################################# Pycaret Feature Selection ##############################################
PyFeatSel = FeatSel.checkbox("Pycaret Feature Selection")
if PyFeatSel:
    if st.session_state.get('FeatSelExecuted') == None:
        df=st.session_state['df']
        st.write("df.shape:",df.shape)
        method = st.selectbox('Choose the feature selection method:',options = [' ','Univariate','Random Forest','Sequential']) 
        if method !=' ':
            tss=TimeSeriesSplit(n_splits=5)
            setup(data=df, target=st.session_state['TargetColStr'], feature_selection=True, 
                remove_multicollinearity=True, feature_selection_method=method,
                fold_strategy=tss, fold=5, data_split_shuffle=False, fold_shuffle=False, 
                #log_experiment=True, experiment_name='pycaret_regression',
                session_id=42)
            setup_df = pull()
            st.dataframe(setup_df)
            st.session_state['df']=get_config('X_transformed').join(get_config('y_transformed'))
            st.write("These are the selected features in the transformed df:")
            st.dataframe(st.session_state['df'], height=205)
            df=st.session_state['df']
            st.write("df.shape:",df.shape)
            st.session_state['FeatSelExecuted']=True
else:
    st.session_state['FeatSelExecuted']= None




st.sidebar.title("3. Model Development:")
################################## Model Dev Expander ###########################
ModelDev=st.sidebar.expander("Choose how you want to develop your model:", expanded=True)
PycaretDefault = ModelDev.checkbox("Pycaret Model Development")
if PycaretDefault:
  if st.session_state.get('ModelDevExecuted') == None:
    df=st.session_state['df']
    st.write("df.shape:",df.shape)
    tss=TimeSeriesSplit(n_splits=5)
    setup(data=df, target=st.session_state['TargetColStr'], feature_selection=False, 
            #preprocess=False, remove_multicollinearity=True,
            fold_strategy=tss, fold=5, data_split_shuffle=False, fold_shuffle=False, 
            #log_experiment=True, experiment_name='pycaret_regression',
            session_id=42)
    st.session_state.setup_df = pull()
    st.write("Model Setup:")
    st.dataframe(st.session_state.setup_df)
    df=get_config('X_transformed').join(get_config('y_transformed'))
    st.write("Target Variable: ",st.session_state['TargetColStr'])
    st.write("df Used:")
    st.dataframe(df, height=205)
    st.write("df.shape:",df.shape)
    Best = compare_models(sort='MAE')
    st.session_state.compare_df = pull()
    st.write("Model Leaderboard:")
    st.dataframe(st.session_state.compare_df)
    Model = create_model(Best)
    st.session_state['Model']=Model
    st.session_state.ModelOutput = pull()
    st.write("Best Model Class: ", type(Model))
    st.write("Model Output:")
    st.dataframe(st.session_state.ModelOutput)
    st.write("Model Parameters:")
    st.dataframe (pd.DataFrame.from_dict(Model.get_params(),orient='index', columns=['value']))
    Prediction = predict_model(Model, verbose=True)
    st.session_state.PredictionDF=pull()
    st.write("Prediction Scores:")
    st.dataframe(st.session_state.PredictionDF)
    st.session_state.df=df
 
    st.write("Model Pipeline:")
    img=plot_model(Model,plot='pipeline', display_format='streamlit',save=True)
    st.image(img)
    img=plot_model(Model,plot='error', display_format='streamlit',save=True)
    st.image(img, use_column_width=True)
    img=plot_model(Model,plot='feature_all', display_format='streamlit', save=True)
    st.image(img)
    img=plot_model(Model,plot='vc', display_format='streamlit',save=True)
    st.image(img)
    save_model(Model, 'best_model')
    st.session_state['ModelDevExecuted'] = True
  else:
    st.write("Model Setup:")
    st.dataframe(st.session_state.setup_df)
    st.write("Model Leaderboard:")
    st.dataframe(st.session_state.compare_df)
    st.write("Model Output:")
    st.dataframe(st.session_state.ModelOutput)
    st.write("Prediction Scores:")
    st.dataframe(st.session_state.PredictionDF)
else:
  st.session_state['ModelDevExecuted'] = None
        


TuneModel = ModelDev.checkbox("Fine tune model")
if TuneModel:
  if st.session_state.get('ModelTuningExecuted') == None:
    df=st.session_state['df']
    Model=st.session_state.Model
    if 'TuneType' not in st.session_state:
        st.session_state.TuneType = ' '
    st.selectbox('Select Tuning model:', options = [' ','Sklearn-Random Search'
                ,'Scikit-Optimize Bayesian','Optuna TPE'], index=None, key='TuneType')
    if st.session_state.TuneType != ' ':
        st.write("Starting optimization with "+st.session_state.TuneType+"...")
        if st.session_state.TuneType=='Sklearn-Random Search':
            TModel = tune_model(Model, optimize='MAE', n_iter=5)
        elif st.session_state.TuneType=='Scikit-Optimize Bayesian':
            TModel = tune_model(Model, optimize='MAE', n_iter=5, search_library='scikit-optimize', search_algorithm='bayesian')
        #elif st.session_state.TuneType=='Tune-Sklearn Bayesian':
        #   TModel = tune_model(Model, optimize='MAE', n_iter=5, search_library='tune-sklearn', search_algorithm='bayesian', verbose=True)
        elif st.session_state.TuneType=='Optuna TPE':
            TModel = tune_model(Model, optimize='MAE', n_iter=5, search_library='optuna', search_algorithm='tpe')

        st.session_state.Tuned_model = pull()
        st.write("Tuned Model with "+st.session_state.TuneType+".   Cross-Validation Results:")
        st.dataframe(st.session_state.Tuned_model)
        st.write(st.session_state.TuneType+" Tuned Model Parameters:")
        st.session_state.TModelParams = pd.DataFrame.from_dict(TModel.get_params(),orient='index', columns=['value'])
        st.dataframe (st.session_state.TModelParams)
        st.session_state.TPrediction = predict_model(TModel, verbose=True)
        st.session_state.TPredictionDF=pull()
        st.write("Prediction Scores with "+st.session_state.TuneType+":")
        st.dataframe(st.session_state.TPredictionDF)
        st.write("Model Pipeline:")
        img=plot_model(Model,plot='pipeline', display_format='streamlit',save=True)
        st.image(img)
        img=plot_model(Model,plot='error', display_format='streamlit',save=True)
        st.image(img, use_column_width=True)
        img=plot_model(Model,plot='feature_all', display_format='streamlit', save=True)
        st.image(img)
        img=plot_model(Model,plot='vc', display_format='streamlit',save=True)
        st.image(img)
        save_model(Model, 'best_model')
        st.session_state['ModelTuningExecuted'] = True
  else:
    st.write("Tuned Model with "+st.session_state.TuneType+".   Cross-Validation Results:")
    st.dataframe(st.session_state.Tuned_model)
    st.write(st.session_state.TuneType+" Tuned Model Parameters:")
    st.session_state.TModelParams = pd.DataFrame.from_dict(TModel.get_params(),orient='index', columns=['value'])
    st.dataframe (st.session_state.TModelParams)
    st.write("Prediction Scores with "+st.session_state.TuneType+":")
    st.dataframe(st.session_state.TPredictionDF)
else:
  st.session_state['ModelTuningExecuted'] = None

st.sidebar.title("4. File Input/Output:")
################################## SaveFile Expander ###########################
SaveFile=st.sidebar.expander("File Operations:", expanded=True)
@st.cache_data
def convert_df(df):
  return df.to_csv(index=None).encode('utf-8')

if 'df' in st.session_state:
  # Create an input box to enter the file name without extension
  file_name = SaveFile.text_input("Enter a file name and hit 'Enter' to be able to download the dataset (no extension needed):", " ")
  if file_name != " ":
    csv = convert_df(df)
    # Append .csv to the file name
    file_name = file_name + ".csv"
    # Create a download button with the file name as a parameter
    SaveFile.download_button("Dowload data as CSV",data=csv, file_name=file_name, mime='text/csv')

Reload = SaveFile.file_uploader("Reload Your Dataset")
if Reload:
    df = st.cache_data(pd.read_csv)(Reload, index_col=None)
    st.session_state['df'] = df
    st.dataframe(st.session_state['df'])


PrintDF=st.sidebar.button("Print df")
if PrintDF and ('df' in st.session_state):
  st.dataframe(st.session_state.df)


