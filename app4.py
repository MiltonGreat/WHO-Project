import streamlit as st
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 1. Data Fetching
@st.cache_data
def fetch_data():
    url_dict = {
        "Life Expectancy": "https://ghoapi.azureedge.net/api/WHOSIS_000001",
        "Infant Mortality": "https://ghoapi.azureedge.net/api/MDG_0000000026",
        "Obesity": "https://ghoapi.azureedge.net/api/NCD_BMI_30A",
        "Hypertension": "https://ghoapi.azureedge.net/api/NCD_HYP_CONTROL_A",
        "Water Access": "https://ghoapi.azureedge.net/api/WSH_WATER_BASIC",
        "Sanitation Access": "https://ghoapi.azureedge.net/api/WSH_SANITATION_SAFELY_MANAGED"
    }

    datasets = {}
    for name, url in url_dict.items():
        response = requests.get(url)
        if response.status_code == 200:
            datasets[name] = pd.DataFrame(response.json()['value'])
        else:
            st.error(f"Failed to fetch data from {url}. Status code: {response.status_code}")
    
    return datasets

# 2. Data Cleaning
def clean_and_prepare_data(df):
    columns_to_drop = [
        'IndicatorCode', 'TimeDimType', 'Value', 'Low', 'High',
        'Id', 'DataSourceDimType', 'DataSourceDim', 'Dim2', 
        'Dim2Type', 'Dim3', 'Dim3Type', 'Comments', 'Date', 
        'TimeDimensionBegin', 'TimeDimensionEnd', 'ParentLocationCode', 
        'Dim1Type', 'Dim1', 'TimeDimensionValue'
    ]
    
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore').copy()
    df_cleaned.rename(columns={'SpatialDim': 'Country', 'ParentLocation': 'Continent'}, inplace=True)
    df_cleaned.dropna(inplace=True)
    df_cleaned.reset_index(drop=True, inplace=True)
    
    return df_cleaned

# 3. Consistent Missing Value Strategy
def consistent_missing_value_strategy(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear').ffill().bfill()
    return df

# 4. Scaling
def min_max_scaling(df, column='NumericValue'):
    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(df[[column]])
    return df

# 5. Feature Engineering
def feature_engineering(datasets):
    def add_change_over_time(df, column='NumericValue'):
        df = df.sort_values(by=['Country', 'TimeDim'])
        df['Change_' + column] = df.groupby('Country')[column].diff().fillna(0)
        return df

    def add_interaction_features(df, interaction_terms):
        for term1, term2 in interaction_terms:
            interaction_feature_name = f'{term1}_x_{term2}'
            df[interaction_feature_name] = df[term1] * df[term2]
        return df

    def categorize_life_expectancy(df):
        bins = [0, 0.6, 0.8, 1.0]
        labels = ['Low', 'Medium', 'High']
        df['LifeExpectancyCategory'] = pd.cut(df['NumericValue'], bins=bins, labels=labels)
        return df

    interaction_terms = [('NumericValue', 'Change_NumericValue')]
    for name, df in datasets.items():
        df = add_change_over_time(df)
        df = add_interaction_features(df, interaction_terms)
        if name == 'Life Expectancy':
            df = categorize_life_expectancy(df)
        datasets[name] = df

    return datasets

# 6. Merging Datasets
def merge_datasets(datasets):
    merged_df = datasets['Life Expectancy']

    for name, df in datasets.items():
        if name != 'Life Expectancy':
            merged_df = merged_df.merge(df, on=['Country', 'TimeDim'], how='outer', suffixes=('', f'_{name.replace(" ", "_")}'))
    
    merged_df.rename(columns={'NumericValue': 'Life_Expectancy'}, inplace=True)
    
    return merged_df

# Function to plot the distribution of a numeric value
def plot_distribution(df, title, add_title=True):
    if df.empty or 'NumericValue' not in df.columns:
        st.warning(f"Not enough data to plot distribution for {title}.")
        return
    plt.figure(figsize=(12, 6))
    sns.histplot(df['NumericValue'], kde=True, bins=30)
    if add_title:
        plt.title(f'Distribution of {title}')
    plt.xlabel('Numeric Value')
    plt.ylabel('Frequency')
    st.pyplot(plt)

# Function to plot a time series analysis
def plot_time_series(df, title, add_title=True):
    if df.empty or 'TimeDim' not in df.columns or 'NumericValue' not in df.columns:
        st.warning(f"Not enough data to plot time series for {title}.")
        return
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df, x='TimeDim', y='NumericValue', hue='Country', legend=False)
    if add_title:
        plt.title(f'Time Series Plot of {title} by Country')
    plt.xlabel('Year')
    plt.ylabel('Numeric Value')
    st.pyplot(plt)

# Function to plot a choropleth map
def plot_choropleth(df, title, add_title=True):
    if df.empty or 'Country' not in df.columns or 'NumericValue' not in df.columns:
        st.warning(f"Not enough data to plot choropleth map for {title}.")
        return
    fig = px.choropleth(df, locations="Country", locationmode="ISO-3", color="NumericValue", hover_name="Country",
                        title=f'Choropleth Map of {title}' if add_title else None, color_continuous_scale=px.colors.sequential.Plasma)
    st.plotly_chart(fig)

# Function to plot a correlation matrix
def plot_correlation_matrix(df, add_title=True):
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=['TimeDim'])
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    if add_title:
        plt.title("Correlation Matrix of Health Metrics and Risk Factors")
    st.pyplot(plt)

# Function to compare top performers with global benchmarks
def global_benchmarks(df):
    adjusted_columns = [
        'Life_Expectancy',
        'NumericValue_Infant_Mortality',
        'NumericValue_Obesity',
        'NumericValue_Hypertension',
        'NumericValue_Water_Access',
        'NumericValue_Sanitation_Access'
    ]
    return df[adjusted_columns].mean()

def compare_top_performers(df, group_col, add_title=True):
    adjusted_columns = [
        'Life_Expectancy',
        'NumericValue_Infant_Mortality',
        'NumericValue_Obesity',
        'NumericValue_Hypertension',
        'NumericValue_Water_Access',
        'NumericValue_Sanitation_Access'
    ]
    global_benchmarks_data = global_benchmarks(df)
    filtered_df = df.dropna(subset=[group_col])
    valid_groups = filtered_df.groupby(group_col)['Life_Expectancy'].apply(lambda x: x.notna().any())
    filtered_df = filtered_df[filtered_df[group_col].isin(valid_groups[valid_groups].index)]
    top_performers = filtered_df.loc[filtered_df.groupby(group_col)['Life_Expectancy'].idxmax()]
    comparison_df = top_performers.set_index(group_col)[adjusted_columns].subtract(global_benchmarks_data)

    plt.figure(figsize=(12, 8))
    comparison_df.plot(kind='bar', figsize=(12, 8))
    if add_title:
        plt.title(f'Comparison of Top-Performing {group_col} with Global Benchmarks')
    plt.ylabel('Difference from Global Benchmark')
    plt.xlabel(group_col)
    plt.axhline(0, color='red', linestyle='--')
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Fetch and process the data
datasets = fetch_data()
cleaned_datasets = {name: clean_and_prepare_data(df) for name, df in datasets.items()}
cleaned_datasets = {name: consistent_missing_value_strategy(df) for name, df in cleaned_datasets.items()}
cleaned_datasets = {name: min_max_scaling(df) for name, df in cleaned_datasets.items()}
feature_engineered_datasets = feature_engineering(cleaned_datasets)
merged_df = merge_datasets(feature_engineered_datasets)

# Dashboard Title
st.markdown("# Global Health Metrics Dashboard")

# Display Merged DataFrame
st.markdown("##### Merged DataFrame")
st.write(merged_df)

# Visualization Sections
sections = [
    ("Life Expectancy", ["Distribution", "Time Series", "Choropleth Map"]),
    ("Infant Mortality", ["Distribution", "Time Series", "Choropleth Map"]),
    ("Obesity", ["Distribution", "Time Series", "Choropleth Map"]),
    ("Hypertension", ["Distribution", "Time Series", "Choropleth Map"]),
    ("Water Access", ["Distribution", "Time Series", "Choropleth Map"]),
    ("Sanitation Access", ["Distribution", "Time Series", "Choropleth Map"])
]

for name, vis_types in sections:
    st.markdown(f"### {name}")
    if "Distribution" in vis_types:
        st.markdown(f"#### Distribution of {name}")
        plot_distribution(cleaned_datasets[name], name)
    if "Time Series" in vis_types:
        st.markdown(f"#### Time Series Analysis of {name}")
        plot_time_series(cleaned_datasets[name], name)
    if "Choropleth Map" in vis_types:
        st.markdown(f"#### Choropleth Map of {name}")
        plot_choropleth(cleaned_datasets[name], name)

# Merged Health Data Section
st.markdown("### Merged Health Data")

st.markdown("#### Correlation Matrix")
plot_correlation_matrix(merged_df)

st.markdown("#### Comparison by Country")
compare_top_performers(merged_df, 'Country')

st.markdown("#### Comparison by Continent")
compare_top_performers(merged_df, 'Continent')

# Top-Performing Regions by Continent
st.markdown("#### Comparison of Top-Performing Regions by Continent")
risk_factors = ['NumericValue_Obesity', 'NumericValue_Hypertension', 'NumericValue_Water_Access', 'NumericValue_Sanitation_Access']
for risk_factor in risk_factors:
    risk_factor_name = risk_factor.split('_')[-1]
    st.markdown(f"#### Comparison of Top-Performing Regions ({risk_factor_name}) with Global Benchmarks")
    benchmark_data = global_benchmarks(merged_df)
    comparison_df = merged_df.groupby('Continent')[[risk_factor]].mean().subtract(benchmark_data.get(risk_factor, 0))
    st.bar_chart(comparison_df)

# Global Health Metrics Menu
st.markdown("### Global Health Metrics Menu")

# Country, Continent, and Year Range Selection
selected_country = st.selectbox('Select a Country', options=merged_df['Country'].dropna().unique())
selected_continent = st.selectbox('Select a Continent', options=merged_df['Continent'].dropna().unique())
year_range = st.slider('Select Year Range', min_value=int(merged_df['TimeDim'].min()), max_value=int(merged_df['TimeDim'].max()), value=(2000, 2020))

# Filter Data based on selections and remove NaN values
filtered_df = merged_df[(merged_df['Country'] == selected_country) & 
                        (merged_df['Continent'] == selected_continent) & 
                        (merged_df['TimeDim'] >= year_range[0]) & 
                        (merged_df['TimeDim'] <= year_range[1])].dropna()

# Display the filtered data
st.write(f"Filtered data for {selected_country}, {selected_continent} from {year_range[0]} to {year_range[1]}:")
st.write(filtered_df)

# Plot the filtered data
if st.checkbox('Show Choropleth Map'):
    plot_choropleth(filtered_df, f"Choropleth Map for {selected_country}, {selected_continent}")

if st.checkbox('Show Time Series'):
    plot_time_series(filtered_df, f"Time Series for {selected_country}, {selected_continent}")

if st.checkbox('Show Distribution Plot'):
    plot_distribution(filtered_df, f"Distribution for {selected_country}, {selected_continent}")

if st.checkbox('Show Correlation Matrix'):
    plot_correlation_matrix(filtered_df)

if st.checkbox('Show Comparison by Country'):
    compare_top_performers(filtered_df, 'Country')

if st.checkbox('Show Comparison by Continent'):
    compare_top_performers(filtered_df, 'Continent')

# Save the filtered data to a CSV file
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="💾 Download CSV",
    data=csv,
    file_name=f"filtered_data_{selected_country}_{selected_continent}_{year_range[0]}-{year_range[1]}.csv",
    mime='text/csv'
)
