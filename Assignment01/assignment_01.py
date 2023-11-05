import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



# Define a list of the 12 categorical variables
categorical_variables = ['drive', 'fuelType', 'fuelType1', 'make', 'model', 'trany',
                         'VClass', 'atvType', 'trans_dscr', 'tCharger', 'sCharger', 'c240Dscr']

# Define a list of the numeric variables you want to visualize for outliers
numeric_variables = ['barrels08', 'barrelsA08', 'charge240', 'city08', 'city08U',
                     'cityA08', 'cityA08U', 'cityCD', 'cityE', 'cityUF', 'co2', 'co2A', 'co2TailpipeAGpm',
                     'co2TailpipeGpm', 'comb08', 'comb08U', 'combA08', 'combA08U', 'combE', 'combinedCD',
                     'combinedUF', 'cylinders', 'displ', 'feScore', 'fuelCost08', 'fuelCostA08',
                     'ghgScore', 'ghgScoreA', 'highway08', 'highway08U', 'highwayA08', 'highwayA08U',
                     'highwayCD', 'highwayE', 'highwayUF', 'hlv', 'hpv', 'lv2', 'lv4', 'pv2', 'pv4',
                     'range', 'rangeCity', 'rangeCityA', 'rangeHwy', 'rangeHwyA', 'UCity', 'UCityA',
                     'UHighway', 'UHighwayA', 'year', 'youSaveSpend']

# Select relevant feature variables based on pearson correlation analysis to use in rms, and rmse
selected_features = ['city08', 'comb08', 'UHighway']


def load_data(url):
    # Load data from the specified URL
    df = pd.read_csv(url)
    return df

def calculate_fuel_efficiency(df):
    # Calculate fuel efficiency as MPG (miles per gallon)
    df['mpgData'] = df['comb08']
    return df

def filter_missing_values(df, numeric_vars):
    # Drop rows with missing values in the selected numeric variables
    df.dropna(subset=numeric_vars, inplace=True)
    return df

# explore categorical variables
def explore_categorical_variables(df, cat_vars):
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(18, 10))
    fig.subplots_adjust(hspace=0.4)  # Adjust vertical spacing between subplots

    for i, var in enumerate(cat_vars):
        row, col = i // 4, i % 4
        ax = axes[row, col]

        if var in df.columns:
            top_15_categories = df[var].value_counts().nlargest(15).index.tolist()
            if top_15_categories:
                sns.countplot(data=df[df[var].isin(top_15_categories)], x=var, ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_xlabel(var)
                ax.set_ylabel('Count')
                ax.set_title(f'Distribution of {var}')

def box_plot_num_var(df, num_vars):
    # Explore numeric variables with box plots to identify outliers
    fig, axes = plt.subplots(nrows=7, ncols=9, figsize=(18, 14))
    fig.subplots_adjust(hspace=0.4, wspace=0.5)

    for i, var in enumerate(num_vars):
        row, col = i // 9, i % 9
        ax = axes[row, col]

        sns.boxplot(data=df, y=var, ax=ax)
        ax.set_ylabel(var)
        ax.set_title(f'{var}')


# Create a heatmap for the correlation matrix
def heatmap_correlation_matrix():
    correlation_matrix = df[numeric_variables].corr()
    plt.figure(figsize=(10, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".0f", vmin=-1, vmax=1)
    plt.xticks(rotation=75, ha='right')
    plt.title('Correlation Matrix Heatmap')
    plt.savefig('heatmap.pdf')
    plt.show()



def fuel_efficiency_vs_fuel_type():
    # Filter out rows with missing values in the fuel efficiency column
    df.dropna(subset=['mpgData'], inplace=True)
    # Create a bar chart to visualize fuel efficiency by fuel type
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='fuelType', y='mpgData', ci=None) 
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Fuel Type')
    plt.ylabel('Miles per Gallon (MPG)') 
    plt.title('Fuel Efficiency by Fuel Type')
    plt.tight_layout()
    # Save the figure as a PDF file
    plt.savefig('fuel_efficiency_by_fuel_type.pdf')
    plt.show()



def fuel_efficiency_vs_price():
    # price
    price_column = 'fuelCost08'
    # Filter out rows with missing values in the price column
    df.dropna(subset=[price_column], inplace=True)
    # Create a scatter plot or regression plot to visualize the relationship between price and fuel efficiency
    plt.figure(figsize=(12, 6))
    sns.regplot(data=df, x=price_column, y='mpgData', scatter_kws={'alpha':0.5})
    plt.xlabel('Price')
    plt.ylabel('Miles per Gallon (MPG)')  # You can adjust the label accordingly
    plt.title('Price vs. Fuel Efficiency')
    plt.tight_layout()
    # Save the figure as a PDF file
    plt.savefig('price_vs_fuel_efficiency.pdf')
    plt.show()


def pearson_correlation(df):
    pearson_corr = df[numeric_variables].corr(method='pearson')
    print(pearson_corr)


def rmse_rms(selected_features, df):
    df[numeric_variables] = df[numeric_variables].fillna(df[numeric_variables].mean())
    # Split the data into training and testing sets.
    X = df[selected_features]
    y = df['UCity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build a simple linear regression model.
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set.
    y_pred = model.predict(X_test)

    # Evaluate the model using mean squared error (MSE) and root mean squared error (RMSE).
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Print evaluation metrics.
    print(f'Mean Squared Error (MSE): {mse:.2f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    

    # Visualize the predicted vs. actual values.
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual UCity')
    plt.ylabel('Predicted UCity')
    plt.title('Actual vs. Predicted UCity')
    plt.show()


if __name__ == "__main__":
    data_url = 'https://raw.githubusercontent.com/DrUzair/MLSD/master/Datasets/vehicles.csv'
    # Load data
    df = load_data(data_url)
    pearson_correlation(df)
    rmse_rms(selected_features, df)
    calculate_fuel_efficiency(df)
    filter_missing_values(df,numeric_variables)
    explore_categorical_variables(df, categorical_variables)
    box_plot_num_var(df, numeric_variables)
    fuel_efficiency_vs_fuel_type()
    fuel_efficiency_vs_price()
    plt.show()