# Toni Salminen
# Data-analytiikka
# 2021

# import all necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.core.reshape.pivot import crosstab
from scipy import stats as st
from scipy.stats import chi2_contingency 
from sklearn.linear_model import LinearRegression
import seaborn as sn
import statistics

# read data and save it in pandas dataframe
def get_data(name):
    data = pd.read_csv(name, delimiter=",")
    return data

# nicer output for viewing a math constant value like "1.-0000E2" etc..
def math_to_string(value):
    return '{0:.10f}'.format(value)

# returns pearson correlation from dataset variables
def get_pearson_correlation(data, column_1, column_2):
    x, y = data[column_1], data[column_2]
    # deal with missing values
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    return st.pearsonr(x[~nas], y[~nas])

# returns spearman correlation from dataset variables
def get_spearman_correlation(data, column_1, column_2):
    x, y = data[column_1], data[column_2]
    # deal with missing values
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    return st.spearmanr(x[~nas], y[~nas])

# nicer output for viewing a math constant value like "1.-0000E2" etc.. in percents
def sig_percent(sig):
    return '{0:.10f}'.format(sig * 100)

# returns String explanation of sig-value
def sig_meaning(sig):
    if sig > 0.1:
        return "sig > 0.1. ei riippuvuutta"
    elif sig > 0.05:
        return "sig 0.05 - 0.1. suuntaa antava"
    elif sig > 0.01:
        return "sig 0.01 - 0.05. melkein merkitsevä (heikko riippuvuus)"
    elif sig > 0.001:
        return "sig 0.001 - 0.01. merkitsevä (riippuvuus)"
    elif sig > 0:
        return"sig 0 - 0.001. erittäin merkitsevä (voimakas riippuvuus)"
    return "Sig 0 tai negatiivinen"

if __name__ == "__main__":

    # Read mammals.csv into pandas DataFrame object
    data = get_data('mammals.csv')

    # for automatic plot() enumeration
    plot_n = 1

    # for saving plotted images on device
    save = False

    # for toggling pop-up graphs produced by the program
    show_graphs = True
    
    # Correlation matrixes for nominal , ordinal and both values
    matrix_data = {'body_wt': data['body_wt'], 'brain_wt': data['brain_wt'], 'non_dreaming': data['non_dreaming'], 'dreaming': data['dreaming'], 'total_sleep': data['total_sleep'], 'life_span': data['life_span'], 'gestation': data['gestation'], 'predation': data['predation'], 'exposure': data['exposure'], 'danger': data['danger']}
    matrix_data_sleep = {'total_sleep': data['total_sleep'], 'dreaming': data['dreaming'], 'non_dreaming': data['non_dreaming']}
    matrix_data_ordinal = {'total_sleep': data['total_sleep'],'dreaming': data['dreaming'],'non_dreaming': data['non_dreaming'],'predation': data['predation'],'exposure': data['exposure'],'danger': data['danger']}
    matrix_data_scale = {'total_sleep': data['total_sleep'],'dreaming': data['dreaming'],'non_dreaming': data['non_dreaming'],'body_wt': data['body_wt'],'brain_wt': data['brain_wt'],'life_span': data['life_span'],'gestation': data['gestation']}
    
    matrix_dataframe = pd.DataFrame(matrix_data, columns=['body_wt','brain_wt','non_dreaming','dreaming','total_sleep','life_span','gestation','predation','exposure','danger'])
    matrix_dataframe_sleep = pd.DataFrame(matrix_data_sleep, columns=['total_sleep','dreaming','non_dreaming'])
    matrix_dataframe_ordinal = pd.DataFrame(matrix_data_ordinal, columns=['total_sleep','dreaming','non_dreaming','predation','exposure','danger'])
    matrix_dataframe_scale = pd.DataFrame(matrix_data_scale, columns=['total_sleep','dreaming','non_dreaming','body_wt','brain_wt','life_span','gestation'])

    correlation_matrix = matrix_dataframe.corr()
    correlation_matrix_sleep = matrix_dataframe_sleep.corr()
    correlation_matrix_ordinal = matrix_dataframe_ordinal.corr()
    correlation_matrix_scale = matrix_dataframe_scale.corr()   

    # plot matrixes and save on device if True
    plot1 = plt.figure(plot_n)
    sn.heatmap(correlation_matrix, annot=True)
    if save:
        plt.savefig(str(plot_n), bbox_inches = "tight")
    plot_n += 1

    plot2 = plt.figure(plot_n)
    sn.heatmap(correlation_matrix_sleep, annot=True)
    if save:
        plt.savefig(str(plot_n), bbox_inches = "tight")
    plot_n += 1

    plot3 = plt.figure(plot_n)
    sn.heatmap(correlation_matrix_ordinal, annot=True)
    if save:
        plt.savefig(str(plot_n), bbox_inches = "tight")
    plot_n += 1

    plot4 = plt.figure(plot_n)
    sn.heatmap(correlation_matrix_scale, annot=True)
    if save:
        plt.savefig(str(plot_n), bbox_inches = "tight")
    plot_n += 1


    # Regression analysis for nominal, ordinal ans scale values

    # columns to check correlation/regression on
    correlation_columns = ['total_sleep', 'dreaming', 'non_dreaming']

    # ordinal columns for Spearman
    ordinal_colums = ['predation','exposure','danger']
    # scale columns for Pearson
    scale_colums = ['body_wt','brain_wt','life_span','gestation']

    # print Pearson, Spearman and chi Square results
    for column in data.columns:

        # Pearson & Chi
        if column in scale_colums:
            for corr_column in correlation_columns:

                # Pearson
                print(f"{column} - {corr_column}, sig: {sig_percent(get_pearson_correlation(data, column, corr_column)[1])}%")
                print(f"{sig_meaning(get_pearson_correlation(data, column, corr_column)[1])}")
                print(get_pearson_correlation(data,column, corr_column))
                print()
            
                # chi square
                print()    
                x,y = data[column], data[corr_column]
                nas = np.logical_or(np.isnan(x), np.isnan(y))
                dice = np.array([x[~nas],y[~nas]])
                chi2_stat, sig, df, expected = chi2_contingency(dice)
                print("===Chi-Square Test Value===")
                print(chi2_stat)
                print("===Degrees of Freedom===")
                print("Df = ", df)
                print("===Significance===")
                print("Sig = ", sig)#math_to_string(sig))
                print("===Contingency Table===")
                print(expected)
            print()

        # Spearman
        if column in ordinal_colums:
            for corr_column in correlation_columns:
                print(f"{column} - {corr_column}, sig: {sig_percent(get_spearman_correlation(data, column, corr_column)[1])}%")
                print(f"{sig_meaning(get_spearman_correlation(data, column, corr_column)[1])}")
                print(get_spearman_correlation(data, column, corr_column))
                print()
            print()
            
    # Scatterplots/Linear regression and sig inteprenations with Pearson correlation (scale)
    for i in range(len(correlation_columns)):
        for j in range(len(scale_colums)):
            if data[correlation_columns[i]] is not data[scale_colums[j]]:
                plot_x = plt.figure(plot_n)
                x,y = data[scale_colums[j]], data[correlation_columns[i]]
                nas = np.logical_or(np.isnan(x), np.isnan(y))
                temp_xy = [pd.DataFrame(x[~nas]).values.reshape(-1,1), pd.DataFrame(y[~nas]).values.reshape(-1,1)]
                model = LinearRegression()
                model.fit(temp_xy[0], temp_xy[1])
                y_predicted = model.predict(temp_xy[0])
                plt.title(f"{correlation_columns[i]} - {scale_colums[j]}. Sig: {sig_percent(get_pearson_correlation(data,scale_colums[j],correlation_columns[i])[1])}%\nCoeff: {math_to_string(get_pearson_correlation(data,scale_colums[j],correlation_columns[i])[0])} - Sig: {math_to_string(get_pearson_correlation(data,scale_colums[j],correlation_columns[i])[1])}\n{sig_meaning(get_pearson_correlation(data,scale_colums[j],correlation_columns[i])[1])}")
                plt.xlabel(scale_colums[j])
                plt.ylabel(correlation_columns[i])
                plt.plot(temp_xy[0], temp_xy[1], 'ro', markerfacecolor = "lightblue", label = "Observed Values")
                plt.plot(temp_xy[0], y_predicted, 'g-', linewidth = 2, label = "Trend Line")
                if save:
                    plt.savefig(str(plot_n), bbox_inches = "tight")
                plot_n += 1

    # Scatterplots/Linear regression and sig inteprenations with Pearson correlation (scale)
    # focusing on 'life_span' variables
    for column in ['total_sleep', 'dreaming', 'non_dreaming','body_wt','brain_wt','gestation']:
        plot_x = plt.figure(plot_n)
        x,y = data[column], data['life_span']
        nas = np.logical_or(np.isnan(x), np.isnan(y))
        temp_xy = [pd.DataFrame(x[~nas]).values.reshape(-1,1), pd.DataFrame(y[~nas]).values.reshape(-1,1)]
        model = LinearRegression()
        model.fit(temp_xy[0], temp_xy[1])
        y_predicted = model.predict(temp_xy[0])
        plt.title(f"life_span - {column}. Sig: {sig_percent(get_pearson_correlation(data,column,'life_span')[1])}%\nCoeff: {math_to_string(get_pearson_correlation(data,column,'life_span')[0])} - Sig: {math_to_string(get_pearson_correlation(data,column,'life_span')[1])}\n{sig_meaning(get_pearson_correlation(data,column,'life_span')[1])}")
        plt.xlabel(column)
        plt.ylabel('life_span')
        plt.plot(temp_xy[0], temp_xy[1], 'ro', markerfacecolor = "lightblue", label = "Observed Values")
        plt.plot(temp_xy[0], y_predicted, 'g-', linewidth = 2, label = "Trend Line")
        if save:
            plt.savefig(str(plot_n), bbox_inches = "tight")
        plot_n += 1

    # Cross tabulations for nominal variables
    
    # prevent array truncation on print command
    pd.set_option('display.max_rows', None)

    print()
    ankka_x = data['Type']
    ankka_y = data['exposure']
    ankka_xy = pd.crosstab([ankka_x], [ankka_y]).T
    print(ankka_xy)
    print()
    ankka_y = data['danger']
    ankka_xy = pd.crosstab([ankka_x], [ankka_y]).T
    print(ankka_xy)
    print()
    ankka_y = data['predation']
    ankka_xy = pd.crosstab([ankka_x], [ankka_y]).T
    print(ankka_xy)
    print()

    cross_x = data['predation']
    cross_y = data['Type']
    cross_xy = pd.crosstab([cross_x], [cross_y]).T 
    print(cross_xy)
    print()
    cross_x = data['exposure']
    cross_xy = pd.crosstab([cross_x], [cross_y]).T
    print(cross_xy)
    print()
    cross_x = data['danger']
    cross_xy = pd.crosstab([cross_x], [cross_y]).T
    print(cross_xy)
    print()

    cross_x = data['predation']
    cross_y = data['species']
    cross_xy = pd.crosstab([cross_y], [cross_x])
    print(cross_xy)
    print()

    cross_x = data['exposure']
    cross_xy = pd.crosstab([cross_y], [cross_x])
    print(cross_xy)
    print()
    
    cross_x = data['danger']
    cross_xy = pd.crosstab([cross_y], [cross_x])
    print(cross_xy)
    print()

    # Median values for all values divided by column
    variables = list(correlation_columns + scale_colums + ordinal_colums)
    medians = {}
    for column in variables:
        print(column)
        print(column, np.nanmin(data[column]),"-",np.nanmax(data[column]),"median",statistics.median(data[column]))
        medians[column] = statistics.median(data[column])
    print(medians)

    # Medians of sleeps as bar plot 
    plot_medians = plt.figure(plot_n)
    plt.bar(['total_sleep', 'non_dreaming', 'dreaming'], [medians['total_sleep'],medians['non_dreaming'],medians['dreaming']],width=0.5)
    plt.title('Sleep medians')
    if save:
        plt.savefig(str(plot_n), bbox_inches = "tight")
    plot_n += 1

    # Linear correlation plots on physical aspects with 'gestation'
    for column in ['body_wt','brain_wt','life_span']:
        plot_x = plt.figure(plot_n)
        x,y = data[column], data['gestation']
        nas = np.logical_or(np.isnan(x), np.isnan(y))
        temp_xy = [pd.DataFrame(x[~nas]).values.reshape(-1,1), pd.DataFrame(y[~nas]).values.reshape(-1,1)]
        model = LinearRegression()
        model.fit(temp_xy[0], temp_xy[1])
        y_predicted = model.predict(temp_xy[0])
        plt.title(f"gestation - {column}. Sig: {sig_percent(get_pearson_correlation(data,column,'gestation')[1])}%\nCoeff: {math_to_string(get_pearson_correlation(data,column,'gestation')[0])} - Sig: {math_to_string(get_pearson_correlation(data,column,'gestation')[1])}\n{sig_meaning(get_pearson_correlation(data,column,'gestation')[1])}")
        plt.xlabel(column)
        plt.ylabel('gestation')
        plt.plot(temp_xy[0], temp_xy[1], 'ro', markerfacecolor = "lightblue", label = "Observed Values")
        plt.plot(temp_xy[0], y_predicted, 'g-', linewidth = 2, label = "Trend Line")
        if save:
            plt.savefig(str(plot_n), bbox_inches = "tight")
        plot_n += 1

    # Comparison of median sleep value with all species as bar plot
    horizontal_bar_data = {'Median': medians['total_sleep'],
        'Actual': np.nan_to_num(data['total_sleep'])
        }
    df = pd.DataFrame(horizontal_bar_data,columns=['Median','Actual'], index = data['species'])
    plot_medians_sleep = plt.figure(plot_n)
    fig,ax = plt.subplots(figsize=(5,0.25*len(data['species'])))
    df.plot.barh(ax=ax)
    plt.title('Total sleep. Median and actual by species')
    plt.ylabel('Species')
    plt.xlabel('Total sleep')
    if save:
        plt.savefig(str(plot_n), bbox_inches = "tight")
    plot_n += 1

    # Show all produced plots
    if show_graphs:
        plt.show()