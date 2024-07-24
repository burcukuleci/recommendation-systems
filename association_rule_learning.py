############################################
# ASSOCIATION RULE LEARNING
############################################

# 1. Data Preprocessing
# 2. Preparing ARL Data Structure (Invoice-Product Matrix)
# 3. Extracting Association Rules
# 4. Recommending Products to Users
# 5. Script


#### BUSINESS PROBLEM : Recommending Products to Users

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)



############################################
# 1. Data Preprocessing
############################################

# dataset:
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011", engine="openpyxl")

df = df_.copy()
df.head()
df.describe()

# Minimum values of the Quantity and Price variables are negative due to returns. We need to remove them.
# We can delete the rows containing missing values since the number of rows is enough.
# We need to trim the outliers that appear as max values.

# detect outlier thresholds using interquartile range method.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# replace the outliers with threshold values (lower and upper limits)

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    # drop missing values
    dataframe.dropna(inplace=True)

    # drop return invoices to not include returns.
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    # do not include negative values of 'Quantity' and 'Price'
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]

    # handle the outliers
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)

# check the missing values
df.isnull().sum()

# check the distribution of values
df.describe()



############################################
# 2. Preparing ARL Data Structure (Invoice-Product Matrix)
############################################

df.head()

# Desired data structure:
# Show whether a product is included in the relevant invoice as 1 or 0 in the relevant product column.

# Method: Count each item's quantity for each invoice, convert counts that are greater than zero to 1 and counts of zeros to 0.

# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
# Invoice
# 536370                              0                                 1                       0
# 536852                              1                                 0                       1
# 536974                              0                                 0                       0
# 537065                              1                                 0                       0
# 537463                              0                                 0                       1



### Practice: Create ARL data structure for France.

# dataframe for France
df_fr = df[df['Country'] == "France"]

# calculate quantity sum of each item for each invoice.
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)

# print the count values as each item name is represented as a column. - dataframe -
# fillna(0): represent NaN values for count as 0.
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

# use applymap to convert quantity values to binary values of 1 and 0.
df_fr.groupby(['Invoice', 'StockCode']). \
    agg({"Quantity": "sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

# define function for creating ARL matrix structure.
# id=True >> analysis will be done with 'StockCode' column instead of 'Description' column.

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


fr_inv_pro_df = create_invoice_product_df(df_fr)


# show quantity values as True or False for better performance.

def create_invoice_product_df_bool(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: True if x > 0 else False)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: True if x > 0 else False)

# ARL dataframe >>>
fr_inv_pro_df_bool = create_invoice_product_df_bool(df_fr, id=True)


# use function check_id to get the item name from a stockcode.
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_fr, 10120)



############################################
# 3. Extracting Association Rules
############################################

# min_support = support threshold value. The calculated support values lower than threshold value will be filtered out.
frequent_itemsets = apriori(fr_inv_pro_df_bool,
                            min_support=0.01,
                            use_colnames=True)

# sort values according to support values in descending order.
frequent_itemsets.sort_values("support", ascending=False)

# itemsets: item combinations

# ---------------
#         support                                           itemsets
# 538    0.773779                                             (POST)
# 387    0.187661                                            (23084)
# ...         ...                                                ...
# 18785  0.010283                       (21086, 22492, 22326, 22727)
# 40654  0.010283  (22659, 23206, 22726, 22727, 22728, 20750, 223...

# extract association rules according to support.
rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

# -------------
#         antecedents       consequents  antecedent support  consequent support   support  confidence       lift  leverage  conviction  zhangs_metric
# 0           (10002)           (21791)            0.020566            0.028278  0.010283    0.500000  17.681818  0.009701    1.943445       0.963255
# 1           (21791)           (10002)            0.028278            0.020566  0.010283    0.363636  17.681818  0.009701    1.539111       0.970899
# 2           (10002)           (21915)            0.020566            0.069409  0.010283    0.500000   7.203704  0.008855    1.861183       0.879265


# filter rules according to support, confidence and lift thresholds.
rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]

# rule number decreased to 84 due to filtering.

# sort rules according to confidence values in descending order to see most possible consequents(Y) for antecendents(X).
rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
sort_values("confidence", ascending=False)


# --- write function for extraction of rules

def create_rules(dataframe, id=True, country="France"):
    # create dataframe for specified country.
    dataframe = dataframe[dataframe['Country'] == country]
    # create invoice-product dataframe.
    dataframe = create_invoice_product_df(dataframe, id)
    # calculate support values.
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    # generate associatian rules.
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

rules = create_rules(df)



############################################
# 4. Recommending Products to Users
############################################

product_id = 22492
check_id(df, product_id)

# Which antecendents values contains 'product_id'?
# What are the consequents corresponding those selected antecedents?

sorted_rules = rules.sort_values("lift", ascending=False)

recommendation_list = []

# i: index, product: values of the column antecedents(X)
for i, product in enumerate(sorted_rules["antecedents"]):
    # j: each product in product set (itemset)
    for j in list(product):
        # consider antecedents(X) with 'product_id'.
        # if the product of interest (product_id) equals to 'product' from an antecedent(X)
        if j == product_id:
            # select first product of corresponding consequent(Y) for product_id in the antecedent of an index i
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

recommendation_list[0:3]

check_id(df, 22326)

# write function

def arl_recommender_metric(rules_df, product_id, metric, rec_count=1):
    '''
    parameters:
        rules_df: association rules dataframe.
        product_id: id of the product that is in the basket to be used for recommendations.
        metric: metric for sorting the itemsets in rules_df.
        rec_count: number of recommended products.
    '''
    # sort rules according to specified metric in descending order
    sorted_rules = rules_df.sort_values(metric, ascending=False)
    recommendation_list = []
    # i: index, product: values of the column antecedents(X)
    for i, product in enumerate(sorted_rules["antecedents"]):
        # j: each product in product set (itemset)
        for j in list(product):
            # consider antecedents(X) with 'product_id'.
            # if the product of interest (product_id) equals to 'product' from an antecedent(X)
            if j == product_id:
                # select first product of corresponding consequent(Y) for product_id in the antecedent of an index i
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

recommended_product_ids=arl_recommender_metric(rules, 22492, metric="lift", rec_count=3)




############################################
# 5. Script
############################################

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


def create_invoice_product_df_bool(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: True if x > 0 else False)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: True if x > 0 else False)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df_bool(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules


def arl_recommender_metric(rules_df, product_id, metric, rec_count=1):
    '''
    parameters:
        rules_df: association rules dataframe.
        product_id: id of the product that is in the basket to be used for recommendations.
        metric: metric for sorting the itemsets in rules_df.
        rec_count: number of recommended products.
    '''
    # sort rules according to specified metric in descending order
    sorted_rules = rules_df.sort_values(metric, ascending=False)
    recommendation_list = []
    # i: index, product: values of the column antecedents(X)
    for i, product in enumerate(sorted_rules["antecedents"]):
        # j: each product in product set (itemset)
        for j in list(product):
            # consider antecedents(X) with 'product_id'.
            # if the product of interest (product_id) equals to 'product' from an antecedent(X)
            if j == product_id:
                # select first product of corresponding consequent(Y) for product_id in the antecedent of an index i
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

df = df_.copy()

# data preprocessing
df = retail_data_prep(df)

# Association Rules Extraction
rules = create_rules(df)

# Recommending products to users
recommended_product_ids=arl_recommender_metric(rules, 22492, metric="lift", rec_count=4)

print(recommended_product_ids)


