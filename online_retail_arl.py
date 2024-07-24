############################################
# ASSOCIATION RULE BASED RECOMMENDER SYSTEM
############################################


# BUSINESS PROBLEM : Recommending Products to Users

# Below is the basket information of 3 different users.
# Make the product suggestion that best suits this basket information using the association rule.
# Product recommendations can be 1 or more than 1.
# Derive decision rules based on 2010-2011 Germany customers.

# product_id of User 1: 21987
# product_id of User 2: 23235
# product_id of User 3: 22747


# 1. Data Preprocessing
# 2. Generating Association Rules through German Customers
# 3. Making Product Recommendations to Users Given the Product IDs in the Basket


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II


############################################
# 1. Data Preprocessing
############################################


## 1.1: Read sheet 'Year 2010-2011' from 'online_retail_II' excel file.

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

df.head()
df.describe()
df.info()

# check missing values
df.isnull().sum()


# Minimum values of the Quantity and Price variables are negative due to returns. We need to remove them.
# We can delete the rows containing missing values since the number of rows is enough.

## 1.2: Drop the observation units whose StockCode is POST. (POST is the price added to each invoice, it does not refer to the product.)
## 1.3: Drop observation units containing empty values.
## 1.4: Remove the values containing C in the Invoice from the data set. (C indicates cancellation of the invoice.)
## 1.5: Filter observation units whose price value is less than zero.
## 1.6: Examine the outliers of the Price and Quantity variables and suppress them with threshold values if necessary.

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


# do steps with a function "retail_data_prep"

def retail_data_prep(dataframe):
    # drop missing values
    dataframe.dropna(inplace=True)
    # drop "Invoice" values containing 'C'
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    # Drop values of 'POST' in "StockCode" column.
    dataframe = dataframe[~dataframe["StockCode"].str.contains("POST", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    # replace outliers with threshold values
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)

df.isnull().sum()
df.describe()


########################
# 2. Generating Association Rules through German Customers
########################

df.head()

## 2.1 : Create invoice-product pivot table as shown below by defining "create_invoice_product_df".

# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
# Invoice
# 536370                              0                                 1                       0
# 536852                              1                                 0                       1
# 536974                              0                                 0                       0
# 537065                              1                                 0                       0
# 537463                              0                                 0                       1

# id=True -- analysis will be done using StockCode instead of product descriptions.

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

def create_invoice_product_df_bool(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: True if x > 0 else False)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: True if x > 0 else False)

invoice_product_df = create_invoice_product_df_bool(df, id=True)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df, 10120)

## 2.2 : Generate association rules by defining "create_rules" and then find rules for the customers from Germany.


def create_rules(dataframe, id=True, country="Germany"):
    # create dataframe for specified country.
    dataframe = dataframe[dataframe['Country'] == country]
    # create invoice-product dataframe. 
    dataframe = create_invoice_product_df(dataframe, id)
    # calculate support values.
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    # generate associatian rules.
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

df = df_.copy()

df = retail_data_prep(df)
rules = create_rules(df)
rules.head()



########################
# 3. Making Product Recommendations to Users Given the Product IDs in the Basket
########################

## 3.1 : Find product descriptions using "check_id" function.

# product_id of User 1: 21987
# product_id of User 2: 23235
# product_id of User 3: 22747


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

product_id_1 = 21987
product_id_2 = 23235
product_id_3 = 22747

product_1 = check_id(df, product_id_1)
product_2 = check_id(df, product_id_2)
product_3 = check_id(df, product_id_3)

## 3.2 : Make product recommendations for 3 users using the "arl_recommender" function.

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


# metric can be decided with "arl_recommender_metric" function.


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

arl_recommender(rules, product_id_1, 1)
arl_recommender(rules, product_id_2, 2)
arl_recommender(rules, product_id_3, 2)


## 3.3: Get the names of recommended products.

def recommended_item_names(dataframe, rules_df, product_id, rec_count):
    # recommended item list 
    rec_item_list = arl_recommender(rules_df, product_id, rec_count)
    recommendation_list = []
    for item_id in rec_item_list:
        item_name = check_id(dataframe, item_id)
        recommendation_list.append(item_name)
    return recommendation_list

recommended_items_1 = recommended_item_names(df, rules, product_id_1, 1)
recommended_items_2 = recommended_item_names(df, rules, product_id_2, 2)
# ['SET OF TEA COFFEE SUGAR TINS PANTRY']
# ['ROUND STORAGE TIN VINTAGE LEAF']

recommended_items_3 = recommended_item_names(df, rules, product_id_3, 2)









