# ONLINE RETAIL ARL RECOMMENDER SYSTEM

#BUSINESS PROBLEM : Recommend products to the user having product of 'product_id' in the basket using association rule learning method.

## 1. Data Preprocessing

Dataframe is created by reading 'online_retail_II.xlsx' excel.

**retail_data_prep(dataframe)** function is used to handle outliers with interquiartile method and to filter unwanted values.

```python
df = retail_data_prep(df)
```

## 2. Generating Association Rules through German Customers

- **2.1**: Create invoice-product pivot table as shown below by defining "create_invoice_product_df".

```python
invoice_product_df = create_invoice_product_df_bool(df, id=True)
invoice_product_df.head()
```

Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
Invoice
536370                              0                                 1                       0
536852                              1                                 0                       1
536974                              0                                 0                       0


- **2.2**: Generate association rules by defining "create_rules" and then find rules for the customers from Germany.

rules = create_rules(df)


## 3. Making Product Recommendations to Users Given the Product IDs in the Basket

- **3.1**: Find product descriptions using "check_id" function.

```python
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

product_id_1 = 21987
product_1 = check_id(df, product_id_1)
```

- **3.2**: Make product recommendations for 3 users using the "arl_recommender" function.

```python
arl_recommender(rules, product_id_1, 1)
arl_recommender(rules, product_id_2, 2)
arl_recommender(rules, product_id_3, 2)
```

- **3.3**: Get the names of recommended products.

```python
def recommended_item_names(dataframe, rules_df, product_id, rec_count):
    # recommended item list 
    rec_item_list = arl_recommender(rules_df, product_id, rec_count)
    recommendation_list = []
    for item_id in rec_item_list:
        item_name = check_id(dataframe, item_id)
        recommendation_list.append(item_name)
    return recommendation_list

recommended_items_2 = recommended_item_names(df, rules, product_id_2, 2)
# ['SET OF TEA COFFEE SUGAR TINS PANTRY']
# ['ROUND STORAGE TIN VINTAGE LEAF']
```