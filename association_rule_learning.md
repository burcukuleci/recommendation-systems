
# ASSOCIATION RULE LEARNING

Products are recommended by rules that are extracted by association rule learning. 

The Apriori algorithm is a basket analysis method that reveals product associations.

Parameters that are used for generating association rules are:

- N: Total number of baskets/transactions.

- **Support**(X, Y) = Freq(X, Y) / N  : Probability of X and Y itemsets to be present in a purchase/invoice. (X, Y) association occurs support(%) of the transactions.

- **Confidence**(X, Y) =  Freq(X, Y) / Freq(X) : If X is purchased, there is a confidence(%) chance to purchase Y.

- **Lift** = Support(X, Y) / Support(X)*Support(Y) : Y is three times more likely to be bought when X is bought than when it is bought independently.

## Project :

- Aim: The step-by-step codes to generate association rules are as follows: data preprocessing, preparing invoice-product matrix, extracting association rules, recommending products to users, and script.

- Method: *apriori* and *assocotion_rules* functions are used from *mlextend.frequent_patterns* library. 

- dataset: online_retail_II.xlsx  [link](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)

**Outline**
1. [Data Preprocessing](#1-data-preprocessing)
2. [Preparing ARL Data Structure (Invoice-Product Matrix)](2-preparing-arl-data-structure-invoice-product-matrix)
3. [Extracting Association Rules](#3-extracting-association-rules)
4. [Recommending Products to Users](#4-recommending-products-to-users)
5. [Script](5-script)

## 1. Data Preprocessing

Dataframe is created by reading 'online_retail_II.xlsx' excel.

**retail_data_prep(dataframe)** function is used to handle outliers with interquiartile method and to filter unwanted values.

## 2. Preparing ARL Data Structure (Invoice-Product Matrix)

Data structure should be pivot table such that rows sare invoice numbers, columns are pproduct names and the values are binary form of product quantities.


        Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET

        Invoice

        536370                              0                                 1                       0

        536852                              1                                 0                       1

        536974                              0                                 0                       0


- Sum product quantities within each invoice. Then use *unstack()* to shown items as columns. Use *fillna(0)* to fill NaN values with zero.

- use *applymap* and *lambda* convert each count values that are greater than zero to 1 and values of zeros to 0.

```python
df.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0). \
                                        applymap(lambda x: 1 if x > 0 else 0)
```

note: representing with True and False instead of 1 and 0 results in better performance.

- Define **create_invoice_product_df(dataframe, id=False)** function to create invoice-product matrix.

```python
fr_inv_pro_df_bool = create_invoice_product_df_bool(df_fr, id=True)
```

## 3. Extracting Association Rules

- Calculate support values.

```python
frequent_itemsets = apriori(fr_inv_pro_df_bool,
                            min_support=0.01,
                            use_colnames=True)
```

- Extract association rules according to support.

```python
rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)
```

***Output:***
```
    antecedents    consequents  antecedent support  consequent support   support  confidence       lift  leverage  conviction  zhangs_metric
0       (10002)        (21791)            0.020566            0.028278  0.010283    0.500000  17.681818  0.009701    1.943445       0.963255
1       (21791)        (10002)            0.028278            0.020566  0.010283    0.363636  17.681818  0.009701    1.539111       0.970899
2       (10002)        (21915)            0.020566            0.069409  0.010283    0.500000   7.203704  0.008855    1.861183       0.879265
``` 

***Columns:***

    antecedents: X

    consequents: The item that is predicted or expected to occur with the antecendent. Y

    antecedent support: Support(X)=Freq(X)/N, Freq(X): count of itemset X, N: total number of itemsets

    consequent support: Support(Y)=Freq(Y)/N, Freq(Y): count of temset Y, N: total number of itemsets

    support: Support(X,Y)=Freq(X,Y)/N, Freq(X,Y): count of an itemset consists of X and Y itemsets.
           
             Probability of X and Y itemsets to be present in a purchase/invoice.

    confidence: Freq(X, Y) / Freq(X)
 
                Probability of itemset Y to be present when itemset X is purchased.

    lift: Support(X, Y) / Support(X)*Support(Y)

          When X is purchased, the probability of purchasing Y increases by a factor of lift value.

- Define function **create_rules** to create rules for specified country. 

```python
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
```

## 4. Recommending Products to Users

Recommend products to the user having the product with 'product_id' in the basket.

***Method:***

Sort rules according to one of the metrics (support, confidence, lift).

Antecedent values that contain 'product_id' will be selected, and the consequents corresponding to those selected antecedents will be used for recommendations. Within the set of consequents, only the first item will be chosen as a recommendation.

- Create function called **arl_recommender_metric** to return list of recommended products.


```python
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
```

## 5. Script

Bring all the functions together to functionalize all steps.
