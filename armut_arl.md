# ARMUT RECOMMENDATION SYSTEM

Create a product recommendation system with association rule learning method to recommended services to a user that having a certain service in the basket.

## 1. Data Preprocessing

- dataset: armut.csv (Note: This dataset is not shared due to Miuul course confidentiality.)

Columns: UserId, ServiceId, CategoryId, CreateDate

- **1.1**: Read 'armut_data.csv' file.

- **1.2**: Create a new variable "Service" to represent these services by combining ServiceId (row[1]) and CategoryId (row[2]) with "_".

- **1.3**: Combine UserId and the date variable you just created as 'YYYY_MM' and assign it to a new variable called "BasketId".

```python
df.head()
```

   UserId  ServiceId  CategoryId          CreateDate Service Year_Month       BasketId
0   25446          4           5 2017-08-06 16:11:00     4_5    2017-08  25446_2017-08
1   22948         48           5 2017-08-06 16:12:00    48_5    2017-08  22948_2017-08


## 2: Extract Association Rules and Recommend Service.

- **2.1**: Create pivot table of "BasketId" and "Service" as shown below.

```python
invoice_product_df.head()
```

Service        0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
BasketId
0_2017-08        0     0      0     0      0     0     0     0     0     0..
0_2017-09        0     0      0     0      0     0     0     0     0     0..
0_2018-01        0     0      0     0      0     0     0     0     0     0..

- **2.2** : Create association rules.

- **2.3** : Use the "arl_recommender" function to recommend 3 services to a user who last received the 2_0 service.

```python
recommended_services=arl_recommender(rules, "2_0", rec_count=3)
```

['22_0', '25_0', '15_1']
