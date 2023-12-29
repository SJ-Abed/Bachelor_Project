# Import libraries
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from datetime import datetime
# Import Class table
def prep_V_3(otli,ksigma,least_order_count):
    if not 'CLASS.csv' in os.listdir():
        os.chdir('..\\..')
    start_time = datetime.now()
    # read class DataFrame and set its columns' names
    class_df = pd.read_csv('CLASS.csv')
    class_df = class_df.rename(columns={"id": "classid", "name": "class_name"})
    class_df = class_df.drop_duplicates()

    # assign classes with no parent to primary parents
    class_df.loc[[1], 'primaryparentid'] = 1
    class_df.loc[[18], 'primaryparentid'] = 42
    class_df.loc[[28], 'primaryparentid'] = 1
    class_df.loc[[34], 'primaryparentid'] = 125
    class_df.loc[[35], 'primaryparentid'] = 94
    class_df.loc[[101], 'primaryparentid'] = 93
    class_df.loc[[102], 'primaryparentid'] = 16
    class_df.loc[[103, 104, 184], 'primaryparentid'] = 103

    # Import Order table

    # read Orders DataFrame and set its columns' name.
    orders_df = pd.read_csv('ORDERS.csv')
    orders_df = orders_df.rename(columns={"id": "bid", "id.1": "cid", "id.2": "catid", 'categoryid': 'classid', "id.3": "iid","quantitystepcount": "quantity", "totaloriginalprice": "price", "?column?": "days"})
    """ bid:    basket ID
        cid:    customer ID
        catid:  category ID
        iid:    item ID
        days:   days' count from that order
    """

    # drop some missing values that contains a small part of the dataset
    # delete rows that contain wrong values
    orders_df.dropna(subset=['classid', 'catid', 'iid', 'quantity', 'price', 'marketid'], inplace=True)
    orders_df = orders_df[orders_df['price'] > 300]
    orders_df = orders_df[orders_df['quantity'] > 0]
    # fill rows with null segmentation label with "unlabeled"
    orders_df.fillna('unlabeled', inplace=True)
    # Convert DataFrame to a category base DataFrame
    orders_df = orders_df.groupby(['bid', 'cid', 'checkoutdate', 'classid', 'catid', 'segmentationlabel', 'days','marketid']).sum().reset_index().sort_values('checkoutdate', ascending=False)

    # remove customers with 2 or less transactions
    customer_order_count = orders_df.drop_duplicates(['cid', 'bid']).groupby('cid').size()
    orders_df = orders_df[orders_df['cid'].isin(customer_order_count[customer_order_count > least_order_count].index)]

    # remove classes that has sold less than 25 times
    class_counts = orders_df.__deepcopy__().groupby('classid').count().reset_index()[['classid', 'bid']].rename(columns={'bid': 'rep'})
    # (class_counts[(class_counts['rep']<100)].merge(class_df)).merge(class_counts,left_on = 'primaryparentid',right_on = 'classid',how = "left")
    rare_items = class_counts[(class_counts['rep'] < 25)]['classid']
    orders_df = orders_df[~orders_df.classid.isin(rare_items)]


    # change format of checkout date from string to time
    def str_to_date(row):
        return datetime.strptime(row, "%Y-%m-%d %H:%M:%S.%f")
    orders_df['checkoutdate'] = orders_df['checkoutdate'].apply(str_to_date)

    # add two columns that show first-time purchasing of each class or category

    # cat_first_checkout
    cat_first_checkout = orders_df.groupby(['cid', 'catid'], as_index=False)['checkoutdate'].min()
    cat_first_checkout.rename(columns={'checkoutdate': 'cat_first_checkout'}, inplace=True)
    orders_df = orders_df.merge(cat_first_checkout, on=['cid', 'catid'])

    # class_first_checkout
    class_first_checkout = orders_df.groupby(['cid', 'classid'], as_index=False)['checkoutdate'].min()
    class_first_checkout.rename(columns={'checkoutdate': 'class_first_checkout'}, inplace=True)
    orders_df = orders_df.merge(class_first_checkout, on=['cid', 'classid'])

    # with these functions we check whether this item/category is reordered or not.
    # if it isn't first order, then the related column takes vlaue 1, otherwise 0
    def check_reordered_cat(row):
        if row.checkoutdate == row.cat_first_checkout:
            return 0
        return 1


    def check_reordered_class(row):
        if row.checkoutdate == row.class_first_checkout:
            return 0
        return 1


    # a value from 0 to 6 to determine day of week
    def get_week_day(row):
        return row.weekday()


    orders_df['day_of_week'] = orders_df['checkoutdate'].apply(get_week_day)
    orders_df['cat_reordered'] = orders_df.apply(check_reordered_cat, axis=1)
    orders_df['class_reordered'] = orders_df.apply(check_reordered_class, axis=1)
    ########################################################################################
    # otli = 10000 #outlier interval days
    # ksigma = 1000 #k*sigma for removing outliers
    def remove_outliers(x):
        if len(x)==1:
            return True
        else:
            return (x <= x.mean() + ksigma * x.std())
    #remove outlier purchases (orders with more than otli days interval with next one)
    df = orders_df.sort_values(by=['cid', 'checkoutdate'],ascending=[True,True]).groupby(["cid", "bid"])['days'].max().reset_index(name='days_of_orders').groupby('cid')[
        'days_of_orders'].diff()
    df = pd.DataFrame({'days_from_last': -df})
    df['days_untill_next'] = orders_df.sort_values(by=['cid', 'checkoutdate'],ascending=[True,False]).groupby(["cid", "bid"])['days'].max().reset_index(name='days_of_orders').groupby('cid')[
        'days_of_orders'].diff(periods = -1)

    df['user_id'] = list(orders_df.groupby(["cid", "bid"])['days'].max().reset_index(level=0)['cid'])
    df['days'] =orders_df.groupby(["cid", "bid"])['days'].max().reset_index()['days']
    df['bid'] =orders_df.groupby(["cid", "bid"])['days'].max().reset_index()['bid']
    df['order_number'] = df.groupby(['user_id']).cumcount() + 1
    df = df.merge(df.groupby('user_id')['order_number'].max().reset_index().rename(columns={'order_number':'total_orders'}),on='user_id')
    #df contains date detail of each csutomer orders. like date, interval between next and previous order,total volume of orders and number of that order.

    before_gap_orders = df[df.days_untill_next > otli]
    total_gap_counter = before_gap_orders.groupby('user_id').count()#.groupby('days').count()
    more_3_gap_cid = total_gap_counter[total_gap_counter['days']>=3].index
    gap_counter = total_gap_counter[total_gap_counter['days']>=3][['days']].reset_index().rename(columns={'days':'gap'})
    gap_counter = before_gap_orders[before_gap_orders['user_id'].isin(more_3_gap_cid)].merge(gap_counter)
    gap_counter['gap_ratio'] = gap_counter['gap']/gap_counter['total_orders']
    gap_counter = gap_counter.groupby(['user_id','gap','gap_ratio','total_orders'])['order_number'].max().reset_index()
    #####important
    #remove customers that their outlier orders constitute more than 10 percent of total orders
    more_than_10p_gap_ratio_cid = gap_counter[gap_counter['gap_ratio']>0.1]['user_id']

    ss2 = before_gap_orders.groupby('user_id').count()#.groupby('days').count()
    exact_1_gap_cid = ss2[ss2['days']==1].index
    s1 = before_gap_orders[before_gap_orders['user_id'].isin(exact_1_gap_cid)]
    #####important
    #remove first order of each customer if the interval between first and second order is more than otli days
    first_gap_bid = s1[(s1.order_number<2)].bid
    #####important
    #remove customers that they outlier orders constitute noticable percentage of all orders
    gapped_cid_1 = s1[((s1.total_orders<6)&(s1.order_number>=2))|(s1.total_orders<4)].user_id


    ss3 = before_gap_orders.groupby('user_id').count()#.groupby('days').count()
    exact_2_gap_cid = ss3[ss3['days']==2].index
    s2 = before_gap_orders[before_gap_orders['user_id'].isin(exact_2_gap_cid)]
    s22 = s2.groupby('user_id')['order_number'].max().reset_index()
    #####important
    #remove first two orders of each customer if the interval between first and second, and second and third order is more than otli days
    first_two_gaps_bid = s2[s2.user_id.isin((s22[s22.order_number<=2]).user_id)].bid
    #####important
    #remove customers that they outlier orders constitute noticable percentage of all orders
    gapped_cid_2 = s2[((s2.total_orders<9)&(~(s2.user_id.isin(s22[s22.order_number<=2]))))|(s2.total_orders<7)].user_id
    ########################################################################################
    orders_df = orders_df[~orders_df.cid.isin(more_than_10p_gap_ratio_cid)]
    orders_df = orders_df[~orders_df.cid.isin(gapped_cid_1)]
    orders_df = orders_df[~orders_df.cid.isin(gapped_cid_2)]
    orders_df = orders_df[~orders_df.bid.isin(first_gap_bid)]
    orders_df = orders_df[~orders_df.bid.isin(first_two_gaps_bid)]



    # Set labels and split prior data

    # 1 for an item that is reordered in the last basket and 0 for otherwise (our y in training)
    def reordered_label(row):
        if row.checkoutdate == row.last_basket_date:
            return 1
        return 0


    # split last basket from whole histore. (last basket for prediction and other for creating features)
    def split_prior_data(orders_dff):
        # last basker shows the "days" value (interval since the day of transaction) for last prior basket of each customer
        last_basket = orders_dff.groupby(['cid'], as_index=False)['days'].min()
        last_basket.rename(columns={'days': 'prior_last_basket'}, inplace=True)
        # add prior_last_basket column to table that shows value of "days" column of customer's last basket
        orders_dff = orders_dff.merge(last_basket, on=['cid'])

        # add last_basket_date column to table that shows value of "checkoutdate" column of customer's last basket
        last_basket_date = orders_dff.groupby(['cid'], as_index=False)['checkoutdate'].max()
        last_basket_date.rename(columns={'checkoutdate': 'last_basket_date'}, inplace=True)
        orders_dff = orders_dff.merge(last_basket_date, on=['cid'])

        # change 'days' column so it shows interval between that trnasaction and last transaction
        orders_dff['days'] -= orders_dff['prior_last_basket']
        orders_dff['prior_last_basket'] = 0

        # split prior data from whole data and save it into proir_data variable
        orders_dff.sort_values(by=['cid', 'checkoutdate'], inplace=True)
        last_so_id = orders_dff.drop_duplicates(subset=['cid'], keep='last')
        prior_data = orders_dff[~orders_dff['bid'].isin(last_so_id['bid'])]

        train_validation_data = orders_dff
        train_validation_data.sort_values(by=['cid', 'checkoutdate'], inplace=True)
        # exclude rows of last basket items that they aren't reordered. (firt purchase of that categories is in the last trnasaction)
        train_validation_data = train_validation_data[(((train_validation_data['cat_reordered'] == 1) &
                                                        (train_validation_data['checkoutdate'] ==
                                                         train_validation_data['last_basket_date']))
                                                       | (train_validation_data['checkoutdate'] !=
                                                          train_validation_data['last_basket_date']))]
        # just keep row of the last time that the category ordered
        train_validation_data.drop_duplicates(inplace=True, subset=['cid', 'catid'], keep='last')
        del train_validation_data['bid']

        # set reordered label for categories that has been reordered in the last basket 1 (exactly our y for training)
        train_validation_data = train_validation_data.merge(last_so_id[['cid', 'bid']], on='cid')
        train_validation_data['reorder_label'] = train_validation_data.apply(reordered_label, axis=1)

        del prior_data['last_basket_date']
        del prior_data['prior_last_basket']
        return prior_data, train_validation_data[['cid', 'bid', 'catid', 'classid', 'reorder_label']]


    prior_data_n_1, test_validation_data = split_prior_data(orders_df.__deepcopy__())
    prior_data_n_2, train_validation_data = split_prior_data(prior_data_n_1.__deepcopy__())
    prior_data_n_3, train_validation_data_2 = split_prior_data(prior_data_n_2.__deepcopy__())


    # Generate Features

    # user feature
    def generate_user_features(prior_data):
        user_features = pd.DataFrame(columns=['user_id'])
        user_features['user_id'] = prior_data['cid'].sort_values().unique()
        user_reorder_rate = prior_data.groupby(["cid", "cat_reordered"])['cat_reordered'].count().groupby(level=0).apply(
            lambda x: x / float(x.sum())).reset_index(name='cat_reorder_rate')
        user_reorder_rate = user_reorder_rate.pivot(index='cid', columns='cat_reordered', values=['cat_reorder_rate'])
        user_reorder_rate = pd.DataFrame(user_reorder_rate.to_records())
        user_reorder_rate.columns = ['user_id', '0', '1']
        user_reorder_rate.set_index("user_id", inplace=True)
        user_reorder_rate.fillna(0, inplace=True)
        user_reorder_rate.reset_index(inplace=True)
        user_features['user_cat_reorder_rate'] = user_reorder_rate['1']

        # Get count of all unique cat for every user
        user_features['user_unique_cats'] = \
            prior_data.groupby(["cid"])['catid'].nunique().reset_index(name='unique')['unique']

        # Get count of all cat ordered by user
        user_features['user_total_cats'] = prior_data.groupby(["cid"])['catid'].size().reset_index(name='count')['count']

        # Get mean cat per user = Average cart size of user
        df = prior_data.groupby(["cid", "bid"])['catid'].count().reset_index(name='cart_size')
        ss = df[df.groupby(["cid"]).cart_size.transform(remove_outliers).eq(1)]
        df = ss.groupby('cid')['cart_size'].mean().reset_index()
        del ss
        user_features['user_avg_cart_size'] = df['cart_size']

        # Get average days between 2 orders for every user
        df = \
            prior_data.groupby(["cid", "bid"])['days'].max().reset_index(name='days_of_orders') \
                .groupby('cid')['days_of_orders'].diff()
        df = pd.DataFrame({'days_between_orders': -df})
        df['user_id'] = list(prior_data.groupby(["cid", "bid"])['days'].max().reset_index(level=0)['cid'])
        df.dropna(inplace=True)
        ss = df[df.groupby(["user_id"]).days_between_orders.transform(remove_outliers).eq(1)]
        df = ss.groupby('user_id', as_index=False)['days_between_orders'].mean()
        del ss
        user_features = user_features.merge(df, on='user_id')

        # get user product reorder ratio
        # number of unique products reordered / number of unique products ordered

        # get user cats reorder ratio
        # number of unique cats reordered / number of unique cats ordered

        df = prior_data.groupby(["cid"])['catid'].nunique().reset_index(name='user_unique_cats')
        df = df.merge(prior_data[prior_data['cat_reordered'] == 1].groupby(["cid"])['catid'].nunique().reset_index(
            name='user_reordered_cats'), on='cid')
        df.fillna(0, inplace=True)
        df['user_reordered_cats_ratio'] = df['user_reordered_cats'] / df['user_unique_cats']
        del user_features['user_unique_cats']
        user_features = user_features.merge(df, left_on='user_id', right_on='cid')
        del user_features['cid']

        # get user classes reorder ratio
        # number of unique classes reordered / number of unique classes ordered
        df = prior_data.groupby(["cid"])['classid'].nunique().reset_index(name='user_unique_classes')
        df = df.merge(prior_data[prior_data['cat_reordered'] == 1].groupby(["cid"])['classid'].nunique().reset_index(
            name='user_reordered_classes'), on='cid')
        df.fillna(0, inplace=True)
        df['user_reordered_classes_ratio'] = df['user_reordered_classes'] / df['user_unique_classes']
        user_features = user_features.merge(df, left_on='user_id', right_on='cid')
        del user_features['cid']

        return user_features


    user_features_n_1 = generate_user_features(prior_data_n_1)
    user_features_n_2 = generate_user_features(prior_data_n_2)
    user_features_n_3 = generate_user_features(prior_data_n_3)


    #  Cat Features :
    def generate_cat_features(prior_data):
        # create an empty dataframe
        product_features = pd.DataFrame(columns=['catid'])

        # add cat_name
        product_features['catid'] = prior_data['catid'].sort_values().unique()

        ############# average days between cat reorder
        df = prior_data.sort_values(by=['cid', 'catid', 'checkoutdate'])[
            ['cid', 'bid', 'catid', 'checkoutdate', 'days', 'price']]
        df['days_since_prior_cat_order'] = -df.groupby(['cid', 'catid'])['days'].transform(lambda x: x.diff())
        ss = df[df.groupby(["catid"]).days_since_prior_cat_order.transform(remove_outliers).eq(1)]
        df3 = ss.groupby(['catid']).mean().reset_index()[['catid', 'days_since_prior_cat_order']]
        ss = df3[df3.days_since_prior_cat_order.transform(remove_outliers).eq(1)]
        df3.fillna(ss.days_since_prior_cat_order.mean(), inplace=True)
        del ss
        df3.rename(columns={'days_since_prior_cat_order': 'days_between_cat_orders'}, inplace=True)
        days_to_next = list(df['days_since_prior_cat_order'].iloc[1:])
        days_to_next.append(np.nan)
        # df = \
        #     prior_data.groupby(["cid", "bid"])['days'].max().reset_index(name='days_of_orders') \
        #         .groupby('cid')['days_of_orders'].diff()
        # df = pd.DataFrame({'days_between_orders': -df})
        df['days_to_next_order'] = days_to_next
        df['price'] = df['price'].astype(float)
        df['days_to_next_order'] = df['days_to_next_order'].replace(0, 1)
        df['price_day_ratio'] = df['price'] / df['days_to_next_order']
        df.loc[abs(df['days_to_next_order']) == 0, "price_day_ratio"] = np.nan
        ss = df[df.groupby(["catid"]).price_day_ratio.transform(remove_outliers).eq(1)]
        df2 = ss.groupby(["catid"])['price_day_ratio'].mean().reset_index(name='ave_price_day_ratio')
        del ss
        # df2.fillna(0, inplace=True)
        product_features = product_features.merge(df2, on=['catid'])
        product_features = product_features.merge(df3, on=['catid'])
        ###############################

        # get reorder_rate for each cat
        # reorder_rate = reorders / total orders
        df = pd.DataFrame({'cat_reorder_rate': prior_data.groupby(['catid', 'cat_reordered'])['cat_reordered']. \
                          count().groupby(level=0).apply(lambda x: x / float(x.sum()))}).reset_index()

        # get data of reordered cat only
        new_df = df[df['cat_reordered'] == 1]
        new_df['cat_reorder_rate'] = new_df['cat_reorder_rate'] * new_df['cat_reordered']

        # handling for cat which were never reordered, hence reorder_rate = 0.0
        new_df_1 = df[(df['cat_reordered'] == 0) & (df['cat_reorder_rate'] == float(1.0))]
        new_df_1['cat_reorder_rate'] = new_df_1['cat_reorder_rate'] * new_df_1['cat_reordered']
        new_df = new_df.append(new_df_1)

        # drop other columns of the new_df and sort values by cat name to align with cat features dataframe
        new_df.drop('cat_reordered', axis=1, inplace=True)
        new_df.sort_values(by='catid', inplace=True)
        new_df = new_df.reset_index(drop=True)

        # add to feat_1 of cat_features dataframe
        product_features['cat_reorder_rate'] = new_df['cat_reorder_rate']

        #  generate boolean values if cat belongs to below classes
        products = orders_df[['catid', 'classid']].drop_duplicates().reset_index()

        products['isMilk'] = products['classid'].apply(lambda x: x == 51).astype(int)
        products['isSeifijat'] = products['classid'].apply(lambda x: x == 57).astype(int)
        products['isFruits'] = products['classid'].apply(lambda x: x == 21).astype(int)
        products['isLabaniat'] = products['classid'].apply(lambda x: x == 2 or x == 55).astype(int)
        products['isProtein'] = products['classid'].apply(lambda x: x == 156 or x == 68).astype(int)
        products['isSnack'] = products['classid'].apply(lambda x: x == 131 or x == 132 or x == 133).astype(int)
        products['isKalayeAsasi'] = products['classid'].apply(
            lambda x: x == 9 or x == 45 or x == 92 or x == 69 or x == 71).astype(int)

        new_product_feat = products[
            ['isMilk', 'isSeifijat', 'isFruits', 'isLabaniat', 'isProtein', 'isSnack', 'isKalayeAsasi']]

        # reduce sparsity using NMF
        # ref:https://www.kaggle.com/themissingsock/matrix-decomposition-with-buyer-data

        nmf = NMF(n_components=3)
        model = nmf.fit(new_product_feat)
        W = model.transform(new_product_feat)
        prod_data = pd.DataFrame(normalize(W))

        prod_data.columns = ['p_reduced_feat_1', 'p_reduced_feat_2', 'p_reduced_feat_3']
        products.drop(['isMilk', 'isSeifijat', 'isFruits', 'isLabaniat', 'isProtein', 'isSnack', 'isKalayeAsasi'],
                      axis=1, inplace=True)

        product_features['p_reduced_feat_1'] = prod_data['p_reduced_feat_1']
        product_features['p_reduced_feat_2'] = prod_data['p_reduced_feat_2']
        product_features['p_reduced_feat_3'] = prod_data['p_reduced_feat_3']

        # merge dept_reorder_rate and aisle_reorder_rate to existing product features

        del df, new_df, new_df_1, new_product_feat, model, prod_data
        return product_features


    def generate_user_cat_features(prior_data):
        # create an empty dataframe
        user_cat_features = pd.DataFrame(columns=['cid', 'catid'])

        # add user and cat to dataframe
        u_t = prior_data.groupby(["cid", "catid"]).size().reset_index()
        user_cat_features["cid"] = u_t["cid"]
        user_cat_features["catid"] = u_t["catid"]

        # How frequently user ordered the cat ?
        # #times user ordered the cat/ #times user placed an order
        df = prior_data.groupby(["cid", "catid"])["cat_reordered"].size()
        df = df / prior_data.groupby(["cid"]).size()
        df = df.reset_index(name='order_rate')
        df.fillna(0., inplace=True)
        user_cat_features["u_t_order_rate"] = df["order_rate"]

        # How frequently user reordered the cat ?
        # #times user reordered the cat/ #times user ordered the cat
        df = prior_data[prior_data["cat_reordered"] == 1].groupby(["cid", "catid"])["cat_reordered"].size()
        df = df / prior_data.groupby(["cid", "catid"]).size()
        df = df.reset_index(name='reorder_rate')
        df.fillna(0., inplace=True)
        user_cat_features["u_t_reorder_rate"] = df["reorder_rate"]

        # Number of orders placed since the cat was last ordered ?

        ############# average days between cat reorder
        df = prior_data.sort_values(by=['cid', 'catid', 'checkoutdate'])[
            ['cid', 'bid', 'catid', 'checkoutdate', 'days', 'price']]
        df['days_since_prior_cat_order'] = -df.groupby(['cid', 'catid'])['days'].transform(lambda x: x.diff())
        ss = df[df.groupby(["cid", "catid"]).days_since_prior_cat_order.transform(remove_outliers).eq(1)]
        df1 = ss.groupby(['cid', 'catid']).mean().reset_index()[['cid', 'catid', 'days_since_prior_cat_order']]
        del ss
        df1.rename(columns={'days_since_prior_cat_order': 'days_between_user_cat_orders'}, inplace=True)
        days_to_next = list(df['days_since_prior_cat_order'].iloc[1:])
        days_to_next.append(np.nan)

        df['days_to_next_order'] = days_to_next
        df['price'] = df['price'].astype(float)
        df['days_to_next_order'] = df['days_to_next_order'].replace(0, 1)
        df['price_day_ratio'] = df['price'] / df['days_to_next_order']
        # df.loc[abs(df['days_to_next_order'])==0, "price_day_ratio"] = np.nan
        ss = df[df.groupby(["cid", "catid"]).price_day_ratio.transform(remove_outliers).eq(1)]
        df2 = ss.groupby(["cid", "catid"])['price_day_ratio'].mean().reset_index(name='user_ave_price_day_ratio')
        del ss
        # df2.fillna(0, inplace=True)
        user_cat_features = user_cat_features.merge(df2, on=['cid', 'catid'])
        user_cat_features = user_cat_features.merge(df1, on=['cid', 'catid'])

        df = prior_data.sort_values(by=['cid', 'catid', 'checkoutdate'])[
            ['cid', 'bid', 'catid', 'checkoutdate', 'days', 'price']].drop_duplicates(subset=['cid', 'catid'], keep='last')
        df['price'] = df['price'].astype(float)
        df['user_days_price_ratio_since_prior'] = df['price'] / (df['days'] + 0.4)
        user_cat_features = user_cat_features.merge(df[['cid', 'catid', 'user_days_price_ratio_since_prior']],
                                                    on=['cid', 'catid'])

        ###############################3

        #  Get Number of orders
        prior_data_order_number = prior_data.groupby('cid').apply(
            lambda x: x.drop_duplicates(subset='bid').reset_index(drop=True).
                reset_index()[['cid', 'bid', 'index']].merge(x, on=['cid', 'bid'])).reset_index(drop=True)

        prior_data_order_number = prior_data_order_number.rename({'index': 'order_number'}, axis='columns')

        # Get last order_number placed by user , subtract with last order_number with the CAT in cart
        df = prior_data_order_number.groupby(["cid", "catid"])['order_number'].max().reset_index()
        df_2 = prior_data_order_number.groupby(["cid"])['order_number'].max().reset_index()
        new_df = pd.merge(df, df_2, how='outer', left_on=['cid'], right_on=['cid'])
        new_df['order_number_diff'] = new_df['order_number_y'] - new_df['order_number_x']
        user_cat_features['u_t_orders_since_last'] = new_df['order_number_diff']

        # Get last order_number placed by user , subtract with last order_number with the CLASS in cart
        df = prior_data_order_number.groupby(["cid", "classid"])['order_number'].max().reset_index()
        df_2 = prior_data_order_number.groupby(["cid"])['order_number'].max().reset_index()
        new_df = pd.merge(df, df_2, how='outer', left_on=['cid'], right_on=['cid'])
        new_df['order_number_diff'] = new_df['order_number_y'] - new_df['order_number_x']
        user_cat_features['u_c_orders_since_last'] = new_df['order_number_diff']

        # max_streak
        def max_streak(row):
            #  Function to calculate the maximum number of orders in a row which contains reorders of a cat
            maxx = 0
            summ = 0
            for i in range(len(row) - 1):
                if row[i + 1] - row[i] == 1:
                    summ += 1
                else:
                    if summ > maxx:
                        maxx = summ
                    summ = 0
            return maxx

        df = prior_data_order_number.groupby(["cid", "catid"])['order_number'].apply(list).reset_index(
            name='max_streak_cat')

        df['max_streak_cat'] = [max_streak(df['max_streak_cat'].iloc[i]) for i in range(len(df))]
        user_product_features = pd.merge(user_cat_features, df, on=["cid", "catid"])

        del df, prior_data_order_number, df2, df_2
        return user_product_features


    cat_features_n_1 = generate_cat_features(prior_data_n_1)
    cat_features_n_2 = generate_cat_features(prior_data_n_2)
    cat_features_n_3 = generate_cat_features(prior_data_n_3)

    user_cat_features_n_1 = generate_user_cat_features(prior_data_n_1)
    user_cat_features_n_2 = generate_user_cat_features(prior_data_n_2)
    user_cat_features_n_3 = generate_user_cat_features(prior_data_n_3)

    # number of zero values
    print(user_cat_features_n_1[user_cat_features_n_1['user_ave_price_day_ratio'] == 0].shape)
    print(user_cat_features_n_2[user_cat_features_n_2['user_ave_price_day_ratio'] == 0].shape)
    print(user_cat_features_n_3[user_cat_features_n_3['user_ave_price_day_ratio'] == 0].shape)
    print(user_cat_features_n_1[user_cat_features_n_1['user_days_price_ratio_since_prior'] == 0].shape)
    print(user_cat_features_n_2[user_cat_features_n_2['user_days_price_ratio_since_prior'] == 0].shape)
    print(user_cat_features_n_3[user_cat_features_n_3['user_days_price_ratio_since_prior'] == 0].shape)
    print(cat_features_n_1[cat_features_n_1['ave_price_day_ratio'] == 0].shape)
    print(cat_features_n_2[cat_features_n_2['ave_price_day_ratio'] == 0].shape)
    print(cat_features_n_3[cat_features_n_3['ave_price_day_ratio'] == 0].shape)

    # number of infinity values
    print(user_cat_features_n_1[user_cat_features_n_1['user_days_price_ratio_since_prior'].isin([np.inf, -np.inf])].shape)
    print(user_cat_features_n_2[user_cat_features_n_2['user_days_price_ratio_since_prior'].isin([np.inf, -np.inf])].shape)
    print(user_cat_features_n_3[user_cat_features_n_3['user_days_price_ratio_since_prior'].isin([np.inf, -np.inf])].shape)
    print(user_cat_features_n_1[user_cat_features_n_1['user_ave_price_day_ratio'].isin([np.inf, -np.inf])].shape)
    print(user_cat_features_n_2[user_cat_features_n_2['user_ave_price_day_ratio'].isin([np.inf, -np.inf])].shape)
    print(user_cat_features_n_3[user_cat_features_n_3['user_ave_price_day_ratio'].isin([np.inf, -np.inf])].shape)
    print(cat_features_n_1[cat_features_n_1['ave_price_day_ratio'].isin([np.inf, -np.inf])].shape)
    print(cat_features_n_2[cat_features_n_2['ave_price_day_ratio'].isin([np.inf, -np.inf])].shape)
    print(cat_features_n_3[cat_features_n_3['ave_price_day_ratio'].isin([np.inf, -np.inf])].shape)

    # replace zero by one (avoiding creating infinity)
    cat_features_n_1['days_between_cat_orders'] = cat_features_n_1['days_between_cat_orders'].replace(0, 1)
    cat_features_n_2['days_between_cat_orders'] = cat_features_n_2['days_between_cat_orders'].replace(0, 1)
    cat_features_n_3['days_between_cat_orders'] = cat_features_n_3['days_between_cat_orders'].replace(0, 1)

    user_cat_features_n_1['days_between_user_cat_orders'] = user_cat_features_n_1['days_between_user_cat_orders'].replace(0,
                                                                                                                          1)
    user_cat_features_n_2['days_between_user_cat_orders'] = user_cat_features_n_2['days_between_user_cat_orders'].replace(0,
                                                                                                                          1)
    user_cat_features_n_3['days_between_user_cat_orders'] = user_cat_features_n_3['days_between_user_cat_orders'].replace(0,
                                                                                                                          1)

    user_features_n_1['days_between_orders'] = user_features_n_1['days_between_orders'].replace(0, 1)
    user_features_n_2['days_between_orders'] = user_features_n_2['days_between_orders'].replace(0, 1)
    user_features_n_3['days_between_orders'] = user_features_n_3['days_between_orders'].replace(0, 1)

    """
    feature : how frequently product was reordered on any given hour ?
    """


    def cat_day(prior_data):
        df = prior_data.groupby(['catid', 'day_of_week'])["cat_reordered"].size()
        df = df / prior_data.groupby(["catid"]).size()
        df = df.reset_index(name='cat_week_reorder_rate')
        return df


    def class_day(prior_data):
        df = prior_data.groupby(['classid', 'day_of_week'])["class_reordered"].size()
        df = df / prior_data.groupby(["classid"]).size()
        df = df.reset_index(name='class_week_reorder_rate')
        return df


    def get_days_since_prior(orders_df):
        customer_order_days = orders_df.sort_values(by='checkoutdate').groupby(["cid", "bid"], as_index=False)['days'].max()
        df = customer_order_days.groupby(['cid']).diff()['days']
        df = pd.DataFrame({'days_since_prior_order': -df})
        df[['cid', 'bid']] = customer_order_days[['cid', 'bid']]
        df.dropna(inplace=True)
        df['days_since_prior_order'] = df['days_since_prior_order'].astype(int)
        days_since_prior = df.merge(orders_df, on=['cid', 'bid'])
        return days_since_prior


    # last_cid_so_prior = days_since_prior.drop_duplicates(subset=['cid'], keep='last')


    test_days_since_prior = get_days_since_prior(orders_df.__deepcopy__())
    train_days_since_prior = get_days_since_prior(prior_data_n_1.__deepcopy__())
    train_days_since_prior_2 = get_days_since_prior(prior_data_n_2.__deepcopy__())

    train_validation_data = train_validation_data.merge(train_days_since_prior[['cid', 'bid', 'days_since_prior_order',
                                                                                'day_of_week', 'checkoutdate']],
                                                        on=['cid', 'bid'])
    train_validation_data_2 = train_validation_data_2.merge(
        train_days_since_prior_2[['cid', 'bid', 'days_since_prior_order',
                                  'day_of_week', 'checkoutdate']],
        on=['cid', 'bid'])

    train_validation_data.drop_duplicates(inplace=True)
    train_days_since_prior = train_days_since_prior[~train_days_since_prior['bid'].isin(train_validation_data['bid'])]
    train_validation_data_2.drop_duplicates(inplace=True)
    train_days_since_prior_2 = train_days_since_prior_2[
        ~train_days_since_prior_2['bid'].isin(train_validation_data_2['bid'])]

    test_validation_data = test_validation_data.merge(
        test_days_since_prior[['cid', 'bid', 'days_since_prior_order', 'day_of_week', 'checkoutdate']],
        on=['cid', 'bid'])
    test_validation_data.drop_duplicates(inplace=True)
    test_days_since_prior = test_days_since_prior[~test_days_since_prior['bid'].isin(test_validation_data['bid'])]


    def cat_days_since_prior(days_since_prior):
        df = days_since_prior.groupby(['catid', 'days_since_prior_order'])["cat_reordered"].size()
        df = df / days_since_prior.groupby(["catid"]).size()
        df = df.reset_index(name='t_days_since_prior_order_reorder_rate')

        return df


    def class_days_since_prior(days_since_prior):
        df = days_since_prior.groupby(['classid', 'days_since_prior_order'])["class_reordered"].size()
        df = df / days_since_prior.groupby(["classid"]).size()
        df = df.reset_index(name='c_days_since_prior_order_reorder_rate')

        return df


    def user_days_since_prior(days_since_prior):
        """
        feature: how frequently user reordered any product given difference between 2 orders in days ?
        """
        df = days_since_prior.groupby(['cid', 'days_since_prior_order'])["cat_reordered"].size()
        df = df / days_since_prior.groupby(["cid"]).size()
        df = df.reset_index(name='u_days_since_prior_order_reorder_rate')

        return df


    def u_t_days_since_prior(days_since_prior):
        df = days_since_prior.groupby(["cid", "catid", "days_since_prior_order"])["cat_reordered"].size()
        df = df / days_since_prior.groupby(["cid", "catid"]).size()
        df = df.reset_index(name='u_t_days_since_prior_reorder_rate')
        return df


    def u_c_days_since_prior(days_since_prior):
        df = days_since_prior.groupby(["cid", "classid", "days_since_prior_order"])["class_reordered"].size()
        df = df / days_since_prior.groupby(["cid", "classid"]).size()
        df = df.reset_index(name='u_c_days_since_prior_reorder_rate')
        return df


    # merge features
    train_validation_merge_1 = train_validation_data.merge(user_features_n_2, left_on='cid', right_on='user_id')
    # train_validation_merge_1 = train_validation_merge_1.merge(product_day(train_days_since_prior), how='left',
    #                                                       on=['iid', 'day_of_week'])
    train_validation_merge_1 = train_validation_merge_1.merge(cat_day(train_days_since_prior), how='left',
                                                              on=['catid', 'day_of_week'])
    train_validation_merge_1 = train_validation_merge_1.merge(class_day(train_days_since_prior), how='left',
                                                              on=['classid', 'day_of_week'])

    train_validation_merge_2 = train_validation_data_2.merge(user_features_n_3, left_on='cid', right_on='user_id')
    # train_validation_merge_1 = train_validation_merge_1.merge(product_day(train_days_since_prior_2), how='left',
    #                                                       on=['iid', 'day_of_week'])
    train_validation_merge_2 = train_validation_merge_2.merge(cat_day(train_days_since_prior_2), how='left',
                                                              on=['catid', 'day_of_week'])
    train_validation_merge_2 = train_validation_merge_2.merge(class_day(train_days_since_prior_2), how='left',
                                                              on=['classid', 'day_of_week'])

    test_validation_merge = test_validation_data.merge(user_features_n_1, left_on='cid', right_on='user_id')
    # test_validation_merge = test_validation_merge.merge(product_day(test_days_since_prior), how='left',
    #                                                     on=['iid', 'day_of_week'])
    test_validation_merge = test_validation_merge.merge(cat_day(test_days_since_prior), how='left',
                                                        on=['catid', 'day_of_week'])
    test_validation_merge = test_validation_merge.merge(class_day(test_days_since_prior), how='left',
                                                        on=['classid', 'day_of_week'])
    ################################

    # product/cat/class
    # train_validation_merge_1 = train_validation_merge_1.merge(product_days_since_prior(train_days_since_prior), how='left',
    #                                                       left_on=['iid', 'days_since_prior_order'],
    #                                                       right_on=['iid', 'days_since_prior_order'])
    train_validation_merge_1 = train_validation_merge_1.merge(cat_days_since_prior(train_days_since_prior), how='left',
                                                              left_on=['catid', 'days_since_prior_order'],
                                                              right_on=['catid', 'days_since_prior_order'])
    train_validation_merge_1 = train_validation_merge_1.merge(class_days_since_prior(train_days_since_prior), how='left',
                                                              left_on=['classid', 'days_since_prior_order'],
                                                              right_on=['classid', 'days_since_prior_order'])

    # train_validation_merge_2 = train_validation_merge_2.merge(product_days_since_prior(train_days_since_prior_2), how='left',
    #                                                       left_on=['iid', 'days_since_prior_order'],
    #                                                       right_on=['iid', 'days_since_prior_order'])
    train_validation_merge_2 = train_validation_merge_2.merge(cat_days_since_prior(train_days_since_prior_2), how='left',
                                                              left_on=['catid', 'days_since_prior_order'],
                                                              right_on=['catid', 'days_since_prior_order'])
    train_validation_merge_2 = train_validation_merge_2.merge(class_days_since_prior(train_days_since_prior_2), how='left',
                                                              left_on=['classid', 'days_since_prior_order'],
                                                              right_on=['classid', 'days_since_prior_order'])

    # test_validation_merge = test_validation_merge.merge(product_days_since_prior(test_days_since_prior), how='left',
    #                                                     left_on=['iid', 'days_since_prior_order'],
    #                                                     right_on=['iid', 'days_since_prior_order'])
    test_validation_merge = test_validation_merge.merge(cat_days_since_prior(test_days_since_prior), how='left',
                                                        left_on=['catid', 'days_since_prior_order'],
                                                        right_on=['catid', 'days_since_prior_order'])
    test_validation_merge = test_validation_merge.merge(class_days_since_prior(test_days_since_prior), how='left',
                                                        left_on=['classid', 'days_since_prior_order'],
                                                        right_on=['classid', 'days_since_prior_order'])

    ################################
    # user days since prior
    train_validation_merge_1 = train_validation_merge_1.merge(user_days_since_prior(train_days_since_prior), how='left',
                                                              left_on=['cid', 'days_since_prior_order'],
                                                              right_on=['cid', 'days_since_prior_order'])

    train_validation_merge_2 = train_validation_merge_2.merge(user_days_since_prior(train_days_since_prior_2), how='left',
                                                              left_on=['cid', 'days_since_prior_order'],
                                                              right_on=['cid', 'days_since_prior_order'])

    test_validation_merge = test_validation_merge.merge(user_days_since_prior(test_days_since_prior), how='left',
                                                        left_on=['cid', 'days_since_prior_order'],
                                                        right_on=['cid', 'days_since_prior_order'])
    ################################
    # up/ut/uc
    # train_validation_merge_1 = train_validation_merge_1.merge(u_p_days_since_prior(train_days_since_prior), how='left',
    #                                                       left_on=["cid", "iid", "days_since_prior_order"],
    #                                                       right_on=["cid", "iid", "days_since_prior_order"])
    train_validation_merge_1 = train_validation_merge_1.merge(u_t_days_since_prior(train_days_since_prior), how='left',
                                                              left_on=["cid", "catid", "days_since_prior_order"],
                                                              right_on=["cid", "catid", "days_since_prior_order"])
    train_validation_merge_1 = train_validation_merge_1.merge(u_c_days_since_prior(train_days_since_prior), how='left',
                                                              left_on=["cid", "classid", "days_since_prior_order"],
                                                              right_on=["cid", "classid", "days_since_prior_order"])

    # train_validation_merge_2 = train_validation_merge_2.merge(u_p_days_since_prior(train_days_since_prior), how='left',
    #                                                       left_on=["cid", "iid", "days_since_prior_order"],
    #                                                       right_on=["cid", "iid", "days_since_prior_order"])
    train_validation_merge_2 = train_validation_merge_2.merge(u_t_days_since_prior(train_days_since_prior_2), how='left',
                                                              left_on=["cid", "catid", "days_since_prior_order"],
                                                              right_on=["cid", "catid", "days_since_prior_order"])
    train_validation_merge_2 = train_validation_merge_2.merge(u_c_days_since_prior(train_days_since_prior_2), how='left',
                                                              left_on=["cid", "classid", "days_since_prior_order"],
                                                              right_on=["cid", "classid", "days_since_prior_order"])

    # test_validation_merge = test_validation_merge.merge(u_p_days_since_prior(test_days_since_prior), how='left',
    #                                                     left_on=["cid", "iid", "days_since_prior_order"],
    #                                                     right_on=["cid", "iid", "days_since_prior_order"])
    test_validation_merge = test_validation_merge.merge(u_t_days_since_prior(test_days_since_prior), how='left',
                                                        left_on=["cid", "catid", "days_since_prior_order"],
                                                        right_on=["cid", "catid", "days_since_prior_order"])
    test_validation_merge = test_validation_merge.merge(u_c_days_since_prior(test_days_since_prior), how='left',
                                                        left_on=["cid", "classid", "days_since_prior_order"],
                                                        right_on=["cid", "classid", "days_since_prior_order"])
    #############################
    # merge product feature
    # train_validation_merge_1 = train_validation_merge_1.merge(product_features_n_2, on='catid')
    # test_validation_merge = test_validation_merge.merge(product_features_n_1, on='catid')
    #############################
    # merge cat feature
    train_validation_merge_1 = train_validation_merge_1.merge(cat_features_n_2, on='catid')
    train_validation_merge_2 = train_validation_merge_2.merge(cat_features_n_3, on='catid')
    test_validation_merge = test_validation_merge.merge(cat_features_n_1, on='catid')
    #############################
    # merge user product feature
    # train_validation_merge = train_validation_merge_1.merge(user_product_features_n_2, on=['cid', 'iid'])
    # test_validation_merge = test_validation_merge.merge(user_product_features_n_1, on=['cid', 'iid'])
    #############################
    # merge user cat feature
    train_validation_merge_1 = train_validation_merge_1.merge(user_cat_features_n_2, on=['cid', 'catid'])
    train_validation_merge_2 = train_validation_merge_2.merge(user_cat_features_n_3, on=['cid', 'catid'])
    test_validation_merge = test_validation_merge.merge(user_cat_features_n_1, on=['cid', 'catid'])

    # handle missing values of new features
    mean_ave_price_day_ratio = train_validation_merge_1['ave_price_day_ratio'].mean()
    train_validation_merge_1['ave_price_day_ratio'] = mean_ave_price_day_ratio
    train_validation_merge_2['ave_price_day_ratio'] = mean_ave_price_day_ratio
    test_validation_merge['ave_price_day_ratio'] = mean_ave_price_day_ratio

    train_validation_merge_1['user_ave_price_day_ratio'] = train_validation_merge_1['user_ave_price_day_ratio'].fillna(
        train_validation_merge_1['ave_price_day_ratio'])
    train_validation_merge_2['user_ave_price_day_ratio'] = train_validation_merge_2['user_ave_price_day_ratio'].fillna(
        train_validation_merge_2['ave_price_day_ratio'])
    test_validation_merge['user_ave_price_day_ratio'] = test_validation_merge['user_ave_price_day_ratio'].fillna(
        test_validation_merge['ave_price_day_ratio'])

    train_validation_merge_1['days_between_user_cat_orders'] = train_validation_merge_1[
        'days_between_user_cat_orders'].fillna(train_validation_merge_1['days_between_cat_orders'])
    train_validation_merge_2['days_between_user_cat_orders'] = train_validation_merge_2[
        'days_between_user_cat_orders'].fillna(train_validation_merge_2['days_between_cat_orders'])
    test_validation_merge['days_between_user_cat_orders'] = test_validation_merge['days_between_user_cat_orders'].fillna(
        test_validation_merge['days_between_cat_orders'])
    train_validation_merge_1.fillna(0, inplace=True)
    train_validation_merge_2.fillna(0, inplace=True)
    test_validation_merge.fillna(0, inplace=True)

    # concatenate prediction n-1 and n-2 dateFrames
    train_validation_merge = pd.concat([train_validation_merge_1, train_validation_merge_2], ignore_index=True)

    # new features
    train_validation_merge['since_prior_days_ratio'] = train_validation_merge['days_since_prior_order'] / \
                                                       train_validation_merge['days_between_orders']
    train_validation_merge['since_prior_days_cat_ratio'] = train_validation_merge['days_since_prior_order'] / \
                                                           train_validation_merge['days_between_cat_orders']
    train_validation_merge['since_prior_days_user_cat_ratio'] = train_validation_merge['days_since_prior_order'] / \
                                                                train_validation_merge['days_between_user_cat_orders']
    train_validation_merge['user_unique_cat_ratio'] = train_validation_merge['user_unique_cats'] / train_validation_merge[
        'user_total_cats']
    train_validation_merge['user_DPR_user_ratio'] = train_validation_merge['user_days_price_ratio_since_prior'] / \
                                                    train_validation_merge['user_ave_price_day_ratio']
    train_validation_merge['user_DPR_tot_ratio'] = train_validation_merge['user_days_price_ratio_since_prior'] / \
                                                   train_validation_merge['ave_price_day_ratio']

    test_validation_merge['since_prior_days_ratio'] = test_validation_merge['days_since_prior_order'] / \
                                                      test_validation_merge['days_between_orders']
    test_validation_merge['since_prior_days_cat_ratio'] = test_validation_merge['days_since_prior_order'] / \
                                                          test_validation_merge['days_between_cat_orders']
    test_validation_merge['since_prior_days_user_cat_ratio'] = test_validation_merge['days_since_prior_order'] / \
                                                               test_validation_merge['days_between_user_cat_orders']
    test_validation_merge['user_unique_cat_ratio'] = test_validation_merge['user_unique_cats'] / train_validation_merge[
        'user_total_cats']
    test_validation_merge['user_DPR_user_ratio'] = test_validation_merge['user_days_price_ratio_since_prior'] / \
                                                   train_validation_merge['user_ave_price_day_ratio']
    test_validation_merge['user_DPR_tot_ratio'] = test_validation_merge['user_days_price_ratio_since_prior'] / \
                                                  train_validation_merge['ave_price_day_ratio']

    # test_validation_merge.isin([np.inf, -np.inf]).sum()

    # print(train_validation_merge_1.shape)
    # print(train_validation_merge_2.shape)

    # Save Prepared Data
    train_validation_merge = train_validation_merge[train_validation_merge['days_since_prior_order']<=otli]
    test_validation_merge = test_validation_merge[test_validation_merge['days_since_prior_order']<=otli]

    n = 1
    if 'Bachelor Project' in os.listdir():
        os.chdir('Bachelor Project\\2-Cat_Base_total')
    for i in os.listdir():
        if 'train_validation_merge_v3' in i:
            n+=1

    train_validation_merge.to_csv(f'train_validation_merge_v3_{n}.csv')
    test_validation_merge.to_csv(f'test_validation_merge_v3_{n}.csv')

    # train_validation_merge = pd.read_csv('train_validation_merge_v2.csv',index_col=0)
    # test_validation_merge = pd.read_csv('test_validation_merge_v2.csv',index_col=0)
    duration = datetime.now() - start_time
    report = f"""Successfully Saved as train_validation_merge_v3_{n}\n
    the parameters were \n\tksigma:{ksigma}\n\totli:{otli}\n\tleast_order_count:{least_order_count}"\n
    running time: {duration}\n
    train_validation_merge:{len(train_validation_merge)}\n
    test_validation_merge:{len(test_validation_merge)}\n"""
    return report

reporter = []
for ksigma in (2,3):
    for otli in (45,60,75):
        for least_order_count in (4,5):
            reporter.append(prep_V_3(otli,ksigma,least_order_count))

report_path = f"preprocessing_report_V_3_no_otli_order_limit_change.txt"
text_file = open(report_path, "w")
# report = f"{ann.summary()}\n\n{metrics.classification_report(y_test, y_pred_ann)}\n\nmetric: {met_str}"
report = '\n\n\n'.join(reporter)
m = text_file.write(report)
text_file.close()

from notify import notifier

notifier.mail(report)



