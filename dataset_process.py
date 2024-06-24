import os
import pandas as pd
import numpy as np

np.random.seed(0)
from recq.tools.io import mkdir
from recq.tools.plot import degree_plot

DATASET = "movielens_20m"
# DATASET = "amazonbook"


DATA_DIR = os.path.join("data_temp", DATASET)
TEMP_DIR = os.path.join(DATA_DIR, "temp")
FIG_DIR = os.path.join("output", "figures", "degree_distribution", DATASET)
mkdir(DATA_DIR)
mkdir(TEMP_DIR)
mkdir(FIG_DIR)

if DATASET == "movielens_20m":
    df = pd.read_csv("/data/ml-20m/ratings.csv",
                     usecols=["userId", "movieId", "rating"])
    # 更改列名
    new_column_names = {"userId": "user", "movieId": "item", "rating": "rating"}
    df = df.rename(columns=new_column_names)
    # df = df[df["rating"] == 5][["user", "item"]]  # 只保留评分为5的、user和item列
    df = df[df["rating"].isin([4.5, 5])][["user", "item"]]

    degree_plot({"implicit": df}, FIG_DIR)
    df.to_csv(os.path.join(TEMP_DIR, "implicit.csv"), index=False)
    print('读取数据完毕')
elif DATASET == "amazonbook":
    # item,user,rating,timestamp
    df = pd.read_csv("/data/amazon-book/Books.csv",
                     header=None, names=["item", "user", "rating"], usecols=[0, 1, 2])
    print("amazonbook读取完毕")
    df = df[df["rating"] == 5][["user", "item"]]  # 只保留评分为5的、user和item列
    # 获取数据集中唯一用户列表
    unique_users = df["user"].unique()
    # 从唯一用户列表中随机选择用户
    selected_users = pd.Series(unique_users).sample(n=1500000, random_state=0)
    df = df[df["user"].isin(selected_users)]

    degree_plot({"implicit": df}, FIG_DIR)
    df.to_csv(os.path.join(TEMP_DIR, "implicit.csv"), index=False)
    print('读取数据完毕')
elif DATASET == "netflix":
    data = {}
    for filename in os.listdir("/Users/qi/Documents/data/netflix/training_set"):
        with open(
            os.path.join("/Users/qi/Documents/data/netflix/training_set", filename)
        ) as f:
            i = f.readline().strip(":\n")
            for l in f:
                splits = l.split(",")
                if int(splits[1]) == 5:
                    u = splits[0]
                    if u not in data:
                        data[u] = [i]
                    else:
                        data[u].append(i)
    # Reduce dataset.
    uni_users = list(data.keys())
    uni_users = np.random.choice(uni_users, 50000, replace=False)
    data = {key: data[key] for key in uni_users}
    users = []
    items = []
    for u, i_list in data.items():
        users.extend([u] * len(i_list))
        items.extend(i_list)
    df = pd.DataFrame({"user": users, "item": items})
    df.to_csv(os.path.join(TEMP_DIR, "implicit.csv"), index=False)
elif DATASET == "ifashion":
    # user_data的格式：
    # user_id,item_id;item_id;...,outfit_id
    data = {}
    with open("/Users/qi/Documents/data/ifashion/user_data.txt") as f:
        for l in f:
            u, _, i = l.strip("\n").split(",")
            if u not in data:
                data[u] = [i]
            else:
                data[u].append(i)
    # 整理成这样：{u1:i1,i2,i3,... ; u2:i1,i3,i4,...}
    # Reduce dataset.  数据集太大了还要减少呢
    uni_users = list(data.keys())
    uni_users = np.random.choice(uni_users, 300000, replace=False)  # 随机选择300000个用户进行后续处理

    data = {key: data[key] for key in uni_users}

    users = []
    items = []
    for u, i_list in data.items():
        users.extend([u] * len(i_list))
        items.extend(i_list)

    df = pd.DataFrame({"user": users, "item": items})
    df.to_csv(os.path.join(TEMP_DIR, "implicit.csv"), index=False)
else:
    raise ValueError("Unsupported dataset.")


def get_df_info(df: pd.DataFrame):  # 取得数据集信息
    n_user = len(df["user"].unique())  # 用户个数
    n_item = len(df["item"].unique())  # 物品个数
    n_interact = len(df)  # 交互次数
    density = n_interact / n_user / n_item  # 交互的密度，越接近1，表示交互越密集。
    print(
        f"#User: {n_user}, #Item: {n_item}, #Interaction: {n_interact}, Density: {density:.5f}"
    )
    return n_user, n_item, n_interact, density


def filter_degree(df: pd.DataFrame, min_u_d, min_i_d):
    # 过滤交互次数过少的用户和项目
    # 交互次数小于min_u_d的用户被过滤
    # 交互次数小于min_i_d的项目被过滤
    print("Before being filtered by degree:")
    get_df_info(df)
    while 1:
        df = df.groupby("user").filter(lambda x: len(x) >= min_u_d)  # groupby可以将同一用户的纪录聚集成一组
        df = df.groupby("item").filter(lambda x: len(x) >= min_i_d)
        u_degree = df["user"].value_counts()
        i_degree = df["item"].value_counts()
        if u_degree.min() == min_u_d and i_degree.min() == min_i_d:
            break
    print("After being filtered by degree:")
    get_df_info(df)
    return df


# 开始统一处理

df = pd.read_csv(os.path.join(TEMP_DIR, "implicit.csv"))
df = filter_degree(df, 5, 5)

degree_plot({"full": df}, FIG_DIR)  # 绘制过滤后数据的度分布图，并将图保存到指定目录 FIG_DIR
df.to_csv(os.path.join(TEMP_DIR, "full.csv"), index=False)  # 过滤后的数据存到full.csv

df = pd.read_csv(os.path.join(TEMP_DIR, "full.csv"))
if DATASET == "movielens_20m":  # max_ips是最大采样概率
    max_ips = 1 / 60
elif DATASET == "amazonbook":  # 数据集密度越低，最大采样率应该越高
    max_ips = 1 / 12
elif DATASET == "netflix":
    max_ips = 1 / 60
elif DATASET == "ifashion":
    max_ips = 1 / 12
# 为每条记录添加一列 ips，表示该记录的采样权重。
# 采样权重取min(1 / len(x), max_ips)，也就是说，物品出现的次数越多，采样ips越低(这里用了最大采样率)
# 这个属于动态采样的一部分吗？
df["ips"] = df.groupby(["item"]).transform(lambda x: min(1 / len(x), max_ips))
# 得到一个名为 unbias 的 DataFrame，包含从原始数据中根据 IPS 采样得到的 30% 的记录，ips最大的前30%（理由？）
unbias = df.sample(frac=0.3, weights="ips", random_state=0)[["user", "item"]]
# train包含未被采样到 unbias 数据集的剩余 70% 的记录。
train = df.drop(unbias.index)[["user", "item"]]

# Move users and items that do not appear in train back to train.
# 把unbias中出现而train中没出现的数据，从unbias中删掉
users = train["user"].unique()
items = train["item"].unique()
unbias = unbias[unbias["user"].isin(users) & unbias["item"].isin(items)]
train = df.drop(unbias.index)[["user", "item"]]
print('训练集长度：', len(train))
print('unbias长度(包括test和valid)', len(unbias))

# Reindex.
# 创建原始用户和物品 ID 到新的整数索引的映射字典。然后更新数据集到新的整数索引
users = train["user"].unique()
items = train["item"].unique()
id2idx_u = dict(zip(users, range(len(users))))
id2idx_i = dict(zip(items, range(len(items))))
for idx in range(len(unbias)):
    unbias.values[idx] = (
        id2idx_u[unbias.values[idx][0]],
        id2idx_i[unbias.values[idx][1]],
    )
for idx in range(len(train)):
    train.values[idx] = id2idx_u[train.values[idx][0]], id2idx_i[train.values[idx][1]]

# 测试集包含2/3的unbias记录，验证集包含1/3的unbias
test = unbias.sample(frac=2 / 3, random_state=0)
valid = unbias.drop(test.index)

print('训练集长度：', len(train))
print('验证集长度：', len(valid))
print('测试集长度：', len(test))
degree_plot({"train": train, "valid": valid, "test": test}, FIG_DIR)  # 到这里验证集什么的划分完了

train.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
valid.to_csv(os.path.join(DATA_DIR, "valid.csv"), index=False)
test.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

df = pd.DataFrame({"id": list(id2idx_u.keys()), "idx": list(id2idx_u.values())})
df.to_csv(os.path.join(DATA_DIR, "id2idx_u.csv"), index=False)

df = pd.DataFrame({"id": list(id2idx_i.keys()), "idx": list(id2idx_i.values())})
df.to_csv(os.path.join(DATA_DIR, "id2idx_i.csv"), index=False)  # 这里把id到整数索引的映射划分完了


# Create train sets with different degrees of bias.使用不同度量方式的偏差创建训练集
def subsample(frac, pow):
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    # 根据物品出现次数len(x)的pow次方，进行抽样，这个值越大，越容易被抽样(为什么要这样做？)。frac是抽样率
    train[str(pow)] = train.groupby(["item"]).transform(lambda x: len(x) ** pow)
    train_skew = train.sample(frac=frac, weights=str(pow), random_state=0)[
        ["user", "item"]
    ]
    # 这里在填补数据
    users = train_skew["user"].unique()
    items = train_skew["item"].unique()
    train_missing = train[~(train["user"].isin(users) & train["item"].isin(items))]
    train_skew = pd.concat([train_skew, train_missing])
    # 画图，和保存train数据集
    degree_plot({"train_" + str(pow): train_skew}, FIG_DIR)
    train_skew.to_csv(os.path.join(DATA_DIR, "train_" + str(pow) + ".csv"), index=False)


subsample(0.7, 0.5)
subsample(0.7, -0.5)
