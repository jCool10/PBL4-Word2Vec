from collections import Counter
from dataclasses import dataclass
import random
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.spatial import distance

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# instacart_path = "https://media.githubusercontent.com/media/tiepvupsu/tabml_data/master/instacart/"
instacart_path ="./data/"

order_df = pd.read_csv(instacart_path + "order_products__prior.csv")

def get_list_orders(order_df: pd.DataFrame) -> List[List[int]]:
    order_df = order_df.sort_values(by=["order_id", "add_to_cart_order"])
    return order_df.groupby("order_id")["product_id"].apply(list).tolist()

all_orders = get_list_orders(order_df)
orders = [order for order in all_orders if len(order) >= 2]

product_df = pd.read_csv(instacart_path + "products.csv", usecols=["product_id", "product_name"])

product_name_by_id = product_df.set_index("product_id").to_dict()["product_name"]

ordered_products = set([product for order in orders for product in order])

product_mapping = {
    "index_by_id": {product_id: ind for ind, product_id in enumerate(ordered_products)},
    "name_by_index": {ind: product_name_by_id[product_id] for ind, product_id in enumerate(ordered_products)}
}

indexed_orders = [
    [product_mapping["index_by_id"][product_id] for product_id in order]
    for order in orders
]

context_window = 5
all_targets = []
all_positive_contexts = []
for order in indexed_orders:
    for i, product in enumerate(order):
        all_targets.append(product)
        positive_context = [
            order[j]
            for j in range(max(0, i - context_window), min(len(order), i + context_window + 1))
            if j != i
        ]
        all_positive_contexts.append(positive_context)

sampling_weights = np.zeros(len(ordered_products))
product_freq = Counter([product for order in indexed_orders for product in order])
for product_index, count in product_freq.items():
    sampling_weights[product_index] = count ** 0.5

num_products = len(ordered_products)
product_sampler = random.choices(range(num_products), weights=sampling_weights, k=10_000_000)

class TargetContextDataset(tf.keras.utils.Sequence):
    def __init__(self, all_targets, all_positive_contexts, product_sampler, num_context_products=10, batch_size=8192, shuffle=True):
        self.all_targets = all_targets
        self.all_positive_contexts = all_positive_contexts
        self.product_sampler = product_sampler
        self.num_context_products = num_context_products
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.all_targets))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.all_targets) // self.batch_size

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_targets = np.array([self.all_targets[index] for index in indexes])
        batch_contexts = np.zeros((len(indexes), self.num_context_products), dtype=np.int32)
        batch_masks = np.zeros((len(indexes), self.num_context_products), dtype=np.float32)

        for i, index in enumerate(indexes):
            positive_contexts = self.all_positive_contexts[index]
            num_pos = len(positive_contexts)
            num_neg = self.num_context_products - num_pos
            mask = np.concatenate((np.ones(num_pos), np.zeros(num_neg)))

            while len(positive_contexts) < self.num_context_products:
                product = next(self.product_sampler)
                if product not in positive_contexts:
                    positive_contexts.append(product)

            batch_contexts[i] = positive_contexts
            batch_masks[i] = mask

        return batch_targets, batch_contexts, batch_masks

sampling_weights = np.array(sampling_weights)
product_sampler = iter(product_sampler)
training_data = TargetContextDataset(all_targets, all_positive_contexts, product_sampler, num_context_products=10)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=num_products, output_dim=50),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(training_data, epochs=50)

embs_arr = model.layers[0].get_weights()[0]

class NearestNeighbor:
    def __init__(self, embeddings, measure='cosine'):
        self.embeddings = embeddings
        self.measure = measure

    def find_nearest_neighbors(self, vector, k=2):
        if self.measure == 'cosine':
            distances = distance.cdist([vector], self.embeddings, 'cosine')[0]
        else:
            distances = distance.cdist([vector], self.embeddings, 'euclidean')[0]

        nearest_indices = np.argsort(distances)[:k]
        return nearest_indices

names = [product_mapping["name_by_index"][i] for i in range(num_products)]
emb_nn = NearestNeighbor(embs_arr, measure="cosine")

sub_name = "Water"
ids = [ind for ind, name in enumerate(names) if sub_name in name]

for ind in ids[:5]:
    print('==========')
    print(f'Similar items of "{names[ind]}":')
    nearest_ids = emb_nn.find_nearest_neighbors(embs_arr[ind], k=5)
    print([names[i] for i in nearest_ids])
