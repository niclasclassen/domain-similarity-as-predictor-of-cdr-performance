# Experimental Setting

**Dataset:** ml-100k

**Processing:** 

Create an intervened set, such that each item has a uniform observed probability:
- sampling a test set with 50% of the original dataset, and the remaining part are normal set.
- resampling data from the test set based on the inverse of the relative item probabilities

Split the dataset (consisting of normal set and intervened set) as follows:
- training set (100% normal and 25% intervened)
- validation set (25% intervened)
- test set (50% intervened)

**Filtering:** Remove users and items with less than 1 interactions

**Evaluation:** Ratio-based Splitting, full sort

**Metrics:** Recall@10, MRR@10, NGCG@10, Hit@10, Precision@10

**Properties:**

```yaml
# Atomic File Format
field_separator: "\t"
seq_separator: " "

# Common Features
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
TIME_FIELD: timestamp
seq_len: ~
# Label for Point-wise DataLoader
LABEL_FIELD: label
# NegSample Prefix for Pair-wise DataLoader
NEG_PREFIX: neg_
# Sequential Model Needed
ITEM_LIST_LENGTH_FIELD: item_length
LIST_SUFFIX: _list
MAX_ITEM_LIST_LENGTH: 50
POSITION_FIELD: position_id
# Knowledge-based Model Needed
HEAD_ENTITY_ID_FIELD: head_id
TAIL_ENTITY_ID_FIELD: tail_id
RELATION_ID_FIELD: relation_id
ENTITY_ID_FIELD: entity_id

# Selectively Loading
load_col:
    inter: [user_id, item_id, rating, timestamp, intervene_mask]
unload_col: ~
unused_col: ~

# Filtering
rm_dup_inter: ~
val_interval: ~
filter_inter_by_user_or_item: True
user_inter_num_interval: '[1,inf)'
item_inter_num_interval: '[1,inf)'

# Benchmark file
benchmark_filename: ['train','valid','test']

# special
INTERVENE_MASK: intervene_mask
```

# Dataset Statistics

| Dataset    | #Users | #Items | #Interactions | Sparsity |
| ---------- | ------ | ------ | ------------- | -------- |
| ml-100k    |    943 | 1,682  |   74,817      | 95.29%   |

# Hyper-parameters
We tune the hyper-parameters of the implemented models and present the range for reference.

For fairness, we tune the common hyper-parameters of methods as following. 
```
learning_rate in [0.01, 0.005, 0.001, 0.0001]
embedding_size in [16, 32, 64]
train_batch_size in [256, 512, 1024, 2048]
```

|                      | Best hyper-parameters                                        | Tuning range                                                 |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **MF**               | learning_rate=0.005<br />embedding_size=16<br />train_batch_size=2048 |- |
| **MF-IPS (User)**    | learning_rate=0.005<br />embedding_size=16<br />train_batch_size=256<br />eta=0.5 | eta in [0.1, 0.2, ..., 1] |
| **MF-IPS (Item)**    | learning_rate=0.001<br />embedding_size=32<br />train_batch_size=1024 |-  |
| **MF-IPS (NB)**      | learning_rate=0.001<br />embedding_size=64<br />train_batch_size=256 |-  |
| **PDA**              | learning_rate=0.005<br />embedding_size=64<br />train_batch_size=2048<br />reg_weight=0.1 | reg_weight in [0.1, 0.01, 0.001] |
| **MACR**             | learning_rate=0.001<br />embedding_size=64<br />train_batch_size=256<br />item_loss_weight=0.05<br />user_loss_weight=0.01<br />mlp_hidden_size=[32, 16, 1]<br />dropout_prob=0<br />c=5 | item_loss_weight in [0.001, 0.005, 0.01, 0.05, 0.1]<br />user_loss_weight in [0.001, 0.005, 0.01, 0.05, 0.1]<br />mlp_hidden_size in [[32, 1], [32, 16, 1]]<br />dropout_prob in [0.1, 0]<br />c in [5, 1, 0] |
| **DICE**             | learning_rate=0.005<br />embedding_size=16<br />train_batch_size=2048<br />dis_pen=0.005<br />int_weight=0.5<br />pop_weight=0.0001 | dis_pen in [0.1, 0.01, 0.005, 0.001]<br />int_weight in [0.1, 0.5, 0.01, 0.05, 0.0001]<br />pop_weight in [0.1, 0.5, 0.01, 0.05, 0.0001]|
| **CausE**            | learning_rate=0.001<br />embedding_size=64<br />train_batch_size=256<br />dis_pen=1 | dis_pen in [1, 0.1, 0.01, 0.001] |
| **Rel-MF**           | learning_rate=0.001<br />embedding_size=32<br />train_batch_size=2048<br />reg_weight=0.001<br />eta=0.0 | reg_weight in [0.1, 0.01, 0.001]<br />eta in [0.0, 0.1, 0.2, ..., 1] |
