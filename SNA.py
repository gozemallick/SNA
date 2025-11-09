#!/usr/bin/env python
# coding: utf-8

# # **SNA PROJECT**
# ### **Title: Graph-Based Recommendation System for Tinder: Leveraging Social Interactions for Personalized Matches**
# 
# ### **This project aims to address this challenge by developing a recommendation system for Tinder that leverages graph representation learning techniques. By modeling the social network of users as a graph and learning embeddings for users based on their interactions and connections, the system aims to generate more personalized and relevant match suggestions**

# ### **TEAM MEMBERS**
# - ### **Jiya Thakur - 229309176**
# - ### **Harshita Batta- 229309044**
# - ### **Faaiz Hasib Mallick -229309195**
# - ### **Pavni Jain- 229309039**
# 

# ### **LIBRARIES**

# In[2]:


get_ipython().system('pip install --user node2vec')


# In[3]:


import ast
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt

import random
import numpy as np
from node2vec import Node2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


# ### **DATA COLLECTION**

# In[30]:


USERS = pd.read_csv("C:/Users/ASUS/Downloads/users.csv")
SWIPES = pd.read_csv("C:/Users/ASUS/Downloads/swipes.csv")


# In[31]:


USERS.head()


# In[6]:


SWIPES.head()


# In[7]:


USERS.info()


# In[8]:


SWIPES.info()


# ### **DATA PREPROCESSING**

# **USERS DATA**

# In[9]:


USERS.drop(columns=['height', 'is_subscribed', 'what_to_find', 'face_detection_probabilities', 'createdAt', 'dob', 'email',
                       'insta_username', 'is_habit_drink', 'is_habit_smoke', 'is_verified',
                       'mobile', 'status', 'type', 'updatedAt', 'verified_at', 'who_to_date'], inplace=True)


# In[10]:


USERS.shape


# In[11]:


USERS.dropna(inplace = True)
USERS.reset_index(inplace = True, drop = True)
USERS.shape


# In[12]:


USERS['bio'] = USERS['bio'].str.replace(r'[^\w\s]', '').str.strip()

interests = []

for i in range(len(USERS)):
    try:
        USERS.loc[i, "college"] = ast.literal_eval(USERS.loc[i, "college"])[0]
    except Exception as e:
        USERS.loc[i, "college"] = None

    interests.append(ast.literal_eval(USERS.loc[i, "interests"]))

USERS["interests"] = interests


# In[13]:


USERS = USERS.dropna().reset_index(drop = True)


# In[14]:


USERS.head()


# **SWIPES DATA**

# In[15]:


SWIPES.columns


# In[16]:


SWIPES.drop(columns=['id', 'p1_extend_at', 'p2_extend_at', 'first_like_unlike_at',
                     'first_msg', 'is_unmatch', 'second_like_unlike_at', 'second_msg', 'unmatch_on'], inplace=True)
SWIPES = SWIPES[SWIPES.like_count == 2].reset_index(drop = True)
SWIPES.drop(columns = ['like_count'], inplace = True)


# In[17]:


SWIPES.head()


# In[18]:


SWIPES.info()


# In[19]:


users_ids = set(USERS["_id"])
SWIPES["is_p1"] = SWIPES["p1"].apply(lambda x: True if x in users_ids else False)
SWIPES["is_p2"] = SWIPES["p2"].apply(lambda x: True if x in users_ids else False)


# In[20]:


SWIPES = SWIPES[(SWIPES["is_p1"]) & (SWIPES["is_p2"])].reset_index(drop = True)
SWIPES.drop(columns = ["is_p1", "is_p2"], inplace = True)
SWIPES


# ### **GRAPH CONSTRUCTION**

# In[21]:


G = nx.DiGraph()

for index, row in USERS.iterrows():
    user_id = row['_id']
    user_attributes = row.drop('_id').to_dict()
    G.add_node(user_id, **user_attributes)

for index, row in SWIPES.iterrows():
    p1 = row['p1']
    p2 = row['p2']
    G.add_edge(p1, p2, interaction_type=row['first_type'])
    G.add_edge(p2, p1, interaction_type=row['second_type'])


# In[22]:


nx.draw(G, with_labels = True)
plt.show()


# ### **GRAPH REPRESENTATION LEARNING**

# In[23]:


node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1)


# In[24]:


node_embeddings = {}
for node in G.nodes():
    try:
        node_embeddings[node] = model.wv[node]
    except Exception as e:
        continue


# In[25]:


users = list(node_embeddings.keys())


# In[26]:


def recommend_similar_users(user_id, k=5):
    user_embedding = node_embeddings[user_id]
    similarities = {}
    for other_user_id in users:
        if other_user_id != user_id:
            other_user_embedding = node_embeddings[other_user_id]
            similarity = cosine_similarity([user_embedding], [other_user_embedding])[0][0]
            similarities[other_user_id] = similarity

    top_similar_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
    return top_similar_users


# In[27]:


user_id = random.choice(users)
top_similar_users = recommend_similar_users(user_id)
print(f"Top 5 similar users for user {user_id}:")
for i in top_similar_users:
    print(f"{i}")


# In[28]:


USERS[USERS["_id"] == user_id].reset_index(drop = True)


# In[29]:


RECOMMENDATIONS = pd.DataFrame(columns = USERS.columns)

for i, s in top_similar_users:
    RECOMMENDATIONS = pd.concat([RECOMMENDATIONS, USERS[USERS["_id"] == i]])

RECOMMENDATIONS.reset_index(drop = True)


# In[ ]:





# In[ ]:





# In[ ]:





# # ============================================================
# # Social Network Analysis: Full Pipeline
# # - Loads users.csv, swipes.csv (use the same folder as notebook)
# # - Builds user graph (mutual-like edges)
# # - Centrality plots: Betweenness, Closeness (with Top-K tables)
# # - Similarity heatmap (Node2Vec cosine if available; else Jaccard)
# # - Community plot (greedy modularity on GCC / k-core optional)
# # - Supervised link prediction with confusion matrix + metrics
# # ============================================================

# In[5]:


import math, random, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------- Config / Paths --------------------------
DATA_DIR = Path("C:/Users/ASUS/Downloads")  # put users.csv and swipes.csv next to the notebook
USERS_PATH  = DATA_DIR / "users.csv"
SWIPES_PATH = DATA_DIR / "swipes.csv"

# -------------------------- Load CSVs --------------------------
USERS  = pd.read_csv(USERS_PATH)
SWIPES = pd.read_csv(SWIPES_PATH, low_memory=False)

# -------------------------- Filter mutual likes + valid IDs --------------------------
if "like_count" in SWIPES.columns:
    SWIPES = SWIPES[SWIPES["like_count"] == 2].reset_index(drop=True)
    SWIPES.drop(columns=["like_count"], inplace=True)

uid_set = set(USERS["_id"])
must_cols = {"p1", "p2"}
if not must_cols.issubset(set(SWIPES.columns)):
    raise ValueError("swipes.csv must have 'p1' and 'p2' columns.")

SWIPES = SWIPES[SWIPES["p1"].isin(uid_set) & SWIPES["p2"].isin(uid_set)].reset_index(drop=True)

# -------------------------- Build Directed Graph --------------------------
G = nx.DiGraph()
for _, row in USERS.iterrows():
    uid = row["_id"]
    attrs = row.drop(labels=["_id"]).to_dict()
    G.add_node(uid, **attrs)

for _, row in SWIPES.iterrows():
    p1, p2 = row["p1"], row["p2"]
    G.add_edge(p1, p2, interaction_type=row.get("first_type", None))
    G.add_edge(p2, p1, interaction_type=row.get("second_type", None))

UG = G.to_undirected()
print(f"Graph: |V|={UG.number_of_nodes()} |E|={UG.number_of_edges()}")

# -------------------------- CENTRALITY PLOTS --------------------------
def plot_hist(values, title, xlabel):
    plt.figure(figsize=(8,5))
    plt.hist(values, bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

num_nodes = UG.number_of_nodes()
k_sample = min(200, max(20, num_nodes // 5))  # approximate betweenness (faster)
try:
    betw = nx.betweenness_centrality(UG, k=k_sample, seed=42)
except Exception:
    betw = nx.betweenness_centrality(UG)

betw_s = pd.Series(betw).sort_values(ascending=False)
print("\nTop 20 by Betweenness:\n", betw_s.head(20))

# Closeness on Giant Component
if num_nodes > 0:
    components = sorted(nx.connected_components(UG), key=len, reverse=True)
    GCC_nodes = list(components[0])
    GCC = UG.subgraph(GCC_nodes).copy()
    close = nx.closeness_centrality(GCC)
    close_s = pd.Series(close).sort_values(ascending=False)
    print("\nTop 20 by Closeness (GCC):\n", close_s.head(20))
else:
    close_s = pd.Series(dtype=float)

plot_hist(list(betw_s.values), "Betweenness Centrality Distribution", "Betweenness")
if not close_s.empty:
    plot_hist(list(close_s.values), "Closeness Centrality Distribution (GCC)", "Closeness")

# -------------------------- SIMILARITY HEATMAP --------------------------
def try_node2vec_embeddings(graph):
    """Return dict node->vector if node2vec is available; else None."""
    try:
        from node2vec import Node2Vec
        node2vec = Node2Vec(graph, dimensions=32, walk_length=10, num_walks=20, workers=1)
        model = node2vec.fit(window=5, min_count=1)
        emb = {}
        for n in graph.nodes():
            k = str(n)
            if k in model.wv:
                emb[n] = model.wv[k]
        return emb
    except Exception as e:
        print(f"[info] Node2Vec not used ({e}). Falling back to Jaccard.")
        return None

def jaccard_sim(graph, u, v):
    Nu, Nv = set(graph.neighbors(u)), set(graph.neighbors(v))
    denom = len(Nu | Nv)
    return (len(Nu & Nv) / denom) if denom > 0 else 0.0

deg_s = pd.Series(dict(UG.degree())).sort_values(ascending=False)
small_set = list(deg_s.head(25).index) if len(deg_s) >= 25 else list(UG.nodes())[:min(25, len(UG))]

embeddings = try_node2vec_embeddings(UG)
if embeddings is not None and len(embeddings) >= len(small_set):
    sim_mat = np.zeros((len(small_set), len(small_set)))
    for i,u in enumerate(small_set):
        for j,v in enumerate(small_set):
            sim_mat[i,j] = 1.0 if i==j else cosine_similarity([embeddings[u]], [embeddings[v]])[0,0]
    title = "Similarity Heatmap (cosine on Node2Vec embeddings)"
else:
    sim_mat = np.zeros((len(small_set), len(small_set)))
    for i,u in enumerate(small_set):
        for j,v in enumerate(small_set):
            sim_mat[i,j] = 1.0 if i==j else jaccard_sim(UG, u, v)
    title = "Similarity Heatmap (Jaccard over neighbor sets)"

plt.figure(figsize=(8,7))
plt.imshow(sim_mat, interpolation="nearest", aspect="auto")
plt.title(title)
plt.xlabel("Users (top by degree)")
plt.ylabel("Users (top by degree)")
plt.tight_layout()
plt.show()

# -------------------------- COMMUNITY PLOT --------------------------
from networkx.algorithms.community import greedy_modularity_communities

# Work on GCC for clarity
if num_nodes > 0:
    comps = sorted(nx.connected_components(UG), key=len, reverse=True)
    H = UG.subgraph(list(comps[0])).copy()
    comms = list(greedy_modularity_communities(H))
    pos = nx.spring_layout(H, seed=42)

    plt.figure(figsize=(10,8))
    for c in comms:  # auto color cycle
        nx.draw_networkx_nodes(H, pos, nodelist=list(c), node_size=20)
    nx.draw_networkx_edges(H, pos, width=0.3)
    plt.title(f"Community Plot (Greedy Modularity) - {len(comms)} communities")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# -------------------------- SUPERVISED LINK PREDICTION --------------------------
# Robust split (temporal if possible; otherwise random)
def do_temporal_split(df, time_cols=("first_like_unlike_at","second_like_unlike_at","createdAt","updatedAt"),
                      train_frac=0.8, min_train=100):
    for tc in time_cols:
        if tc in df.columns:
            try:
                times = pd.to_numeric(df[tc], errors="coerce")
                tmp = df.assign(_t=times).dropna(subset=["_t"]).sort_values("_t").reset_index(drop=True)
                if len(tmp) >= (min_train + 10):
                    split_idx = int(train_frac * len(tmp))
                    tr = tmp.iloc[:split_idx][["p1","p2"]]
                    te = tmp.iloc[split_idx:][["p1","p2"]]
                    if len(tr) >= min_train and len(te) >= 10:
                        return tr, te
            except Exception:
                pass
    return None, None

train_edges_df, test_edges_df = do_temporal_split(SWIPES)
if (train_edges_df is None) or (test_edges_df is None):
    train_edges_df, test_edges_df = train_test_split(SWIPES[["p1","p2"]], test_size=0.2, random_state=42, shuffle=True)

train_edges = list(map(tuple, train_edges_df.values))
test_pos    = list(map(tuple, test_edges_df.values))

# Train graph
H_train = nx.Graph()
H_train.add_nodes_from(UG.nodes())
H_train.add_edges_from(train_edges)

# Negative sampling
def sample_non_edges(G, n, rng=random.Random(42), max_trials_factor=50):
    res, tried = [], 0
    nodes = list(G.nodes())
    existing = set(map(tuple, map(sorted, G.edges())))
    max_trials = max(1, n * max_trials_factor)
    while len(res) < n and tried < max_trials:
        u, v = rng.sample(nodes, 2)
        key = tuple(sorted((u, v)))
        if key not in existing:
            res.append((u, v))
        tried += 1
    return res

train_neg = sample_non_edges(H_train, len(train_edges))
test_neg  = sample_non_edges(UG,      len(test_pos))

# Safe small validation split
def safe_split(lst, test_frac=0.2, min_len=10):
    if len(lst) < min_len:
        return lst, []
    test_size = max(1, int(test_frac * len(lst)))
    tr, va = train_test_split(lst, test_size=test_size, random_state=42)
    return tr, va

tr_pos, val_pos = safe_split(train_edges, test_frac=0.2, min_len=10)
tr_neg, val_neg = safe_split(train_neg,   test_frac=0.2, min_len=10)

print(f"\nSplit sizes -> train_pos={len(tr_pos)}, val_pos={len(val_pos)}, test_pos={len(test_pos)}")
print(f"               train_neg={len(tr_neg)}, val_neg={len(val_neg)}, test_neg={len(test_neg)}")

# Node2Vec embeddings on train graph (optional)
def fit_node2vec(graph, dimensions=64, walk_length=20, num_walks=100, p=1.0, q=1.0, workers=2, window=10):
    try:
        from node2vec import Node2Vec
        n2v = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length,
                       num_walks=num_walks, p=p, q=q, workers=workers)
        model = n2v.fit(window=window, min_count=1)
        emb = {}
        for n in graph.nodes():
            k = str(n)
            if k in model.wv:
                emb[n] = model.wv[k]
        return emb
    except Exception as e:
        print(f"[info] Node2Vec skipped ({e}). Using topology-only features.")
        return None

grid = [
    dict(dimensions=64, walk_length=20, num_walks=80,  p=1.0, q=1.0),
    dict(dimensions=64, walk_length=20, num_walks=80,  p=0.5, q=2.0),
    dict(dimensions=64, walk_length=20, num_walks=80,  p=2.0, q=0.5),
    dict(dimensions=128,walk_length=30, num_walks=100, p=1.0, q=1.0),
]

def topo_feats(G, u, v):
    Nu, Nv = set(G.neighbors(u)), set(G.neighbors(v))
    cn = len(Nu & Nv)
    j  = cn/len(Nu | Nv) if len(Nu | Nv)>0 else 0.0
    aa = 0.0
    for w in (Nu & Nv):
        deg = G.degree(w)
        if deg > 1:
            aa += 1.0/math.log(deg)
    ra = 0.0
    for w in (Nu & Nv):
        deg = G.degree(w)
        if deg > 0:
            ra += 1.0/deg
    pa = G.degree(u) * G.degree(v)
    return [cn, j, aa, ra, pa]

def build_features(emb, G, pairs):
    X = []
    for u, v in pairs:
        # embedding features
        if emb is not None and (u in emb) and (v in emb):
            eu, ev = np.asarray(emb[u]), np.asarray(emb[v])
            cos = float(cosine_similarity([eu],[ev])[0,0])
            had = eu*ev
            l1  = np.abs(eu-ev)
            l2  = float(np.linalg.norm(eu-ev))
            avg = 0.5*(eu+ev)
            emb_feats = [cos, l2] + list(had[:10]) + list(l1[:10]) + list(avg[:10])
        else:
            emb_feats = [0.0, 0.0] + [0.0]*30
        X.append(emb_feats + topo_feats(G, u, v))
    return np.array(X, dtype=float)

# Choose Node2Vec config (grid if we have a validation set; else default)
if len(val_pos) >= 10 and len(val_neg) >= 10:
    best_cfg, best_ap = None, -1.0
    for cfg in grid:
        emb = fit_node2vec(H_train, **cfg)
        X_val = build_features(emb, H_train, val_pos + val_neg)
        y_val = np.array([1]*len(val_pos) + [0]*len(val_neg))
        scores = X_val[:,0]  # cosine feature if embeddings present; 0 otherwise
        ap = average_precision_score(y_val, scores)
        if ap > best_ap:
            best_ap, best_cfg = ap, cfg
    print("Best Node2Vec config (AP):", best_cfg, "AP=", round(best_ap, 4))
else:
    best_cfg = dict(dimensions=64, walk_length=20, num_walks=80, p=1.0, q=1.0)
    print("Validation too small; using default Node2Vec config:", best_cfg)

emb_train = fit_node2vec(H_train, **best_cfg)

# Train classifier
X_tr = build_features(emb_train, H_train, tr_pos + tr_neg)
y_tr = np.array([1]*len(tr_pos) + [0]*len(tr_neg))

clf = LogisticRegression(max_iter=300, class_weight="balanced")
clf.fit(X_tr, y_tr)

# Test set + metrics
X_te = build_features(emb_train, H_train, test_pos + test_neg)
y_te = np.array([1]*len(test_pos) + [0]*len(test_neg))
te_scores = clf.predict_proba(X_te)[:,1]

roc  = roc_auc_score(y_te, te_scores)
ap   = average_precision_score(y_te, te_scores)
K    = int((y_te == 1).sum())
order = np.argsort(-te_scores)
preds = np.zeros_like(y_te)
preds[order[:K]] = 1

cm = confusion_matrix(y_te, preds, labels=[0,1])
acc = accuracy_score(y_te, preds)
prec, rec, f1, _ = precision_recall_fscore_support(y_te, preds, average="binary", zero_division=0)

print("\nTEST METRICS (supervised link prediction)")
print("ROC-AUC:", round(roc,4), "  PR-AUC:", round(ap,4))
print("Accuracy:", round(acc,4), "  Precision:", round(prec,4), "  Recall:", round(rec,4), "  F1:", round(f1,4))
print("Confusion Matrix [[TN FP],[FN TP]]:\n", cm)

plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix (Top-K threshold)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# In[ ]:




