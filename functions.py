# THIS FILE CONTAINS ONLY FUNCTION DEFINITIONS
import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np
from collections import Counter, defaultdict
import pandas as pd
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import random


def community_similarity_vs_pathlength(
    nodes,
    G,
    features,
    labels,
    samples=500,
    title="Community similarity vs average shortest path"
):


    groups = defaultdict(list)
    for n in nodes:
        groups[labels[n]].append(n)

    comm_ids = sorted(groups.keys())

    avg_sims = []
    avg_paths = []
    sizes = []   # <-- community sizes

    # -----------------------------
    # compute metrics per community
    # -----------------------------
    for c in comm_ids:
        members = groups[c]
        sizes.append(len(members))

        if len(members) < 2:
            avg_sims.append(np.nan)
            avg_paths.append(np.nan)
            continue

        # cosine similarity (sampling)
        U = random.choices(members, k=samples)
        V = random.choices(members, k=samples)

        X = np.vstack([features[u] for u in U])
        Y = np.vstack([features[v] for v in V])

        sims = cosine_similarity(X, Y).diagonal()
        avg_sims.append(sims.mean())

        # shortest path (largest connected component)
        subG = G.subgraph(members)
        if nx.is_connected(subG):
            avg_paths.append(nx.average_shortest_path_length(subG))
        else:
            lcc = max(nx.connected_components(subG), key=len)
            avg_paths.append(
                nx.average_shortest_path_length(subG.subgraph(lcc))
            )

    avg_sims = np.array(avg_sims)
    avg_paths = np.array(avg_paths)
    sizes = np.array(sizes)

    plt.figure(figsize=(6, 5))

    # log-scaled sizes for visibility
    size_scale = 20
    log_sizes = np.log(sizes)

    plt.scatter(
        avg_paths,
        avg_sims,
        s=size_scale * log_sizes,
        alpha=0.7
    )

    plt.xlabel("Average shortest path length")
    plt.ylabel("Average cosine similarity")
    plt.title(title)
    plt.grid(True)
    plt.show()




def sampled_group_cosine_matrix(
    nodes,
    features,
    labels,
    samples_per_pair=300,
    title="Cosine similarity"
):

    # -----------------------------
    # group nodes by label
    # -----------------------------
    groups = defaultdict(list)
    for n in nodes:
        groups[labels[n]].append(n)

    group_ids = sorted(groups.keys())
    idx = {g: i for i, g in enumerate(group_ids)}
    n = len(group_ids)

    # -----------------------------
    # compute matrix
    # -----------------------------
    M = np.zeros((n, n))

    for g1 in group_ids:
        for g2 in group_ids:
            U = random.choices(groups[g1], k=samples_per_pair)
            V = random.choices(groups[g2], k=samples_per_pair)

            X = np.vstack([features[u] for u in U])
            Y = np.vstack([features[v] for v in V])

            sims = cosine_similarity(X, Y).diagonal()
            M[idx[g1], idx[g2]] = sims.mean()

    # -----------------------------
    # plot (lower triangle only)
    # -----------------------------
    plt.figure(figsize=(5, 5))
    plt.axis("off")

    for i in range(n):
        for j in range(i + 1):  # lower triangle
            val = M[i, j]
            txt = f"{val:.2f}".replace("0.", ".")
            plt.text(j, i, txt, ha="center", va="center")

    plt.xticks(range(n), group_ids)
    plt.yticks(range(n), group_ids)
    plt.title(title)
    plt.show()

    return M, group_ids


def distance_k_cosine_similarity(G, features, max_dist=3, n_samples=10000):
    """
    Compute cosine similarity for node pairs at graph distance k = 1..max_dist,
    plus random node pairs.

    """


    nodes = list(G.nodes())
    sims = {"random": []}

    # -----------------------------
    # RANDOM pairs
    # -----------------------------
    while len(sims["random"]) < n_samples:
        u, v = random.sample(nodes, 2)
        if u in features and v in features:
            sim = cosine_similarity(
                features[u].reshape(1, -1),
                features[v].reshape(1, -1)
            )[0, 0]
            sims["random"].append(sim)

    sims["random"] = np.array(sims["random"])

    # -----------------------------
    # DISTANCE-k pairs
    # -----------------------------
    for k in range(1, max_dist + 1):
        sims[k] = []

        while len(sims[k]) < n_samples:
            u = random.choice(nodes)
            if u not in features:
                continue

            # BFS layers from u
            lengths = nx.single_source_shortest_path_length(G, u, cutoff=k)
            candidates = [v for v, d in lengths.items() if d == k and v in features]

            if not candidates:
                continue

            v = random.choice(candidates)

            sim = cosine_similarity(
                features[u].reshape(1, -1),
                features[v].reshape(1, -1)
            )[0, 0]

            sims[k].append(sim)

        sims[k] = np.array(sims[k])

    return sims



def plot_num_artists_vs_avg_similarity(
    G,
    features,
    random_samples=20,
    seed=42
):
    random.seed(seed)

    neigh_sims_by_k = defaultdict(list)
    rand_sims_by_k = defaultdict(list)

    all_nodes = list(features.keys())

    for u in G.nodes():
        if u not in features:
            continue

        neighbors = [v for v in G.neighbors(u) if v in features]
        if len(neighbors) == 0:
            continue

        k_artists = int(np.sum(features[u]))

        # --- neighbor similarity ---
        neigh_sims = []
        for v in neighbors:
            sim = cosine_similarity(
                features[u].reshape(1, -1),
                features[v].reshape(1, -1)
            )[0, 0]
            neigh_sims.append(sim)

        neigh_sims_by_k[k_artists].append(np.mean(neigh_sims))

        # --- random similarity ---
        rand_nodes = random.sample(all_nodes, min(random_samples, len(all_nodes)))
        rand_sims = []

        for v in rand_nodes:
            if v == u:
                continue
            sim = cosine_similarity(
                features[u].reshape(1, -1),
                features[v].reshape(1, -1)
            )[0, 0]
            rand_sims.append(sim)

        if len(rand_sims) > 0:
            rand_sims_by_k[k_artists].append(np.mean(rand_sims))

    # aggregate
    x = sorted(neigh_sims_by_k.keys())
    y_neigh = [np.mean(neigh_sims_by_k[k]) for k in x]
    y_rand  = [np.mean(rand_sims_by_k[k])  for k in x]

    # plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y_neigh, linewidth=1, label="Neighbors")
    plt.plot(x, y_rand,  linewidth=1, linestyle="--", label="Random nodes")

    plt.xlabel("Number of liked artists")
    plt.ylabel("Average cosine similarity")
    plt.title("Music taste similarity as a function of the number of liked artists")
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_degree_distribution_loglog(G):

    degrees = [d for _, d in G.degree()]
    degree_counts = Counter(degrees)
    
    # Remove zero-degree nodes (log scale issue)
    if 0 in degree_counts:
        del degree_counts[0]
    
    # Sort by degree
    k = sorted(degree_counts.keys())
    pk = [degree_counts[i] / G.number_of_nodes() for i in k]
    
    # Plot
    plt.figure()
    plt.loglog(k, pk, marker='o', linestyle='None')
    plt.xlabel("Degree k")
    plt.ylabel("P(k)")
    plt.show()


def community_country_percentages(nodes, labels_louvain, country):
    """
    nodes: iterable of node IDs
    labels_louvain: dict {node -> community id}
    country: dict {node -> country code}

    Returns:
    {community_id: {country_code: percentage}}
    """

    # 🔑 ALL country codes present in the graph
    all_countries = sorted(set(country.values()))

    community_nodes = defaultdict(list)

    # group nodes by community
    for u in nodes:
        if u in labels_louvain and u in country:
            community_nodes[labels_louvain[u]].append(u)

    result = {}

    for comm, comm_nodes in community_nodes.items():
        counts = Counter(country[u] for u in comm_nodes)
        total = sum(counts.values())

        # include ALL countries, even if 0%
        result[comm] = {
            c: 100 * counts.get(c, 0) / total
            for c in all_countries
        }

    return result




def plot_community_country_distribution(country_dist, title=None):
    """
    country_dist: dict {community_id: {country_code: percentage}}
    """

    # infer community ids
    comm_ids = sorted(country_dist.keys())
    comm_labels = dict(zip(comm_ids, ["C1", "C2", "C3", "C4", "C5"][:len(comm_ids)]))

    # build DataFrame: rows = countries, columns = communities
    df = pd.DataFrame(country_dist).fillna(0)

    # reorder and relabel communities
    df = df[comm_ids]
    df = df.rename(columns=comm_labels)

    # infer ALL country codes present in the graph
    all_countries = sorted(df.index)

    # ensure all countries appear (explicit reindex for safety)
    df = df.reindex(all_countries, fill_value=0)

    # plot: each bar = community, stacked by country
    ax = df.T.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6),
        colormap="tab20"
    )

    ax.set_xlabel("Community")
    ax.set_ylabel("Percentage of nodes")
    ax.set_ylim(0, 100)

    ax.set_title(
        title if title else
        "Geographic composition of Louvain communities"
    )

    ax.legend(
        title="Country",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8
    )

    plt.tight_layout()
    plt.show()


def shuffle_country_labels(country):
    """
    Randomly permutes country labels across nodes
    while preserving the global country distribution.

    country: dict {node_id: country_code}
    """


    nodes = list(country.keys())
    labels = list(country.values())

    random.shuffle(labels)

    return dict(zip(nodes, labels))

