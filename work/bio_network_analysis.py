import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx

from causalnex.structure import StructureModel
from causalnex.structure.pytorch import from_pandas
from causalnex.network import BayesianNetwork
from causalnex.evaluation import classification_report
from causalnex.evaluation import roc_auc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

import optuna

import warnings
warnings.simplefilter('ignore')

# データの読み込み・前処理
data = pd.read_excel('../data/231010-20_AI用qPCRデータ.xlsx', index_col=0, header=1)
data = data.dropna()
data = data.reset_index(drop=True)

# データの正規化
scaler = StandardScaler()
normalized_data_array = scaler.fit_transform(data)
# conversion array to Dataframe
normalized_df = pd.DataFrame(normalized_data_array, columns=data.columns)

# パラメータ調整
def objective(trial):
    # Optunaでチューニングするハイパーパラメータ
    max_iter = trial.suggest_int('max_iter', 100, 1000)
    threshold = trial.suggest_float('threshold', 0.0, 1.0)
    lasso_beta = trial.suggest_float('lasso_beta', 1e-4, 1e-1, log=True)  # ログスケールでlassoの値を探索
    ridge_beta = trial.suggest_float('ridge_beta', 1e-4, 1e-1, log=True)  # リッジ正則化の係数を探索
    use_bias = trial.suggest_categorical('use_bias', [True, False])

    # StructureModelのインスタンスを作成
    sm = StructureModel()

    # NOTEARSアルゴリズムを用いて構造学習を実施
    # ここでfrom_pandasのパラメータをOptunaのtrialを通してチューニング
    sm, loss_value = from_pandas(normalized_df, 
                                 max_iter=max_iter,
                                 w_threshold=threshold,
                                 lasso_beta=lasso_beta,
                                 ridge_beta=ridge_beta,
                                 use_bias=use_bias,
                                 )

    # 学習された構造のスコアを計算（スコアリング方法はプロジェクトにより異なる）
    score = calculate_score(sm)

    return score

# スコアリング関数（例：エッジ数でスコアリング）
def calculate_score(sm):
    return -len(sm.edges)  # エッジの数が少ないほどスコアが高くなるように設定

study = optuna.create_study(direction='maximize')  # スコアを最大化するように設定
study.optimize(objective, n_trials=100)  # 100回の試行で最適化

# 最適なハイパーパラメータを出力
print(study.best_params)

# cuda割り当て
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# 最適な閾値を取得
best_max_iter = study.best_params['max_iter']
best_threshold = study.best_params['threshold']
my_driven_hidden_layer_units = [2] # 中間層は2層
best_lasso_beta = study.best_params['lasso_beta']
best_ridge_beta = study.best_params['ridge_beta']
best_use_bias = study.best_params['use_bias']

# 最適な閾値で構造学習を実施
best_sm, loss_value = from_pandas(normalized_df,
                                  max_iter=best_max_iter,
                                  w_threshold=best_threshold,
                                  hidden_layer_units=my_driven_hidden_layer_units,
                                  lasso_beta=best_lasso_beta,
                                  ridge_beta=best_ridge_beta,
                                  use_bias=best_use_bias,
                                  )

best_sm.threshold_till_dag()

# 損失関数の出力
print("損失関数:", loss_value)
print("ノード:", best_sm.nodes)

# エッジが伸びていないノードを排除, 関係性の強いエッジを太くする
edge_width = [d["weight"]*1 for (u, v, d) in best_sm.edges(data=True)]
# 上記の処置を施したものを新たなグラフとして保存
sm_l = best_sm.get_largest_subgraph()
print("構造モデル:", sm_l)

# ネットワーク図を描画
fig, ax = plt.subplots(figsize=(16,16))
nx.draw_circular(sm_l,
                 with_labels=True,
                 font_size=20,
                 node_size=3000,
                 arrowsize=20,
                 alpha=0.5,
                 width=edge_width,
                 ax=ax)

plt.savefig("./output/best_network.png", format="png", dpi=300)

# Discretize the features
# Excluding the first column which seems to be an index or identifier
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
discretised_data = discretizer.fit_transform(normalized_df.iloc[:, :])

# Creating a new DataFrame for the discretized data
discretised_data = pd.DataFrame(discretised_data, columns=normalized_df.columns)

# Display the first few rows of the discretized data
discretised_data.head()

# 構造モデルのノードの確認
nodes = list(sm_l.nodes)
print('nodes:', nodes)

bn = BayesianNetwork(sm_l)
train, test = train_test_split(discretised_data, train_size=0.9, test_size=0.1)
bn = bn.fit_node_states(discretised_data)
bn = bn.fit_cpds(train)

for i in range(len(nodes)):
    print(nodes[i])
    print(bn.cpds[nodes[i]])

for n in bn.nodes:
    roc, auc = roc_auc(bn, test, n)
    print(n, auc)

for i in range(len(nodes)):
    node = nodes[i]
    cpd = bn.cpds[node]
    
    # CPDをDataFrameに変換（CausalNexのCPDオブジェクトがDataFrameに直接変換可能であると仮定）
    cpd_df = pd.DataFrame(cpd)

    # ヒートマップを描画
    plt.figure(figsize=(10, 8))
    sns.heatmap(cpd_df, annot=True, cmap='viridis')
    plt.title(f'CPD of Node: {node}')
    plt.ylabel('Parent states' if cpd_df.shape[0] > 1 else 'State')
    plt.xlabel('Node states')
    plt.show()

