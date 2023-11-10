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