# -*- coding: utf-8 -*-
"""
Kaggle House Prices: Advanced Regression Techniques
Solution by Jules (Kaggle Grandmaster)

Core Strategy:
1. 极致的数据清洗与异常值处理
2. 深度特征工程 (Box-Cox, TotalSF, Categorical encoding)
3. 多样化模型库 (Linear, Tree-based, Boosting)
4. Stacking与Blending集成策略
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from mlxtend.regressor import StackingCVRegressor

import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 1. 数据加载与预处理 (Data Loading & Preprocessing)
# ------------------------------------------------------------------------------
import os

# 获取当前脚本所在目录
base_path = os.path.dirname(os.path.abspath(__file__))

print("正在加载数据...")
train = pd.read_csv(os.path.join(base_path, 'train.csv'))
test = pd.read_csv(os.path.join(base_path, 'test.csv'))

# 保存ID用于提交
train_ID = train['Id']
test_ID = test['Id']
train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)

# ------------------------------------------------------------------------------
# 1.1 异常值处理 (Outlier Removal)
# ------------------------------------------------------------------------------
# 根据Kaggle讨论区经验，GrLivArea > 4000 且 SalePrice < 300000 的通常是异常值
# 注意：仅在训练集中删除
print("处理异常值...")
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

# ------------------------------------------------------------------------------
# 1.2 目标变量平滑 (Target Variable Smoothing)
# ------------------------------------------------------------------------------
# 使用 log1p 使数据更符合正态分布，适用于 RMSLE 评估指标
train["SalePrice"] = np.log1p(train["SalePrice"])
y_train = train.SalePrice.values

# 合并数据集以便统一进行特征工程
ntrain = train.shape[0]
ntest = test.shape[0]
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print(f"合并后的数据维度: {all_data.shape}")

# ------------------------------------------------------------------------------
# 1.3 缺失值填充 (Missing Value Imputation)
# ------------------------------------------------------------------------------
print("正在填充缺失值...")

# LotFrontage: 房子的临街宽度通常与邻居相似，按 Neighborhood 分组取中位数
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

# 对于特定分类特征，缺失意味着"无" (None)
cols_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
             'MasVnrType']
for col in cols_none:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna("None")

# 对于特定数值特征，缺失意味着 0
cols_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars',
             'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
             'MasVnrArea']
for col in cols_zero:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna(0)

# MSZoning: 用众数填充
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

# Utilities: 几乎全都是 'AllPub'，如果存在该列且极其单一，可以考虑删除，这里简单填充
all_data['Utilities'] = all_data['Utilities'].fillna("AllPub")

# Functional: 文档说缺失即为 'Typ'
all_data["Functional"] = all_data["Functional"].fillna("Typ")

# Electrical, KitchenQual, Exterior1st, Exterior2nd, SaleType: 用众数填充
for col in ['Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# 检查是否还有缺失值
total_missing = all_data.isnull().sum().sum()
print(f"剩余缺失值总数: {total_missing}")

# ------------------------------------------------------------------------------
# 2. 黄金特征工程 (Advanced Feature Engineering)
# ------------------------------------------------------------------------------
print("开始特征工程...")

# 2.1 数据类型转换 (Type Corrections)
# 这些特征虽然是数字，但实际上代表类别
for col in ['MSSubClass', 'OverallCond', 'YrSold', 'MoSold']:
    if col in all_data.columns:
        all_data[col] = all_data[col].apply(str)

# 2.2 创建强关联聚合特征 (Advanced Feature Engineering)
# 房屋总面积 = 地下室 + 一楼 + 二楼
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

# 房屋总卫浴数
all_data['TotalBath'] = all_data['FullBath'] + 0.5 * all_data['HalfBath'] + all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath']

# 房屋总房龄 (注意 YrSold 之前已转为 str，需转回 int 计算)
all_data['TotalAge'] = all_data['YrSold'].astype(int) - all_data['YearBuilt']

# 是否有游泳池/二楼/地下室/车库/壁炉等二值特征
all_data['HasPool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['Has2ndFloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasGarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasBsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasFireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# 2.3 偏度校正 (Skewness Correction)
numeric_feats = all_data.select_dtypes(include=[np.number]).columns
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness = skewness[abs(skewness['Skew']) > 0.75]
print(f"对 {skewness.shape[0]} 个偏度较高的数值特征进行 Box-Cox 转换")

skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)

# 2.4 独热编码 (One-Hot Encoding)
all_data = pd.get_dummies(all_data)
print(f"编码后特征总数: {all_data.shape[1]}")

# 重新分割 Train 和 Test
X_train = all_data[:ntrain]
X_test = all_data[ntrain:]

# ------------------------------------------------------------------------------
# 3. 模型构建 (Modeling)
# ------------------------------------------------------------------------------
print("构建模型库...")

# 定义交叉验证策略
kf = KFold(n_splits=10, shuffle=True, random_state=42)

def rmsle_cv(model):
    # 使用负均方误差，因为 cross_val_score 期望越大越好
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=kf))
    return(rmse)

# 3.1 基础模型 (Base Models)
# 使用 RobustScaler 应对离群点

# Lasso
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))

# ElasticNet
enet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

# Kernel Ridge Regression
krr = make_pipeline(RobustScaler(), KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5))

# Ridge
ridge = make_pipeline(RobustScaler(), Ridge(alpha=0.6, random_state=1))

# SVR
svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))

# Gradient Boosting
gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5)

# XGBoost
xgboost = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state=7, nthread=-1)

# LightGBM
lightgbm = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

# CatBoost
catboost = CatBoostRegressor(iterations=3000, learning_rate=0.01,
                             depth=6, l2_leaf_reg=1,
                             eval_metric='RMSE', random_seed=42, verbose=0)

# 3.2 Stacking 模型
# 使用 Lasso 作为 Meta-model (加入 SVR, Ridge 到基础模型)
stack_gen = StackingCVRegressor(regressors=(lasso, enet, krr, ridge, svr, gboost, xgboost, lightgbm),
                                meta_regressor=lasso,
                                use_features_in_secondary=True)

# ------------------------------------------------------------------------------
# 4. 训练与预测 (Training & Prediction)
# ------------------------------------------------------------------------------
print("开始训练模型 (这可能需要几分钟)...")

# 辅助函数：训练并打印分数
def fit_model(model, name):
    print(f"Training {name}...")
    try:
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error training {name}: {e}")
        return None

lasso_model_full = fit_model(lasso, "Lasso")
ridge_model_full = fit_model(ridge, "Ridge")
svr_model_full = fit_model(svr, "SVR")
enet_model_full = fit_model(enet, "ElasticNet")
krr_model_full = fit_model(krr, "KernelRidge")
gboost_model_full = fit_model(gboost, "GradientBoosting")
xgb_model_full = fit_model(xgboost, "XGBoost")
lgb_model_full = fit_model(lightgbm, "LightGBM")
cat_model_full = fit_model(catboost, "CatBoost")

print("Training Stacking Regressor...")
try:
    stack_gen_model = stack_gen.fit(np.array(X_train), np.array(y_train))
except Exception as e:
    print(f"Stacking failed: {e}")
    stack_gen_model = None

# ------------------------------------------------------------------------------
# 5. 模型融合 (Ensembling/Blending)
# ------------------------------------------------------------------------------
print("计算最终融合结果...")

def predict(model, X):
    if model is not None:
        # Check if model expects numpy array (stacking often does)
        if isinstance(model, StackingCVRegressor):
             return model.predict(np.array(X))
        return model.predict(X)
    return np.zeros(X.shape[0])

# 获取各模型预测值
# 0.15*Lasso + 0.15*Ridge + 0.2*XGB + 0.1*LGBM + 0.1*CatBoost + 0.3*Stacking

pred_lasso = predict(lasso_model_full, X_test)
pred_ridge = predict(ridge_model_full, X_test)
pred_xgb = predict(xgb_model_full, X_test)
pred_lgb = predict(lgb_model_full, X_test)
pred_cat = predict(cat_model_full, X_test)
pred_stack = predict(stack_gen_model, X_test)

# 加权融合
# 确保所有模型都成功训练，否则回退到简单平均或可用模型
preds = []
weights = []

if lasso_model_full: preds.append(pred_lasso); weights.append(0.15)
if ridge_model_full: preds.append(pred_ridge); weights.append(0.15)
if xgb_model_full: preds.append(pred_xgb); weights.append(0.20)
if lgb_model_full: preds.append(pred_lgb); weights.append(0.10)
if cat_model_full: preds.append(pred_cat); weights.append(0.10)
if stack_gen_model: preds.append(pred_stack); weights.append(0.30)

# 归一化权重
weights = np.array(weights)
weights = weights / weights.sum()

final_pred = np.zeros_like(pred_lasso)
for i, p in enumerate(preds):
    final_pred += p * weights[i]

# 还原预测值 (expm1)
final_pred = np.expm1(final_pred)

# ------------------------------------------------------------------------------
# 6. 生成提交文件 (Submission)
# ------------------------------------------------------------------------------
submission = pd.DataFrame()
submission['Id'] = test_ID
submission['SalePrice'] = final_pred
submission.to_csv('submission.csv', index=False)

print("完成！提交文件已保存为 submission.csv")
print("Top 1% 的荣耀在向你招手！")
