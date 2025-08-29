# Student Performance – ML App (clean rebuild)
import os, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, streamlit as st
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False

st.set_page_config(page_title="Student Performance – ML App", layout="wide")
st.title("📚 Student Performance – ML App")
st.caption("Upload data, pick features & model, train, compare, and export.")

@st.cache_data(show_spinner=False)
def load_csv(file):
    return pd.read_csv(file) if not isinstance(file, str) else pd.read_csv(file)

def ohe_safe():
    try:  return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError: return OneHotEncoder(handle_unknown="ignore", sparse=False)

def detect_num_cat(df, features, thresh=0.6):
    num, cat = [], []
    for c in features:
        col_num = pd.to_numeric(df[c], errors="coerce")
        if col_num.notna().mean() >= thresh:
            df[c] = col_num; num.append(c)
        else:
            cat.append(c)
    return num, cat, df

def build_pre(num_cols, cat_cols, standardize=True):
    num_steps=[("imputer", SimpleImputer(strategy="median"))]
    if standardize: num_steps.append(("scaler", StandardScaler(with_mean=True)))
    numeric = Pipeline(num_steps)
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", ohe_safe())])
    trf=[]
    if num_cols: trf.append(("num", numeric, num_cols))
    if cat_cols: trf.append(("cat", categorical, cat_cols))
    return ColumnTransformer(trf, remainder="drop")

def feature_names(ct, num_cols, cat_cols):
    names=[]
    if num_cols: names += list(num_cols)
    if cat_cols:
        try: names += ct.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols).tolist()
        except Exception: names += [f"{c}__encoded" for c in cat_cols]
    return names

# Sidebar – data
st.sidebar.header("1) Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if not uploaded:
    st.info("Upload a CSV to begin.")
    st.stop()
data = load_csv(uploaded)

# Sidebar – columns
st.sidebar.header("2) Columns")
all_cols = list(data.columns)
target_col = st.sidebar.selectbox("Target (y)", options=all_cols, index=(all_cols.index("exam_score") if "exam_score" in all_cols else 0))
default_feats = [c for c in ["study_hours_per_day","attendance_percentage","sleep_hours","social_media_hours","mental_health_rating"] if c in all_cols]
feature_cols = st.sidebar.multiselect("Features (X)", options=[c for c in all_cols if c != target_col],
                                      default=(default_feats or [c for c in all_cols if c != target_col]))
if not feature_cols:
    st.warning("Pick at least one feature."); st.stop()

test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
standardize = st.sidebar.checkbox("Standardize numeric features", value=True)

# Sidebar – model
st.sidebar.header("3) Model")
model_name = st.sidebar.selectbox("Choose model", ["Linear Regression","Random Forest","SVR","XGBoost (if installed)"])
params={}
if model_name=="Random Forest":
    params["n_estimators"]=st.sidebar.slider("n_estimators",50,1000,300,50)
    params["max_depth"]=st.sidebar.slider("max_depth (0=None)",0,50,0,1)
    params["min_samples_split"]=st.sidebar.slider("min_samples_split",2,20,2,1)
    params["min_samples_leaf"]=st.sidebar.slider("min_samples_leaf",1,20,1,1)
elif model_name=="SVR":
    params["C"]=st.sidebar.slider("C",0.01,100.0,1.0,0.01)
    params["epsilon"]=st.sidebar.slider("epsilon",0.0,5.0,0.1,0.01)
    params["kernel"]=st.sidebar.selectbox("kernel",["rbf","linear","poly","sigmoid"],index=0)
    standardize=True
elif model_name=="XGBoost (if installed)":
    if not XGB_OK: st.sidebar.warning("xgboost not installed"); 
    params["n_estimators"]=st.sidebar.slider("n_estimators",50,1000,400,50)
    params["learning_rate"]=st.sidebar.slider("learning_rate",0.01,0.5,0.1,0.01)
    params["max_depth"]=st.sidebar.slider("max_depth",1,20,6,1)
    params["subsample"]=st.sidebar.slider("subsample",0.5,1.0,1.0,0.05)
    params["colsample_bytree"]=st.sidebar.slider("colsample_bytree",0.5,1.0,1.0,0.05)

st.sidebar.header("4) Train")
seed = st.sidebar.number_input("random_state", value=42, step=1)
train_btn = st.sidebar.button("🚀 Train model")

# Data peek
with st.expander("Peek at the data", expanded=True):
    st.write(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")
    st.dataframe(data.head(30), use_container_width=True)
    miss = data.isna().sum().sort_values(ascending=False)
    if miss.sum()>0: st.write("Missing values:"); st.dataframe(miss.to_frame("missing"), use_container_width=True)
    num_cols_all = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols_all)>=2:
        cols = num_cols_all if target_col in num_cols_all else num_cols_all+[target_col]
        corr = data[cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6,4)); im = ax.imshow(corr, aspect="auto")
        ax.set_title("Correlation (numeric)")
        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); st.pyplot(fig, clear_figure=True)

# Train & evaluate
if train_btn:
    df = data[feature_cols+[target_col]].copy()
    y = pd.to_numeric(df[target_col], errors="coerce")
    if y.isna().all(): st.error("Target has no numeric values."); st.stop()
    y = y.fillna(y.median())
    num_cols, cat_cols, df = detect_num_cat(df, feature_cols, 0.6)
    st.info(f"Numeric: {num_cols or 'None'}"); st.info(f"Categorical: {cat_cols or 'None'}")
    X = df[feature_cols].copy()
    pre = build_pre(num_cols, cat_cols, standardize)
    if model_name=="Linear Regression":
        model=LinearRegression()
    elif model_name=="Random Forest":
        model=RandomForestRegressor(n_estimators=params["n_estimators"],
                                    max_depth=None if params["max_depth"]==0 else params["max_depth"],
                                    min_samples_split=params["min_samples_split"],
                                    min_samples_leaf=params["min_samples_leaf"],
                                    random_state=seed, n_jobs=-1)
    elif model_name=="SVR":
        model=SVR(C=params["C"], epsilon=params["epsilon"], kernel=params["kernel"])
    else:
        if not XGB_OK: st.error("Install xgboost or pick another model."); st.stop()
        model=XGBRegressor(n_estimators=params["n_estimators"], learning_rate=params["learning_rate"],
                           max_depth=params["max_depth"], subsample=params["subsample"],
                           colsample_bytree=params["colsample_bytree"], random_state=seed)
    pipe = Pipeline([("pre", pre), ("model", model)])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
    with st.spinner("Training..."): pipe.fit(Xtr, ytr)
    ypred = pipe.predict(Xte)
    mse = mean_squared_error(yte, ypred); rmse=float(np.sqrt(mse))
    mae = mean_absolute_error(yte, ypred); r2 = r2_score(yte, ypred)
   if train_btn:
    df = data[feature_cols + [target_col]].copy()
    y = pd.to_numeric(df[target_col], errors="coerce")
    if y.isna().all():
        st.error("Target has no numeric values.")
        st.stop()
    y = y.fillna(y.median())
    num_cols, cat_cols, df = detect_num_cat(df, feature_cols, 0.6)
    st.info(f"Numeric: {num_cols or 'None'}")
    st.info(f"Categorical: {cat_cols or 'None'}")
    X = df[feature_cols].copy()
    pre = build_pre(num_cols, cat_cols, standardize)

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=None if params["max_depth"] == 0 else params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=seed,
            n_jobs=-1
        )
    elif model_name == "SVR":
        model = SVR(C=params["C"], epsilon=params["epsilon"], kernel=params["kernel"])
    else:
        if not XGB_OK:
            st.error("Install xgboost or pick another model.")
            st.stop()
        model = XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            random_state=seed
        )

    pipe = Pipeline([("pre", pre), ("model", model)])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)

    with st.spinner("Training..."):
        pipe.fit(Xtr, ytr)

    ypred = pipe.predict(Xte)
    mse = mean_squared_error(yte, ypred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(yte, ypred)
    r2 = r2_score(yte, ypred)

    st.subheader("📈 Performance")
    st.table({"MAE": [mae], "RMSE": [rmse], "R²": [r2]})

tab1, tab2, tab3 = st.tabs(["📊 Plots","⭐ Importance / Coefficients","🔎 Predictions sample"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        ax.scatter(yte, ypred, alpha=0.7)
        mn, mx = float(min(yte.min(), ypred.min())), float(max(yte.max(), ypred.max()))
        ax.plot([mn, mx], [mn, mx], "--", linewidth=2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig, clear_figure=True)
    with c2:
        fig2, ax2 = plt.subplots()
        ax2.hist(yte - ypred, bins=30)
        ax2.set_title("Residuals")
        st.pyplot(fig2, clear_figure=True)

with tab2:
    try:
        feats = feature_names(pipe.named_steps["pre"], num_cols, cat_cols)
    except Exception:
        feats = [f"f{i}" for i in range(pipe.named_steps["pre"].transform(Xte[:1]).shape[1])]
    m = pipe.named_steps["model"]
    if hasattr(m, "feature_importances_"):
        imp = pd.DataFrame({"feature": feats, "importance": m.feature_importances_}).sort_values("importance", ascending=False)
        st.dataframe(imp.head(30), use_container_width=True)
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        top = imp.head(15)
        ax3.barh(top["feature"][::-1], top["importance"][::-1])
        ax3.set_title("Feature Importance (Top 15)")
        st.pyplot(fig3, clear_figure=True)
    elif isinstance(m, LinearRegression) and hasattr(m, "coef_"):
        coefs = np.ravel(m.coef_)
        dfc = pd.DataFrame({"feature": feats, "coefficient": coefs})
        dfc["abs"] = dfc["coefficient"].abs()
        dfc = dfc.sort_values("abs", ascending=False).drop(columns="abs")
        st.dataframe(dfc.head(50), use_container_width=True)
        fig4, ax4 = plt.subplots(figsize=(6, 6))
        top = dfc.copy()
        top["abs"] = top["coefficient"].abs()
        top = top.sort_values("abs", ascending=False).head(15)
        ax4.barh(top["feature"][::-1], top["coefficient"][::-1])
        ax4.set_title("Linear Coefficients (Top 15 by |coef|)")
        st.pyplot(fig4, clear_figure=True)
    else:
        st.info("Use RF/XGBoost for importances or Linear for coefficients.")

with tab3:
    preview = pd.DataFrame({"y_true": yte[:25].to_numpy(), "y_pred": ypred[:25]})
    st.dataframe(preview, use_container_width=True)
    st.subheader("📦 Export")
    st.download_button("Download trained model (.pkl)", data=pickle.dumps(pipe),
                       file_name=f"model_{model_name.replace(' ', '_').lower()}.pkl", mime="application/octet-stream")
    preds = pd.DataFrame({"y_true": yte, "y_pred": ypred}).reset_index(names="row")
    st.download_button("Download predictions (.csv)", data=preds.to_csv(index=False).encode("utf-8"),
                       file_name="predictions.csv", mime="text/csv")


st.markdown("---"); st.subheader("🏁 Compare models with cross-validation")
with st.expander("Show comparison settings", expanded=False):
    k=st.slider("CV folds (k)",3,10,5); r=st.slider("Repeats",1,5,2)
    c1,c2,c3,c4=st.columns(4)
    use_lr=c1.checkbox("Linear Regression",True); use_rf=c2.checkbox("Random Forest",True)
    use_svr=c3.checkbox("SVR (rbf)",True); use_xgb=c4.checkbox("XGBoost",XGB_OK)
    run_cmp=st.button("Run comparison")

if "run_cmp" in locals() and run_cmp:
    dfc=data[feature_cols+[target_col]].copy()
    yc=pd.to_numeric(dfc[target_col],errors="coerce").fillna(dfc[target_col].median())
    numc,catc,dfc=detect_num_cat(dfc, feature_cols, 0.6); Xc=dfc[feature_cols].copy()
    prec=build_pre(numc,catc, standardize)
    models={}
    if use_lr: models["Linear Regression"]=LinearRegression()
    if use_rf: models["Random Forest"]=RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=-1)
    if use_svr: models["SVR (rbf)"]=SVR(C=1.0,epsilon=0.1,kernel="rbf")
    if use_xgb and XGB_OK: models["XGBoost"]=XGBRegressor(n_estimators=400,learning_rate=0.1,max_depth=6,
                                                         subsample=1.0,colsample_bytree=1.0,random_state=42)
    if not models: st.warning("Pick at least one model."); 
    else:
        cv=RepeatedKFold(n_splits=k, n_repeats=r, random_state=42); rows=[]
        for name,mdl in models.items():
            pc=Pipeline([("pre",prec),("model",mdl)])
            scores=cross_validate(pc,Xc,yc,cv=cv,n_jobs=-1,error_score="raise",
                                  scoring={"MAE":"neg_mean_absolute_error","MSE":"neg_mean_squared_error","R2":"r2"})
            rows.append({"Model":name,"MAE":-scores["test_MAE"].mean(),
                         "RMSE":float(np.sqrt(-scores["test_MSE"].mean())),
                         "R²":scores["test_R2"].mean()})
        board=pd.DataFrame(rows).sort_values(["RMSE","MAE","R²"],ascending=[True,True,False]).reset_index(drop=True)
        st.dataframe(board, use_container_width=True)
        fig,ax=plt.subplots(figsize=(6,4)); ax.bar(board["Model"], board["R²"])
        ax.set_ylabel("R²"); ax.set_title("Model comparison (R², CV)"); plt.xticks(rotation=20); st.pyplot(fig, clear_figure=True)
