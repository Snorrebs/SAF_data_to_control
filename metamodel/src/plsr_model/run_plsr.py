from data_loader import load_db, load_regcoeff
from utils import cols_by_tag, build_xis
from model import fit_pls_on_xis, predict_from_regcoeff
from eval import metrics

def main():
    MODE = "both"   # "direct" | "fit" | "both"

    """Tags from the excel sheet, and y (should only be one target)"""
    x_tags = ["ge1","ge2","ge3","ge4","ge62","ge63","ge64","ge6","ge7","ge8","ge10","ge11","ge12"]
    y_tag  = "ge14"

    # load data
    db = load_db()

    # Get headers by tags
    base_cols = cols_by_tag(db, x_tags)
    y_col = cols_by_tag(db, y_tag)
    y_col = y_col[0]


    print("Base X columns:", base_cols)
    print("Y column:", y_col)

    # Using regkoeff from NORCE
    if MODE in ("direct","both"):

        coeffs = load_regcoeff()
        y_true = db[y_col].to_numpy()
        y_pred = predict_from_regcoeff(db, coeffs, y_col).to_numpy()
        m = metrics(y_true, y_pred)
        print(f"[NORCE] {y_col}  R^2={m['r2']:.3f}, RMSE={m['rmse']:.3f}")

    #FIT PLS on XIS features 
    if MODE in ("fit","both"):

        X, names = build_xis(db, base_cols)
        y = db[y_col].astype(float).to_numpy()[:, None]
        pls = fit_pls_on_xis(X, y, n_components=len(x_tags), scale=True)
        yhat = pls.predict(X)
        m = metrics(y, yhat)
        print(f"[fit  ] {y_col}  R^2={m['r2']:.3f}, RMSE={m['rmse']:.3f}, XIS features={X.shape[1]}")

if __name__ == "__main__":
    main()
