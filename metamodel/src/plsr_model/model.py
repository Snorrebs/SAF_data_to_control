from sklearn.cross_decomposition import PLSRegression
import pandas as pd
import re

def fit_pls_on_xis(X, y, n_components, scale=False):
    """ 
        X: ndarray (N x P) from build_xis
        y: ndarray (N x 1) target 
        n_components: number of PLS components"""

    pls = PLSRegression(n_components=n_components, scale=scale)  # center=True default
    pls.fit(X, y)
    return pls

_INTERACTION_RE = re.compile(r"^(.*?, ge\d+)\*(.*?, ge\d+)$") # match "ge1*ge2" but also "some prefix, ge1*some other, ge2"

def _series_for_term(term: str, db: pd.DataFrame) -> pd.Series:
    """This insane chat solution works for now, but could be simplified."""
    if term == "B_0":
        return pd.Series(1.0, index=db.index)
    if term.endswith("**2"):
        base = term[:-3]
        return db[base] ** 2
    m = _INTERACTION_RE.match(term)
    if m:
        a, b = m.group(1), m.group(2)
        return db[a] * db[b]
    return db[term]


def predict_from_regcoeff(db: pd.DataFrame, coeffs: pd.DataFrame, y_col: str) -> pd.Series:
    """Predict y_col from NORCE reg coeff."""
    beta = coeffs[y_col]
    terms = beta.index.tolist()
    term_matrix = pd.DataFrame({t: _series_for_term(t, db) for t in terms})
    y_pred = term_matrix.dot(beta)
    return y_pred