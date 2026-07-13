import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("C:/Users/adity/FSW_Defect_Detection/dataset/internal_defects/quantification/quantification_ds.csv")

# Width model
X_width = df[["w"]]
y_width = df["defect width"]

width_model = LinearRegression()
width_model.fit(X_width, y_width)

# Height model
X_height = df[["h"]]
y_height = df["defect height"]

height_model = LinearRegression()
height_model.fit(X_height, y_height)

# Depth model
X_depth = df[["cy", "area"]]
y_depth = df["depth"]

depth_model = LinearRegression()
depth_model.fit(X_depth, y_depth)

print("\nWIDTH MODEL")
print(
    f"defect_width = "
    f"{width_model.coef_[0]:.4f} * w + "
    f"{width_model.intercept_:.4f}"
)

print("\nHEIGHT MODEL")
print(
    f"defect_height = "
    f"{height_model.coef_[0]:.4f} * h + "
    f"{height_model.intercept_:.4f}"
)

print("\nDEPTH MODEL")
print(
    f"depth = "
    f"{depth_model.coef_[0]:.4f} * cy + "
    f"{depth_model.coef_[1]:.4f} * area + "
    f"{depth_model.intercept_:.4f}"
)

"""
def predict_defect(cy, w, h, area):
    pred_width = width_model.predict([[w]])[0]
    pred_height = height_model.predict([[h]])[0]
    pred_depth = depth_model.predict([[cy, area]])[0]

    return pred_width, pred_height, pred_depth
"""

loo = LeaveOneOut()

def evaluate_model(model, X, y, name):
    y_pred = cross_val_predict(model, X, y, cv=loo)

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    print(f"\n===== {name.upper()} MODEL =====")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

# Width
evaluate_model(
    LinearRegression(),
    X_width,
    y_width,
    "Defect Width"
)

# Height
evaluate_model(
    LinearRegression(),
    X_height,
    y_height,
    "Defect Height"
)

# Depth
evaluate_model(
    LinearRegression(),
    X_depth,
    y_depth,
    "Depth"
)