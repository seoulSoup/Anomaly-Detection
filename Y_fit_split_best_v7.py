
import ast
import re
import numpy as np
import joblib

# ============================================================
# 1. LOAD FITTED MODEL
# ============================================================

MODEL_PATH = "complex_admittance_model.joblib"

model = joblib.load(MODEL_PATH)

FREQ = model.get("frequency_hz", 10000.0)
OMEGA = 2 * np.pi * FREQ

# pF -> nS conversion at the measurement frequency
B_PER_PF = OMEGA * 1e-3

assert model["impedance_mode"] == "SharedPath"
assert model["pair_mode"] == "Sqrt"
assert model["pinfit"] == "Individual"

t = np.asarray(model["common_params"], dtype=float)
v = np.asarray(model["individual_deviations"], dtype=float)

# Stage 1: common complex pin admittance [nS]
y0 = complex(t[0], t[1])

# Shared complex impedance [MOhm]
z_shared = complex(t[3], t[4])

# Stage 2: individual deviations [nS]
dg = v[:44] - np.mean(v[:44])
db = v[44:] - np.mean(v[44:])

y_pin = y0 + dg + 1j * db

# Cross-SMU coefficient explicitly ignored.
CROSS_PF = 0.0

# Fixed local pair coefficients [pF]
LOCAL_PF = {
    "even": 1.544,
    "odd": 0.605,
    "dist2": 0.175
}

# Pin -> SMU mapping
PIN_SMU = {}

for g, (start, end) in enumerate([
    (1, 6),
    (7, 14),
    (15, 22),
    (23, 30),
    (31, 38),
    (39, 44)
]):
    for p in range(start, end + 1):
        PIN_SMU[p] = g


# ============================================================
# 2. PIN PARSER
# ============================================================

def parse_pins(value):
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple, set)):
                value = parsed
            else:
                value = re.findall(r"\d+", value)
        except (ValueError, SyntaxError):
            value = re.findall(r"\d+", value)

    pins = tuple(sorted(int(p) for p in value))

    if not pins:
        raise ValueError("Empty pin selection")

    if len(pins) != len(set(pins)):
        raise ValueError("Duplicate pins")

    if any(p < 1 or p > 44 for p in pins):
        raise ValueError("Pin must be between 1 and 44")

    return pins


# ============================================================
# 3. SHARED PATH AGGREGATION
#
# y_pin unit   : nS
# z_shared unit: MOhm
#
# 1 MOhm * 1 nS = 0.001
# ============================================================

def shared_aggregate(pins):
    y_sum = np.sum(
        y_pin[np.asarray(pins, dtype=int) - 1]
    )

    denominator = (
        1.0 + 1e-3 * z_shared * y_sum
    )

    if abs(denominator) < 1e-12:
        raise ZeroDivisionError(
            "SharedPath denominator is near zero"
        )

    return y_sum / denominator


# ============================================================
# 4. PAIR SQRT AGGREGATION
#
# Sum:
#   Sum of coefficients for all H-L pairs.
#
# MaxPerPin:
#   Mean of:
#       Sum of maximum coefficients for each H pin
#       Sum of maximum coefficients for each L pin
#
# Sqrt:
#   sqrt(Sum * MaxPerPin)
#
# Output in pF.
# ============================================================

def pair_sqrt(high, low):
    pair_values = np.zeros(
        (len(high), len(low)),
        dtype=float
    )

    for i, h in enumerate(high):
        for j, l in enumerate(low):

            if PIN_SMU[h] != PIN_SMU[l]:
                # Cross-SMU contribution ignored
                c = 0.0

            else:
                d = abs(h - l)

                if d == 1:
                    if max(h, l) % 2 == 0:
                        c = LOCAL_PF["even"]
                    else:
                        c = LOCAL_PF["odd"]

                elif d == 2:
                    c = LOCAL_PF["dist2"]

                else:
                    c = 0.0

            pair_values[i, j] = c

    pair_sum = np.sum(pair_values)

    max_h = np.max(pair_values, axis=1).sum()
    max_l = np.max(pair_values, axis=0).sum()

    max_per_pin = 0.5 * (max_h + max_l)

    return float(
        np.sqrt(
            max(pair_sum * max_per_pin, 0.0)
        )
    )


# ============================================================
# 5. INDEPENDENT OPEN CAPACITANCE PREDICTION
# ============================================================

def predict_open_cap(high, low, return_detail=False):
    high = parse_pins(high)
    low = parse_pins(low)

    if set(high) & set(low):
        raise ValueError("High and Low cannot overlap")

    # Complex admittance aggregation on both sides.
    yh = shared_aggregate(high)
    yl = shared_aggregate(low)

    denominator = yh + yl

    if abs(denominator) < 1e-12:
        raise ZeroDivisionError(
            "H/L series admittance denominator near zero"
        )

    y_line_ns = yh * yl / denominator

    # Fixed local pair contribution.
    c_pair_pf = pair_sqrt(high, low)

    # Pair increment as capacitive susceptance [nS].
    y_pair_ns = 1j * B_PER_PF * c_pair_pf

    # Total complex admittance [nS].
    y_total_ns = y_line_ns + y_pair_ns

    # Parallel equivalent capacitance [pF].
    c_pred_pf = y_total_ns.imag / B_PER_PF

    if not return_detail:
        return float(c_pred_pf)

    return {
        "C_pred_pF": float(c_pred_pf),
        "C_line_pF": float(y_line_ns.imag / B_PER_PF),
        "C_pair_pF": float(c_pair_pf),
        "G_pred_nS": float(y_total_ns.real),
        "Y_pred_S": complex(y_total_ns * 1e-9)
    }


# ============================================================
# 6. EXAMPLES
# ============================================================

high = list(range(15, 45))
low = list(range(7, 14))

# Return capacitance only [pF]
c_pred = predict_open_cap(high, low)

print(f"Predicted Open C = {c_pred:.6f} pF")

# Return decomposition
detail = predict_open_cap(
    high, low, return_detail=True
)

print(detail)

# DataFrame example:
#
# open_df["C_pred_pF"] = [
#     predict_open_cap(h, l)
#     for h, l in zip(
#         open_df["high_pins"],
#         open_df["low_pins"]
#     )
# ]
