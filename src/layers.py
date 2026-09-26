"""
Layer parameters as (low, high) ranges, from the research pass.
Values are fractional gains/losses applied to light reaching the next layer,
or (for harvesting layers) conversion efficiency.
"""

LAYERS = {
    "moth_eye_antiglare":      {"reflection_loss_cut": (0.75, 0.95)},  # cuts ~4% reflection to under 1%
    "dye_concentrator":        {"efficiency": (0.01, 0.03)},           # LSC, transparent, real-world record ~3%
    "firefly_extraction":      {"display_efficiency_gain": (0.15, 0.61)},  # OLED light-out gain
    "structural_color_display":{"reflectance": (0.6, 0.8), "power_mw_cm2": (0.5, 1.7)},
    "compound_eye_backpanel":  {"efficiency": (0.15, 0.30)},           # opaque, back-of-device cell
    "indoor_cell_baseline":    {"efficiency": (0.10, 0.30)},           # commercial indoor PV (Exeger-class)
}

SCENARIOS_LUX = {
    "pocket": 0,
    "indoor_300lux": 300,
    "indoor_500lux": 500,
    "window_light": 2000,
    "overcast_outdoor": 10000,
    "full_sun": 100000,
}
