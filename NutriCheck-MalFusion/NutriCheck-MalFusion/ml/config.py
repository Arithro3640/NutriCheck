"""
Central schema for the Child Malnutrition (MalFusion) project.

Everything about the dataset lives here so the form, validation, preprocessing
and training all stay in sync. Edit this file if your CSV columns ever change.
"""

# --- File locations -----------------------------------------------------------
import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

BASE_CSV    = os.path.join(DATA_DIR, "child_malnutrition_labeled.csv")
NEW_CSV     = os.path.join(DATA_DIR, "new_records.csv")        # admin-saved records
MODEL_PATH  = os.path.join(MODEL_DIR, "malfusion.joblib")      # trained pipeline
STATUS_PATH = os.path.join(MODEL_DIR, "status.json")           # model status report

# Columns that exist in the raw CSV but are never used as model inputs.
DROP_COLS = ["Timestamp", "Guardian Name", "Middle Upper Arm Circumference(cm)"]

TARGET = "Nutritional_Status"

# Human-friendly description of every output class.
CLASS_INFO = {
    "Normal": {
        "label": "Normal",
        "tone": "good",
        "blurb": "The child's nutritional status looks healthy for their age.",
    },
    "Mild Malnutrition": {
        "label": "Mild Malnutrition",
        "tone": "watch",
        "blurb": "Early signs of undernutrition. Closer monitoring and feeding support are advised.",
    },
    "MAM": {
        "label": "Moderate Acute Malnutrition (MAM)",
        "tone": "warn",
        "blurb": "Moderate acute malnutrition. A nutrition programme and follow-up are recommended.",
    },
    "SAM": {
        "label": "Severe Acute Malnutrition (SAM)",
        "tone": "danger",
        "blurb": "Severe acute malnutrition. Urgent referral to a health facility is recommended.",
    },
}

# --- Feature schema -----------------------------------------------------------
# Order here MUST match the training column order. Each entry drives the form.
#   kind: "number" or "choice"
#   group: which section it appears in on the form
FEATURES = [
    # ---- Child basics ----
    {"name": "Gender", "label": "Gender", "kind": "choice",
     "choices": ["Female", "Male"], "group": "Child basics",
     "help": "Biological sex of the child."},
    {"name": "Age(days)", "label": "Age (in days)", "kind": "number",
     "min": 0, "max": 2000, "step": 1, "default": 253, "group": "Child basics",
     "help": "How many days old the child is. 1 year ≈ 365 days."},

    # ---- Body measurements ----
    {"name": "Weight(Kg)", "label": "Weight (kg)", "kind": "number",
     "min": 0.5, "max": 30, "step": 0.1, "default": 7.7, "group": "Body measurements",
     "help": "Current body weight in kilograms."},
    {"name": "Height(cm)", "label": "Height / Length (cm)", "kind": "number",
     "min": 30, "max": 130, "step": 0.1, "default": 67.1, "group": "Body measurements",
     "help": "Standing height or lying length in centimetres."},
    {"name": "Odema", "label": "Oedema (swelling)", "kind": "choice",
     "choices": ["No", "Yes"], "group": "Body measurements",
     "help": "Swelling, usually in the feet — an important malnutrition sign."},

    # ---- Birth & pregnancy ----
    {"name": "Birth Weight(Kg)", "label": "Birth weight (kg)", "kind": "number",
     "min": 0.5, "max": 6, "step": 0.1, "default": 2.9, "group": "Birth & pregnancy",
     "help": "Weight recorded at birth."},
    {"name": "Multiple Baby", "label": "Babies in this birth", "kind": "number",
     "min": 1, "max": 4, "step": 1, "default": 1, "group": "Birth & pregnancy",
     "help": "1 = single baby, 2 = twins, etc."},
    {"name": "Birth Complication", "label": "Birth maturity", "kind": "choice",
     "choices": ["Mature", "Premature", "No"], "group": "Birth & pregnancy",
     "help": "Whether the baby was born mature, premature, or no complication recorded."},
    {"name": "Pregnancy Duration(month)", "label": "Pregnancy duration (months)", "kind": "number",
     "min": 5, "max": 11, "step": 0.5, "default": 9, "group": "Birth & pregnancy",
     "help": "How many months the pregnancy lasted."},
    {"name": "Birth Procedure", "label": "Delivery type", "kind": "choice",
     "choices": ["Normal", "C-Section"], "group": "Birth & pregnancy",
     "help": "How the child was delivered."},
    {"name": "Mother's nutritional status during pregnancy",
     "label": "Mother's nutrition in pregnancy", "kind": "choice",
     "choices": ["Nutrition", "Undernutrition", "Overnutrition"], "group": "Birth & pregnancy",
     "help": "Mother's nutritional state during pregnancy."},
    {"name": "Mother's Education", "label": "Mother's education", "kind": "choice",
     "choices": ["Primary", "Secondary", "Higher Secondary"], "group": "Birth & pregnancy",
     "help": "Highest education level the mother completed."},

    # ---- Feeding ----
    {"name": "Breastfeeding status", "label": "Breastfeeding status", "kind": "choice",
     "choices": ["Natural", "Less", "Stopped", "No"], "group": "Feeding",
     "help": "Current breastfeeding pattern."},
    {"name": "Breastfeeding Duration(days)", "label": "Breastfeeding so far (days)", "kind": "number",
     "min": 0, "max": 2000, "step": 1, "default": 181, "group": "Feeding",
     "help": "Total number of days the child has been breastfed."},
    {"name": "Formula Feeding", "label": "Formula feeding", "kind": "choice",
     "choices": ["No", "Yes"], "group": "Feeding",
     "help": "Whether the child is given formula milk."},
    {"name": "Is Other Food Given or Not in the First Six Months",
     "label": "Other food before 6 months", "kind": "choice",
     "choices": ["No", "Yes"], "group": "Feeding",
     "help": "Was anything other than breast milk given in the first six months?"},

    # ---- Health & symptoms ----
    {"name": "Diarea", "label": "Diarrhoea", "kind": "choice",
     "choices": ["No", "Acute", "Persistent", "Chronic"], "group": "Health & symptoms",
     "help": "Type of diarrhoea, if any."},
    {"name": "Duration of Diarea (Day)", "label": "Diarrhoea duration (days)", "kind": "number",
     "min": 0, "max": 60, "step": 1, "default": 0, "group": "Health & symptoms",
     "help": "How many days diarrhoea has lasted (0 if none)."},
    {"name": "Fever", "label": "Fever", "kind": "choice",
     "choices": ["No", "Yes"], "group": "Health & symptoms",
     "help": "Does the child currently have a fever?"},
    {"name": "Pneumonia", "label": "Pneumonia", "kind": "choice",
     "choices": ["No", "Acute", "Chronic"], "group": "Health & symptoms",
     "help": "Pneumonia status, if any."},
    {"name": "Vomit", "label": "Vomiting", "kind": "choice",
     "choices": ["No", "Yes"], "group": "Health & symptoms",
     "help": "Is the child vomiting?"},
    {"name": "Cough", "label": "Cough", "kind": "choice",
     "choices": ["No", "Yes"], "group": "Health & symptoms",
     "help": "Does the child have a cough?"},
    {"name": "Appetite", "label": "Good appetite", "kind": "choice",
     "choices": ["Yes", "No"], "group": "Health & symptoms",
     "help": "Is the child eating well?"},
    {"name": "Chronic Disease", "label": "Chronic disease", "kind": "choice",
     "choices": ["No", "Yes"], "group": "Health & symptoms",
     "help": "Any long-term illness?"},
    {"name": "Jaundis", "label": "Jaundice", "kind": "choice",
     "choices": ["No", "Yes"], "group": "Health & symptoms",
     "help": "Yellowing of skin or eyes."},
    {"name": "Bronchiolitis", "label": "Bronchiolitis", "kind": "choice",
     "choices": ["No", "Yes"], "group": "Health & symptoms",
     "help": "Inflammation of the small airways in the lungs."},
    {"name": "Tuberculosis", "label": "Tuberculosis", "kind": "choice",
     "choices": ["No", "Yes"], "group": "Health & symptoms",
     "help": "Has TB been diagnosed?"},
    {"name": "immunisation", "label": "Immunisation", "kind": "choice",
     "choices": ["Completed", "Partially Completed"], "group": "Health & symptoms",
     "help": "Is the child's vaccination schedule complete?"},
]

# Ordered section list for the form layout.
GROUP_ORDER = [
    "Child basics",
    "Body measurements",
    "Birth & pregnancy",
    "Feeding",
    "Health & symptoms",
]

GROUP_ICONS = {
    "Child basics": "child",
    "Body measurements": "ruler",
    "Birth & pregnancy": "heart",
    "Feeding": "bottle",
    "Health & symptoms": "stethoscope",
}

FEATURE_NAMES   = [f["name"] for f in FEATURES]
NUMERIC_COLS    = [f["name"] for f in FEATURES if f["kind"] == "number"]
CATEGORICAL_COLS = [f["name"] for f in FEATURES if f["kind"] == "choice"]


def feature_by_name(name):
    for f in FEATURES:
        if f["name"] == name:
            return f
    return None
