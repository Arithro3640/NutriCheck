# NutriCheck — Child Malnutrition Detection (MalFusion)

A friendly, full-stack web app that estimates a young child's **nutritional
status** (Normal / Mild Malnutrition / MAM / SAM) from simple measurements and
health history, using the proposed **MalFusion** ensemble model.

- Clean, low-friction form so health workers can use it without hesitation
- Live result with confidence gauge and per-class probabilities
- **Admin panel**: add new records (saved to CSV), one-click **data check +
  auto-fix + retrain**, and a full **model status** dashboard
- **Custom themes** (Nurture, Sunrise, Calm, Night) + a nurturing background
- HTML / CSS / JavaScript frontend with a Flask (Python) backend

---

## 1. What is MalFusion?

MalFusion is a soft-voting ensemble taken from your research notebook:

| Component      | Weight | Role                          |
|----------------|:------:|-------------------------------|
| Random Forest  |   2    | tuned, robust base learner    |
| SVM (RBF)      |   2    | strong individual leader      |
| XGBoost        |   1    | gradient boosting             |
| CatBoost       |   1    | gradient boosting             |

The full preprocessing chain (winsorising outliers, median/mode imputation,
label encoding, clinical feature engineering, scaling, and SMOTE balancing)
is reproduced exactly and bundled with the model.

> If `xgboost`, `catboost`, or `imbalanced-learn` are not installed, the app
> automatically falls back to scikit-learn equivalents so it **always runs**.
> Install the full `requirements.txt` to get the complete model and the higher
> accuracy reported in the notebook.

---

## 2. Run it in VS Code (step by step)

### Prerequisites
- Python 3.9 or newer installed
- VS Code with the Python extension (recommended)

### Steps

1. **Open the folder** in VS Code: `File → Open Folder…` → select this project.

2. **Open a terminal** in VS Code: `Terminal → New Terminal`.

3. **(Recommended) Create a virtual environment**

   Windows:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
   macOS / Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Start the app**
   ```bash
   python app.py
   ```
   On the **first run** it trains the MalFusion model automatically (takes a few
   seconds). You'll see `Initial MalFusion model ready.`

6. **Open the site** — visit **http://127.0.0.1:5000** in your browser.

To stop the server, press `Ctrl + C` in the terminal.

---

## 3. Using the app

### Check a child (home page)
Fill the form — sensible defaults are pre-filled, so you only adjust what you
know. The three key fields are **age, weight, and height**. Click
**Assess nutrition status** to see the result, confidence, and guidance.

### Admin panel (`/admin`)
1. Click **Admin** in the top bar and enter the passcode.
   **Default passcode: `admin123`** (change it — see below).
2. **Add a new record** → fills one row and saves it to
   `data/new_records.csv`.
3. **Check data only** → scans all data and lists problems + the preprocessing
   that will fix them (no training).
4. **Train model now** → checks data, auto-fixes issues, retrains MalFusion on
   the base dataset **plus** your new records, and updates the status panel
   (accuracy, F1, precision, recall, components, last trained, row counts).
5. **Download saved records (CSV)** → exports `new_records.csv`.

---

## 4. Change the admin passcode

Open `app.py` and edit:
```python
ADMIN_PASSCODE = os.environ.get("ADMIN_PASSCODE", "admin123")
```
Or set an environment variable before running:
```bash
# macOS / Linux
export ADMIN_PASSCODE="your-strong-code"
# Windows (PowerShell)
$env:ADMIN_PASSCODE="your-strong-code"
```

---

## 5. Project structure

```
malnutrition-app/
├── app.py                  # Flask backend + all API routes
├── requirements.txt
├── README.md
├── ml/
│   ├── config.py           # feature schema (drives form, validation, training)
│   ├── pipeline.py         # MalFusion model + preprocessing + data health check
│   └── trainer.py          # load data, train, save model + status
├── data/
│   └── child_malnutrition_labeled.csv   # base dataset
│   └── new_records.csv     # (created when you add records in admin)
├── models/                 # trained model + status.json (created on first run)
├── templates/
│   ├── base.html           # shared layout, theme switcher, icons
│   ├── index.html          # prediction form + result
│   └── admin.html          # admin panel
└── static/
    ├── css/style.css       # themes, background, components
    └── js/
        ├── theme.js        # theme switcher
        ├── main.js         # prediction logic
        └── admin.js        # admin logic
```

---

## 6. Retrain from the terminal (optional)

You can train without opening the admin panel:
```bash
python -m ml.trainer
```

---

## 7. Notes

- This tool is a **screening aid**, not a diagnosis. A child with worrying
  signs should always be seen by a qualified health worker.
- If you change the dataset columns, update `ml/config.py` to match.
