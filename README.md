# House Price Prediction Using an Extreme Learning Machine

BSc CSIT (CSC383 Neural Networks, Semester VI) project work, Tribhuvan University.

An Extreme Learning Machine is implemented **from scratch in NumPy** — a
single-hidden-layer feedforward network whose input weights and hidden biases are
drawn once at random and never updated, and whose output weights are obtained in a
single closed-form least-squares solve. The trained model is served by a Flask web
application with a form, a JSON API and a model card.

Full documentation, in the TU project-report format:
`report/House_Price_Prediction_ELM_Project_Report.pdf`

## Layout

| Path | Contents |
| --- | --- |
| `elm/core.py` | The learning algorithm: `ELMRegressor` (fit, predict, save, load) and the five activation functions |
| `elm/utils.py` | Data loading, cleaning, ratio features, one-hot encoding, scaling, splitting, regression metrics |
| `scripts/data_prep.py` | Cleans the raw CSV and writes the split and the scaler statistics |
| `scripts/run_experiments.py` | Hidden-layer sweep, activation sweep, ridge sweep, ten-draw stability study, baselines; writes every metric to `models/results.json` |
| `scripts/convergence_study.py` | Timing of the single solve by width, and the back-propagation learning curve at several epoch budgets |
| `app/app.py` | Flask application: form page, form submission, JSON prediction API, model information endpoint |
| `tests/` | 37 automated unit and system tests |
| `report/` | Figure generators, report generator, PDF builder, verification script, and the built PDF |
| `models/` | Trained model, scaler statistics, split, metrics records |

## Reproducing everything

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install numpy pandas flask matplotlib scikit-learn pytest reportlab pymupdf

# 1. data -> models/splits.npz, models/scaler.npz
.venv/Scripts/python.exe scripts/data_prep.py

# 2. every metric quoted in the report -> models/results.json
.venv/Scripts/python.exe scripts/run_experiments.py

# 3. timing study and the back-propagation learning curve
.venv/Scripts/python.exe scripts/convergence_study.py

# 4. tests
.venv/Scripts/python.exe -m pytest tests -v

# 5. run the application
.venv/Scripts/python.exe -m flask --app app.app run --port 5000
#    then open http://127.0.0.1:5000/

# 6. rebuild the report figures and the PDF
.venv/Scripts/python.exe report/figs_results.py
.venv/Scripts/python.exe report/figs_diagrams.py
.venv/Scripts/python.exe report/gen_spec.py
.venv/Scripts/python.exe report/pdf_build_local.py report/spec.json \
    -o report/House_Price_Prediction_ELM_Project_Report.pdf
.venv/Scripts/python.exe report/verify_pdf.py
```

## Measured result

California housing, 20,433 cleaned block groups, 15 features, 16,346 / 4,087 split.

| Model | Training time | Test RMSE | Test R² | Test MAPE |
| --- | --- | --- | --- | --- |
| Mean predictor | 0.00 s | $115,743 | -0.0002 | 63.52% |
| Linear regression | — | — | — | — |
| **ELM, 512 sigmoid neurons** | **25.1 s** | **$57,406** | **0.7539** | **23.25%** |
| Back-propagation MLP, 512 neurons, 300 epochs | 145.7 s | $54,464 | 0.7785 | 20.49% |

The ELM reaches within five per cent of the iterative network's error at roughly
one sixth of the training cost. Draw the accurate conclusion, not the flattering
one: the ELM is a much cheaper way to get an almost-as-good model, not a better
model. See Section 5.3 of the report for the full analysis.