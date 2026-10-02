# [MOF CO₂ Adsorption Predictor](https://mof-demo-ebon.vercel.app/)

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-505050?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![React](https://img.shields.io/badge/React-149ECA?style=for-the-badge&logo=react&logoColor=white)](https://react.dev/)
[![Vercel](https://img.shields.io/badge/Vercel-171717?style=for-the-badge&logo=vercel&logoColor=white)](https://vercel.com/)

Compare material shortlists with a learned neural regressor running in the browser.

## What you can explore

- Six structural inputs, input-range checks, and one-feature sensitivity comparisons.
- Saved shortlist predictions and downloadable results.
- Held-out neural-model and Ridge-baseline metrics.

The model is trained on 3,000 synthetic records from the repository generator. Its target is CO₂ adsorption in mol/kg at 298 K and 1 bar. These results describe the synthetic dataset; experimental performance has not been established.

## Reproduce the browser model

```bash
pip install numpy pandas scikit-learn
python scripts/train_browser_model.py
```

Training exports weights and scaler parameters to `web/model.json`. [Model details](docs/browser_model.md) explain the dataset and evaluation. The original TensorFlow pipeline remains in `src/`.
