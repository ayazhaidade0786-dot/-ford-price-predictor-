# Ford Car Price Predictor — Deployment Package

This folder contains everything needed to run and deploy the Ford price prediction app.

## Files
| File | Purpose |
|---|---|
| `ford_data.csv` | Training data (your uploaded dataset) |
| `train_model.py` | Trains the model and saves the 4 `.pkl` files below. Already run once — re-run only if you change the data or pipeline. |
| `ford_price_model.pkl` | Trained Linear Regression model (R² ≈ 0.84) |
| `scaler.pkl` | StandardScaler fitted on numeric columns |
| `model_columns.pkl` | Exact column order the model expects (needed to align one-hot encoded input) |
| `dropdown_options.pkl` | Valid dropdown values for Model / Transmission / Fuel Type |
| `app.py` | The Streamlit web app |
| `requirements.txt` | Python packages needed to run the app |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
This opens the app at `http://localhost:8501`.

## Deploy for free (Streamlit Community Cloud)

1. **Create a GitHub repo** and push these files to it:
   - `app.py`
   - `requirements.txt`
   - `ford_price_model.pkl`
   - `scaler.pkl`
   - `model_columns.pkl`
   - `dropdown_options.pkl`

   (You don't need to push `ford_data.csv` or `train_model.py` — the app only needs the 4 `.pkl` files.)

   ```bash
   git init
   git add app.py requirements.txt ford_price_model.pkl scaler.pkl model_columns.pkl dropdown_options.pkl
   git commit -m "Ford price predictor app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```

2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

3. Click **"Create app"** → choose **"Deploy a public app from GitHub"**.

4. Select your repo, branch (`main`), and main file path (`app.py`).

5. Click **Deploy**. Streamlit installs `requirements.txt` automatically and gives you a public URL like:
   `https://<your-app-name>.streamlit.app`

That URL is what you share or put on your presentation slide.

## Notes
- The `.pkl` files together are small (well under GitHub's 100MB limit), so no special handling needed.
- If you ever retrain the model (e.g. more data, different algorithm), just re-run `train_model.py` and push the updated `.pkl` files — no changes needed to `app.py` as long as the column structure stays the same.
