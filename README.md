# NBA Game Outcome Prediction Project

## Project Overview

This project tries to predict if the home team will win an NBA game. It is a binary classification problem (win or lose). The data comes from the Kaggle "Basketball" dataset. We create many new features (like win rates, schedule difficulty, injuries) and train machine learning models to make predictions.

## Project Structure

The project has several Jupyter notebooks. They are listed in the order you should run them.

---

### **preliminary.ipynb**
- Loads NBA data from Kaggle using KaggleHub.
- Shows what data is in each file.
- Creates simple features like score difference.
- Trains a basic logistic regression model.
- This gives us a first baseline (about 59% accuracy).

---

### **Week8.ipynb**
- Adds new features like:
  - Win rate in the last 10 games.
  - Win rate in the last 3 months.
  - Days since last game.
  - Number of games in the last 7 days.
  - Back-to-back game flag.
- Saves the result to `game_stats_2000_onwards.csv`.

---

### **Week8GradientBoostingRegressor.ipynb**
- Combines feature files.
- Uses Gradient Boosting Regressor to predict win (0 or 1).
- R² score and accuracy (~64%) are better than the baseline.
- Proves that new features are helpful.

---

### **Week9.ipynb**
- Tests a random forest model.
- Accidentally uses final scores (data leakage).
- Gets 100% accuracy (cheating).
- Notes the mistake and gives a fix: only use pre-game data.

---

### **Week9_correct_clean.ipynb**
- Applies the fix:
  - Splits each game into home and away rows.
  - Adds rolling averages over the last 10 games.
  - Merges all features.
- Adds win/loss as a label called `target`.
- Saves the clean data as `NBA_cleaned.csv`.

---

### **Week9_training.ipynb**
- Loads `NBA_cleaned.csv`.
- Handles missing values (drops or fills with mean).
- Trains Random Forest and Gradient Boosting models.
- Accuracy is between 64%–66%.
- Helps us know which features are important.

---

### **week10.ipynb**
- Combines all feature steps again.
- Adds:
  - Playoff flag.
  - Injury data (number of inactive players for both teams).
- Creates final feature file for training.

---

### **FinalTraining.ipynb**
- Uses the full feature set (22 features).
- Removes useless columns.
- Trains final models (can tune parameters here).
- Prepares the final dataset with all good features.

---

### **Demo.ipynb**
- Loads final dataset (e.g., `nba_22features.csv`).
- Uses **2016 playoffs** as test data.
- Trains on other games, tests on 2016 playoffs.
- Shows prediction results (accuracy and game-by-game).
- Compares predictions with real results (e.g., NBA Finals).

---

## How to Use

- Run notebooks in the above order to recreate everything.
- To skip to results:
  - Run `FinalTraining.ipynb` to train model.
  - Run `Demo.ipynb` to test on 2016 playoffs.


## Installation Requirements

Please ensure the following Python packages are installed:

```bash
pip install pandas scikit-learn numpy kagglehub lightgbm matplotlib xgboost
```

**Generated Intermediate Datasets**：
- `game_stats_2000_onwards.csv`
- `new_table_with_win_rates_2000_onwards.csv`
- `game_with_rolling_features.csv`
- `selected_game_features.csv`
- `NBA_cleaned.csv`
- `nba_22features.csv`