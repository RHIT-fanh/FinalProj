# NBA Game Outcome Prediction Project

## Project Overview

This project aims to predict whether the home team will win an NBA game, formulated as a binary classification task. The dataset is based on the comprehensive historical NBA records provided in the Kaggle "Basketball" dataset. The project performs extensive feature engineering (e.g., recent win rates, schedule intensity, player injuries) to train classification models that predict game results.

## Project Structure

The project consists of multiple Jupyter Notebook files, organized chronologically by development stage. Each notebook focuses on a specific component:

- **preliminary.ipynb**: Data exploration and baseline model. Downloads the Kaggle dataset using KaggleHub, loads all original CSV files, and prints out their feature columns. Then, it constructs simple baseline features (e.g., differences in total scores between home and away teams) and trains a logistic regression model as a baseline classifier to evaluate initial performance.

- **Week8.ipynb**: Feature engineering – Win rate and schedule-related features. For games after 1999, it calculates recent performance metrics, including 10-game win rates, 3-month win rates for both home and away teams. It also computes schedule features such as days since last game, number of games in the past 7 days, and back-to-back status. The resulting feature-enriched tables are saved as `game_stats_2000_onwards.csv` for later use.

- **Week8GradientBoostingRegressor.ipynb**: Preliminary model testing. Loads the feature datasets (`game_stats_2000_onwards.csv` and `new_table_with_win_rates_2000_onwards.csv`) and merges them for training. A Gradient Boosting Regressor is used to model the win/loss outcome as 0/1. The model's R² score and rounded prediction accuracy (≈64%) demonstrate improvement over the baseline (≈59%), validating the value of engineered features.

- **Week9.ipynb**: Model refinement and error identification. Attempts to train a random forest classifier directly on the game data, but mistakenly includes features like actual scores which directly determine outcomes. This leads to 100% training accuracy, revealing a data leakage issue. The notebook documents this and proposes a fix: to use only pre-game historical statistics.

- **Week9_correct_clean.ipynb**: Data cleaning and feature reconstruction. Implements the fix by restructuring the data: each game is split into two rows (home and away teams) and computes rolling averages over the last 10 games (points, rebounds, assists, margin). These are merged back into game records, forming a new dataset saved as `game_with_rolling_features.csv`. It also integrates previous schedule/win rate features and maps the actual win/loss result to a binary label called `target`. The finalized feature set is saved as **NBA_cleaned.csv**.

- **Week9_training.ipynb**: Model training and evaluation. Loads `NBA_cleaned.csv`, handles missing values (e.g., dropping early games with insufficient data or filling missing entries with means), and trains both Random Forest and Gradient Boosting classifiers. Model performance is evaluated on a test set (average accuracy: 64%–66%), helping to determine the most informative features.

- **week10.ipynb**: Integration and expansion. Combines steps from previous notebooks to regenerate the final feature set (similar to NBA_cleaned.csv) and explores additional features. These include playoff flag (regular season or playoffs) and injury statistics (home/away inactive player counts) from `inactive_players.csv`, resulting in new features like `home_injury_count` and `away_injury_count`. The enriched dataset is saved for final model training.

- **FinalTraining.ipynb**: Final model training. Uses the fully processed dataset (including 22 features) from week10 to train the final model. This notebook may include parameter tuning or comparison of multiple algorithms to optimize prediction performance. After excluding irrelevant features, it trains and evaluates the final model. The resulting dataset contains features such as recent win rates, rolling stats, injury counts, and playoff indicators.

- **Demo.ipynb**: Project demonstration and case analysis. Loads the finalized feature dataset (e.g., `nba_22features.csv`) and sets aside all **2016 playoff games** as a test set. A logistic regression model is trained on the remaining data and used to predict the 2016 playoff outcomes. The notebook outputs a classification report and prediction accuracy for the entire 2016 playoffs, along with game-by-game predictions for the NBA Finals, compared to actual results.

All notebooks should be run sequentially to ensure consistent data generation and model training. In practice, you can directly run `FinalTraining.ipynb` to build the model and then `Demo.ipynb` to assess performance on the 2016 playoffs.

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