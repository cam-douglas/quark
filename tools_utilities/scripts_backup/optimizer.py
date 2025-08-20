import torch
import optuna
from model import MyBrainModel, train_model_with_sam

def optimize_model():
    def objective(trial):
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
        dropout = trial.suggest_uniform('dropout', 0.1, 0.5)

        model = MyBrainModel(dropout=dropout)
        score = train_model_with_sam(model, lr=lr)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10, timeout=600)

    print("✅ Optimization complete. Best score:", study.best_value)