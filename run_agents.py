"""
Agentic Model Improvement Loop
================================

Two-agent system for automated fare prediction model improvement:

  ┌─────────────┐     recommendations     ┌──────────────┐
  │   Analyst   │ ─────────────────────► │   Engineer   │
  │   Agent     │                        │   Agent      │
  │             │ ◄───────────────────── │              │
  └─────────────┘   RMSE + model file    └──────────────┘
         ▲                                      │
         └──────── updated predictions ─────────┘

Workflow per iteration:
  1. Analyst: compute residual statistics on current best model predictions
  2. Analyst: call Claude to identify patterns and generate recommendations
  3. Engineer: pick top recommendation not yet tried
  4. Engineer: call Claude to generate modified model code
  5. Engineer: run the new model, parse RMSE
  6. If improved → promote to new best model
  7. Analyst receives updated predictions for next iteration

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python3 run_agents.py [--max-iterations N] [--base-model model_v4.py]
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

from agent_analyst import AnalystAgent
from agent_engineer import EngineerAgent


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_module(path: str):
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location("_model", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_model_in_process(model_path: str) -> dict:
    """
    Load a model module, run its pipeline, and return predictions + metadata.
    Runs in the same process for speed (avoids subprocess overhead when we
    need the actual arrays for residual analysis).
    """
    print(f"[Orchestrator] Loading model: {model_path}")
    mod = load_module(model_path)

    train, test = mod.load_data()
    global_traj = mod.build_global_trajectory_features(train)

    # Some models (v5+) need extra args — detect by signature
    import inspect
    prep_sig = inspect.signature(mod.prepare_data)
    prep_params = list(prep_sig.parameters.keys())

    if "offer_trend_feats" in prep_params:
        offer_ts         = mod.build_offer_timeseries(train, test)
        train_trend      = mod.build_offer_trend_features(train, offer_ts)
        test_trend       = mod.build_offer_trend_features(test,  offer_ts)
        train_data       = mod.prepare_data(train, train, global_traj, train_trend)
        test_data        = mod.prepare_data(test,  train, global_traj, test_trend)
    else:
        train_data = mod.prepare_data(train, train, global_traj)
        test_data  = mod.prepare_data(test,  train, global_traj)

    train_data = train_data.dropna(subset=["price_at_obs"]).reset_index(drop=True)
    test_data  = test_data.dropna(subset=["price_at_obs"]).reset_index(drop=True)

    feature_cols = mod.get_features()
    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data["target_price"]
    X_test  = test_data[feature_cols].fillna(0)
    y_test  = test_data["target_price"]

    # Train LightGBM residual model (consistently best across versions)
    baseline_train = np.maximum(train_data["price_at_obs"], train_data["expected_minfare"])
    baseline_test  = test_data["baseline_pred"].values
    y_resid_train  = y_train - baseline_train

    model = lgb.LGBMRegressor(
        n_estimators=2000, max_depth=8, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        num_leaves=127, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_resid_train)
    resid_pred  = model.predict(X_test)
    predictions = baseline_test + resid_pred

    rmse = float(np.sqrt(np.mean((y_test.values - predictions) ** 2)))

    feat_importance = dict(zip(feature_cols, model.feature_importances_))

    print(f"[Orchestrator] RMSE = {rmse:.4f}")

    return {
        "model_path":       model_path,
        "rmse":             rmse,
        "predictions":      predictions,
        "test_data":        test_data,
        "feature_cols":     feature_cols,
        "feat_importance":  feat_importance,
        "baseline_rmse":    float(np.sqrt(np.mean((y_test.values - baseline_test) ** 2))),
    }


def model_description(model_path: str) -> str:
    code = Path(model_path).read_text()
    # Return the module docstring
    lines = code.splitlines()
    if lines and lines[0].startswith('"""'):
        end = next((i for i, l in enumerate(lines[1:], 1) if '"""' in l), 10)
        return "\n".join(lines[:end+1])
    return f"Model at {model_path}"


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class AgenticLoop:
    """
    Coordinates the analyst ↔ engineer loop.

    State is persisted to `agent_state.json` after each iteration so the
    run can be inspected or resumed.
    """

    def __init__(
        self,
        base_model_path: str = "model_v4.py",
        max_iterations:  int = 5,
        min_improvement: float = 0.05,   # RMSE must drop by at least this much
        state_file:      str  = "agent_state.json",
        workdir:         str  = ".",
    ):
        self.base_model_path = base_model_path
        self.max_iterations  = max_iterations
        self.min_improvement = min_improvement
        self.state_file      = Path(workdir) / state_file
        self.workdir         = workdir

        self.analyst  = AnalystAgent()
        self.engineer = EngineerAgent(workdir=workdir)

        self.state = {
            "best_model_path":     base_model_path,
            "best_rmse":           None,
            "baseline_rmse":       None,
            "iterations":          [],
            "all_recommendations": [],
        }

    def _save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    def _print_header(self):
        print("\n" + "="*70)
        print("  AGENTIC MODEL IMPROVEMENT LOOP")
        print("="*70)
        print(f"  Base model : {self.base_model_path}")
        print(f"  Max iters  : {self.max_iterations}")
        print(f"  Min δRMSE  : {self.min_improvement}")
        print("="*70 + "\n")

    def _print_iteration_summary(self, it: dict):
        print(f"\n{'─'*60}")
        print(f"  Iteration {it['iteration']} summary")
        print(f"{'─'*60}")
        print(f"  Recommendation : {it['recommendation_title']}")
        print(f"  New RMSE       : {it['new_rmse']:.4f}")
        print(f"  Improvement    : {it['improvement']:+.4f}")
        print(f"  Accepted       : {'YES ✓' if it['accepted'] else 'NO ✗'}")
        print(f"{'─'*60}\n")

    def run(self):
        self._print_header()

        # ── Iteration 0: run the base model ──────────────────────────────────
        print(f"[Orchestrator] Running base model: {self.base_model_path}")
        result = run_model_in_process(self.base_model_path)

        self.state["best_model_path"] = self.base_model_path
        self.state["best_rmse"]       = result["rmse"]
        self.state["baseline_rmse"]   = result["baseline_rmse"]
        self._save_state()

        current_result     = result
        tried_titles: list[str] = []

        print(f"\n[Orchestrator] Base RMSE = {result['rmse']:.4f}  "
              f"(baseline = {result['baseline_rmse']:.4f})\n")

        # ── Main loop ─────────────────────────────────────────────────────────
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'='*70}")
            print(f"  ITERATION {iteration} / {self.max_iterations}")
            print(f"  Current best RMSE = {self.state['best_rmse']:.4f}")
            print(f"{'='*70}\n")

            # ── Analyst phase ────────────────────────────────────────────────
            prev_recs = self.state["all_recommendations"]
            analysis  = self.analyst.analyze(
                test_data                = current_result["test_data"],
                predictions              = current_result["predictions"],
                feature_cols             = current_result["feature_cols"],
                feature_importances      = current_result["feat_importance"],
                current_model_description= model_description(self.state["best_model_path"]),
                iteration                = iteration,
                previous_recommendations = prev_recs,
            )

            recommendations = analysis.get("recommendations", [])
            self.state["all_recommendations"].extend(recommendations)

            # Pick highest-priority untried recommendation
            rec = None
            for r in sorted(recommendations, key=lambda x: x.get("priority", 99)):
                if r.get("title", "") not in tried_titles:
                    rec = r
                    break

            if rec is None:
                print("[Orchestrator] No new recommendations. Stopping.")
                break

            tried_titles.append(rec.get("title", ""))

            # ── Engineer phase ───────────────────────────────────────────────
            eng_result = self.engineer.implement(
                recommendation    = rec,
                base_model_path   = self.state["best_model_path"],
                iteration         = iteration,
                current_best_rmse = self.state["best_rmse"],
            )

            improvement = eng_result["improvement"]
            accepted    = eng_result["success"] and improvement >= self.min_improvement

            iter_record = {
                "iteration":            iteration,
                "recommendation_title": rec.get("title"),
                "recommendation_type":  rec.get("type"),
                "rationale":            rec.get("rationale"),
                "new_model_path":       eng_result.get("new_model_path"),
                "new_rmse":             eng_result["rmse"],
                "previous_rmse":        self.state["best_rmse"],
                "improvement":          improvement,
                "accepted":             accepted,
            }
            self.state["iterations"].append(iter_record)

            self._print_iteration_summary(iter_record)

            if accepted:
                # Promote new model to best
                self.state["best_model_path"] = eng_result["new_model_path"]
                self.state["best_rmse"]       = eng_result["rmse"]

                # Reload predictions from new best model for next analyst run
                print(f"[Orchestrator] New model accepted — reloading predictions...")
                try:
                    current_result = run_model_in_process(eng_result["new_model_path"])
                except Exception as e:
                    print(f"[Orchestrator] Warning: could not reload new model ({e}). "
                          "Keeping previous predictions for next analyst run.")
            else:
                if not eng_result["success"]:
                    print(f"[Orchestrator] Engineer failed to produce a working model.")
                else:
                    print(f"[Orchestrator] Improvement {improvement:+.4f} below threshold "
                          f"({self.min_improvement}). Keeping current best.")

            self._save_state()

        # ── Final summary ─────────────────────────────────────────────────────
        self._print_final_summary()

    def _print_final_summary(self):
        print("\n" + "="*70)
        print("  AGENTIC LOOP — FINAL SUMMARY")
        print("="*70)
        print(f"  Baseline RMSE      : {self.state['baseline_rmse']:.4f}")
        print(f"  Starting RMSE      : {self.state['iterations'][0]['previous_rmse'] if self.state['iterations'] else self.state['best_rmse']:.4f}")
        print(f"  Best RMSE achieved : {self.state['best_rmse']:.4f}")
        bl = self.state['baseline_rmse']
        best = self.state['best_rmse']
        print(f"  Total improvement  : {(bl - best) / bl * 100:+.2f}% vs baseline")
        print(f"  Best model         : {self.state['best_model_path']}")
        print()
        print(f"  {'Iter':<5} {'Recommendation':<40} {'RMSE':>8} {'Δ':>8} {'Accept':>7}")
        print(f"  {'─'*5} {'─'*40} {'─'*8} {'─'*8} {'─'*7}")
        for it in self.state["iterations"]:
            print(f"  {it['iteration']:<5} "
                  f"{str(it['recommendation_title'])[:40]:<40} "
                  f"{it['new_rmse']:>8.4f} "
                  f"{it['improvement']:>+8.4f} "
                  f"{'YES' if it['accepted'] else 'NO':>7}")
        print("="*70 + "\n")
        print(f"Full state saved to: {self.state_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Agentic fare prediction model improvement loop"
    )
    parser.add_argument(
        "--base-model", default="model_v4.py",
        help="Base model file to start from (default: model_v4.py)"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=5,
        help="Maximum number of analyst→engineer iterations (default: 5)"
    )
    parser.add_argument(
        "--min-improvement", type=float, default=0.05,
        help="Minimum RMSE drop to accept a new model (default: 0.05)"
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    loop = AgenticLoop(
        base_model_path = args.base_model,
        max_iterations  = args.max_iterations,
        min_improvement = args.min_improvement,
    )
    loop.run()


if __name__ == "__main__":
    main()
