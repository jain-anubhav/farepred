"""
Analyst Agent — analyzes model residuals and recommends improvements.

Responsibilities:
  1. Run the current best model on train/test data to get predictions
  2. Compute rich residual statistics sliced by DFD, market, airline,
     price tier, offer_date season, and existing features
  3. Use Claude to reason about which patterns imply actionable improvements
  4. Return structured JSON recommendations for the engineer agent
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import anthropic


# ────────────────────────────────────────────────────────────────────────────
# Statistical analysis helpers
# ────────────────────────────────────────────────────────────────────────────

def _rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def _top_segments(df, col, n=6):
    """Return the n groups with highest RMSE for a categorical column."""
    g = df.groupby(col).apply(
        lambda x: pd.Series({
            'rmse':         float(np.sqrt((x['residual']**2).mean())),
            'mean_residual': float(x['residual'].mean()),
            'n':             len(x),
        })
    ).reset_index()
    return g.nlargest(n, 'rmse')[g.columns.tolist()].to_dict('records')


def _bucket_analysis(df, col, bins, labels):
    """RMSE + mean residual by binned numeric column."""
    df = df.copy()
    df['_bucket'] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    g = df.groupby('_bucket').apply(
        lambda x: pd.Series({
            'rmse':         float(np.sqrt((x['residual']**2).mean())),
            'mean_residual': float(x['residual'].mean()),
            'n':             len(x),
        })
    ).reset_index()
    g['_bucket'] = g['_bucket'].astype(str)
    return g.to_dict('records')


def compute_residual_statistics(test_data, predictions, feature_cols):
    """
    Compute multi-dimensional residual statistics for analyst reasoning.
    """
    df = test_data.copy()
    df['pred']         = predictions
    df['residual']     = df['target_price'] - predictions
    df['abs_residual'] = np.abs(df['residual'])

    stats = {}

    # ── Overall ──────────────────────────────────────────────────────────────
    r = df['residual']
    stats['overall'] = {
        'rmse':               float(np.sqrt((r**2).mean())),
        'baseline_rmse':      _rmse(df['target_price'], df['baseline_pred']),
        'mean_residual':      float(r.mean()),
        'std_residual':       float(r.std()),
        'pct_positive':       float((r > 0).mean()),   # positive = underpredict
        'p25_abs':            float(np.percentile(df['abs_residual'], 25)),
        'p50_abs':            float(np.percentile(df['abs_residual'], 50)),
        'p75_abs':            float(np.percentile(df['abs_residual'], 75)),
        'p95_abs':            float(np.percentile(df['abs_residual'], 95)),
    }

    # ── By DFD bucket ────────────────────────────────────────────────────────
    stats['by_dfd_bucket'] = _bucket_analysis(
        df, 'dfd',
        bins=[-1, 4, 9, 19, 29, 39],
        labels=['0-4', '5-9', '10-19', '20-29', '30-39'],
    )

    # ── By exact DFD (spot check at key dfds) ────────────────────────────────
    key_dfds = [0, 5, 10, 15, 20, 25, 30, 35, 39]
    stats['by_key_dfd'] = [
        {
            'dfd': int(d),
            'rmse': _rmse(df[df['dfd']==d]['target_price'], df[df['dfd']==d]['pred']),
            'baseline_rmse': _rmse(df[df['dfd']==d]['target_price'], df[df['dfd']==d]['baseline_pred']),
            'mean_residual': float(df[df['dfd']==d]['residual'].mean()),
            'n': int((df['dfd']==d).sum()),
        }
        for d in key_dfds if (df['dfd']==d).sum() > 0
    ]

    # ── By market (pathod_id) ────────────────────────────────────────────────
    stats['worst_markets'] = _top_segments(df, 'pathod_id', n=8)
    mkt = df.groupby('pathod_id').agg(
        rmse=('residual', lambda x: float(np.sqrt((x**2).mean()))),
        mean_res=('residual', 'mean'),
        n=('residual', 'count'),
    ).reset_index()
    stats['market_rmse_spread'] = {
        'min': float(mkt['rmse'].min()),
        'max': float(mkt['rmse'].max()),
        'std': float(mkt['rmse'].std()),
        'n_markets': int(len(mkt)),
    }

    # ── By airline ───────────────────────────────────────────────────────────
    stats['by_airline'] = _top_segments(df, 'airline_enc', n=6)

    # ── By departure day of week ─────────────────────────────────────────────
    stats['by_depdow'] = _bucket_analysis(
        df, 'depdow', bins=[-0.5+i for i in range(8)],
        labels=[str(i) for i in range(7)],
    )

    # ── By departure hour ────────────────────────────────────────────────────
    stats['by_dephour'] = _bucket_analysis(
        df, 'dephour',
        bins=[-1, 6, 12, 18, 24],
        labels=['0-6', '7-12', '13-18', '19-24'],
    )

    # ── By price tier (quartiles of actual price) ────────────────────────────
    q = df['target_price'].quantile([.25, .5, .75]).tolist()
    stats['by_price_tier'] = _bucket_analysis(
        df, 'target_price',
        bins=[-np.inf, q[0], q[1], q[2], np.inf],
        labels=['low', 'med-low', 'med-high', 'high'],
    )

    # ── By offer_date month ──────────────────────────────────────────────────
    if 'offer_month' in df.columns:
        stats['by_offer_month'] = _bucket_analysis(
            df, 'offer_month', bins=list(range(1, 14)), labels=[str(i) for i in range(1, 13)],
        )

    # ── Baseline vs ML improvement ───────────────────────────────────────────
    df['ml_improvement']   = df['abs_residual'] < (df['target_price'] - df['baseline_pred']).abs()
    df['baseline_correct'] = (df['target_price'] - df['baseline_pred']).abs() < df['abs_residual']
    stats['ml_vs_baseline'] = {
        'pct_ml_better':       float(df['ml_improvement'].mean()),
        'pct_baseline_better': float(df['baseline_correct'].mean()),
        'worst_ml_dfd_buckets': (
            df[df['baseline_correct']]
            .groupby(pd.cut(df[df['baseline_correct']]['dfd'],
                            bins=[-1,4,9,19,29,39],
                            labels=['0-4','5-9','10-19','20-29','30-39']))
            ['residual'].count()
            .to_dict()
        ),
    }

    # ── Feature correlations with |residual| (top informative features) ──────
    numeric_feats = [c for c in feature_cols if c in df.columns and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]
    corrs = {}
    for c in numeric_feats:
        try:
            corrs[c] = float(df[c].corr(df['abs_residual']))
        except Exception:
            pass
    top_pos = sorted(corrs.items(), key=lambda x: x[1], reverse=True)[:8]
    top_neg = sorted(corrs.items(), key=lambda x: x[1])[:5]
    stats['feature_corr_with_abs_residual'] = {
        'highest_positive': [{'feature': k, 'corr': round(v, 4)} for k, v in top_pos],
        'highest_negative': [{'feature': k, 'corr': round(v, 4)} for k, v in top_neg],
        'note': 'positive corr = higher feature value → larger error',
    }

    # ── Residual by (dfd bucket, price tier) interaction ────────────────────
    df['dfd_b']   = pd.cut(df['dfd'],          bins=[-1,9,19,29,39], labels=['0-9','10-19','20-29','30-39'])
    df['price_b'] = pd.cut(df['target_price'], bins=[-np.inf, q[0], q[1], np.inf], labels=['low','mid','high'])
    interaction = df.groupby(['dfd_b', 'price_b']).agg(
        rmse=('residual', lambda x: round(float(np.sqrt((x**2).mean())), 2)),
        mean_res=('residual', lambda x: round(float(x.mean()), 2)),
        n=('residual', 'count'),
    ).reset_index()
    interaction['dfd_b']   = interaction['dfd_b'].astype(str)
    interaction['price_b'] = interaction['price_b'].astype(str)
    stats['dfd_x_price_interaction'] = interaction.to_dict('records')

    # ── Market competition effect ────────────────────────────────────────────
    if 'market_n_airlines' in df.columns:
        n_max = int(df['market_n_airlines'].max()) + 1
        stats['by_n_airlines'] = _bucket_analysis(
            df, 'market_n_airlines',
            bins=[-0.5 + i for i in range(n_max + 1)],
            labels=[str(i) for i in range(1, n_max + 1)],
        )

    return stats


# ────────────────────────────────────────────────────────────────────────────
# Analyst Agent
# ────────────────────────────────────────────────────────────────────────────

class AnalystAgent:
    """
    Uses residual statistics + Claude reasoning to generate feature
    engineering recommendations for the engineer agent.
    """

    def __init__(self, model_name: str = "claude-sonnet-4-6"):
        self.client     = anthropic.Anthropic()
        self.model_name = model_name

    def analyze(
        self,
        test_data: pd.DataFrame,
        predictions: np.ndarray,
        feature_cols: list[str],
        feature_importances: dict,
        current_model_description: str,
        iteration: int,
        previous_recommendations: list[dict] | None = None,
    ) -> dict:
        """
        Analyze residuals and return structured recommendations.

        Returns:
            {
              "analysis": str,
              "recommendations": [
                {
                  "priority": int,
                  "type": "feature_engineering" | "model_tuning" | "target_transform",
                  "title": str,
                  "idea": str,
                  "rationale": str,
                  "implementation_hint": str,
                }
              ]
            }
        """
        print(f"[Analyst] Computing residual statistics (iteration {iteration})...")
        stats = compute_residual_statistics(test_data, predictions, feature_cols)

        # Top feature importances to give Claude context on what the model already knows
        top_feats = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:20]

        prev_str = ""
        if previous_recommendations:
            prev_titles = [r.get('title', '') for r in previous_recommendations]
            prev_str = (
                f"\n\nPrevious recommendations already tried: {prev_titles}. "
                "Do NOT suggest the same ideas again."
            )

        prompt = f"""You are an expert ML analyst helping improve a flight fare prediction model.

## Model description
{current_model_description}

## Top-20 most important features (already in the model)
{json.dumps([{'feature': k, 'importance': v} for k, v in top_feats], indent=2)}

## Residual analysis (actual - predicted)
Positive residual = model underpredicts (actual > predicted).
Negative residual = model overpredicts (actual < predicted).

{json.dumps(stats, indent=2, default=str)}
{prev_str}

## Your task
Identify the 3–5 most impactful opportunities to reduce RMSE. Focus on:
- Which data slices show systematic bias (consistent over/under prediction)?
- Which patterns suggest missing features or under-represented structure?
- What new signals could help the model in its weakest areas?

For each recommendation be SPECIFIC and ACTIONABLE — describe exactly what
feature to compute and from which columns, not just a vague direction.

Return ONLY valid JSON in this exact structure:
{{
  "analysis": "2-4 sentence summary of the key residual patterns found",
  "recommendations": [
    {{
      "priority": 1,
      "type": "feature_engineering",
      "title": "Short title (5-10 words)",
      "idea": "Specific description of what to build",
      "rationale": "Which residual pattern this addresses and why it helps",
      "implementation_hint": "Pseudocode or specific pandas/numpy logic"
    }}
  ]
}}"""

        print(f"[Analyst] Querying Claude for recommendations...")
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = message.content[0].text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON block
            import re
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            result = json.loads(m.group()) if m else {"analysis": raw, "recommendations": []}

        print(f"[Analyst] Analysis: {result.get('analysis', '')[:200]}")
        print(f"[Analyst] Generated {len(result.get('recommendations', []))} recommendations:")
        for r in result.get('recommendations', []):
            print(f"  [{r.get('priority')}] {r.get('title')}")

        return result
