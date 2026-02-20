"""
Engineer Agent — implements analyst recommendations as new model features.

Responsibilities:
  1. Receive a structured recommendation from the analyst
  2. Read the current best model file as a base
  3. Use Claude to generate new feature engineering code that extends the base
  4. Write a self-contained model file (model_agent_vN.py) that can be run standalone
  5. Execute it and parse RMSE results
  6. Return whether the change improved the model
"""

import json
import os
import re
import subprocess
import sys
import anthropic
from pathlib import Path


TIMEOUT_SECONDS = 600  # 10 min per model run


class EngineerAgent:
    """
    Uses Claude to generate new feature engineering code, integrates it into
    the model pipeline, and evaluates whether it improves RMSE.
    """

    def __init__(self, model_name: str = "claude-sonnet-4-6", workdir: str = "."):
        self.client     = anthropic.Anthropic()
        self.model_name = model_name
        self.workdir    = Path(workdir)

    def implement(
        self,
        recommendation: dict,
        base_model_path: str,
        iteration: int,
        current_best_rmse: float,
    ) -> dict:
        """
        Generate and test a new model version based on the recommendation.

        Returns:
            {
              "success": bool,
              "new_model_path": str,
              "rmse": float,
              "improvement": float,  # positive = better
              "details": str,
            }
        """
        output_path = self.workdir / f"model_agent_v{iteration}.py"

        print(f"[Engineer] Implementing: {recommendation.get('title')}")
        print(f"[Engineer]   Type: {recommendation.get('type')}")

        # ── Load base model code ──────────────────────────────────────────────
        base_code = Path(base_model_path).read_text()

        # ── Ask Claude to generate the modified model ─────────────────────────
        generated_code = self._generate_model_code(
            base_code, recommendation, iteration, output_path.name
        )

        if generated_code is None:
            return {
                "success": False, "new_model_path": None,
                "rmse": current_best_rmse, "improvement": 0.0,
                "details": "Code generation failed.",
            }

        # ── Write and execute ─────────────────────────────────────────────────
        output_path.write_text(generated_code)
        print(f"[Engineer] Wrote {output_path} ({len(generated_code)} chars)")

        result = self._run_model(str(output_path))

        if not result["success"]:
            # Try one self-correction pass
            print(f"[Engineer] Execution failed, attempting self-correction...")
            corrected = self._self_correct(generated_code, result["stderr"], recommendation)
            if corrected:
                output_path.write_text(corrected)
                result = self._run_model(str(output_path))

        if not result["success"]:
            return {
                "success": False, "new_model_path": str(output_path),
                "rmse": current_best_rmse, "improvement": 0.0,
                "details": f"Execution error: {result['stderr'][-500:]}",
            }

        new_rmse   = result["rmse"]
        improvement = current_best_rmse - new_rmse   # positive = better

        print(f"[Engineer] Result: RMSE={new_rmse:.4f}  improvement={improvement:+.4f}")

        return {
            "success":        True,
            "new_model_path": str(output_path),
            "rmse":           new_rmse,
            "improvement":    improvement,
            "details":        result["stdout"][-1500:],
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _generate_model_code(
        self,
        base_code: str,
        recommendation: dict,
        iteration: int,
        filename: str,
    ) -> str | None:
        """Ask Claude to extend the base model with the recommended feature."""

        prompt = f"""You are an expert ML engineer working on a flight fare prediction model.

## Your task
Extend the model by adding the following new feature:

**Title**: {recommendation.get('title')}
**Idea**: {recommendation.get('idea')}
**Rationale**: {recommendation.get('rationale')}
**Implementation hint**: {recommendation.get('implementation_hint')}

## Base model code
Below is the complete current model (model_v4.py). Your job is to generate
a MODIFIED version saved as `{filename}` that adds the new feature.

```python
{base_code}
```

## Rules for the generated code
1. Preserve the exact same `main()` function structure and FINAL SUMMARY print format
2. The FINAL SUMMARY must still print lines like:
   `  ModelName   RMSE  +/-pct%`
   and include the line `Baseline   XX.XXXX  (reference)`
3. Add your new feature in `build_flight_features()` and list it in `get_features()`
4. The new feature must be derived only from columns already in the dataframe or
   from the training set (passed as `train_df`) — no external data
5. Handle NaN values gracefully with `.fillna(0)` or `.fillna(method)`
6. Add a brief comment explaining the new feature
7. Do NOT change the model hyperparameters or training logic
8. The file must be self-contained and runnable with `python3 {filename}`

Return ONLY the complete Python code — no markdown, no explanation."""

        print(f"[Engineer] Asking Claude to generate model code...")
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=8000,
                messages=[{"role": "user", "content": prompt}],
            )
            code = message.content[0].text.strip()

            # Strip markdown if wrapped
            if code.startswith("```python"):
                code = code[9:]
            elif code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            return code.strip()

        except Exception as e:
            print(f"[Engineer] Code generation error: {e}")
            return None

    def _self_correct(self, original_code: str, error_msg: str, recommendation: dict) -> str | None:
        """Ask Claude to fix a code error."""
        prompt = f"""The following Python code produced an error when run. Fix it.

## Error
```
{error_msg[-1000:]}
```

## Code
```python
{original_code}
```

Return ONLY the corrected Python code — no markdown, no explanation.
The feature being added: {recommendation.get('title')}"""

        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=8000,
                messages=[{"role": "user", "content": prompt}],
            )
            code = message.content[0].text.strip()
            if code.startswith("```python"):
                code = code[9:]
            elif code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            return code.strip()
        except Exception as e:
            print(f"[Engineer] Self-correction error: {e}")
            return None

    def _run_model(self, model_path: str) -> dict:
        """Execute a model file and parse RMSE from its stdout."""
        print(f"[Engineer] Running {model_path}...")
        try:
            proc = subprocess.run(
                [sys.executable, model_path],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                cwd=str(self.workdir),
            )
            stdout = proc.stdout
            stderr = proc.stderr

            if proc.returncode != 0:
                return {"success": False, "rmse": None, "stdout": stdout, "stderr": stderr}

            # Parse RMSE from FINAL SUMMARY section
            # Look for the best non-baseline model RMSE
            rmse = self._parse_best_rmse(stdout)
            if rmse is None:
                return {"success": False, "rmse": None, "stdout": stdout,
                        "stderr": "Could not parse RMSE from output"}

            return {"success": True, "rmse": rmse, "stdout": stdout, "stderr": stderr}

        except subprocess.TimeoutExpired:
            return {"success": False, "rmse": None, "stdout": "", "stderr": "Timeout"}
        except Exception as e:
            return {"success": False, "rmse": None, "stdout": "", "stderr": str(e)}

    def _parse_best_rmse(self, stdout: str) -> float | None:
        """
        Parse the best model RMSE from the FINAL SUMMARY output.
        Looks for lines like:  '  ModelName   59.1234   +7.18%'
        and returns the minimum RMSE (excluding baseline).
        """
        # Find FINAL SUMMARY section
        if "FINAL SUMMARY" not in stdout:
            # Fall back to finding any RMSE= pattern
            matches = re.findall(r'RMSE=(\d+\.\d+)', stdout)
            if matches:
                return float(matches[-1])
            return None

        summary_block = stdout.split("FINAL SUMMARY")[-1]
        rmses = []

        for line in summary_block.splitlines():
            # Skip baseline line
            if '(reference)' in line or 'Baseline' in line.split()[0:1]:
                continue
            # Match RMSE value: a float with 4 decimal places
            m = re.search(r'\b(\d{2,3}\.\d{4})\b', line)
            if m:
                rmses.append(float(m.group(1)))

        return min(rmses) if rmses else None
