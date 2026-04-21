"""
agents/experiment.py — ExperimentAgent for running code-based experiments.

Responsible for:
- Generating experiment code via LLM
- AST-based safety checking before execution
- Running experiments in isolated subprocess
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import config
from models.hypothesis import Hypothesis, ResearchGoal
from utils.llm import get_llm_completion, parse_json_response, ensure_str
from utils.safety import check_code_safety

logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    openai = None


class ExperimentAgent:
    """Agent that generates and runs python code to simulate experiments or analyze data"""
    
    def __init__(self, use_local_llm: bool = True):
        self.name = "Experiment"
        self.experiments_run = 0
        self.llm_client = None
        
        if use_local_llm and openai:
            try:
                self.llm_client = config.get_openai_client()
            except Exception:
                self.llm_client = None

    async def run_experiment(self, hypothesis: Hypothesis, goal: ResearchGoal) -> str:
        if not self.llm_client:
            return "Simulation skipped: No LLM available for experimental design."
            
        print(f"   🧪 [Experiment Agent] Designing experiment for hypothesis: {hypothesis.title}")
        
        prompt = f"""
        Research Goal: {goal.title}
        Hypothesis: {hypothesis.title}
        Mechanism: {hypothesis.mechanism}
        Predictions: {', '.join(hypothesis.testable_predictions)}
        
        You are an AI Scientist tasked with empirically validating this hypothesis.
        Write a Python 3 script that uses standard libraries (numpy, scipy, sklearn) to run a simulation or statistical test for the predictions of this hypothesis. 
        It MUST be completely self-contained, without assuming external files exist unless you scrape them. Do not use UI libraries.
        It MUST print a clear summary of the results to stdout at the end, concluding whether the data supports the hypothesis.
        
        Output ONLY the python code inside a ```python block. No other explanation.
        """
        
        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                json_mode=False
            )
            content = response.choices[0].message.content.strip()
            
            code = ""
            if "```python" in content:
                code = content.split("```python")[1].split("```")[0].strip()
            elif "```" in content:
                code = content.split("```")[1].split("```")[0].strip()
            else:
                code = content

            if not code:
                return "Failed to generate experimental code."

            # --- AST Safety Check before execution ---
            is_safe, reason = check_code_safety(code)
            if not is_safe:
                msg = f"⛔ Experiment blocked by safety filter: {reason}"
                print(f"      {msg}")
                hypothesis.experimental_results = msg
                return msg

            print(f"      ✓ Safety check passed. Executing experiment script...")

            import tempfile
            import subprocess

            env = os.environ.copy()
            for key in ('OPENAI_API_KEY', 'NCBI_API_KEY', 'ANTHROPIC_API_KEY'):
                env.pop(key, None)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(code)
                script_path = f.name

            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    [sys.executable, '-S', script_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=env
                )

                output = result.stdout
                if result.stderr:
                    output += f"\nErrors/Warnings:\n{result.stderr}"

                if not output.strip():
                    output = "Script ran successfully but produced no output."
                    
                hypothesis.experimental_results = f"Experimental Results:\n{output[:1500]}"
                self.experiments_run += 1
                return hypothesis.experimental_results
            except subprocess.TimeoutExpired:
                hypothesis.experimental_results = "Experiment simulation timed out after 30 seconds."
                return hypothesis.experimental_results
            finally:
                if os.path.exists(script_path):
                    os.remove(script_path)
                    
        except Exception as e:
            hypothesis.experimental_results = f"Experiment implementation failed: {e}"
            return hypothesis.experimental_results


__all__ = ["ExperimentAgent"]
