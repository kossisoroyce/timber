"""Generate a sample XGBoost model JSON for testing the CLI."""
import json
from pathlib import Path

model = {
    "learner": {
        "learner_model_param": {
            "num_feature": "4",
            "num_class": "0",
            "base_score": "0.5",
        },
        "gradient_booster": {
            "model": {
                "gbtree_model_param": {"num_trees": "3"},
                "trees": [
                    {
                        "split_indices": [0, 1, 2, 0, 0, 0, 0],
                        "split_conditions": [0.5, 0.3, 0.7, 0.1, -0.2, 0.15, -0.1],
                        "left_children": [1, 3, 5, -1, -1, -1, -1],
                        "right_children": [2, 4, 6, -1, -1, -1, -1],
                        "default_left": [1, 1, 1, 1, 1, 1, 1],
                    },
                    {
                        "split_indices": [2, 3, 0, 0, 0, 0, 0],
                        "split_conditions": [0.6, 0.4, 0.8, 0.05, -0.15, 0.12, -0.08],
                        "left_children": [1, 3, 5, -1, -1, -1, -1],
                        "right_children": [2, 4, 6, -1, -1, -1, -1],
                        "default_left": [1, 1, 1, 1, 1, 1, 1],
                    },
                    {
                        "split_indices": [1, 0, 3, 0, 0, 0, 0],
                        "split_conditions": [0.55, 0.35, 0.9, 0.08, -0.12, 0.18, -0.05],
                        "left_children": [1, 3, 5, -1, -1, -1, -1],
                        "right_children": [2, 4, 6, -1, -1, -1, -1],
                        "default_left": [1, 1, 1, 1, 1, 1, 1],
                    },
                ],
                "tree_info": [0, 0, 0],
            },
            "gbtree_model_param": {"num_trees": "3"},
        },
        "objective": {"name": "reg:squarederror"},
        "feature_names": ["age", "income", "score", "tenure"],
    },
    "version": [2, 0, 0],
}

out = Path(__file__).parent / "sample_model.json"
out.write_text(json.dumps(model, indent=2))
print(f"Wrote {out}")
