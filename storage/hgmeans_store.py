import os
import json


class HGMeansStore(object):
    def __init__(self, checkpoint_dir_path, eval_dir_path):
        self.checkpoint_dir_path = checkpoint_dir_path
        os.makedirs(checkpoint_dir_path, exist_ok=True)
        self.eval_dir_path = eval_dir_path
        os.makedirs(eval_dir_path, exist_ok=True)

    def save_checkpoint(self, key, checkpoint):
        file_path = os.path.join(self.checkpoint_dir_path, f"{key}.json")
        with open(file_path, "w") as f:
            json.dump(checkpoint, f,)

    def load_checkpoint(self, key):
        try:
            file_path = os.path.join(self.checkpoint_dir_path, f"{key}.json")
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def save_evaluation(self, key, evaluation):
        file_path = os.path.join(self.eval_dir_path, f"{key}.json")
        with open(file_path, "w") as f:
            json.dump(evaluation, f)

    def load_evaluation(self, key):
        file_path = os.path.join(self.eval_dir_path, f"{key}.json")
        with open(file_path, "r") as f:
            return json.load(f)
