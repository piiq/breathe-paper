"""Inference helpers."""
import os
from typing import Dict, List, Tuple
import pickle as pkl  # nosec

import torch

import soundfile as sf
import numpy as np

from breathe.models import densenet
from breathe.audio_utils import trim_and_norm, get_5sec_clips, extract_specs_for_clip
from breathe.dataloaders import datasetnormal


class Ensemble3(object):
    """Ensemble of 3 Densenet models."""

    ENSEMBLE_WEIGHTS_PATH = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    )

    PREDICTION_PICKLE_PATH = os.path.join(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "models", "prediction.pkl")
        )
    )

    DEVICE = torch.device("cpu")  # pylint: disable=no-member

    models: Dict

    def __init__(self, models: List[Tuple[str, str]]) -> None:

        if len(models) != 3:
            print("Failed to initialize a 3 model Ensemble")
            return

        self.models = {}

        for model in models:
            model_name, model_file = model

            checkpoint_path = os.path.join(self.ENSEMBLE_WEIGHTS_PATH, model_file)
            densenet_model = densenet.DenseNet("ESC", False).to(self.DEVICE)
            checkpoint = torch.load(checkpoint_path, map_location=self.DEVICE)
            densenet_model.load_state_dict(checkpoint["model"])
            densenet_model.eval()

            self.models[model_name] = densenet_model

    def get_prediction(
        self, clips: List, sample_rate: int, selected_model: str
    ) -> List:
        """Get predictions for each clip.

        Parameters
        ----------
        clips : List
            A list of audio clips
        sample_rate : int
            Sampling rate of the audio file
        selected_model : str
            Densenet model key

        Returns
        -------
        List
            A list of predicted classes for each clip.
        """
        values = []
        for clip in clips:
            clip, specs = extract_specs_for_clip(
                clip, sample_rate, 0, None, [0.025, 0.1, 0.175]
            )
            new_entry = {}
            new_entry["audio"] = clip.numpy()
            new_entry["values"] = np.array(specs)
            new_entry["target"] = 3
            values.append(new_entry)

        if os.path.exists(self.PREDICTION_PICKLE_PATH):
            os.remove(self.PREDICTION_PICKLE_PATH)
        with open(self.PREDICTION_PICKLE_PATH, "wb") as handler:
            pkl.dump(values, handler, protocol=pkl.HIGHEST_PROTOCOL)

        prediction_dataloader = datasetnormal.fetch_dataloader(
            self.PREDICTION_PICKLE_PATH, "ESC", 100, 4
        )

        predictions = []
        with torch.no_grad():
            for _, data in enumerate(prediction_dataloader):
                inputs = data[0].to(self.DEVICE)

                outputs = self.models[selected_model](inputs)

                _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
                predictions.append(predicted.tolist())
        return predictions

    def infer_file(self, file: str):
        """Run ensemble inference on a file.

        Parameters
        ----------
        file : str
            Path to an audio file
        """
        try:
            sample_waveform, sample_rate = sf.read(file, dtype="float32")
            sample_waveform = sample_waveform.T
        except Exception as e:
            return {"success": False, "error": str(e)}

        try:
            tn_sample = trim_and_norm(sample_waveform)
        except Exception as e:
            return {"success": False, "error": str(e)}

        try:
            clips = get_5sec_clips(tn_sample, sample_rate)
        except Exception as e:
            return {"success": False, "error": str(e)}

        try:
            response = {}
            for model_name in self.models:
                predictions = self.get_prediction(clips, sample_rate, model_name)
                predictions_count = {}
                for value in set(predictions[0]):
                    predictions_count[value] = predictions[0].count(value)
                file_prediction = list(predictions_count.keys())[
                    list(predictions_count.values()).index(
                        max(list(predictions_count.values()))
                    )
                ]  # keywithmaxval
                response[model_name] = {
                    "file_prediction": file_prediction,
                    "clip_predictions": predictions,
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

        return response
