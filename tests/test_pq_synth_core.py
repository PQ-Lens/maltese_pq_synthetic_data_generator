import os
import tempfile
import unittest
import json
from unittest.mock import patch

import pandas as pd

import pq_synth_core as core


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _a1_row():
    return {
        "Date": "13/03/2026",
        "PQ No.": "12345",
        "MP": "Chris Said",
        "Ministry (EN)": "Prime Minister",
        "Title (EN)": "Transport - Traffic Data",
        "Question (EN)": "Can the Minister provide monthly traffic counts?",
        "Answer (EN)": "The requested data will be tabled in the coming sitting.",
    }


class ProviderResolutionTests(unittest.TestCase):
    def test_default_provider_is_gemini(self):
        with patch.dict(os.environ, {}, clear=True):
            provider, model = core.resolve_provider_and_model()
        self.assertEqual(provider, "gemini")
        self.assertEqual(model, "gemini-flash-latest")

    def test_function_arg_overrides_env_provider(self):
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "gemini",
                "MISTRAL_MODEL": "mistral-small-latest",
            },
            clear=True,
        ):
            provider, model = core.resolve_provider_and_model(provider="mistral")
        self.assertEqual(provider, "mistral")
        self.assertEqual(model, "mistral-small-latest")

    def test_unknown_provider_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                core.resolve_provider_and_model(provider="unknown")

    def test_model_arg_overrides_env_model(self):
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "gemini",
                "GEMINI_MODEL": "gemini-legacy",
            },
            clear=True,
        ):
            _, model = core.resolve_provider_and_model(model="gemini-flash-latest")
        self.assertEqual(model, "gemini-flash-latest")

    def test_missing_selected_provider_key_fails_early(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "mistral"}, clear=True):
            with self.assertRaises(ValueError):
                core.call_llm_chat_json(
                    messages=[{"role": "user", "content": "[]"}],
                    provider="mistral",
                    model="mistral-small-latest",
                )


class GenerationTests(unittest.TestCase):
    @patch("pq_synth_core.requests.post")
    def test_generate_set_gemini(self, mock_post):
        mock_post.return_value = FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        "["
                                        "{"
                                        '"Date":"13/03/2026",'
                                        '"PQ No.":"12345",'
                                        '"MP":"Chris Said",'
                                        '"Ministry (EN)":"Prime Minister",'
                                        '"Title (EN)":"Transport - Traffic Data",'
                                        '"Question (EN)":"Can the Minister provide monthly traffic counts?",'
                                        '"Answer (EN)":"The requested data will be tabled in the coming sitting."'
                                        "}"
                                        "]"
                                    )
                                }
                            ]
                        }
                    }
                ]
            }
        )

        with patch.dict(
            os.environ,
            {
                "GOOGLE_API_KEY": "gemini-key",
                "GEMINI_MODEL": "gemini-flash-latest",
            },
            clear=True,
        ):
            df = core.generate_set("A1", n=1, provider="gemini")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 1)
        self.assertListEqual(list(df.columns), core.QUESTION_COLUMNS)

    @patch("pq_synth_core.requests.post")
    def test_generate_set_mistral(self, mock_post):
        mock_post.return_value = FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "{"
                                '"rows":[{'
                                '"Date":"13/03/2026",'
                                '"PQ No.":"12345",'
                                '"MP":"Chris Said",'
                                '"Question (EN)":"Can the Minister confirm the publication date?",'
                                '"Answer (EN)":"The publication is expected in the coming months."'
                                "}]"
                                "}"
                            )
                        }
                    }
                ]
            }
        )

        with patch.dict(
            os.environ,
            {
                "MISTRAL_API_KEY": "mistral-key",
                "MISTRAL_MODEL": "mistral-small-latest",
            },
            clear=True,
        ):
            df = core.generate_set("B2", n=1, provider="mistral")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 1)
        self.assertListEqual(list(df.columns), core.ANSWER_COLUMNS)

    @patch("pq_synth_core.requests.post")
    def test_run_and_save_writes_csv(self, mock_post):
        payload_text = json.dumps([_a1_row()])
        mock_post.return_value = FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": payload_text
                                }
                            ]
                        }
                    }
                ]
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.dict(
                os.environ,
                {
                    "GOOGLE_API_KEY": "gemini-key",
                    "GEMINI_MODEL": "gemini-flash-latest",
                },
                clear=True,
            ):
                df = core.run_and_save("A1", n=1, provider="gemini", output_dir=tmp_dir)

            self.assertEqual(df.shape[0], 1)
            created_files = os.listdir(tmp_dir)
            self.assertTrue(any(name.startswith("A1_") and name.endswith(".csv") for name in created_files))


if __name__ == "__main__":
    unittest.main()
