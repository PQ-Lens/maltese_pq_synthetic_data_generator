import os
import tempfile
import unittest
import json
from unittest.mock import patch

import pandas as pd
import requests

import pq_synth_core as core
import maltese_pq_synthetic_generator as generator


class FakeResponse:
    def __init__(self, payload, status_code=200, reason="OK"):
        self._payload = payload
        self.status_code = status_code
        self.reason = reason
        self.text = json.dumps(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} {self.reason}", response=self)
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

    @patch("maltese_pq_synthetic_generator.time.sleep")
    @patch("maltese_pq_synthetic_generator.requests.post")
    def test_transient_http_error_is_retried_and_redacted(self, mock_post, mock_sleep):
        mock_post.side_effect = [
            FakeResponse({"error": "temporary"}, status_code=503, reason="Service Unavailable"),
            FakeResponse(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": json.dumps([_a1_row()])
                                    }
                                ]
                            }
                        }
                    ]
                }
            ),
        ]

        with patch.dict(
            os.environ,
            {
                "GOOGLE_API_KEY": "secret-gemini-key",
                "GEMINI_MODEL": "gemini-flash-latest",
            },
            clear=True,
        ):
            df = generator.generate_set(
                "A1",
                n=1,
                provider="gemini",
                model="gemini-flash-latest",
                api_retries=1,
                api_retry_backoff=0,
            )

        self.assertEqual(df.shape[0], 1)
        self.assertEqual(mock_post.call_count, 2)
        self.assertFalse(mock_sleep.called)

    @patch("maltese_pq_synthetic_generator.requests.post")
    def test_exhausted_http_error_does_not_expose_api_key(self, mock_post):
        mock_post.return_value = FakeResponse(
            {"error": "temporary"},
            status_code=503,
            reason="Service Unavailable",
        )

        with patch.dict(
            os.environ,
            {
                "GOOGLE_API_KEY": "secret-gemini-key",
                "GEMINI_MODEL": "gemini-flash-latest",
            },
            clear=True,
        ):
            with self.assertRaises(generator.ProviderRequestError) as ctx:
                generator.call_llm_chat_json(
                    messages=[{"role": "user", "content": "[]"}],
                    provider="gemini",
                    model="gemini-flash-latest",
                    api_retries=0,
                )

        self.assertNotIn("secret-gemini-key", str(ctx.exception))
        self.assertIn("HTTP 503", str(ctx.exception))

    def test_configured_model_ids_include_env_model_lists(self):
        with patch.dict(
            os.environ,
            {
                "OLLAMA_MODELS": "llama3.1:8b,qwen2.5:7b",
                "GOOGLE_API_KEY": "gemini-key",
                "GEMINI_MODELS": "gemini-2.5-flash,gemini-flash-latest",
                "MISTRAL_API_KEY": "mistral-key",
                "MISTRAL_MODELS": "mistral-small-latest",
            },
            clear=True,
        ):
            ids = generator.get_configured_model_ids()

        self.assertIn("ollama:llama3.1:8b", ids)
        self.assertIn("ollama:qwen2.5:7b", ids)
        self.assertIn("gemini:gemini-2.5-flash", ids)
        self.assertIn("mistral:mistral-small-latest", ids)


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

    @patch("maltese_pq_synthetic_generator.requests.post")
    def test_generate_set_ollama(self, mock_post):
        mock_post.return_value = FakeResponse(
            {
                "message": {
                    "content": json.dumps({"rows": [_a1_row()]})
                }
            }
        )

        with patch.dict(
            os.environ,
            {
                "OLLAMA_MODELS": "llama3.1:8b",
                "OLLAMA_BASE_URL": "http://localhost:11434",
            },
            clear=True,
        ):
            df = generator.generate_set("A1", n=1, provider="ollama", model="llama3.1:8b")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 1)
        self.assertListEqual(list(df.columns), generator.QUESTION_COLUMNS)

    @patch("maltese_pq_synthetic_generator.run_and_save")
    def test_run_all_continues_after_category_failure(self, mock_run_and_save):
        success_df = pd.DataFrame([_a1_row()], columns=generator.QUESTION_COLUMNS)
        success_metrics = pd.DataFrame([{"status": "success"}])
        mock_run_and_save.side_effect = [
            RuntimeError("temporary provider failure key=secret"),
            (success_df, success_metrics),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = generator.run_all_and_save(
                n=1,
                categories=["A1", "A2"],
                provider="ollama",
                model="llama3.1:8b",
                output_dir=tmp_dir,
                continue_on_error=True,
            )

            self.assertIn("A2", result)
            self.assertEqual(len(result.failures), 1)
            self.assertEqual(result.failures[0].category_id, "A1")
            self.assertNotIn("key=secret", result.failures[0].error)
            self.assertIn("key=<redacted>", result.failures[0].error)
            self.assertTrue(any(name.endswith("_failed_metrics.csv") for name in os.listdir(tmp_dir)))
            self.assertIsNotNone(result.aggregate_path)
            self.assertTrue(result.aggregate_path.exists())

            aggregate = pd.read_csv(result.aggregate_path)
            self.assertEqual(aggregate.shape[0], 1)
            self.assertEqual(aggregate.loc[0, "Category ID"], "A2")
            self.assertEqual(aggregate.loc[0, "Category Type"], "question")
            self.assertIn("Ministry (EN)", aggregate.columns)

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
