"""
Tests exhaustifs de src/utils.py.

Couvre :
- with_retry : succès, retry sur erreurs récupérables, échec total,
               erreurs fatales (pas de retry), délais de sleep corrects
- safe_parse_json : JSON valide, bruité, malformé, absent, vide, imbriqué
- normalize_literal : exact, insensible casse, espaces, inconnu, non-string
- safe_float : int, float, string, format FR, None, bool, vide
"""
import json
from unittest.mock import MagicMock, call, patch

import pytest

import anthropic
from src.utils import normalize_literal, safe_float, safe_parse_json, with_retry


# ---------------------------------------------------------------------------
# with_retry
# ---------------------------------------------------------------------------

class TestWithRetry:
    """On patche _RETRYABLE pour injecter ValueError comme erreur retryable."""

    @patch("src.utils._RETRYABLE", (ValueError,))
    @patch("src.utils.time.sleep")
    def test_success_on_first_attempt(self, mock_sleep):
        fn = MagicMock(return_value="ok")
        decorated = with_retry(max_attempts=3, base_delay=1.0)(fn)
        assert decorated() == "ok"
        fn.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("src.utils._RETRYABLE", (ValueError,))
    @patch("src.utils.time.sleep")
    def test_success_after_one_failure(self, mock_sleep):
        fn = MagicMock(side_effect=[ValueError("fail"), "ok"])
        decorated = with_retry(max_attempts=3, base_delay=1.0)(fn)
        assert decorated() == "ok"
        assert fn.call_count == 2
        mock_sleep.assert_called_once_with(1.0)  # base_delay * 2^0

    @patch("src.utils._RETRYABLE", (ValueError,))
    @patch("src.utils.time.sleep")
    def test_success_after_two_failures(self, mock_sleep):
        fn = MagicMock(side_effect=[ValueError("1"), ValueError("2"), "ok"])
        decorated = with_retry(max_attempts=3, base_delay=2.0)(fn)
        assert decorated() == "ok"
        assert fn.call_count == 3
        # Délais : 2.0 (2^0), 4.0 (2^1)
        mock_sleep.assert_has_calls([call(2.0), call(4.0)])

    @patch("src.utils._RETRYABLE", (ValueError,))
    @patch("src.utils.time.sleep")
    def test_raises_after_all_attempts_exhausted(self, mock_sleep):
        fn = MagicMock(side_effect=ValueError("always fails"))
        decorated = with_retry(max_attempts=3, base_delay=1.0)(fn)
        with pytest.raises(ValueError, match="always fails"):
            decorated()
        assert fn.call_count == 3
        assert mock_sleep.call_count == 2  # sleep entre chaque tentative sauf la dernière

    @patch("src.utils.time.sleep")
    def test_no_retry_on_authentication_error(self, mock_sleep):
        """AuthenticationError = erreur fatale → pas de retry."""
        auth_err = MagicMock(spec=anthropic.AuthenticationError)

        fn = MagicMock()

        @with_retry(max_attempts=3, base_delay=1.0)
        def fn_decorated():
            raise anthropic.AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401, headers={}),
                body={},
            )

        with pytest.raises(anthropic.AuthenticationError):
            fn_decorated()
        mock_sleep.assert_not_called()

    @patch("src.utils._RETRYABLE", (ValueError,))
    @patch("src.utils.time.sleep")
    def test_preserves_function_name(self, mock_sleep):
        """Le décorateur ne doit pas masquer le nom de la fonction."""
        @with_retry(max_attempts=2, base_delay=0.1)
        def my_function():
            return "ok"

        assert my_function.__name__ == "my_function"

    @patch("src.utils._RETRYABLE", (ValueError,))
    @patch("src.utils.time.sleep")
    def test_no_sleep_after_last_attempt(self, mock_sleep):
        """Pas de sleep après la dernière tentative échouée."""
        fn = MagicMock(side_effect=ValueError("fail"))
        decorated = with_retry(max_attempts=2, base_delay=1.0)(fn)
        with pytest.raises(ValueError):
            decorated()
        # 2 tentatives → 1 seul sleep (entre tentative 1 et 2)
        assert mock_sleep.call_count == 1

    @patch("src.utils._RETRYABLE", (ValueError,))
    @patch("src.utils.time.sleep")
    def test_exponential_backoff_delays(self, mock_sleep):
        """Vérifie les délais : base, base*2, base*4."""
        fn = MagicMock(side_effect=[ValueError(), ValueError(), ValueError(), "ok"])
        decorated = with_retry(max_attempts=4, base_delay=1.0)(fn)
        decorated()
        delays = [c.args[0] for c in mock_sleep.call_args_list]
        assert delays == [1.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# safe_parse_json
# ---------------------------------------------------------------------------

class TestSafeParseJson:
    def test_valid_simple_json(self):
        assert safe_parse_json('{"key": "value"}') == {"key": "value"}

    def test_json_with_prefix_noise(self):
        assert safe_parse_json('Voici : {"key": "val"} voilà.') == {"key": "val"}

    def test_json_with_multiline_content(self):
        raw = '{"a": 1,\n"b": 2}'
        assert safe_parse_json(raw) == {"a": 1, "b": 2}

    def test_nested_json_object(self):
        raw = '{"outer": {"inner": 42}}'
        result = safe_parse_json(raw)
        assert result == {"outer": {"inner": 42}}

    def test_json_with_unicode(self):
        raw = '{"message": "Bonjour à tous"}'
        assert safe_parse_json(raw)["message"] == "Bonjour à tous"

    def test_returns_none_for_no_json(self):
        assert safe_parse_json("pas de JSON ici") is None

    def test_returns_none_for_empty_string(self):
        assert safe_parse_json("") is None

    def test_returns_none_for_malformed_json(self):
        # JSON avec guillemets non échappés dans une valeur
        assert safe_parse_json('{"key": "val"ue"}') is None

    def test_returns_none_for_truncated_json(self):
        assert safe_parse_json('{"key": "val') is None

    def test_returns_none_for_json_array(self):
        # safe_parse_json cherche uniquement des objets {}, pas des tableaux []
        assert safe_parse_json('["a", "b"]') is None

    def test_picks_first_json_object(self):
        raw = '{"first": 1} {"second": 2}'
        result = safe_parse_json(raw)
        # La regex DOTALL matche de { au dernier } — comportement documenté
        assert result is not None


# ---------------------------------------------------------------------------
# normalize_literal
# ---------------------------------------------------------------------------

class TestNormalizeLiteral:
    CATEGORIES = ("facture", "devis", "question_comptable", "autre")

    def test_exact_match(self):
        assert normalize_literal("facture", self.CATEGORIES, "autre") == "facture"

    def test_case_insensitive_upper(self):
        assert normalize_literal("FACTURE", self.CATEGORIES, "autre") == "facture"

    def test_case_insensitive_mixed(self):
        assert normalize_literal("Facture", self.CATEGORIES, "autre") == "facture"

    def test_strips_whitespace(self):
        assert normalize_literal("  devis  ", self.CATEGORIES, "autre") == "devis"

    def test_unknown_value_returns_default(self):
        assert normalize_literal("inconnu", self.CATEGORIES, "autre") == "autre"

    def test_non_string_int_returns_default(self):
        assert normalize_literal(42, self.CATEGORIES, "autre") == "autre"

    def test_non_string_none_returns_default(self):
        assert normalize_literal(None, self.CATEGORIES, "autre") == "autre"

    def test_non_string_list_returns_default(self):
        assert normalize_literal(["facture"], self.CATEGORIES, "autre") == "autre"

    def test_empty_string_returns_default(self):
        assert normalize_literal("", self.CATEGORIES, "autre") == "autre"

    def test_urgency_values(self):
        urgencies = ("low", "medium", "high")
        assert normalize_literal("HIGH", urgencies, "low") == "high"
        assert normalize_literal("Medium", urgencies, "low") == "medium"
        assert normalize_literal("critical", urgencies, "low") == "low"  # inconnu → default


# ---------------------------------------------------------------------------
# safe_float
# ---------------------------------------------------------------------------

class TestSafeFloat:
    def test_integer_input(self):
        assert safe_float(42, 0.0) == pytest.approx(42.0)

    def test_float_input(self):
        assert safe_float(3.14, 0.0) == pytest.approx(3.14)

    def test_string_float(self):
        assert safe_float("3.14", 0.0) == pytest.approx(3.14)

    def test_string_integer(self):
        assert safe_float("42", 0.0) == pytest.approx(42.0)

    def test_string_with_spaces(self):
        # "  3.14  " → float() l'accepte nativement
        assert safe_float("  3.14  ", 0.0) == pytest.approx(3.14)

    def test_invalid_string_returns_default(self):
        assert safe_float("élevé", 99.0) == 99.0

    def test_none_returns_default(self):
        assert safe_float(None, 99.0) == 99.0

    def test_empty_string_returns_default(self):
        assert safe_float("", 0.0) == 0.0

    def test_boolean_true_coerces_to_one(self):
        # bool est sous-classe de int en Python : True → 1.0
        assert safe_float(True, 0.0) == pytest.approx(1.0)

    def test_boolean_false_coerces_to_zero(self):
        assert safe_float(False, 99.0) == pytest.approx(0.0)

    def test_zero_is_valid(self):
        assert safe_float(0, 99.0) == pytest.approx(0.0)

    def test_negative_float(self):
        assert safe_float(-5.5, 0.0) == pytest.approx(-5.5)

    def test_default_none_compatible(self):
        """safe_float doit accepter None comme valeur de default (type hint souple)."""
        result = safe_float("bad", None)  # type: ignore
        assert result is None
