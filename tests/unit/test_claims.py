"""Tests for splitting answers into claims and recognising abstentions."""

from __future__ import annotations

import pytest

from aegis.domain.services.claims import (
    grounding_figures,
    is_abstention,
    split_claims,
    split_premises,
    strip_reply_word,
    ungrounded_figures,
)


class TestSplitClaims:
    def test_one_claim_per_sentence(self) -> None:
        text = "Tiene 15 días para devolver un colchón. Debe presentar la factura de compra."

        assert split_claims(text) == [
            "Tiene 15 días para devolver un colchón.",
            "Debe presentar la factura de compra.",
        ]

    @pytest.mark.parametrize(
        "text",
        [
            "Cada punto vale Q0.10 en compras futuras.",
            "El nivel Plata da 1.5 puntos por cada Q10.",
            "Tienda Modelo, S.A. es una empresa ficticia del ejemplo.",
        ],
    )
    def test_a_full_stop_inside_a_sentence_does_not_split_it(self, text: str) -> None:
        assert split_claims(text) == [text]

    def test_spanish_opening_marks_start_a_sentence(self) -> None:
        text = "Puede apartarlo durante 24 horas. ¿Necesita algo más de la tienda?"

        assert len(split_claims(text)) == 2

    def test_list_items_are_separate_claims(self) -> None:
        text = (
            "Se aceptan estos pagos:\n"
            "- Efectivo en cualquier tienda.\n"
            "- Tarjetas Visa y Mastercard."
        )

        assert split_claims(text) == [
            "Se aceptan estos pagos:",
            "Efectivo en cualquier tienda.",
            "Tarjetas Visa y Mastercard.",
        ]

    def test_a_fragment_too_short_to_judge_joins_the_next_claim(self) -> None:
        assert split_claims("Sí. Puede comprar como invitado en la tienda en línea.") == [
            "Sí. Puede comprar como invitado en la tienda en línea."
        ]

    def test_a_short_trailing_fragment_joins_the_previous_claim(self) -> None:
        assert split_claims("Los puntos vencen a los 12 meses. ¡Gracias!") == [
            "Los puntos vencen a los 12 meses. ¡Gracias!"
        ]

    def test_a_short_answer_is_kept_whole(self) -> None:
        assert split_claims("No.") == ["No."]

    def test_punctuation_and_blank_lines_carry_no_claim(self) -> None:
        assert split_claims("\n\n  ...  \n- \n") == []


class TestIsAbstention:
    @pytest.mark.parametrize(
        "text",
        [
            "El contexto proporcionado no contiene información suficiente para responder esta "
            "pregunta.",
            "EL CONTEXTO NO CONTIENE INFORMACION SUFICIENTE.",
            "The provided context does not contain enough information to answer this question.",
        ],
    )
    def test_the_prompted_refusals_are_abstentions(self, text: str) -> None:
        assert is_abstention(text)

    def test_an_answer_is_not_an_abstention(self) -> None:
        assert not is_abstention("Los domingos las tiendas abren de 10:00 a 19:00.")

    def test_a_refusal_followed_by_claims_is_still_judged(self) -> None:
        text = (
            "El contexto no contiene información suficiente. Sin embargo, la tienda abre a las 9. "
            "También tiene parqueo gratuito. Y entrega a domicilio los domingos."
        )

        assert not is_abstention(text)


class TestSplitPremises:
    def test_hard_line_breaks_inside_a_sentence_are_rejoined(self) -> None:
        text = (
            "El plazo es de 30 días calendario siguientes\na la fecha de compra. Sin excepciones."
        )

        assert split_premises(text) == [
            "El plazo es de 30 días calendario siguientes a la fecha de compra.",
            "Sin excepciones.",
        ]

    def test_table_rows_are_premises_of_their_own(self) -> None:
        text = "Horario de las tiendas\nLunes a sábado | 9:00 a 20:00\nDomingo | 10:00 a 19:00"

        assert split_premises(text) == [
            "Horario de las tiendas",
            "Lunes a sábado: 9:00 a 20:00",
            "Domingo: 10:00 a 19:00",
        ]

    def test_short_sentences_are_kept_whole_and_separate(self) -> None:
        assert split_premises("Sí. No hay entregas los domingos.") == [
            "Sí.",
            "No hay entregas los domingos.",
        ]


class TestStripReplyWord:
    @pytest.mark.parametrize(
        ("claim", "expected"),
        [
            ("No, the stores are closed on 25 December.", "The stores are closed on 25 December."),
            ("Sí, se aceptan transferencias bancarias.", "Se aceptan transferencias bancarias."),
            ("Si, se aceptan transferencias bancarias.", "Se aceptan transferencias bancarias."),
            ("No. Seasonal products have 7 days.", "Seasonal products have 7 days."),
            ("YES: points expire after 12 months.", "Points expire after 12 months."),
        ],
    )
    def test_a_leading_reply_word_is_dropped(self, claim: str, expected: str) -> None:
        assert strip_reply_word(claim) == expected

    @pytest.mark.parametrize(
        "claim",
        [
            "No hay entregas los domingos.",
            "Nothing is delivered on Sundays.",
            "Siempre hay entregas los lunes.",
            "Las tiendas cierran a las 20:00.",
        ],
    )
    def test_a_negation_or_an_ordinary_word_is_kept(self, claim: str) -> None:
        assert strip_reply_word(claim) == claim

    def test_a_claim_is_kept_when_too_little_would_remain(self) -> None:
        assert strip_reply_word("No, gracias.") == "No, gracias."


class TestFigures:
    def _ungrounded(self, claim: str, *context: str) -> list[str]:
        return ungrounded_figures(claim, grounding_figures(context))

    def test_a_figure_the_context_states_is_grounded(self) -> None:
        assert (
            self._ungrounded("El plazo es de 15 días.", "Para colchones el plazo es de 15 días.")
            == []
        )

    def test_a_figure_the_context_never_states_is_named(self) -> None:
        assert self._ungrounded("El plazo es de 60 días.", "El plazo es de 30 días.") == ["60"]

    def test_a_figure_from_another_case_is_left_to_the_nli_model(self) -> None:
        """30 is in the context, for another case; only an absent figure is caught here."""
        context = "El plazo general es de 30 días; para colchones, 15 días."
        assert self._ungrounded("Para colchones el plazo es de 30 días.", context) == []

    @pytest.mark.parametrize(
        ("claim", "context"),
        [
            ("Compras de Q1.000 o más.", "Compras de Q1,000 o más."),
            ("Cada punto vale Q0,10.", "Cada punto vale Q0.10."),
            ("Each point is worth Q0.1.", "Cada punto vale Q0.10."),
            ("Hasta el 50% del valor.", "Hasta el 50 % del valor."),
            ("Pedidos de más de Q20000.", "Pedidos de más de Q20,000."),
            ("Requires two references: 2 of them.", "dos referencias comerciales"),
            ("Phone support closes at 8:00 p.m.", "de 8:00 a 20:00"),
            ("Stores close at 8 pm on weekdays.", "Lunes a sábado: 9:00 a 20:00"),
            ("Stores close at 20:00 on weekdays.", "Lunes a sábado: 9:00 a 20:00"),
            ("A warranty of 12-month duration.", "12 meses de garantía"),
        ],
    )
    def test_the_same_value_written_another_way_is_grounded(self, claim: str, context: str) -> None:
        assert self._ungrounded(claim, context) == []

    def test_a_thousands_separator_is_not_read_as_a_decimal(self) -> None:
        """Q2,000 is two thousand, so it does not ground a claim of 2 days."""
        assert self._ungrounded("Takes 2 business days.", "Pedidos mayores de Q2,000.") == ["2"]

    def test_a_number_word_in_the_claim_is_not_checked(self) -> None:
        assert self._ungrounded("Coverage grows by five years.", "uno, dos o tres años") == []

    def test_a_claim_without_figures_has_none_to_ground(self) -> None:
        assert self._ungrounded("Las plantas no se devuelven.", "Nada aquí.") == []

    def test_every_text_given_grounds_figures(self) -> None:
        grounded = grounding_figures(["El envío cuesta Q55.", "¿Cuánto cuesta un pedido de Q300?"])
        assert ungrounded_figures("Su pedido de Q300 paga Q55.", grounded) == []
