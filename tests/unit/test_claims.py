"""Tests for splitting answers into claims and recognising abstentions."""

from __future__ import annotations

import pytest

from aegis.domain.services.claims import is_abstention, split_claims


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
