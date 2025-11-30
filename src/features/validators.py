# src/features/validators.py
import pandas as pd


def check_no_leakage(df: pd.DataFrame) -> None:
    """
    Valida se o DataFrame não contém vazamento de informação (data leakage).

    O objetivo é garantir que variáveis que não deveriam estar presentes
    nas features — como a variável alvo 'is_fraud' ou outras colunas
    derivadas diretamente dela — não estejam incluídas no conjunto de dados
    usado para treinamento ou inferência.

    Args:
        df (pd.DataFrame): DataFrame contendo as features e possivelmente a
                           variável alvo.

    Raises:
        ValueError: Se a coluna 'is_fraud' ou qualquer outra coluna proibida
                    for encontrada nas features.
    """
    # Exemplo de checagem simples: verificar se 'is_fraud' está presente
    if "is_fraud" in df.columns:
        raise ValueError(
            "Data leakage detectado: coluna 'is_fraud' não deve estar nas features."
        )
