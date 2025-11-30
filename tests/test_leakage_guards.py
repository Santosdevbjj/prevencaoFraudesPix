import pandas as pd
import pytest
from src.features.validators import check_no_leakage # Assumindo que check_no_leakage existe

# Assumindo que check_no_leakage() verifica a presença de 'is_fraud' ou 'target_encoded_...'
# e que levanta um ValueError se encontrar alguma delas.

def test_leakage_present_should_fail():
    """
    CENÁRIO 1: Testa se o validador FALHA quando a variável alvo ('is_fraud') 
    está indevidamente presente nas features.
    """
    
    # Simula um DataFrame que tem a target, que é um vazamento direto
    df_leak = pd.DataFrame({
        "feature_A": [1, 2],
        "is_fraud": [0, 1]  # Vazamento Direto
    })

    # O teste SÓ passa se a função levantar um ValueError (ou a exceção que você usa)
    with pytest.raises(ValueError, match="Variável 'is_fraud' encontrada"):
        check_no_leakage(df_leak)

def test_no_leakage_should_pass():
    """
    CENÁRIO 2: Testa se o validador PASSA (não levanta exceção) quando 
    apenas features legítimas estão presentes.
    """
    
    # Simula um DataFrame limpo
    df_clean = pd.DataFrame({
        "feature_A": [1, 2],
        "time_since_last_pix": [100, 200]
    })
    
    # A função deve rodar sem levantar exceções
    try:
        check_no_leakage(df_clean)
    except Exception as e:
        pytest.fail(f"Validador levantou exceção inesperada em dados limpos: {e}")
