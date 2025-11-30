import pandas as pd
import pytest
from src.features.time_windows import features_time_based

def test_shift_prevents_leakage_and_counts_correctly():
    """
    Verifica se a feature 'num_pix_last_1h' exclui a transação atual (anti-leakage) 
    e se a contagem funciona corretamente para a próxima transação.
    """
    
    # 1. Setup do DataFrame de Teste
    # Três transações do mesmo usuário:
    # T0: 00:00:00
    # T1: 00:30:00 (Dentro da janela de 1h de T0)
    # T2: 01:30:00 (Fora da janela de 1h de T0, mas dentro da de T1)
    
    df_test = pd.DataFrame({
        "user_id": [1, 1, 1],
        "timestamp": pd.to_datetime(["2025-01-01 00:00:00", 
                                      "2025-01-01 00:30:00", 
                                      "2025-01-01 01:30:00"]),
        "amount": [100, 200, 300]
    }).sort_values('timestamp').reset_index(drop=True)
    
    # 2. Aplica Feature Engineering
    df_result = features_time_based(df_test)

    # 3. Assertions (Validações Cruciais)
    
    # Validação A (Anti-Leakage): A primeira transação deve ter 0, mesmo que não haja 1h de histórico.
    # Se fosse 1, haveria vazamento.
    assert df_result["numpixlast_1h"].iloc[0] == 0, "Falha no Anti-Leakage: T0 deve ser 0."
    
    # Validação B (Contagem Correta): A segunda transação (T1, 00:30:00) deve contar 1 evento anterior (T0).
    assert df_result["numpixlast_1h"].iloc[1] == 1, "Falha na Contagem: T1 deve contar 1 evento."

    # Validação C (Contagem Acumulada): A terceira transação (T2, 01:30:00) deve contar 2 eventos anteriores (T0 e T1).
    # Pois T0 (00:00:00) e T1 (00:30:00) ainda estão dentro da janela de 1 hora (00:30:00 - 01:30:00).
    assert df_result["numpixlast_1h"].iloc[2] == 2, "Falha na Contagem: T2 deve contar 2 eventos."

