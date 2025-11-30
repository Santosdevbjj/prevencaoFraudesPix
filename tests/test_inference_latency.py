import time
import numpy as np
import pandas as pd
import joblib
import pytest
from src.modeling.inference import infer_single # Assume que infer_single está correto

# 1. Mock de Dados de Entrada para o Teste
@pytest.fixture
def sample_feature_row():
    """Cria uma linha de dados simulada com 12 features numéricas (exemplo)."""
    # Esta linha deve simular as features que o seu modelo XGBoost espera.
    # Exemplo simplificado (você deve ajustá-lo ao número exato de colunas do seu X):
    return pd.Series({
        "amount": 100.50,
        "senderaccountagedays": 300,
        "internalriskscore": 750.0,
        "maxcreditlimit": 15000.0,
        "historicaldefaultrate": 0.01,
        "timesincelastlogin": 3600,
        "changeofdevice": 0,
        "isfirsttimerecipient": 1,
        "numpixlast1h": 5,
        "avgamountlast7d": 500.0,
        "stdamountlast30d": 1000.0,
        "numuniquerecipients24h": 3,
    })

# 2. Mock do Modelo (para não depender de um arquivo salvo)
# Em um teste real, você carregaria o modelo salvo, mas para isolamento,
# podemos simular um modelo leve ou garantir que um modelo real exista.
# VAMOS ASSUMIR QUE O ARQUIVO models/artifacts/model_xgb.joblib EXISTE E É LEVE.

def test_latency_under_50ms(sample_feature_row):
    """Testa se a latência média da inferência é menor que 50 ms."""
    
    # 1. CARREGAR O ARTEFATO DO MODELO (Correção de 'job' para 'joblib')
    # O caminho deve ser ajustado, mas o ideal é que seja o modelo treinado.
    try:
        model = joblib.load('models/artifacts/model_xgb.joblib')
    except FileNotFoundError:
        # Se o modelo não foi treinado ainda, pule o teste.
        pytest.skip("Modelo não encontrado. Treine o modelo primeiro (python src/pipelines/trainandeval.py).")
        return

    N_RUNS = 100  # Número de execuções para média
    latencies_ms = []

    # 2. MEDIR LATÊNCIA
    # Simula o cálculo de features columns que são usadas no modelo
    feature_cols = sample_feature_row.index.tolist()

    for _ in range(N_RUNS):
        # infere_single retorna (probabilidade, tempo_em_ms)
        _, dt_ms = infer_single(model, sample_feature_row, feature_cols)
        latencies_ms.append(dt_ms)

    # 3. VALIDAÇÃO
    average_latency = np.mean(latencies_ms)
    p95_latency = np.percentile(latencies_ms, 95)
    
    # Assert do Requisito de Negócio (Latência < 50ms)
    print(f"\nLatência Média: {average_latency:.2f} ms")
    print(f"Latência P95: {p95_latency:.2f} ms")
    
    # Usamos o P95 para o teste ser mais rigoroso e simular a performance da maioria dos usuários.
    # Você pode escolher o P50 (média) ou P95.
    assert p95_latency < 50.0, f"Latência P95 ({p95_latency:.2f} ms) excede 50 ms."

