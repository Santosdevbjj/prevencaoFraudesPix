import pandas as pd
import argparse
from pathlib import Path

# Importação dos módulos do projeto
from src.data.simulate_transactions import simulate
from src.features.time_windows import features_time_based
from src.features.categorical import encode_categorical
from src.utils.io import write_parquet

def build_dataset(nrows: int, fraud_rate: float, output_path: str):
    """
    Orquestra a pipeline completa de criação do dataset:
    Simulação -> Feature Engineering Temporal -> Codificação Categórica.
    
    Args:
        nrows (int): Número de transações a simular.
        fraud_rate (float): Proporção de fraude no dataset (ex: 0.008).
        output_path (str): Caminho completo para salvar o arquivo Parquet final.
    """
    print(f"Iniciando simulação de {nrows:,} transações...")
    
    # 1. Simulação dos Dados Brutos
    df = simulate(nrows=nrows, fraud_rate=fraud_rate)
    
    print("Iniciando Feature Engineering Temporal (Rolling Windows)...")
    # 2. Feature Engineering Temporal (Anti-Leakage)
    df = features_time_based(df)
    
    print("Iniciando Codificação Categórica...")
    # 3. Codificação Categórica
    df = encode_categorical(df)
    
    print(f"Dataset final criado com {len(df.columns)} colunas.")
    
    # 4. Escrita do Arquivo Final
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_parquet(df, output_path)
    
    print(f"Dataset salvo com sucesso em: {output_path}")


if __name__ == "__main__":
    # Define o parser para receber argumentos de linha de comando
    parser = argparse.ArgumentParser(
        description="Orquestra a criação do dataset de fraude do início ao fim."
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=200_000,
        help="Número de linhas (transações) a simular."
    )
    parser.add_argument(
        "--fraud_rate",
        type=float,
        default=0.008,
        help="Taxa de fraude a simular (0.008 = 0.8%)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/dataset.parquet",
        help="Caminho onde o dataset final processado será salvo."
    )
    
    args = parser.parse_args()
    
    build_dataset(
        nrows=args.nrows,
        fraud_rate=args.fraud_rate,
        output_path=args.output_path
    )
