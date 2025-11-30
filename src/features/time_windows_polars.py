# src/features/time_windows_polars.py
import polars as pl


def features_time_based_polars(df: pl.DataFrame) -> pl.DataFrame:
    """
    Gera features temporais baseadas em janelas de tempo usando Polars.

    Features criadas:
    - num_pix_last_1h: número de transações nos últimos 60 minutos.
    - avg_amount_last_7d: média dos valores transacionados nos últimos 7 dias.
    - std_amount_last_30d: desvio padrão dos valores nos últimos 30 dias.
    - num_unique_recipients_24h: número de recebedores únicos em uma janela de 24h.

    Args:
        df (pl.DataFrame): DataFrame contendo colunas 'user_id', 'timestamp',
                           'amount' e 'recipient_id'.

    Returns:
        pl.DataFrame: DataFrame original com as novas features adicionadas.
    """
    # Garantir ordenação por usuário e timestamp
    df = df.sort(["user_id", "timestamp"])

    # num_pix_last_1h: rolling count
    df = df.with_columns(
        pl.col("timestamp")
        .rolling_count(window="1h", by="user_id")
        .alias("num_pix_last_1h")
    )

    # avg_amount_last_7d: média dos amounts nos últimos 7 dias
    df = df.with_columns(
        pl.col("amount")
        .rolling_mean(window="7d", by="user_id", closed="left")
        .alias("avg_amount_last_7d")
    )

    # std_amount_last_30d: desvio padrão nos últimos 30 dias
    df = df.with_columns(
        pl.col("amount")
        .rolling_std(window="30d", by="user_id", closed="left")
        .alias("std_amount_last_30d")
    )

    # num_unique_recipients_24h: contagem de recebedores únicos
    # Polars não tem rolling nunique direto, usamos groupby_dynamic
    df_unique = (
        df.groupby_dynamic(
            index_column="timestamp",
            every="1h",
            period="24h",
            by="user_id",
            closed="left",
        )
        .agg(pl.col("recipient_id").n_unique().alias("num_unique_recipients_24h"))
    )

    # Join de volta ao dataframe original
    df = df.join(df_unique, on=["user_id", "timestamp"], how="left")

    return df
