# Sistema de Prevenção de Fraudes em Pix em Tempo Real


<p align="center">
  <img src="data/assets/prevencao-fraude-pix.png" alt="Prevenção de Fraudes no Pix" width="600"/>
</p>




---

## 📑 Sumário

- [1. Problema de Negócio](#1-problema-de-negócio)
- [2. Contexto](#2-contexto)
- [3. Premissas da Análise](#3-premissas-da-análise)
- [4. Planejamento da Solução](#4-planejamento-da-solução)
- [5. Limpeza e Preparação dos Dados](#5-limpeza-e-preparação-dos-dados)
- [6. Feature Engineering](#6-feature-engineering)
- [7. Modelagem e Treinamento](#7-modelagem-e-treinamento)
- [8. Performance do Modelo](#8-performance-do-modelo)
- [9. Insights Técnicos e de Negócio](#9-insights-técnicos-e-de-negócio)
- [10. Resultados para o Negócio](#10-resultados-para-o-negócio)
- [11. Modelo em Produção](#11-modelo-em-produção)
- [12. Stack Tecnológica e Decisões](#12-stack-tecnológica-e-decisões)
- [13. Estrutura do Repositório](#13-estrutura-do-repositório)
- [14. Como Executar](#14-como-executar)
- [15. Próximos Passos](#15-próximos-passos)
- [Contato](#contato)

---

## 1. Problema de Negócio

**Qual problema este projeto resolve?**

O Pix movimenta trilhões de reais por ano no Brasil. E exatamente por ser instantâneo e irreversível, se tornou o alvo preferido de fraudadores. O dinheiro some em segundos — e o banco não tem como desfazer a transação depois.

O desafio concreto que este projeto resolve é:

> *"Como uma fintech ou banco digital pode detectar uma transação Pix fraudulenta em menos de 50ms — antes de ela ser concluída — minimizando as perdas financeiras sem bloquear clientes legítimos e destruir a experiência do usuário?"*

Hoje a maioria das instituições usa **regras estáticas** (ex.: bloquear transações acima de R$ X às Y horas) como baseline. Essas regras geram dois problemas críticos: deixam fraudes sofisticadas passarem (Falsos Negativos) e bloqueiam clientes legítimos (Falsos Positivos), que abandona a plataforma.

A resposta está em um **pipeline de Machine Learning em tempo real** que aprende padrões comportamentais de fraude, calibra o limiar de decisão pelo custo financeiro real e entrega predições em produção dentro das restrições de latência do Pix.

---

## 2. Contexto

Fintechs e bancos digitais com operações de Pix convivem com três pressões simultâneas:

- **Pressão financeira:** cada fraude não detectada representa perda direta. O custo de um Falso Negativo (fraude aprovada) pode ser dezenas ou centenas de vezes maior que o custo de um Falso Positivo (transação legítima bloqueada).
- **Pressão de UX:** bloquear transações legítimas gera churn. Um cliente bloqueado erroneamente migra para o concorrente.
- **Pressão regulatória:** o Banco Central exige rastreabilidade, auditabilidade e governança nas decisões automatizadas que afetam clientes.

O cenário antes desta solução:

- Detecção baseada em **regras manuais estáticas**, que não se adaptam a novos padrões de fraude.
- Sem otimização financeira do limiar de decisão — o threshold era fixo, sem considerar o custo real de cada tipo de erro.
- Ausência de **monitoramento de drift**: o modelo degrada silenciosamente sem alertas.
- Pipeline de dados sem validações anti-leakage, colocando em risco a confiabilidade das predições.

Este projeto constrói do zero um **sistema completo**: da simulação de dados à inferência em produção, passando por engenharia de atributos, modelagem comparativa e otimização de threshold por custo financeiro.

---

## 3. Premissas da Análise

Para a construção do sistema, foram adotadas as seguintes premissas:

- As transações são simuladas com padrões realistas de comportamento suspeito: horários atípicos, valores discrepantes, múltiplas transações em janelas curtas e destinatários desconhecidos.
- O **rótulo de fraude** (`is_fraud`) é a fonte oficial para treino e avaliação — sem ambiguidade.
- **Data leakage temporal é um risco crítico:** o split treino/teste respeita rigorosamente a linha do tempo (sem embaralhamento aleatório). Features de janela temporal só usam dados do passado em relação a cada transação.
- O **custo de um Falso Negativo** (fraude aprovada) é significativamente superior ao custo de um Falso Positivo (bloqueio indevido). O modelo foi calibrado com essa assimetria.
- Latência máxima de inferência: **50ms por transação** — limite operacional para não impactar a experiência do Pix.
- O dataset é **desbalanceado** por natureza (fraudes são raras em relação ao volume total), o que impacta a escolha de métricas e a estratégia de treinamento.

---

## 4. Planejamento da Solução

A solução foi estruturada em quatro camadas independentes e orquestradas:

**Camada 1 — Simulação e Ingestão de Dados**
Geração de transações Pix sintéticas com padrões realistas de fraude e comportamento legítimo. Validação de schema para garantir consistência antes de qualquer processamento.

**Camada 2 — Feature Engineering Temporal**
Construção de atributos comportamentais em janelas de tempo (1min, 5min, 1h) com Polars para alta performance. Encoding de variáveis categóricas e guardrails anti-leakage.

**Camada 3 — Modelagem Comparativa**
Treinamento de dois modelos: Regressão Logística (baseline interpretável) e XGBoost (alta performance). Avaliação com métricas financeiramente relevantes. Otimização de threshold por custo real.

**Camada 4 — Inferência em Produção**
Pipeline de scoring em tempo real com carregamento de artefatos, pré-processamento idêntico ao treino e validação de latência máxima (< 50ms).

**Ferramentas planejadas:**

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.12 |
| Processamento de dados | Polars, Pandas, PyArrow |
| Feature Engineering | Polars (janelas temporais), Scikit-learn |
| Modelagem | XGBoost, LightGBM, Regressão Logística |
| Qualidade de código | Black, Ruff, Isort, Mypy, Pytest |
| Automação e CI/CD | Makefile, Poetry, GitHub Actions, pre-commit |
| Notebooks | JupyterLab (EDA, Feature Inspection, Threshold Analysis) |

---

## 5. Limpeza e Preparação dos Dados

### Simulação de Transações (`simulate_transactions.py`)

As transações são geradas com controle de seed para reprodutibilidade, incluindo:

- Campos: `id`, `timestamp`, `valor`, `origem`, `destino`, `chave_pix`, `dispositivo`, `is_fraud`.
- Padrões de comportamento suspeito: sazonalidade de fraudes por horário, valores discrepantes, alta frequência em janelas curtas.

### Validação de Schema (`schemas.py`)

Antes de qualquer transformação, o schema é validado:

- Tipos esperados: `timestamp` como `datetime64`, `valor` como `float64`, `is_fraud` como `int`.
- Qualquer inconsistência interrompe o pipeline com erro rastreável — zero tolerância a dados corrompidos chegando ao modelo.

### Guardrails Anti-Leakage (`validators.py`)

A maior armadilha em dados temporais é usar informações do futuro para treinar o modelo, inflando artificialmente as métricas.

Os validadores garantem:

- Features de janela temporal só usam dados **anteriores** ao timestamp de cada transação.
- Split treino/teste respeita a **ordem cronológica** — nenhum embaralhamento aleatório.
- Labels futuras nunca chegam ao conjunto de treino.

---

## 6. Feature Engineering

### Janelas Temporais com Polars (`timewindowspolars.py`)

A maior diferença entre uma transação legítima e uma fraudulenta frequentemente está no **comportamento recente** do usuário, não na transação isolada.

Features construídas por janela temporal:

| Feature | Janela | Lógica |
|---|---|---|
| `count_1min` | 1 minuto | Número de transações enviadas no último minuto |
| `sum_valor_5min` | 5 minutos | Soma dos valores enviados nos últimos 5 minutos |
| `count_dest_1h` | 1 hora | Número de destinatários únicos na última hora |
| `freq_hora` | 1 hora | Frequência de transações na hora corrente |

**Por que Polars?** O processamento de janelas temporais em Pandas com `.rolling()` por usuário é lento em grandes volumes. Polars executa agregações colunares vetorizadas com performance 5–10x superior, mantendo a latência de inferência dentro dos 50ms exigidos.

### Encoding Categórico (`categorical.py`)

- `tipo_chave_pix` (CPF, email, telefone, aleatória): One-Hot Encoding.
- `dispositivo` (mobile, web, API): Target Encoding com validação de leakage.
- Normalizações numéricas: StandardScaler salvo como artefato para uso idêntico na inferência.

---

## 7. Modelagem e Treinamento

### Baseline — Regressão Logística (`train_logreg.py`)

O baseline interpretável serve para dois propósitos:

1. **Referência mínima de performance:** qualquer modelo mais complexo precisa superar o baseline para justificar sua complexidade.
2. **Interpretabilidade regulatória:** coeficientes explicáveis para auditoria pelo Banco Central.

Pipeline: carregamento → split temporal → escalonamento → treino → salvamento de artefatos e métricas.

### Modelo Principal — XGBoost (`train_xgboost.py`)

XGBoost foi escolhido por três razões técnicas:

- **Robustez com dados desbalanceados** via parâmetro `scale_pos_weight`.
- **Importância de features nativa** — fundamental para auditabilidade.
- **Performance superior ao baseline** em PR-AUC, que é a métrica correta para classes desbalanceadas.

Pipeline: carregamento → split temporal → hiperparametrização → treino → salvamento de artefatos, relatórios e curvas.

### Otimização de Threshold por Custo Financeiro (`03thresholdanalysis.ipynb`)

**Este é o pulo do gato do projeto.**

AUC alto não garante lucro. O threshold padrão de 0.5 não é ótimo para fraudes.

A função de custo total utilizada:

```
Custo Total = (custo_fraude × Falsos Negativos) + (custo_operacional × Falsos Positivos)
```

O notebook varia o threshold de 0.1 a 0.9, calcula o custo total em cada ponto e seleciona o limiar que **minimiza o prejuízo financeiro** — não o erro estatístico. Isso significa que o modelo pode ter uma acurácia menor mas gerar menos perda para o negócio.

---

## 8. Performance do Modelo

### Métricas de Avaliação (`evaluate.py`)

Para datasets desbalanceados como fraudes, **acurácia é uma métrica enganosa**. Um modelo que classifica tudo como legítimo teria 99%+ de acurácia e seria completamente inútil.

As métricas relevantes utilizadas:

| Métrica | Por que é relevante |
|---|---|
| **PR-AUC** | Principal métrica — captura performance em classes desbalanceadas |
| **ROC-AUC** | Comparação entre modelos em diferentes thresholds |
| **Recall (Fraudes)** | Percentual de fraudes detectadas — minimiza Falsos Negativos |
| **Precisão (Fraudes)** | Percentual de alertas corretos — minimiza Falsos Positivos |
| **F1-Score** | Equilíbrio entre Precisão e Recall |
| **Latência de inferência** | SLA operacional: < 50ms por transação |

### Comparativo de Modelos

| Modelo | ROC-AUC | PR-AUC | Latência média |
|---|---|---|---|
| Regressão Logística (baseline) | Baseline | Baseline | < 10ms |
| **XGBoost** | **Superior ao baseline** | **Superior ao baseline** | **< 50ms** ✅ |

Os relatórios completos com curvas ROC, curvas PR e confusion matrices são salvos automaticamente em `models/reports/` a cada execução de treino.

---

## 9. Insights Técnicos e de Negócio

A construção do sistema revelou aprendizados que vão além da modelagem:

**Performance vs. Latência — a decisão que muda tudo**
O processamento de janelas temporais com Polars foi a decisão técnica mais impactante do projeto. A substituição de processamento linha a linha por agregações vetorizadas foi a diferença entre um pipeline que funcionaria em análise batch e um sistema que opera dentro dos 50ms do Pix.

**Data Leakage é o inimigo silencioso**
Em dados temporais, usar informações futuras para treinar é um erro sutil e devastador — as métricas de treino ficam infladas e o modelo colapsa em produção. Os validadores em `validators.py` são a primeira linha de defesa contra esse risco.

**O threshold ótimo não é 0.5**
Em fraudes, o custo assimétrico entre Falsos Negativos e Falsos Positivos muda completamente a decisão de threshold. O modelo calibrado por custo financeiro gerou um threshold significativamente abaixo de 0.5 — priorizando a detecção de fraudes mesmo ao custo de mais alertas manuais.

**Código de Cientista de Dados é código de produção**
A implementação de Type Hinting com Mypy e Linting com Ruff em pipelines de Data Science foi fundamental para evitar bugs em tempo de execução. Um pipeline que vai para produção sem tipagem estática é uma bomba-relógio operacional.

**O modelo de custo muda a conversa com o negócio**
Apresentar o threshold analysis para o CFO com valores em R$ — "cada ponto percentual de Recall recupera R$ X em fraudes" — é muito mais efetivo do que apresentar AUC. Isso posiciona o cientista de dados como resolvedor de problemas financeiros, não operador de algoritmos.

---

## 10. Resultados para o Negócio

### Impacto Direto

| Dimensão | Baseline (Regras Estáticas) | Sistema ML |
|---|---|---|
| Detecção de fraudes | Regras fixas, não adaptativas | Modelo aprende novos padrões comportamentais |
| Threshold de decisão | Fixo, sem otimização financeira | Calibrado por custo real (FN × custo_fraude + FP × custo_operacional) |
| Latência de inferência | — | < 50ms por transação ✅ |
| Auditabilidade | Regras documentadas manualmente | Artefatos versionados, relatórios automáticos, importância de features |
| Adaptabilidade | Requer intervenção manual | Pipeline de re-treino automatizável |
| Governança | Baixa | CI/CD + testes + tipagem estática + versionamento |

### Para a Diretoria

- **Redução de perdas por fraude:** o modelo XGBoost com threshold otimizado detecta significativamente mais fraudes do que o baseline de regras estáticas, com controle do volume de Falsos Positivos.
- **Latência garantida:** decisões em < 50ms — o cliente não percebe o processo de scoring.
- **Governança e auditabilidade:** todos os modelos, métricas e relatórios são versionados e rastreáveis.

### Para o Diretor Financeiro

A função de custo total implementada no `03thresholdanalysis.ipynb` permite:

- Simular o impacto financeiro de cada threshold antes de colocá-lo em produção.
- Quantificar o trade-off: "subir o threshold em 0.1 reduz X alertas manuais mas aumenta Y fraudes aprovadas, custando R$ Z a mais."
- Recalibrar trimestralmente conforme o perfil de fraudes evolui.

---

## 11. Modelo em Produção

O sistema foi desenhado para inferência em produção com as seguintes garantias:

### Pipeline de Inferência (`inference.py`)

```bash
poetry run python src/modeling/inference.py \
  --input data/processed/new_batch.parquet \
  --model models/artifacts/xgb_model.joblib \
  --output models/reports/inference_scores.parquet
```

O que acontece internamente:

1. Carregamento dos artefatos treinados (modelo + scaler + encoders).
2. Aplicação do **mesmo pré-processamento do treino** — zero divergência entre treino e produção.
3. Cálculo das features de janela temporal com Polars.
4. Scoring e geração de probabilidade de fraude para cada transação.
5. Validação de latência: teste automatizado em `testinferencelatency.py` garante que o SLA de 50ms é respeitado.

### Teste de Latência Automatizado (`tests/testinferencelatency.py`)

O CI/CD bloqueia qualquer deploy que quebre o limite de latência. A cada commit, o GitHub Actions roda o pipeline de testes — incluindo o teste de latência — antes de qualquer merge.

### Monitoramento Contínuo

Os KPIs monitorados em produção:

| KPI | Meta |
|---|---|
| Taxa de fraude detectada (Recall) | Acima do baseline |
| Taxa de Falsos Positivos | Abaixo do limiar financeiro definido |
| Latência P95 de inferência | < 50ms |
| Drift de features | Alerta automático quando distribuições mudam |

---

## 12. Stack Tecnológica e Decisões

| Componente | Escolha | Justificativa Técnica |
|---|---|---|
| **Linguagem** | Python 3.12 | Versão LTS com melhorias de performance e tipagem estática aprimorada |
| **Processamento** | Polars + PyArrow | Agregações vetorizadas em janelas temporais 5–10x mais rápidas que Pandas — essencial para < 50ms |
| **Modelagem** | XGBoost + Regressão Logística | XGBoost para performance; LogReg para baseline interpretável e auditabilidade regulatória |
| **Qualidade** | Ruff + Mypy + Black + Isort | Código de produção exige tipagem estática e lint rigoroso — bug em pipeline de fraude tem custo financeiro |
| **Automação** | Makefile + Poetry | Comandos padronizados e ambiente reproduzível — qualquer pessoa clona e executa sem fricção |
| **CI/CD** | GitHub Actions | Testes, linters e validação de latência a cada commit — zero regressão silenciosa |
| **Notebooks** | JupyterLab | EDA, inspeção de features e threshold analysis — camada analítica transparente para negócio |

### Trade-offs Documentados

**Polars vs. Pandas:** Polars foi escolhido para o processamento de janelas temporais por performance superior, embora Pandas seja mais comum no mercado. O trade-off é legibilidade — mas com docstrings adequadas, a manutenção é gerenciável.

**XGBoost vs. LightGBM:** XGBoost foi priorizado por ter suporte nativo mais maduro ao parâmetro `scale_pos_weight` para classes desbalanceadas. LightGBM está disponível no projeto para comparação futura.

**Modularidade vs. simplicidade:** Scripts detalhados foram priorizados em detrimento de uma estrutura monolítica, aumentando o número de arquivos mas garantindo testabilidade isolada de cada componente.

---

## 13. Estrutura do Repositório

```text
prevencaoFraudesPix/
│
├── data/
│   ├── raw/                        # Transações brutas simuladas e logs
│   ├── interim/                    # Dados pós-validação de schema e limpeza
│   └── processed/                  # Dataset final com features para treino e inferência
│
├── models/
│   ├── artifacts/                  # Modelos treinados (.joblib), encoders e scalers
│   └── reports/                    # Métricas (JSON), curvas ROC/PR e confusion matrices (PNG)
│
├── src/
│   ├── data/
│   │   ├── simulate_transactions.py # Gera transações Pix sintéticas com padrões de fraude e seed control
│   │   ├── schemas.py               # Schema de colunas com tipos e obrigatoriedade — valida antes de processar
│   │   └── utils.py                 # Helpers de leitura/escrita e geração de amostras controladas
│   │
│   ├── features/
│   │   ├── timewindowspolars.py     # Agregações em janelas 1min/5min/1h com Polars — núcleo do pipeline de features
│   │   ├── categorical.py           # One-Hot e Target Encoding de tipo de chave Pix e dispositivo
│   │   └── validators.py            # Guardrails anti-leakage: impede labels futuras e valida split temporal
│   │
│   ├── modeling/
│   │   ├── train_logreg.py          # Pipeline de treino com Regressão Logística — baseline interpretável
│   │   ├── train_xgboost.py         # Pipeline avançado com XGBoost — modelo principal de produção
│   │   ├── evaluate.py              # Métricas (ROC-AUC, PR-AUC, F1), curvas e relatórios comparativos
│   │   └── inference.py             # Scoring em tempo real: carrega artefatos, pré-processa e prediz risco
│   │
│   ├── utils/
│   │   ├── io.py                    # API de caminhos padronizados: load/save parquet, CSV e artefatos joblib
│   │   ├── metrics.py               # Métricas customizadas, análise de threshold e função de custo financeiro
│   │   └── config.py                # Parâmetros globais: janelas temporais, caminhos, seed, limites de latência
│   │
│   └── pipelines/
│       ├── build_dataset.py         # Orquestra: simulação → validação → features → dataset final processado
│       └── trainandeval.py          # Treina modelos, compara métricas, salva artefatos e relatórios
│
├── tests/
│   ├── testtimewindows.py           # Valida janelas temporais: consistência e ausência de valores incorretos
│   ├── testleakageguards.py         # Testa validators para garantir zero data leakage
│   └── testinferencelatency.py      # Mede latência de inferência e bloqueia se ultrapassar 50ms
│
├── notebooks/
│   ├── 01_eda.ipynb                 # EDA: distribuição de valores, taxa de fraude por hora/dia, sazonalidade
│   ├── 02featureinspection.ipynb    # Importância de features, correlações, SHAP e drift potencial
│   └── 03thresholdanalysis.ipynb    # Trade-off precisão/recall por threshold — seleção do limiar por custo financeiro
│
├── .github/
│   └── workflows/
│       ├── ci.yaml                  # Pipeline principal: Pytest + qualidade + build de artefatos
│       ├── quality.yml              # Black (check), Ruff, Isort (check), Mypy
│       └── auto-fix.yml             # Correções automáticas de formatação (opcional)
│
├── pyproject.toml                   # Dependências, ferramentas de qualidade (Black, Ruff, Isort, Mypy) e config do projeto
├── requirements.txt                 # Dependências de produção (alternativa ao Poetry)
├── dev-requirements.txt             # Ferramentas de desenvolvimento (Black, Ruff, Isort, Mypy, Pytest)
├── setup.cfg                        # Configurações de Flake8 alinhadas ao Black (88 chars)
├── .pre-commit-config.yaml          # Hooks: Black, Ruff (--fix), Isort — executados antes de cada commit
├── Makefile                         # Comandos padronizados: setup, data, train, quality, test
└── .gitignore                       # Ignora ambientes, caches, dados grandes e artefatos de modelos
```

### Imagens da estrutura real do repositório

<img width="877" height="1270" alt="Screenshot_20251205-192939" src="https://github.com/user-attachments/assets/4c653f2c-4860-4093-88e9-39669e263aae" />
<img width="849" height="1134" alt="Screenshot_20251205-193104" src="https://github.com/user-attachments/assets/59ca820c-f2e4-44c5-bded-c49f4b483868" />

---

## 14. Como Executar

### Pré-requisitos

**Hardware mínimo:**
- CPU: 4+ núcleos | RAM: 16 GB | Armazenamento: 10–20 GB (SSD recomendado)

**Software:**
- Python 3.12
- Poetry 1.7+
- Git e GNU Make
- JupyterLab (para notebooks)

---

### Passo 1 — Instalar Poetry e clonar o repositório

```bash
# Linux/macOS
curl -sSL https://install.python-poetry.org | python3 -

# Windows — via installer oficial: https://python-poetry.org/docs/#installation

git clone https://github.com/Santosdevbjj/prevencaoFraudesPix.git
cd prevencaoFraudesPix
```

---

### Passo 2 — Criar ambiente e instalar dependências

```bash
poetry install
poetry run pre-commit install
```

---

### Passo 3 — Verificar qualidade do código

```bash
poetry run black src tests
poetry run ruff check src tests --fix
poetry run isort src tests
poetry run mypy src
```

Ou com Makefile:

```bash
make quality
```

---

### Passo 4 — Construir o dataset

```bash
poetry run python src/pipelines/build_dataset.py \
  --input data/raw/transactions.parquet \
  --output data/processed/dataset.parquet
```

---

### Passo 5 — Treinar e avaliar modelos

```bash
poetry run python src/pipelines/trainandeval.py \
  --input data/processed/dataset.parquet \
  --models-out models/artifacts/ \
  --reports-out models/reports/
```

---

### Passo 6 — Inferência em tempo real

```bash
poetry run python src/modeling/inference.py \
  --input data/processed/new_batch.parquet \
  --model models/artifacts/xgb_model.joblib \
  --output models/reports/inference_scores.parquet
```

---

### Passo 7 — Explorar notebooks

```bash
poetry run jupyter lab
```

Abrir na ordem:

1. **`notebooks/01_eda.ipynb`** — distribuição de transações, sazonalidade e taxa de fraude por horário.
2. **`notebooks/02featureinspection.ipynb`** — importância de features, correlações e visualização SHAP.
3. **`notebooks/03thresholdanalysis.ipynb`** — threshold analysis com custo financeiro e seleção do limiar ótimo.

---

### Passo 8 — Executar testes

```bash
pytest -v tests/
```

Os testes cobrem janelas temporais, guardrails anti-leakage e latência de inferência. O CI/CD bloqueia merges que quebrem qualquer um desses testes.

---

## 15. Próximos Passos

Com base nos resultados obtidos, os próximos passos recomendados são:

- **Device Intelligence:** integrar dados de dispositivo, geolocalização e biometria comportamental para enriquecer as features do modelo.
- **Re-treino automatizado por drift:** implementar detector de drift de features e acionar re-treino automaticamente quando a distribuição dos dados muda significativamente.
- **Microservices:** refatorar o pipeline de inferência para uma API REST (FastAPI) com contrato OpenAPI — permitindo integração com sistemas de risco existentes.
- **A/B Testing:** comparar o modelo ML com as regras estáticas em produção com tráfego real para quantificar o impacto financeiro.
- **Cobertura de testes:** expandir para 80%+ de cobertura, incluindo testes de contrato para o schema de entrada da inferência.
- **Model Registry:** versionar modelos com MLflow ou equivalente — rastreabilidade completa de qual modelo está em produção e quando foi treinado.

---

## Referências e Documentação Técnica

- 📊 [Relatório para o CEO](./docs/relatorio_ceo.md) — impacto estratégico e roadmap
- 💼 [Relatório para o CFO](./docs/relatorio_cfo.md) — modelo de custo e threshold analysis em R$
- ⚙️ [Relatório Operacional](./docs/relatorio_operacional.md) — KPIs, playbook de incidentes e monitoramento

---

**Autor:** Sergio Santos

[![Portfólio Sérgio Santos](https://img.shields.io/badge/Portfólio-Sérgio_Santos-111827?style=for-the-badge&logo=githubpages&logoColor=00eaff)](https://portfoliosantossergio.vercel.app)

[![LinkedIn Sérgio Santos](https://img.shields.io/badge/LinkedIn-Sérgio_Santos-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/santossergioluiz)

---


