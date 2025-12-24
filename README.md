# Sistema de Prevenção de Fraudes em Pix  em Tempo Real  


<img width="1080" height="670" alt="CienciaDadosPix" src="https://github.com/user-attachments/assets/f3ae5290-64d6-4e21-909e-152e4092b308" />



---

**Visão Geral**

Este projeto implementa um sistema de detecção de fraudes em transações Pix em tempo real, utilizando pipelines de ciência de dados, modelos de machine learning e boas práticas de engenharia de software.

O objetivo é prevenir perdas financeiras e melhorar a experiência do cliente, oferecendo decisões rápidas com baixa latência, mantendo alta precisão e reprodutibilidade.

---

**Motivação**

Criei este projeto para aplicar conhecimento em ciência de dados, engenharia de dados e machine learning em um problema real de fintechs e bancos digitais: fraudes em pagamentos instantâneos.

Minha intenção foi:

• Simular cenários de transações reais, incluindo padrões suspeitos;

• Construir pipelines escaláveis e auditáveis;

• Aplicar modelos interpretáveis (logística) e de alto desempenho (XGBoost/LightGBM);

• Garantir governança de dados, qualidade de código e métricas de negócio.


---

**Problema que Resolve**

Fraudes em Pix podem gerar perdas milionárias e prejudicar a confiança do cliente. O desafio é:

• Detectar fraudes em tempo real;

• Minimizar falsos positivos que impactam a experiência do usuário;

• Otimizar custo financeiro agregado, considerando custo de fraude e operação;

• Manter alta performance e monitoramento contínuo.


---




🛠️ **Decisões Técnicas & Stack Estratégica**

As escolhas abaixo visam equilibrar performance computacional, manutenibilidade de código e valor de negócio:

| Componente | Escolha | Motivação |
|---|---|---|
| Linguagem | Python 3.12 | Versão estável com melhorias de performance e tipagem. |
| Processamento | Polars & PyArrow | Manipulação de grandes volumes de dados com performance superior ao Pandas. |
| Modelagem | XGBoost & LogReg | Comparação entre um modelo de alta performance e um baseline interpretável. |
| Qualidade | Ruff, Mypy & Black | Garantia de código limpo, tipagem estática e padronização automática. |
| Automação | Makefile & Poetry | Padronização de comandos e gestão rigorosa de dependências. |
| CI/CD | GitHub Actions | Execução automática de testes e linters a cada novo commit. |
| Arquitetura | Pipelines Modulares | Separação clara entre Ingestão, Feature Engineering e Modelagem. |




---


**Trade-offs:**

• Optei por Polars para performance em janelas temporais, embora Pandas seja mais comum;

• Mantive regressão logística para interpretabilidade antes de aplicar modelos complexos;

• Scripts detalhados foram priorizados em modularidade, mesmo aumentando a quantidade de arquivos.


---

**Tecnologias Utilizadas**

• Ciência de Dados: Pandas, Polars, NumPy, PyArrow, Scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn

• Qualidade: Black, Ruff, Isort, Mypy, Pytest

• DevOps/Infra: Git, GitHub Actions, Poetry, pre-commit, Makefile

• Ambiente: Python 3.12, JupyterLab/Notebook

---

**Como Executar**

**Pré-requisitos*"

• CPU 4+ núcleos, 16GB RAM, SSD recomendado

• Python 3.12, Poetry 1.7+, Git, Make

• JupyterLab para notebooks

**Passos**

**1. Clone o repositório**

   ```
   git clone https://github.com/Santosdevbjj/prevencaoFraudesPix.git
cd prevencaoFraudesPix
```


**2.Instale dependências e pre-commit**

```
poetry install
poetry run pre-commit install
```


**3. Qualidade do código**

```
poetry run black src tests
poetry run ruff check src tests --fix
poetry run isort src tests
poetry run mypy src
```

**4. Execução de pipelines**

```
poetry run python src/pipelines/build_dataset.py
poetry run python src/pipelines/trainandeval.py
poetry run python src/modeling/inference.py --input data/processed/realtimebatch.parquet --output models/reports/inferenceoutput.parquet
```

**5. Notebooks**

```
poetry run jupyter lab
# 01_eda.ipynb, 02featureinspection.ipynb, 03thresholdanalysis.ipynb
```

---

**Principais Aprendizados**

• Importância de separar regras de negócio, pipelines e modelos;

• Uso de Polars e PyArrow aumentou performance em janelas temporais;

• Gerenciamento de thresholds dinâmicos e métricas de custo foi essencial para decisões financeiras;

• Pipelines modularizadas e CI/CD reduzem risco operacional e facilitam manutenção;

• Experiência prática em detecção de fraude real-time, métricas de latência e trade-offs de precisão/recall.

---


**Próximos Passos**

• Integração com sistemas de risco e device intelligence;

• Automatizar re-treino baseado em drift de dados;

• Expandir cobertura de testes e refatorar pipelines para microservices;

• Explorar novos modelos interpretáveis para auditoria regulatória.


---

**Impacto de Negócio**

• Redução de perdas por fraude: detecção em tempo real com ROC-AUC e PR-AUC superiores ao baseline;

• Latência média: <50 ms por transação, mantendo UX;

• Custo otimizado: minimização de custo total = custofraude × FN + custooperacional × FP;

• Governança e auditabilidade: versionamento de modelos, artefatos e relatórios.




---
---


**Passo a passo para criar e executar o projeto**

**Preparação do ambiente**

- **Hardware recomendado:**
  - CPU de múltiplos núcleos (4+), 16 GB RAM para treinos confortáveis.
  - 10–20 GB de disco para artefatos e datasets intermediários.
- **Software:**
  - Python 3.12.
  - Poetry 1.7+ para gerenciar dependências e ambientes.
  - Git e GitHub para versionamento e CI.
  - Make (GNU Make) para comandos padronizados.
  - JupyterLab/Notebook para exploração.


---

**Instalação**

**1. Instale Python 3.12 e pip.**
**2. Instale Poetry:**
   - Linux/macOS: curl -sSL https://install.python-poetry.org | python3 -
   - Windows: via installer oficial do Poetry.
**3. Clone o repositório:**
   - git clone https://github.com/Santosdevbjj/prevencaoFraudesPix.git
   - cd prevencaoFraudesPix
**4. Crie e ative o ambiente:**
   - poetry install (ou poetry install --no-root se não for empacotar o projeto)
**5. Instale pre-commit e ative hooks:**
   - poetry run pre-commit install


---

**Execução de qualidade local**

- **Formatação e lint:**
  - poetry run black src tests
  - poetry run ruff check src tests --fix
  - poetry run isort src tests
- **Tipos:**
  - poetry run mypy src

**Execução de pipelines**

- Construir dataset:
  - poetry run python src/pipelines/build_dataset.py
- Treinar e avaliar:
  - poetry run python src/pipelines/trainandeval.py
- Inferência:
  - poetry run python src/modeling/inference.py

---


**Prevenção de Fraudes em Pix — Sistema de detecção em tempo real**

**Descrição**

Este projeto implementa um pipeline completo para simulação de transações, engenharia de atributos, treinamento de modelos (logística e XGBoost), avaliação e inferência em tempo real. 

A organização segue práticas de Data Science, com qualidade garantida por CI, testes e linters.

---


**Tecnologias utilizadas**

- Linguagem: Python 3.12.
- **Ciência de Dados:**
  - Pandas, Polars, NumPy, PyArrow.
  - Scikit-learn, XGBoost, LightGBM.
  - Matplotlib, Seaborn para visualização.
- **Qualidade:**
  - Black (formatação), Ruff (lint), Isort (imports), Mypy (tipos).
  - Pytest (testes).
- **DevOps:**
  - GitHub Actions (CI), pre-commit, Makefile, Poetry.
 
---


**Requisitos**

- **Hardware:**
  - CPU 4+ cores, 16 GB RAM, SSD recomendado.
- **Software:**
  - Python 3.12, Poetry 1.7+, Git, Make.
  - JupyterLab para notebooks.


---

**Instalação e execução**

**1. Clone o repositório:**
   - git clone https://github.com/Santosdevbjj/prevencaoFraudesPix.git
   - cd prevencaoFraudesPix
     
**2. Instale dependências:**
   - poetry install
     
**3. Qualidade:**
   - poetry run black src tests
   - poetry run ruff check src tests --fix
   - poetry run isort src tests
   - poetry run mypy src
     
**4. Pipelines:**
   - poetry run python src/pipelines/build_dataset.py
   - poetry run python src/pipelines/trainandeval.py

**5. Notebooks:**
   - poetry run jupyter lab
   - Abra notebooks/01eda.ipynb, 02featureinspection.ipynb, 03threshold_analysis.ipynb.


--- 

**Estrutura do repositório**


<img width="877" height="1270" alt="Screenshot_20251205-192939" src="https://github.com/user-attachments/assets/4c653f2c-4860-4093-88e9-39669e263aae" />
<img width="849" height="1134" alt="Screenshot_20251205-193104" src="https://github.com/user-attachments/assets/59ca820c-f2e4-44c5-bded-c49f4b483868" />



 
---

**Explicação das pastas e arquivos**

- **data/**
  - **raw/:** dados brutos simulados ou provenientes de logs.
  - **interim/:** dados intermediários após limpeza/validações.
  - **processed/:** dataset final com features para treino/inferência.
- **models/**
  - **artifacts/:** modelos treinados, encoders e escalers salvos.
  - **reports/:** relatórios de métricas, curvas ROC/PR, confusion matrices.
- **src/data/**
  - **simulate_transactions.py:** gera transações Pix sintéticas com rótulos de fraude, distribuição de valores e padrões temporais.
  - **schemas.py:** esquemas de colunas (tipos, obrigatoriedade) para garantir consistência.
  - **utils.py:** utilidades de ingestão, validação básica e seed control.
- **src/features/**
  - **timewindowspolars.py:** agregações em janelas de tempo (ex.: contagem por minuto, soma de valores por hora) usando Polars para alta performance.
  - **categorical.py:** encoding/normalização de variáveis categóricas (ex.: tipo de chave Pix, dispositivo).
  - **validators.py:** guardrails de vazamento e sanidade (ex.: impedir uso de labels futuras).
- **src/modeling/**
  - **train_logreg.py:** pipeline de treinamento com regressão logística, baseline interpretável.
  - **train_xgboost.py:** pipeline avançado com XGBoost, foco em desempenho.
  - **evaluate.py:** avaliação com métricas (ROC-AUC, PR-AUC, F1, precisão/recall), curvas e relatórios.
  - **inference.py:** função de predição em tempo real, com carregamento de artefatos e pré-processamento idêntico ao treino.
- **src/utils/**
  - **io.py:** leitura/escrita de datasets e artefatos com caminhos padronizados.
  - **metrics.py:** métricas customizadas e utilitários de limiar.
  - **config.py:** configuração central de caminhos, parâmetros de treino e janelas.
- **src/pipelines/**
  - **build_dataset.py:** orquestra ingestão, validações, engenharia de atributos e persiste dataset final.
  - **trainandeval.py:** treina modelos, salva artefatos e gera relatórios comparativos.
- **tests/**
  - **testtimewindows.py:** testa agregações temporais e consistência das janelas.
  - **testleakageguards.py:** testa validadores para evitar data leakage.
  - **testinferencelatency.py:** mede latência de inferência e verifica limites.
- **notebooks/**
  - **01_eda.ipynb:** análise exploratória, distribuição de valores, taxas de fraude, sazonalidade.
  - **02featureinspection.ipynb:** importância de features, correlações, drift potencial.
  - **03thresholdanalysis.ipynb:** análise de trade-off entre precisão, recall e custo financeiro.
- **.github/workflows/**
  - **ci.yaml:** pipeline principal de testes e qualidade.
  - **quality.yml:** lint/format/type checks.
  - **auto-fix*.yml:** pipelines opcionais para correção automática.
- **Configuração:**
  - **pyproject.toml:** dependências, ferramentas de qualidade e configuração do projeto.
  - **requirements.txt** e dev-requirements.txt: referências alternativas de dependências.
  - **setup.cfg:** configurações de ferramentas (se necessário).
  - **.pre-commit-config.yaml:** hooks de qualidade.
  - **.gitignore:** arquivos ignorados.
  - **Makefile:** comandos padronizados (ex.: build, train, eval).


---

**Como executar**

- **Construção de dataset:**
  - poetry run python src/pipelines/build_dataset.py
- **Treino e avaliação:**
  - poetry run python src/pipelines/trainandeval.py
- **Inferência em tempo real (exemplo):**
  - poetry run python src/modeling/inference.py --input data/processed/realtimebatch.parquet --output models/reports/inferenceoutput.parquet


---

**Exemplos de execução dos notebooks**

- **Inicie Jupyter:**
  - poetry run jupyter lab
- **01_eda.ipynb:**
  - Carregue data/processed/dataset.parquet.
  - Plote distribuição de valores por hora; calcule taxa de fraude por janela.
- **02featureinspection.ipynb:**
  - Carregue artefatos do XGBoost.
  - Extraia importâncias e compare com logística; **visualize SHAP.**
- **03thresholdanalysis.ipynb:**
  - Carregue models/reports/metrics.json.
  - Varie thresholds de 0.1 a 0.9; calcule custo esperado: custofraude × falsosnegativos + custooperacional × falsospositivos; selecione limiar ótimo.

---

**Documentação das bibliotecas**

- **pandas, numpy:** manipulação e operações numéricas.
- **polars:** processamento colunar de alto desempenho, ideal para janelas temporais.
- **pyarrow:** interoperabilidade e parquet/arrow.
- **scikit-learn:** modelos clássicos, métricas, pipelines.
- **xgboost, lightgbm:** gradient boosting eficientes.
- **joblib:** serialização de artefatos.
- **matplotlib, seaborn:** visualização.
- **pytest:** testes automatizados.
- **black, ruff, isort, mypy:** qualidade do código.



---

**Detalhamento dos arquivos:**

**pyproject.toml**

- Define nome, versão, descrição, licença.
- Lista dependências de runtime e grupo dev (black, ruff, isort, mypy).
- Configurações de Black, Ruff, Isort, Mypy.
- Dica: se não for empacotar, usar package-mode = false ou poetry install --no-root.

**requirements.txt e dev-requirements.txt**

- Alternativas ao Poetry para ambientes onde pip é preferido.
- requirements.txt: libs de produção (pandas, numpy, scikit-learn, xgboost, lightgbm, polars, pyarrow, matplotlib, seaborn, joblib, pytest).
- dev-requirements.txt: ferramentas de qualidade (black, ruff, isort, mypy).

**Makefile**

- Targets típicos:
  - make setup → poetry install, pre-commit install.
  - make data → build_dataset.
  - make train → trainandeval.
  - make quality → black, ruff, isort, mypy.
  - make test → pytest.

**src/data**

- simulate_transactions.py:
  - Gera transações com campos: id, timestamp, valor, origem/destino, chave Pix, dispositivo, rótulo fraude.
  - Controla seed, sazonalidade e padrões de comportamento suspeito.
- schemas.py:
  - Especifica schema com dtypes (ex.: timestamp como datetime64, valor como float).
  - Funções para validar conformidade do dataset.
- utils.py:
  - Helpers de leitura/escrita e geração de amostras controladas.

**src/features**

- timewindowspolars.py:
  - Funções de agregação por janelas: contagem/valor acumulado por 1min/5min/1h; número de destinatários únicos; frequência de transações.
  - Implementadas em Polars para performance.
- categorical.py:
  - Encoding de categorias: one-hot/target encoding; normalizações.
- validators.py:
  - Regras anti-vazamento: impede uso de labels futuras; valida separação temporal treino/teste.

**src/modeling**

- train_logreg.py:
  - Pipeline: load dataset, split, scale, treina regressão logística, salva artefatos e métricas.
- train_xgboost.py:
  - Similar ao acima; otimiza hiperparâmetros (grid/bayes opcional).
- evaluate.py:
  - Calcula métricas clássicas; gera relatórios e gráficos, salva em models/reports.
- inference.py:
  - Carrega artefatos, aplica o mesmo pré-processamento, prediz risco de fraude; otimiza latência.

**src/utils**

- io.py:
  - API de caminho padrão; funções de load/save parquet/csv, artefatos joblib.
- metrics.py:
  - Métricas customizadas; funções para análise de threshold e custos.
- config.py:
  - Parâmetros globais (janelas, caminhos de dados, semente, limites de latência).

**src/pipelines**

- build_dataset.py:
  - Orquestra simulador → valida schema → features → salva processed.
- trainandeval.py:
  - Treina modelos, compara métricas, salva artefatos e relatórios (gráficos, JSON).

**data/**

- raw/: insumos brutos (simulados).
- interim/: dados pós-validações.
- processed/: dataset final com features e rótulos.

**models/**

- artifacts/: .joblib/.json dos modelos e pré-processadores.
- reports/: métricas e gráficos (.png, .json).

**tests/**

- testtimewindows.py: garante janelas corretas e sem inconsistências.
- testleakageguards.py: verifica ausência de vazamento.
- testinferencelatency.py: assegura latência máxima definida e estabilidade.

**notebooks/**

- **01_eda.ipynb:**
  - Explora distribuição de transações, sazonalidade, padrões de fraude.
- **02featureinspection.ipynb:**
  - Analisa importância de features, correlação, SHAP opcional.
- **03thresholdanalysis.ipynb:**
  - Seleciona thresholds ótimos por custo.

**.gitignore**

- Ignora ambientes, caches, dados grandes, artefatos.

**.github/workflows**

- **ci.yaml:**
  - Pytest, qualidade básica e build de artefatos.
- quality.yml:
  - Black (check), Ruff, Isort (check), Mypy.
- auto-fix*.yml:
  - Pipelines auxiliares que aplicam correções automáticas (opcional).

**setup.cfg**

- Centralização de configs para ferramentas que oferecem integração via setup.cfg (se aplicável).

**.pre-commit-config.yaml**

- Hooks: Black, Ruff (com --fix), Isort.
- Opcional: integração com Mypy e Pytest.

---

**Exemplos:**

**Construção de dataset**

`bash
poetry run python src/pipelines/build_dataset.py \
  --input data/raw/transactions.parquet \
  --output data/processed/dataset.parquet
`

**Treino e avaliação**

`bash
poetry run python src/pipelines/trainandeval.py \
  --input data/processed/dataset.parquet \
  --models-out models/artifacts/ \
  --reports-out models/reports/
`

**Inferência**

`bash
poetry run python src/modeling/inference.py \
  --input data/processed/new_batch.parquet \
  --model models/artifacts/xgb_model.joblib \
  --output models/reports/inference_scores.parquet
`

**Notebooks**

- **01_eda.ipynb:**
  - Carregue dataset: data/processed/dataset.parquet.
  - Crie gráficos: histograma de valores, taxa de fraude por hora/dia.
- **02featureinspection.ipynb:**
  - Importâncias: carregue xgb_model.joblib, plote features por ganho.
- **03thresholdanalysis.ipynb:**
  - Analise métricas por threshold e calcule custo esperado; selecione limiar de decisão.

---


​🧠 **Aprendizados e Desafios Técnicos**

​O desenvolvimento deste sistema trouxe desafios que foram além da modelagem preditiva, exigindo uma mentalidade de Engenharia de Machine Learning (MLOps):

• **​Performance vs. Latência:** O maior desafio técnico foi garantir que o cálculo de janelas temporais (contagem de transações no último minuto/hora) fosse performático.

A substituição do processamento linha a linha por agregações vetorizadas no Polars foi a decisão chave para manter a latência de inferência dentro dos limites aceitáveis para o Pix.

• **​Prevenção de Data Leakage:** Em dados temporais, é fácil cometer o erro de usar informações do futuro para treinar o modelo. Implementei validadores rigorosos (validators.py) para garantir que as janelas de agregação e o split treino/teste respeitassem estritamente a linha do tempo.

• **​Trade-off Financeiro:** Aprendi que a melhor métrica de Machine Learning (como um AUC alto) nem sempre é a melhor métrica de negócio. 

• O uso do **Threshold Analysis** me permitiu entender que, em fraudes, o custo de um Falso Negativo (deixar a fraude passar) é muito superior ao de um Falso Positivo (bloqueio indevido), e calibrei o modelo para minimizar o prejuízo financeiro total, não apenas o erro estatístico.

• **​Qualidade de Software em Dados:** A implementação de **Type Hinting (Mypy) e Linting (Ruff)** em um pipeline de Data Science foi fundamental para evitar bugs em tempo de execução, demonstrando que código de cientista de dados também deve ser código de produção.




---



## Relatórios executivos


**Relatório para o CEO**

- **Resumo estratégico:**
  - O sistema reduz perdas por fraude em Pix com decisão em tempo real, preservando experiência do cliente.
- **Benefícios:**
  - Diminuição de fraude estimada em X% com incremento Y% em precisão, mantendo latência média inferior a 50 ms por transação (meta de produção).
  - Arquitetura replicável e audítavel, com governança de dados e qualidade de código.
- **Riscos e mitigação:**
  - Drift de dados monitorado com re-treinos periódicos e alertas.
  - Falsos positivos manejados por thresholds dinâmicos e revisão manual em casos de alto impacto.
- **Próximos passos:**
  - **Integração com sistemas de risco:** AB testing; ampliação de features comportamentais e device intelligence.


---


**Relatório para o diretor financeiro**

- **Impacto econômico:**
  - **Modelo de custo:** Custo total = custofraude × FN + custooperacional × FP.
  - Otimização de limiar minimiza custo agregado, com simulações nos notebooks.
- **Resultados:**
  - ROC-AUC e PR-AUC superiores ao baseline (logística), XGBoost melhor performance.
  - Cenários por sazonalidade e horários de pico avaliados.
- **Governança:**
  - Registros de modelos e relatórios em models/reports; versionamento por commit/tag.
- **Recomendação:**
  - Adoção do limiar 𝜏 que minimiza custo sob restrições de SLA e UX; recalibração trimestral.
 

    ---

    

**Relatório para o gerente financeiro**

- **Operação:**
  - **Processo diário:** ingestão → features → scoring em tempo real → revisão de alertas.
  - **Monitoramento de KPIs:** taxa de fraude, precisão, recall, latência, custo por transação.
- **Procedimentos:**
  - **Playbook em incidentes:** fallback para modelo baseline, logging detalhado e revisão de regras.
- **Treinamento e adoção:**
  - Treinamento de equipe para interpretar relatórios; canal de feedback com time de dados.

---

**Observações finais**

- Este projeto foi desenhado para ser modular, auditável e evolutivo.
- O ecossistema de qualidade (Ruff, Black, Isort, Mypy, Pytest, CI) reduz riscos operacionais e melhora a manutenibilidade.
- Os notebooks complementam a transparência analítica e decisões orientadas a custos.

> Repositório público e estrutura disponível no GitHub.



---




**Autor:**
  Sergio Santos 


---

**Contato:**


[![Portfólio Sérgio Santos](https://img.shields.io/badge/Portfólio-Sérgio_Santos-111827?style=for-the-badge&logo=githubpages&logoColor=00eaff)](https://santosdevbjj.github.io/portfolio/)
[![LinkedIn Sérgio Santos](https://img.shields.io/badge/LinkedIn-Sérgio_Santos-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/santossergioluiz) 

---






