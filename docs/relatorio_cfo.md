# Relatório Financeiro — CFO

**Projeto:** Sistema de Prevenção de Fraudes em Pix em Tempo Real
**Responsável:** Sérgio Santos — Cientista de Dados
**Data:** 2026 | **Versão:** 1.0
**Classificação:** Confidencial — Uso Interno

---

## 📌 Sumário Executivo Financeiro

Este relatório traduz a performance do sistema de detecção de fraudes em Pix em **impacto financeiro mensurável em Reais** — eliminando a ambiguidade de métricas estatísticas e conectando diretamente cada decisão técnica ao resultado no caixa.

**Os três números que importam:**

| Indicador | O que representa |
|---|---|
| **Custo de um Falso Negativo (FN)** | Valor médio de uma fraude aprovada — perda direta e irreversível |
| **Custo de um Falso Positivo (FP)** | Custo operacional de um bloqueio indevido + risco de churn |
| **Threshold ótimo** | O ponto de decisão que **minimiza o custo total** para a instituição |

> **A premissa central deste relatório:** a melhor métrica de Machine Learning não é necessariamente a mais lucrativa. Um modelo com AUC de 0.95 pode custar mais do que um modelo com AUC de 0.88 se o threshold de decisão estiver errado. Este relatório mostra como calibrar o threshold pelo custo real — não pela estatística.

---

## 📑 Sumário

- [1. O Modelo de Custo](#1-o-modelo-de-custo)
- [2. Parâmetros Financeiros de Referência](#2-parâmetros-financeiros-de-referência)
- [3. A Lógica do Threshold e seu Impacto Financeiro](#3-a-lógica-do-threshold-e-seu-impacto-financeiro)
- [4. Threshold Analysis em R$ — Simulações](#4-threshold-analysis-em-r-simulações)
- [5. Comparativo Financeiro: Regras Estáticas vs. ML](#5-comparativo-financeiro-regras-estáticas-vs-ml)
- [6. Sensibilidade Financeira por Cenário](#6-sensibilidade-financeira-por-cenário)
- [7. ROI do Projeto](#7-roi-do-projeto)
- [8. Governança Financeira do Modelo](#8-governança-financeira-do-modelo)
- [9. Recomendações ao CFO](#9-recomendações-ao-cfo)
- [10. Glossário Financeiro](#10-glossário-financeiro)

---

## 1. O Modelo de Custo

### A função que governa todas as decisões do sistema

Todo sistema de detecção de fraudes comete dois tipos de erro. A questão não é eliminar os erros — é **escolher conscientemente qual erro custa menos** para a instituição.

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   Custo Total = (C_FN × FN) + (C_FP × FP)                  ║
║                                                              ║
║   Onde:                                                      ║
║   C_FN  = Custo médio de uma fraude aprovada (R$)           ║
║   FN    = Número de fraudes aprovadas (Falsos Negativos)    ║
║   C_FP  = Custo de um bloqueio indevido (R$)                ║
║   FP    = Número de bloqueios indevidos (Falsos Positivos)  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Por que essa fórmula muda tudo

A maioria dos sistemas de fraude é calibrada para maximizar métricas estatísticas (AUC, F1). Essa abordagem ignora um fato crítico: **os dois tipos de erro têm custos completamente diferentes**.

Em fraudes Pix:

- **Falso Negativo (fraude aprovada):** perda direta do valor transacionado + custo de reembolso ao cliente + custo operacional de investigação + risco regulatório. Tipicamente na ordem de **centenas a milhares de reais** por ocorrência.

- **Falso Positivo (bloqueio indevido):** custo de atendimento ao cliente + risco de churn + impacto no NPS. Tipicamente na ordem de **dezenas de reais** por ocorrência — mas com volume muito maior.

Essa assimetria é o fundamento de toda a calibração do modelo.

---

## 2. Parâmetros Financeiros de Referência

Os valores abaixo são **referências de mercado para modelagem**. A instituição deve substituí-los pelos valores reais da sua operação para calibração precisa do threshold.

### Custo de um Falso Negativo — C_FN

| Componente de Custo | Valor de Referência | Observação |
|---|---|---|
| Valor médio da transação fraudulenta | R$ 800 – R$ 3.500 | Varia conforme perfil da base |
| Reembolso ao cliente lesado | 100% do valor (regulatório) | Obrigatório em fraudes confirmadas |
| Custo de investigação e back-office | R$ 150 – R$ 400 por caso | Analistas de fraude + sistemas |
| Risco regulatório (multas BCB) | Variável | Depende de histórico de conformidade |
| **C_FN estimado (conservador)** | **R$ 1.200** | Base para modelagem neste relatório |
| **C_FN estimado (agressivo)** | **R$ 4.500** | Inclui casos de alto valor |

### Custo de um Falso Positivo — C_FP

| Componente de Custo | Valor de Referência | Observação |
|---|---|---|
| Custo de atendimento (SAC/chat) | R$ 15 – R$ 45 por contato | Cliente ligando para desbloquear |
| Probabilidade de churn | 8% – 15% dos clientes bloqueados | Clientes que abandonam após bloqueio |
| LTV médio do cliente | R$ 600 – R$ 2.400/ano | Depende do produto e perfil |
| Custo de churn por bloqueio | R$ 48 – R$ 360 | Probabilidade × LTV |
| Impacto em NPS | Não monetizado diretamente | Mas afeta CAC futuro |
| **C_FP estimado (conservador)** | **R$ 65** | Base para modelagem neste relatório |
| **C_FP estimado (agressivo)** | **R$ 400** | Inclui churn de clientes premium |

### Razão de Assimetria

```
Razão de Assimetria = C_FN ÷ C_FP

Cenário conservador: R$ 1.200 ÷ R$ 65  = 18,5x
Cenário agressivo:   R$ 4.500 ÷ R$ 400 = 11,25x
```

**Interpretação:** aprovar uma fraude custa entre **11 a 19 vezes mais** do que bloquear um cliente indevido. Isso justifica um sistema mais sensível à detecção de fraudes — mesmo que isso gere mais bloqueios indevidos.

---

## 3. A Lógica do Threshold e seu Impacto Financeiro

### O que é o threshold de decisão

O modelo de Machine Learning não retorna uma decisão binária — ele retorna uma **probabilidade de fraude** entre 0 e 1 para cada transação.

```
Probabilidade 0.03  → Muito provavelmente legítima
Probabilidade 0.47  → Incerta — depende do threshold
Probabilidade 0.91  → Muito provavelmente fraude
```

O **threshold** é o ponto de corte: acima dele, a transação é sinalizada como fraude. Abaixo, é aprovada.

### Por que o threshold padrão de 0.5 está errado para fraudes

O threshold de 0.5 assume que os dois tipos de erro têm **o mesmo custo**. Já vimos que não têm — fraudes custam 11 a 19 vezes mais do que bloqueios indevidos.

A consequência prática: **o threshold ótimo para minimizar custo financeiro é significativamente menor que 0.5**.

```
Threshold 0.5  → Equilibra erros estatisticamente (mas não financeiramente)
Threshold 0.3  → Detecta mais fraudes, aceita mais bloqueios indevidos
Threshold 0.2  → Maximiza detecção, custo operacional de revisão maior
Threshold 0.4  → Zona intermediária — avaliada no Threshold Analysis
```

### Como encontrar o threshold ótimo

O notebook `03thresholdanalysis.ipynb` varia o threshold sistematicamente de 0.05 a 0.95 e calcula o **Custo Total em R$** para cada ponto:

```python
# Lógica implementada no notebook
for threshold in np.arange(0.05, 0.95, 0.05):
    FP = sum((probabilidade >= threshold) & (real == 0))
    FN = sum((probabilidade < threshold) & (real == 1))
    
    custo_total = (C_FN * FN) + (C_FP * FP)
    
    resultados.append({
        'threshold': threshold,
        'FN': FN,
        'FP': FP,
        'custo_total_R$': custo_total
    })

threshold_otimo = resultados.loc[resultados['custo_total_R$'].idxmin()]
```

O threshold que produz o **menor Custo Total em R$** é o threshold ótimo — não o que maximiza AUC ou F1.

---

## 4. Threshold Analysis em R$ — Simulações

### Cenário Base: 500.000 transações/mês | Taxa de fraude: 0,15%

**Volume de referência:**
- Total de transações: 500.000/mês
- Fraudes reais: 750/mês (0,15%)
- Transações legítimas: 499.250/mês

---

### Simulação por Threshold — Custo Total Mensal (C_FN = R$ 1.200 | C_FP = R$ 65)

| Threshold | Fraudes Detectadas | Fraudes Aprovadas (FN) | Bloqueios Indevidos (FP) | Custo FN (R$) | Custo FP (R$) | **Custo Total (R$)** |
|---|---|---|---|---|---|---|
| 0,10 | 742 (98,9%) | 8 | 4.820 | R$ 9.600 | R$ 313.300 | **R$ 322.900** |
| 0,15 | 735 (98,0%) | 15 | 3.245 | R$ 18.000 | R$ 210.925 | **R$ 228.925** |
| 0,20 | 720 (96,0%) | 30 | 1.980 | R$ 36.000 | R$ 128.700 | **R$ 164.700** |
| 0,25 | 705 (94,0%) | 45 | 1.120 | R$ 54.000 | R$ 72.800 | **R$ 126.800** |
| **0,30** | **683 (91,1%)** | **67** | **612** | **R$ 80.400** | **R$ 39.780** | **✅ R$ 120.180** |
| 0,35 | 660 (88,0%) | 90 | 340 | R$ 108.000 | R$ 22.100 | **R$ 130.100** |
| 0,40 | 630 (84,0%) | 120 | 180 | R$ 144.000 | R$ 11.700 | **R$ 155.700** |
| 0,50 | 570 (76,0%) | 180 | 75 | R$ 216.000 | R$ 4.875 | **R$ 220.875** |
| 0,60 | 495 (66,0%) | 255 | 30 | R$ 306.000 | R$ 1.950 | **R$ 307.950** |
| 0,70 | 390 (52,0%) | 360 | 12 | R$ 432.000 | R$ 780 | **R$ 432.780** |

> ✅ **Threshold ótimo identificado: 0,30** — Custo Total de **R$ 120.180/mês** no cenário base.

---

### Visualização da Curva de Custo

```
Custo Total (R$)
    │
330k┤ ●
    │  ╲
230k┤   ●
    │    ╲
165k┤     ●
    │      ╲
127k┤       ●
    │        ╲
120k┤         ● ← ÓTIMO (threshold 0.30)
    │          ╲
130k┤           ●
    │            ╲___
156k┤                ●
    │                 ╲____
221k┤                       ●
    │                        ╲_______
308k┤                                 ●
    │
    └──────────────────────────────────────
   0.10  0.15  0.20  0.25  0.30  0.35  0.40  0.50
                                         Threshold
```

**Leitura da curva:**
- À esquerda do ótimo: custo alto por excesso de bloqueios indevidos (FP dominante)
- À direita do ótimo: custo alto por excesso de fraudes aprovadas (FN dominante)
- No ótimo: equilíbrio financeiro entre os dois tipos de erro

---

### Comparativo com threshold padrão 0.5

| Métrica | Threshold 0.5 (padrão) | Threshold 0.30 (ótimo) | Ganho |
|---|---|---|---|
| Fraudes detectadas | 570 (76%) | 683 (91,1%) | +113 fraudes/mês |
| Fraudes aprovadas (FN) | 180 | 67 | -113 fraudes/mês |
| Bloqueios indevidos (FP) | 75 | 612 | +537 bloqueios/mês |
| Custo com FN (R$) | R$ 216.000 | R$ 80.400 | -R$ 135.600 |
| Custo com FP (R$) | R$ 4.875 | R$ 39.780 | +R$ 34.905 |
| **Custo Total (R$)** | **R$ 220.875** | **R$ 120.180** | **-R$ 100.695/mês** |

> **Conclusão:** a simples otimização do threshold — sem mudar o modelo — reduz o custo total em **R$ 100.695/mês** (R$ 1.208.340/ano) neste cenário.

---

## 5. Comparativo Financeiro: Regras Estáticas vs. ML

### Limitações financeiras das regras estáticas

Regras estáticas operam com um threshold fixo e implícito — geralmente definido por intuição da equipe de risco, não por otimização financeira. Isso resulta em:

- **Threshold sub-ótimo:** definido uma vez, nunca recalibrado conforme o custo dos erros muda.
- **Sem aprendizado:** novos padrões de fraude não são detectados até que alguém atualize a regra manualmente.
- **Sem auditabilidade de custo:** não existe visibilidade de quanto cada decisão de aprovação/bloqueio custa.

### Análise financeira comparativa

Assumindo que o sistema de regras estáticas opera com comportamento equivalente ao threshold 0.5 (comum em sistemas de regras conservadoras):

| Dimensão | Regras Estáticas | Sistema ML (threshold ótimo) | Diferença |
|---|---|---|---|
| Fraudes aprovadas/mês | ~180 | ~67 | -113 fraudes |
| Perda por fraudes/mês | R$ 216.000 | R$ 80.400 | **-R$ 135.600** |
| Bloqueios indevidos/mês | ~75 | ~612 | +537 bloqueios |
| Custo operacional FP/mês | R$ 4.875 | R$ 39.780 | +R$ 34.905 |
| **Custo Total/mês** | **R$ 220.875** | **R$ 120.180** | **-R$ 100.695** |
| **Custo Total/ano** | **R$ 2.650.500** | **R$ 1.442.160** | **-R$ 1.208.340** |

> **O sistema ML reduz o custo total anual em R$ 1.208.340** neste cenário — pagando o investimento no projeto em menos de um mês de operação.

---

## 6. Sensibilidade Financeira por Cenário

A análise de sensibilidade mostra como o ganho financeiro varia conforme o volume de transações e a taxa de fraude da operação real.

### Variação por Volume de Transações

*(Taxa de fraude constante: 0,15% | Threshold ótimo aplicado)*

| Volume Mensal | Fraudes/mês | Economia vs. threshold 0.5 | Economia Anual |
|---|---|---|---|
| 100.000 tx/mês | 150 fraudes | R$ 20.139/mês | **R$ 241.668/ano** |
| 250.000 tx/mês | 375 fraudes | R$ 50.348/mês | **R$ 604.170/ano** |
| **500.000 tx/mês** | **750 fraudes** | **R$ 100.695/mês** | **R$ 1.208.340/ano** |
| 1.000.000 tx/mês | 1.500 fraudes | R$ 201.390/mês | **R$ 2.416.680/ano** |
| 5.000.000 tx/mês | 7.500 fraudes | R$ 1.006.950/mês | **R$ 12.083.400/ano** |

### Variação por Taxa de Fraude

*(Volume constante: 500.000 tx/mês | Threshold ótimo aplicado)*

| Taxa de Fraude | Fraudes/mês | Economia vs. threshold 0.5 | Economia Anual |
|---|---|---|---|
| 0,05% | 250 fraudes | R$ 33.565/mês | **R$ 402.780/ano** |
| 0,10% | 500 fraudes | R$ 67.130/mês | **R$ 805.560/ano** |
| **0,15%** | **750 fraudes** | **R$ 100.695/mês** | **R$ 1.208.340/ano** |
| 0,25% | 1.250 fraudes | R$ 167.825/mês | **R$ 2.013.900/ano** |
| 0,50% | 2.500 fraudes | R$ 335.650/mês | **R$ 4.027.800/ano** |

> **Quanto maior o volume e a taxa de fraude, maior o retorno financeiro do sistema ML** — o custo do projeto é essencialmente fixo, enquanto o ganho escala com a operação.

### Variação por Custo de Fraude (C_FN)

*(Volume: 500.000 tx/mês | Taxa: 0,15% | Threshold ótimo)*

| Custo médio por fraude (C_FN) | Economia mensal | Economia anual |
|---|---|---|
| R$ 500 | R$ 42.000/mês | **R$ 504.000/ano** |
| R$ 800 | R$ 67.200/mês | **R$ 806.400/ano** |
| **R$ 1.200** | **R$ 100.695/mês** | **R$ 1.208.340/ano** |
| R$ 2.000 | R$ 167.825/mês | **R$ 2.013.900/ano** |
| R$ 4.500 | R$ 377.100/mês | **R$ 4.525.200/ano** |

---

## 7. ROI do Projeto

### Investimento estimado

| Item | Custo Estimado |
|---|---|
| Desenvolvimento do sistema (já realizado) | — |
| Infraestrutura de produção (cloud/API) | R$ 3.000 – R$ 8.000/mês |
| Manutenção e monitoramento (1 analista parcial) | R$ 5.000 – R$ 10.000/mês |
| Re-treino trimestral | R$ 2.000 – R$ 5.000/trimestre |
| **Custo operacional total estimado** | **R$ 9.000 – R$ 20.000/mês** |

### ROI no cenário base (500.000 tx/mês)

| Métrica | Valor |
|---|---|
| Economia gerada pelo sistema | R$ 100.695/mês |
| Custo operacional (ponto médio) | R$ 14.500/mês |
| **Lucro líquido do sistema** | **R$ 86.195/mês** |
| **ROI mensal** | **594%** |
| **Payback do projeto** | **< 1 mês de operação** |
| **Economia líquida anual** | **R$ 1.034.340/ano** |

> O sistema se paga integralmente no **primeiro mês de operação em produção**.

---

## 8. Governança Financeira do Modelo

### Rastreabilidade de decisões

Cada transação processada pelo sistema registra automaticamente:

| Dado registrado | Finalidade |
|---|---|
| Probabilidade de fraude atribuída | Auditoria de decisões individuais |
| Threshold vigente no momento da decisão | Rastreabilidade histórica de calibrações |
| Versão do modelo utilizado | Controle de qual modelo tomou cada decisão |
| Timestamp da predição | Conformidade regulatória |
| Features utilizadas (resumo) | Explicabilidade para o Banco Central |

### Política de recalibração do threshold

O threshold ótimo não é permanente — ele muda conforme:

- O volume de transações cresce.
- O perfil de fraudes evolui (novas modalidades).
- O custo médio das fraudes muda (sazonalidade, inflação).
- A operação expande para novos produtos ou públicos.

**Recomendação:** recalibração trimestral com os dados reais da operação. O notebook `03thresholdanalysis.ipynb` está preparado para receber os parâmetros financeiros atualizados e recalcular o threshold ótimo em minutos.

### Alertas e monitoramento contínuo de KPIs

| KPI Financeiro | Frequência de monitoramento | Ação se fora do limite |
|---|---|---|
| Custo total de fraude mensal | Diário | Acionar recalibração de threshold |
| Custo médio por fraude aprovada | Semanal | Atualizar C_FN no modelo de custo |
| Taxa de bloqueios indevidos | Diário | Verificar drift de features |
| Custo operacional de FP | Mensal | Revisar C_FP no modelo de custo |
| Variação do Custo Total vs. meta | Mensal | Relatório para CFO |

### Controle de versão de modelos

Todos os modelos são versionados com:

- Data de treino.
- Parâmetros financeiros (C_FN e C_FP) utilizados na calibração do threshold.
- Threshold vigente.
- Métricas de performance no momento do deploy.

Isso garante rastreabilidade completa: para qualquer transação histórica, é possível saber **exatamente qual modelo, com qual threshold e quais parâmetros financeiros** tomou a decisão.

---

## 9. Recomendações ao CFO

### Ação imediata — Parametrização financeira real

O modelo de custo utilizado neste relatório usa valores de referência de mercado. Para calibração precisa do threshold com os dados reais da instituição, são necessários:

**Dados solicitados à área de Risco e Finanças:**

- [ ] Valor médio das fraudes Pix confirmadas nos últimos 12 meses (C_FN base).
- [ ] Custo médio de atendimento por cliente bloqueado indevidamente (SAC + back-office).
- [ ] Taxa de churn observada após bloqueio indevido.
- [ ] LTV médio do cliente da base Pix.
- [ ] Volume mensal de transações Pix atual.
- [ ] Taxa de fraude atual (confirmada pela área de Risco).

Com esses dados, o threshold ótimo pode ser recalculado em **menos de 1 hora** e o modelo entra em produção calibrado com os números reais da operação.

### Recomendação 1 — Adotar threshold dinâmico por segmento

Diferentes segmentos de clientes têm perfis de risco e LTV diferentes. A recomendação de médio prazo é adotar thresholds diferenciados:

| Segmento | Threshold sugerido | Justificativa |
|---|---|---|
| Cliente PF — baixo volume | 0,35 | Menor C_FN esperado, maior risco de churn |
| Cliente PF — alto volume | 0,25 | Maior C_FN, cliente tolera mais fricção |
| Cliente PJ | 0,20 | C_FN muito alto, fraudes de maior valor |
| API / integração | 0,15 | Risco elevado, menor impacto de UX |

### Recomendação 2 — Revisão trimestral obrigatória

Recalibração do threshold a cada trimestre com dados reais atualizados. O custo da não-recalibração é equivalente a operar com o threshold errado — potencialmente custando dezenas a centenas de milhares de reais por mês.

### Recomendação 3 — Budget para escala

O custo operacional do sistema é essencialmente fixo independente do volume de transações. Isso significa que o **ROI cresce proporcionalmente ao crescimento da operação** — sem crescimento proporcional do custo.

Projeção de economia líquida conforme a base de usuários cresce:

| Horizonte | Volume projetado | Economia líquida anual estimada |
|---|---|---|
| Hoje | 500k tx/mês | R$ 1.034.340/ano |
| 12 meses | 1M tx/mês | R$ 2.068.680/ano |
| 24 meses | 2,5M tx/mês | R$ 5.171.700/ano |
| 36 meses | 5M tx/mês | R$ 10.343.400/ano |

---

## 10. Glossário Financeiro

| Termo | Definição para este relatório |
|---|---|
| **Falso Negativo (FN)** | Fraude real que o sistema aprovou — perda financeira direta |
| **Falso Positivo (FP)** | Transação legítima que o sistema bloqueou — custo operacional + risco de churn |
| **C_FN** | Custo médio total de uma fraude aprovada (valor perdido + reembolso + operacional) |
| **C_FP** | Custo médio total de um bloqueio indevido (atendimento + risco de churn × LTV) |
| **Threshold** | Limiar de probabilidade acima do qual o sistema classifica a transação como fraude |
| **Threshold Ótimo** | Valor de threshold que minimiza o Custo Total = (C_FN × FN) + (C_FP × FP) |
| **ROC-AUC** | Métrica estatística de separabilidade do modelo — NÃO é a métrica de decisão financeira |
| **PR-AUC** | Área sob a curva Precisão-Recall — mais relevante que ROC-AUC para fraudes (dados desbalanceados) |
| **Recall (Sensibilidade)** | % de fraudes reais que o sistema detectou — diretamente ligado ao C_FN |
| **Precisão** | % dos alertas de fraude que eram fraudes reais — diretamente ligado ao C_FP |
| **Drift** | Mudança na distribuição dos dados em produção que degrada o modelo ao longo do tempo |
| **LTV** | Lifetime Value — receita total esperada de um cliente durante seu relacionamento com a instituição |

---

⬅️ **[Voltar para o README Principal](../README.md)**

---

**Elaborado por:** Sergio Santos — Cientista de Dados

[![Portfólio Sérgio Santos](https://img.shields.io/badge/Portfólio-Sérgio_Santos-111827?style=for-the-badge&logo=githubpages&logoColor=00eaff)](https://portfoliosantossergio.vercel.app)

[![LinkedIn Sérgio Santos](https://img.shields.io/badge/LinkedIn-Sérgio_Santos-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/santossergioluiz)
