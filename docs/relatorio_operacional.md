# Relatório Operacional — Gerente de Risco e Operações

**Projeto:** Sistema de Prevenção de Fraudes em Pix em Tempo Real
**Responsável:** Sérgio Santos — Cientista de Dados
**Data:** 2026 | **Versão:** 1.0
**Classificação:** Interno — Equipe de Risco, Dados e Engenharia

---

## 📌 Resumo Operacional

Este documento é o **guia operacional do dia a dia** do sistema de prevenção de fraudes em Pix. Ele cobre três áreas críticas para quem opera e monitora o sistema em produção:

- **KPIs:** quais métricas acompanhar, com qual frequência e quais limites disparam ação.
- **Monitoramento:** como identificar degradação do modelo antes que ela gere prejuízo.
- **Playbook de Incidentes:** o que fazer — passo a passo — quando algo sai errado.

> **Premissa operacional:** um modelo de fraude que não é monitorado ativamente é um modelo que degrada silenciosamente. Fraudes evoluem. O comportamento dos usuários muda. Sem monitoramento, o sistema perde efetividade sem que ninguém perceba — até que os números de fraude comecem a subir.

---

## 📑 Sumário

- [1. Arquitetura Operacional](#1-arquitetura-operacional)
- [2. KPIs Operacionais](#2-kpis-operacionais)
- [3. Dashboard de Monitoramento](#3-dashboard-de-monitoramento)
- [4. Monitoramento de Drift](#4-monitoramento-de-drift)
- [5. Processo Operacional Diário](#5-processo-operacional-diário)
- [6. Playbook de Incidentes](#6-playbook-de-incidentes)
- [7. Procedimento de Re-treino](#7-procedimento-de-re-treino)
- [8. Gestão de Threshold](#8-gestão-de-threshold)
- [9. Procedimento de Rollback](#9-procedimento-de-rollback)
- [10. Canais de Comunicação e Escalada](#10-canais-de-comunicação-e-escalada)
- [11. Checklist de Operação](#11-checklist-de-operação)
- [12. Glossário Operacional](#12-glossário-operacional)

---

## 1. Arquitetura Operacional

### Fluxo de uma transação em produção

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRANSAÇÃO PIX INICIADA                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼ (< 5ms)
┌─────────────────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING (Polars)                                    │
│  • Janelas temporais: count_1min, sum_valor_5min, count_dest_1h │
│  • Encoding: tipo_chave_pix, dispositivo                        │
│  • Normalização: StandardScaler (artefato salvo)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼ (< 10ms)
┌─────────────────────────────────────────────────────────────────┐
│  SCORING — XGBoost (inference.py)                               │
│  • Carrega artefato: models/artifacts/xgb_model.joblib          │
│  • Retorna: probabilidade de fraude [0.0 – 1.0]                 │
│  • Registra: probabilidade + versão do modelo + timestamp        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼ (< 5ms)
┌─────────────────────────────────────────────────────────────────┐
│  DECISÃO — Comparação com Threshold                              │
│  • P(fraude) ≥ threshold → SINALIZA para revisão                │
│  • P(fraude) < threshold → APROVA automaticamente               │
│  • Threshold vigente: registrado no log de cada decisão         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
    ┌──────────────────┐     ┌─────────────────────────┐
    │ APROVADA         │     │ SINALIZADA               │
    │ < 50ms total     │     │ → Fila de revisão manual │
    │ Cliente não      │     │ → Notificação à equipe   │
    │ percebe processo │     │   de risco               │
    └──────────────────┘     └─────────────────────────┘
```

### Componentes em produção

| Componente | Localização | Responsável |
|---|---|---|
| Pipeline de features | `src/features/timewindowspolars.py` | Time de Dados |
| Modelo XGBoost | `models/artifacts/xgb_model.joblib` | Time de Dados |
| Serviço de inferência | `src/modeling/inference.py` | Engenharia |
| Logs de decisão | `models/reports/` | Time de Dados |
| Testes de latência | `tests/testinferencelatency.py` | Engenharia |
| CI/CD | `.github/workflows/ci.yaml` | Engenharia |

---

## 2. KPIs Operacionais

### 2.1 KPIs Primários — Monitoramento Diário

Estes são os indicadores que determinam se o sistema está funcionando corretamente. Qualquer desvio dos limites abaixo exige ação imediata.

| KPI | Descrição | Meta | Limite de Alerta | Limite Crítico |
|---|---|---|---|---|
| **Taxa de Detecção (Recall)** | % das fraudes reais identificadas pelo sistema | > baseline de regras estáticas | Queda de 5pp vs. semana anterior | Queda de 10pp vs. semana anterior |
| **Taxa de Falsos Positivos (FPR)** | % das transações legítimas bloqueadas indevidamente | < 0,15% | > 0,20% | > 0,30% |
| **Latência P50 de Inferência** | Tempo mediano de resposta do modelo | < 30ms | > 40ms | > 50ms |
| **Latência P95 de Inferência** | Tempo de resposta no percentil 95 | < 50ms | > 60ms | > 80ms |
| **Latência P99 de Inferência** | Tempo de resposta no percentil 99 | < 80ms | > 100ms | > 120ms |
| **Disponibilidade do Serviço** | % do tempo com sistema respondendo | > 99,9% | < 99,5% | < 99,0% |
| **Taxa de Erros de Inferência** | % de predições que retornaram erro | < 0,01% | > 0,05% | > 0,1% |

### 2.2 KPIs Financeiros — Monitoramento Semanal

| KPI | Descrição | Fórmula | Frequência |
|---|---|---|---|
| **Custo Total de Fraude** | Custo financeiro total dos dois tipos de erro | (C_FN × FN) + (C_FP × FP) | Semanal |
| **Perda por Fraudes Aprovadas** | Valor total de fraudes não detectadas | Soma dos valores das transações FN | Diário |
| **Custo Operacional de FP** | Custo total dos bloqueios indevidos | C_FP × quantidade de FP | Semanal |
| **Custo Médio por Fraude** | Valor médio das fraudes aprovadas | Total perdido ÷ quantidade de FN | Mensal |
| **ROI do Sistema** | Retorno sobre o investimento no modelo | Economia gerada ÷ custo operacional | Mensal |

### 2.3 KPIs de Qualidade do Modelo — Monitoramento Semanal

| KPI | Descrição | Meta | Ação se fora do limite |
|---|---|---|---|
| **PR-AUC** | Área sob curva Precisão-Recall | > PR-AUC do baseline | Investigar drift; considerar re-treino |
| **ROC-AUC** | Área sob curva ROC | > ROC-AUC do baseline | Investigar drift; considerar re-treino |
| **F1-Score (Fraudes)** | Equilíbrio entre Precisão e Recall | > F1 do baseline | Investigar drift |
| **Distribuição de Scores** | Histograma das probabilidades geradas | Estável vs. semana anterior | Possível drift — investigar |
| **PSI (Population Stability Index)** | Drift nas features de entrada | PSI < 0,10 | PSI 0,10–0,25: atenção; > 0,25: re-treino |

### 2.4 KPIs de Experiência do Cliente — Monitoramento Semanal

| KPI | Descrição | Meta | Fonte de dados |
|---|---|---|---|
| **Contatos pós-bloqueio** | Volume de clientes que acionam SAC após bloqueio | Tendência de queda | CRM / SAC |
| **Tempo de resolução de bloqueio** | Tempo médio para liberar transação bloqueada indevidamente | < 2 horas | Sistema de tickets |
| **NPS pós-bloqueio** | Satisfação do cliente após ser bloqueado | > NPS médio geral | Pesquisa pós-atendimento |
| **Churn pós-bloqueio** | % de clientes que cancelam após bloqueio indevido | < 8% | Analytics de produto |

---

## 3. Dashboard de Monitoramento

### 3.1 Visão sugerida do painel operacional diário

```
╔══════════════════════════════════════════════════════════════════╗
║          SISTEMA DE FRAUDES PIX — PAINEL OPERACIONAL             ║
║                    Atualização: Tempo Real                        ║
╠══════════════════════╦═══════════════╦═══════════════════════════╣
║  LATÊNCIA            ║  DETECÇÃO     ║  VOLUME                   ║
║  P50:  28ms  ✅      ║  Recall: 91%  ║  Tx hoje:    48.230       ║
║  P95:  44ms  ✅      ║  FPR:  0,12%  ║  Alertas:       58        ║
║  P99:  71ms  ✅      ║  PR-AUC: 0.87 ║  Revisão manual: 12       ║
╠══════════════════════╬═══════════════╬═══════════════════════════╣
║  CUSTO HOJE          ║  MODELO       ║  SISTEMA                  ║
║  Fraudes: R$ 3.600   ║  v2.1.0       ║  Disponib: 99,97%  ✅     ║
║  FP op:   R$ 754     ║  Threshold:   ║  Erros:     0,003% ✅     ║
║  Total:   R$ 4.354   ║  0,30         ║  CI/CD: ✅ Passing        ║
╠══════════════════════╩═══════════════╩═══════════════════════════╣
║  ALERTAS ATIVOS                                                   ║
║  ✅ Nenhum alerta crítico ativo                                   ║
╚══════════════════════════════════════════════════════════════════╝
```

### 3.2 Logs de decisão — estrutura de cada registro

Cada transação processada gera um log com a seguinte estrutura:

```json
{
  "transaction_id": "pix_20250315_084523_abc123",
  "timestamp": "2025-03-15T08:45:23.412Z",
  "fraud_probability": 0.73,
  "threshold": 0.30,
  "decision": "FLAGGED",
  "model_version": "xgb_v2.1.0",
  "latency_ms": 34,
  "top_features": {
    "count_1min": 8,
    "sum_valor_5min": 4200.00,
    "count_dest_1h": 5,
    "dispositivo": "api"
  }
}
```

### 3.3 Relatórios automáticos gerados pelo pipeline

| Relatório | Localização | Conteúdo | Quando gerado |
|---|---|---|---|
| Métricas do modelo | `models/reports/metrics.json` | PR-AUC, ROC-AUC, F1, Recall, Precisão | A cada treino |
| Curva ROC | `models/reports/roc_curve.png` | Curva ROC comparativa (LogReg vs. XGBoost) | A cada treino |
| Curva PR | `models/reports/pr_curve.png` | Curva Precisão-Recall | A cada treino |
| Confusion Matrix | `models/reports/confusion_matrix.png` | Matriz de confusão com valores absolutos | A cada treino |
| Threshold Analysis | `models/reports/threshold_analysis.json` | Custo total por threshold | A cada treino |
| Latência de inferência | `models/reports/latency_report.json` | P50, P95, P99 de latência | A cada teste |

---

## 4. Monitoramento de Drift

### O que é drift e por que é crítico

Drift é a mudança gradual no comportamento dos dados em produção em relação aos dados de treino. Para sistemas de fraude, drift é inevitável — fraudadores adaptam suas técnicas continuamente.

Existem dois tipos de drift que o sistema monitora:

**Data Drift (Drift de Features)**
A distribuição dos dados de entrada muda. Exemplo: o volume médio de transações por usuário aumentou significativamente, ou um novo tipo de dispositivo começou a aparecer com frequência.

**Concept Drift (Drift de Conceito)**
A relação entre as features e o rótulo de fraude muda. Exemplo: transações que antes tinham probabilidade 0.2 de ser fraude agora são fraudulentas 60% do tempo — o padrão de fraude evoluiu.

### Métrica de detecção de drift — PSI (Population Stability Index)

```
PSI = Σ (% atual - % treino) × ln(% atual / % treino)

PSI < 0,10   → Distribuição estável — sem ação necessária
PSI 0,10–0,25 → Mudança moderada — monitorar com atenção
PSI > 0,25   → Mudança significativa — acionar protocolo de re-treino
```

### Features monitoradas para drift

| Feature | Frequência de verificação | Ação se PSI > 0,25 |
|---|---|---|
| `count_1min` | Diária | Acionar protocolo de investigação |
| `sum_valor_5min` | Diária | Acionar protocolo de investigação |
| `count_dest_1h` | Diária | Acionar protocolo de investigação |
| `tipo_chave_pix` | Semanal | Verificar novos tipos de chave |
| `dispositivo` | Semanal | Verificar novos dispositivos |
| Distribuição de `fraud_probability` | Diária | Se mudança > 15%: alerta imediato |

### Sinais de alerta de drift no comportamento do modelo

Além do PSI, estes sinais indicam possível drift e devem acionar investigação:

- Queda de 5+ pontos percentuais no Recall vs. semana anterior.
- Aumento de 30%+ no volume de alertas sem explicação por sazonalidade.
- Distribuição de scores se concentrando em faixas muito altas (> 0,9) ou muito baixas (< 0,1).
- Aumento de reclamações de clientes sobre bloqueios em produtos ou regiões específicas.
- Novo tipo de golpe reportado pela mídia ou pelo Banco Central.

---

## 5. Processo Operacional Diário

### Rotina matinal — primeiros 15 minutos do dia

```
08:00 — Verificar dashboard: latência, disponibilidade, taxa de erros
08:05 — Verificar volume de alertas de fraude vs. dia anterior
08:08 — Verificar distribuição de scores (sem mudanças abruptas?)
08:10 — Revisar logs de erros de inferência do período noturno
08:12 — Verificar status do CI/CD: todos os testes passando?
08:15 — Se tudo normal: registrar "sistema operacional" no canal de comunicação
```

### Rotina semanal — segunda-feira

```
Segunda 09:00 — Gerar relatório semanal de KPIs
Segunda 09:30 — Calcular PSI das principais features vs. semana de treino
Segunda 10:00 — Revisar métricas de qualidade do modelo (PR-AUC, Recall, F1)
Segunda 10:30 — Calcular custo total da semana: (C_FN × FN) + (C_FP × FP)
Segunda 11:00 — Enviar relatório semanal para gerência de risco e CFO
```

### Rotina mensal — primeira semana do mês

```
Semana 1 — Relatório mensal de performance do modelo
Semana 1 — Revisão de threshold: o custo ótimo mudou?
Semana 1 — Revisão de cobertura de testes (meta: > 80%)
Semana 1 — Verificar se há novos padrões de fraude não cobertos pelas features atuais
Semana 1 — Reunião com equipe de risco: novos tipos de golpe reportados?
```

### Rotina trimestral

```
Trimestre 1 — Recalibração formal do threshold com dados reais acumulados
Trimestre 2 — Avaliação de re-treino do modelo com dados mais recentes
Trimestre 3 — Revisão de features: novas variáveis a incluir?
Trimestre 4 — Relatório anual de impacto financeiro e governança
```

---

## 6. Playbook de Incidentes

### Classificação de incidentes

| Severidade | Critério | Tempo de resposta | Escalada |
|---|---|---|---|
| **P1 — Crítico** | Sistema fora do ar OU latência > 80ms OU taxa de erros > 0,1% | Imediato (< 15 min) | Engenharia + Gerência + CTO |
| **P2 — Alto** | Latência P95 > 60ms OU Recall caiu 10+ pp OU FPR > 0,30% | < 1 hora | Engenharia + Time de Dados |
| **P3 — Médio** | Latência P95 > 50ms OU Recall caiu 5+ pp OU PSI > 0,25 | < 4 horas | Time de Dados |
| **P4 — Baixo** | KPI fora do limite de atenção OU drift moderado detectado | < 24 horas | Time de Dados |

---

### INCIDENTE P1-A: Sistema de Inferência Fora do Ar

**Sintomas:** requisições de scoring não respondem OU retornam erro 500 sistematicamente.

```
PASSO 1 — Confirmação (0–5 minutos)
  ├── Verificar logs do serviço de inferência
  ├── Testar endpoint manualmente: poetry run python src/modeling/inference.py --test
  └── Confirmar: é falha do modelo ou falha de infraestrutura?

PASSO 2 — Contenção imediata (5–15 minutos)
  ├── ATIVAR FALLBACK: redirecionar tráfego para sistema de regras estáticas
  ├── Notificar gerência de risco: "Sistema ML em fallback — regras estáticas ativas"
  └── Abrir canal de incidente no Slack: #incidentes-fraude-pix

PASSO 3 — Diagnóstico (15–30 minutos)
  ├── Verificar se artefato existe: ls -la models/artifacts/xgb_model.joblib
  ├── Verificar consumo de memória e CPU do servidor
  ├── Checar logs de erro completos: tail -n 500 logs/inference.log
  └── Verificar última versão deployada: git log --oneline -5

PASSO 4 — Resolução
  ├── SE artefato corrompido → Restaurar última versão estável do Model Registry
  ├── SE erro de código → Rollback para versão anterior (ver Seção 9)
  ├── SE problema de infraestrutura → Acionar Engenharia de Plataforma
  └── SE causa desconhecida → Manter fallback e escalar para CTO

PASSO 5 — Retorno ao normal
  ├── Validar que sistema ML responde corretamente em staging
  ├── Monitorar latência por 30 minutos antes de remover fallback
  ├── Comunicar gerência: "Sistema ML restaurado às HH:MM"
  └── Registrar post-mortem em docs/postmortems/

COMUNICAÇÃO OBRIGATÓRIA:
  → Gerência de Risco: alertar em < 5 minutos do P1
  → CFO: se fallback durar > 1 hora (impacto financeiro direto)
  → Banco Central: se indisponibilidade afetar clientes (verificar obrigação regulatória)
```

---

### INCIDENTE P1-B: Latência Acima de 80ms (Violação de SLA)

**Sintomas:** P95 de latência ultrapassa 80ms consistentemente por mais de 5 minutos.

```
PASSO 1 — Confirmação (0–5 minutos)
  ├── Verificar teste de latência: poetry run pytest tests/testinferencelatency.py -v
  ├── Medir latência atual: poetry run python src/modeling/inference.py --latency-check
  └── Confirmar se é spike pontual ou degradação contínua

PASSO 2 — Identificação da causa (5–20 minutos)
  ├── Verificar load do servidor (CPU, memória, I/O)
  ├── Verificar se features de janela temporal estão lentas (Polars)
  ├── Verificar se artefato está sendo carregado a cada request (erro grave)
  └── Verificar se há volume anormal de transações (pico de demanda)

PASSO 3 — Ações por causa identificada
  ├── SE alto volume → Verificar Auto Scaling; considerar cache de features
  ├── SE artefato recarregando → Corrigir para carregamento único na inicialização
  ├── SE CPU saturada → Escalar instância; notificar Engenharia
  └── SE Polars lento → Verificar tamanho das janelas; considerar otimização

PASSO 4 — Monitoramento pós-resolução
  └── Acompanhar P95 por 1 hora após intervenção antes de fechar incidente
```

---

### INCIDENTE P2-A: Queda Abrupta no Recall (> 10 pontos percentuais)

**Sintomas:** taxa de detecção de fraudes cai abruptamente em relação à semana anterior.

```
PASSO 1 — Confirmação (0–15 minutos)
  ├── Calcular Recall das últimas 24h vs. média das últimas 2 semanas
  ├── Verificar se a queda é real ou causada por mudança no volume de fraudes
  └── Consultar equipe de risco: houve mudança no perfil de fraudes reportadas?

PASSO 2 — Diagnóstico (15–60 minutos)
  ├── Calcular PSI das features principais: há drift significativo?
  ├── Verificar distribuição de scores: está concentrada diferente do habitual?
  ├── Verificar se novo tipo de golpe foi reportado externamente
  └── Verificar se houve mudança não documentada nos dados de entrada

PASSO 3 — Ações por causa identificada
  ├── SE drift confirmado (PSI > 0,25) → Acionar protocolo de re-treino (Seção 7)
  ├── SE novo tipo de fraude → Coletar exemplos; agendar re-treino com novos dados
  ├── SE mudança nos dados de entrada → Corrigir pipeline de ingestão
  └── SE causa desconhecida → Manter monitoramento intensivo; escalar para Time de Dados

PASSO 4 — Comunicação
  └── Notificar gerência de risco: volume de fraudes pode aumentar enquanto investigamos
```

---

### INCIDENTE P2-B: Explosão de Falsos Positivos (FPR > 0,30%)

**Sintomas:** volume anormal de bloqueios indevidos — clientes reclamando de transações legítimas bloqueadas.

```
PASSO 1 — Confirmação (0–15 minutos)
  ├── Calcular FPR das últimas 4 horas
  ├── Verificar volume de contatos no SAC relacionados a bloqueios
  └── Confirmar: o threshold foi alterado recentemente?

PASSO 2 — Ação imediata se impacto for severo
  ├── SE FPR > 0,50%: considerar aumentar threshold temporariamente em 0,05
  ├── Documentar a mudança de threshold com timestamp e responsável
  └── Notificar CFO se impacto de churn for estimado em > R$ 50.000

PASSO 3 — Diagnóstico
  ├── Verificar se há evento externo causando comportamento atípico (feriado, promoção)
  ├── Verificar se houve mudança recente no modelo ou nas features
  ├── Calcular PSI das features: há drift que esteja inflando os scores?
  └── Analisar perfil dos falsos positivos: há padrão específico? (região, produto, dispositivo)

PASSO 4 — Resolução
  ├── SE evento pontual → Monitorar até normalizar; reverter threshold se necessário
  ├── SE drift de features → Acionar re-treino com dados mais recentes
  └── SE erro de código → Rollback (Seção 9)
```

---

### INCIDENTE P3-A: Drift Detectado (PSI > 0,25)

**Sintomas:** PSI de uma ou mais features principais ultrapassa 0,25.

```
PASSO 1 — Identificação (< 4 horas)
  ├── Identificar qual(is) feature(s) apresentam drift
  ├── Comparar distribuição atual vs. distribuição do período de treino
  └── Investigar causa do drift: mudança de comportamento ou mudança no pipeline?

PASSO 2 — Avaliação de impacto
  ├── O Recall está sendo afetado?
  ├── O custo total de fraude aumentou na última semana?
  └── O drift é pontual (evento específico) ou tendência contínua?

PASSO 3 — Decisão
  ├── SE drift pontual + modelo ainda performando → Monitorar por mais 7 dias
  ├── SE drift contínuo OU modelo degradando → Acionar protocolo de re-treino (Seção 7)
  └── SE drift nas features de entrada → Investigar pipeline de dados
```

---

## 7. Procedimento de Re-treino

### Quando re-treinar

O re-treino deve ser acionado quando qualquer um dos critérios abaixo for atendido:

- PSI > 0,25 em qualquer feature principal por mais de 7 dias consecutivos.
- Queda de 5+ pontos percentuais no Recall confirmada por 3 dias consecutivos.
- Novo tipo de fraude documentado pela equipe de risco com volume > 10 casos.
- Recalibração trimestral programada (independente de degradação detectada).
- Mudança significativa nos parâmetros financeiros (C_FN ou C_FP) que justifique novo threshold.

### Protocolo de re-treino passo a passo

```
FASE 1 — PREPARAÇÃO (responsável: Time de Dados)

  1.1 Coletar dados do período de drift
      poetry run python src/pipelines/build_dataset.py \
        --input data/raw/transactions_new_period.parquet \
        --output data/processed/dataset_retrain.parquet

  1.2 Validar schema e guardrails anti-leakage
      poetry run pytest tests/testleakageguards.py -v

  1.3 Documentar período de treino e razão do re-treino
      → Registrar em docs/model_changelog.md

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FASE 2 — TREINO E AVALIAÇÃO (responsável: Time de Dados)

  2.1 Treinar novo modelo
      poetry run python src/pipelines/trainandeval.py \
        --input data/processed/dataset_retrain.parquet \
        --models-out models/artifacts/candidate/ \
        --reports-out models/reports/candidate/

  2.2 Comparar métricas: novo modelo vs. modelo em produção
      → PR-AUC novo ≥ PR-AUC atual? (condição obrigatória)
      → Recall novo ≥ Recall atual?
      → Latência nova ≤ 50ms?

  2.3 Recalibrar threshold com parâmetros financeiros atualizados
      → Abrir notebooks/03thresholdanalysis.ipynb
      → Inserir C_FN e C_FP atuais
      → Identificar novo threshold ótimo
      → Documentar: threshold anterior vs. novo

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FASE 3 — VALIDAÇÃO (responsável: Time de Dados + Engenharia)

  3.1 Testar em staging com batch histórico
      poetry run python src/modeling/inference.py \
        --input data/processed/validation_batch.parquet \
        --model models/artifacts/candidate/xgb_model.joblib \
        --output models/reports/candidate/validation_output.parquet

  3.2 Validar latência do novo modelo
      poetry run pytest tests/testinferencelatency.py -v
      → P95 deve ser < 50ms

  3.3 Validar que CI/CD passa completamente
      → Todos os testes devem passar antes do deploy

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FASE 4 — DEPLOY (responsável: Engenharia)

  4.1 Fazer backup do modelo atual
      cp models/artifacts/xgb_model.joblib \
         models/artifacts/archive/xgb_model_v{versão_atual}_{data}.joblib

  4.2 Deploy do novo modelo
      cp models/artifacts/candidate/xgb_model.joblib \
         models/artifacts/xgb_model.joblib

  4.3 Atualizar versão no registro
      → Atualizar docs/model_changelog.md com nova versão

  4.4 Monitorar intensivamente por 48 horas pós-deploy
      → KPIs a cada 30 minutos nas primeiras 4 horas
      → KPIs a cada 2 horas nas próximas 44 horas

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FASE 5 — COMUNICAÇÃO (responsável: Time de Dados)

  5.1 Comunicar ao gerente de risco: novo modelo em produção
  5.2 Comunicar ao CFO: novo threshold e impacto financeiro estimado
  5.3 Registrar em docs/model_changelog.md:
      - Versão anterior e nova
      - Período de dados de treino
      - Métricas comparativas
      - Threshold anterior e novo
      - Parâmetros financeiros utilizados
      - Responsável pelo re-treino
```

---

## 8. Gestão de Threshold

### Política de alteração de threshold

O threshold é o parâmetro mais sensível do sistema — uma alteração errada pode custar centenas de milhares de reais. Toda alteração deve seguir o processo:

```
1. JUSTIFICATIVA documentada: por que o threshold precisa mudar?
   (ex.: novo C_FN calculado, drift detectado, A/B testing)

2. CÁLCULO: rodar 03thresholdanalysis.ipynb com parâmetros atualizados

3. APROVAÇÃO: gerente de risco + CFO devem aprovar antes do deploy

4. REGISTRO: documentar em docs/threshold_changelog.md
   - Data da alteração
   - Threshold anterior e novo
   - C_FN e C_FP utilizados no cálculo
   - Responsável pela aprovação
   - Impacto esperado em FN e FP

5. MONITORAMENTO: acompanhar KPIs por 72 horas pós-alteração
```

### Threshold changelog — estrutura do registro

```markdown
## Alteração de Threshold — {DATA}

**Threshold anterior:** 0,30
**Threshold novo:** 0,25
**Motivo:** Aumento de 40% no volume de fraudes em contas PJ detectado
**C_FN utilizado:** R$ 2.800 (atualizado pela área de risco)
**C_FP utilizado:** R$ 65 (sem alteração)
**Custo total esperado anterior:** R$ 120.180/mês
**Custo total esperado novo:** R$ 108.420/mês
**Aprovado por:** [Gerente de Risco] + [CFO]
**Responsável técnico:** [Nome]
```

### Alterações emergenciais de threshold

Em situações onde há evidência clara de ataque em andamento, o threshold pode ser reduzido temporariamente sem passar pelo processo completo — mas com comunicação imediata e registro obrigatório:

```
Threshold de emergência: autorizado pelo gerente de risco
Documentação: registrar em docs/threshold_changelog.md em até 2 horas
Revisão: obrigatória em até 24 horas para confirmar se mantém ou reverte
```

---

## 9. Procedimento de Rollback

### Quando fazer rollback

- Novo modelo em produção com Recall 5+ pp abaixo do modelo anterior após 24 horas.
- Novo modelo causando latência P95 > 60ms consistentemente.
- Novo modelo gerando taxa de erros > 0,05%.
- Comportamento inesperado não diagnosticado nas primeiras 48 horas pós-deploy.

### Procedimento de rollback

```
PASSO 1 — Decisão de rollback (responsável: Time de Dados)
  ├── Confirmar que o problema é do modelo, não de infraestrutura
  ├── Documentar razão do rollback
  └── Comunicar: gerente de risco + Engenharia

PASSO 2 — Execução (responsável: Engenharia)

  # Identificar versão anterior no arquivo
  ls -lt models/artifacts/archive/

  # Restaurar modelo anterior
  cp models/artifacts/archive/xgb_model_v{versão_anterior}_{data}.joblib \
     models/artifacts/xgb_model.joblib

  # Validar que o sistema responde com o modelo restaurado
  poetry run python src/modeling/inference.py --test
  poetry run pytest tests/testinferencelatency.py -v

PASSO 3 — Confirmação (responsável: Time de Dados)
  ├── Monitorar KPIs por 30 minutos pós-rollback
  ├── Confirmar que Recall e latência voltaram ao normal
  └── Comunicar: "Rollback concluído — sistema restaurado para v{versão}"

PASSO 4 — Post-mortem (responsável: Time de Dados)
  └── Documentar em docs/postmortems/ o que causou a falha do novo modelo
```

---

## 10. Canais de Comunicação e Escalada

### Matriz de responsabilidade por incidente

| Situação | Time de Dados | Engenharia | Gerente de Risco | CFO | CTO |
|---|---|---|---|---|---|
| P1 — Sistema fora do ar | Diagnóstico | Resolução | Notificar ✉️ | Se > 1h ✉️ | Se > 2h ✉️ |
| P1 — Latência crítica | Diagnóstico | Resolução | — | — | Se > 2h ✉️ |
| P2 — Recall caiu 10pp | Diagnóstico + Resolução | Suporte | Notificar ✉️ | — | — |
| P2 — FPR explodiu | Diagnóstico + Resolução | Suporte | Notificar ✉️ | Se > R$ 50k ✉️ | — |
| P3 — Drift detectado | Diagnóstico + Resolução | — | Informar ✉️ | — | — |
| Re-treino | Execução | Deploy | Aprovar ✅ | Aprovar threshold ✅ | — |
| Alteração de threshold | Proposta | — | Aprovar ✅ | Aprovar ✅ | — |

### Canais por severidade

| Severidade | Canal primário | Tempo de notificação |
|---|---|---|
| P1 | Chamada telefônica + Slack #incidentes-p1 | Imediato |
| P2 | Slack #incidentes-fraude-pix + e-mail | < 30 minutos |
| P3 | Slack #monitoramento-fraude | < 4 horas |
| P4 | E-mail semanal de KPIs | Próximo relatório semanal |

---

## 11. Checklist de Operação

### ✅ Checklist diário

```
MANHÃ
[ ] Dashboard: latência P50, P95, P99 dentro dos limites
[ ] Dashboard: taxa de erros < 0,01%
[ ] Dashboard: disponibilidade > 99,9%
[ ] Volume de alertas de fraude: dentro do padrão histórico?
[ ] Distribuição de scores: sem mudança abrupta?
[ ] CI/CD: todos os testes passando?
[ ] Logs de inferência: sem erros não tratados?

FIM DO DIA
[ ] Registrar anomalias identificadas durante o dia
[ ] Atualizar status no canal de monitoramento
[ ] Se qualquer KPI ficou fora do limite → registrar e acionar P3 ou acima
```

### ✅ Checklist semanal

```
[ ] Calcular PSI das 5 features principais
[ ] Calcular Recall da semana vs. semana anterior
[ ] Calcular FPR da semana
[ ] Calcular custo total: (C_FN × FN) + (C_FP × FP)
[ ] Revisar volume de contatos SAC por bloqueio indevido
[ ] Gerar e enviar relatório semanal para gerência de risco e CFO
[ ] Se PSI > 0,25 em qualquer feature → abrir P3
[ ] Se Recall caiu 5pp → abrir P3
```

### ✅ Checklist de re-treino

```
[ ] Razão do re-treino documentada em docs/model_changelog.md
[ ] Dataset de re-treino validado (schema + anti-leakage)
[ ] Novo modelo com PR-AUC ≥ modelo atual
[ ] Novo modelo com latência P95 < 50ms
[ ] Threshold recalibrado com C_FN e C_FP atualizados
[ ] Aprovação do gerente de risco registrada
[ ] Aprovação do CFO (threshold) registrada
[ ] Backup do modelo atual realizado em models/artifacts/archive/
[ ] CI/CD passando completamente antes do deploy
[ ] Monitoramento intensivo agendado para 48h pós-deploy
[ ] Comunicação enviada para gerência de risco e CFO
```

### ✅ Checklist de rollback

```
[ ] Razão do rollback documentada
[ ] Modelo anterior identificado em models/artifacts/archive/
[ ] Sistema validado com modelo restaurado (inference.py --test)
[ ] Teste de latência passando após restauração
[ ] KPIs monitorados por 30 min pós-rollback
[ ] Comunicação enviada: gerente de risco + Engenharia
[ ] Post-mortem agendado para até 48h após rollback
```

---

## 12. Glossário Operacional

| Termo | Definição operacional |
|---|---|
| **Recall** | % das fraudes reais que o sistema detectou. Queda = mais fraudes passando. |
| **FPR (False Positive Rate)** | % das transações legítimas bloqueadas. Aumento = mais clientes sendo bloqueados. |
| **Threshold** | Limiar de probabilidade que determina aprovação ou bloqueio. Menor threshold = mais sensível a fraudes. |
| **PSI** | Population Stability Index. Mede drift de features. PSI > 0,25 = re-treino necessário. |
| **Drift** | Mudança no comportamento dos dados em produção vs. dados de treino. Causa degradação do modelo. |
| **Latência P95** | Tempo de resposta do sistema para 95% das transações. SLA: < 50ms. |
| **Fallback** | Sistema de regras estáticas ativado quando o modelo ML não está disponível. |
| **Artefato** | Arquivo salvo do modelo treinado (.joblib). Contém os parâmetros aprendidos durante o treino. |
| **Re-treino** | Processo de treinar o modelo com dados mais recentes para recuperar performance degradada. |
| **Rollback** | Processo de reverter para a versão anterior do modelo quando o novo apresenta problemas. |
| **C_FN** | Custo financeiro médio de uma fraude aprovada (Falso Negativo). |
| **C_FP** | Custo financeiro médio de um bloqueio indevido (Falso Positivo). |
| **Post-mortem** | Documento de análise pós-incidente: o que aconteceu, por quê e como evitar. |

---

⬅️ **[Voltar para o README Principal](../README.md)**

---

**Elaborado por:** Sergio Santos — Cientista de Dados

[![Portfólio Sérgio Santos](https://img.shields.io/badge/Portfólio-Sérgio_Santos-111827?style=for-the-badge&logo=githubpages&logoColor=00eaff)](https://portfoliosantossergio.vercel.app)

[![LinkedIn Sérgio Santos](https://img.shields.io/badge/LinkedIn-Sérgio_Santos-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/santossergioluiz)
