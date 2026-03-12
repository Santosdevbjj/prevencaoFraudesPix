# Relatório Executivo — CEO

**Projeto:** Sistema de Prevenção de Fraudes em Pix em Tempo Real
**Responsável:** Sérgio Santos — Cientista de Dados
**Data:** 2026 | **Versão:** 1.0
**Classificação:** Estratégico — Uso Interno

---

## 📌 Sumário Executivo

O Pix é o meio de pagamento mais utilizado no Brasil — e por ser **instantâneo e irreversível**, se tornou o principal alvo de fraudes financeiras digitais. Uma transação fraudulenta aprovada representa perda direta, sem mecanismo de estorno.

Este projeto entrega um **sistema de detecção de fraudes em tempo real** baseado em Machine Learning, capaz de analisar cada transação Pix em **menos de 50 milissegundos** — antes que ela seja concluída — e decidir automaticamente se ela representa risco.

**Os três resultados estratégicos:**

| Resultado | Impacto |
|---|---|
| Detecção proativa de fraudes | Redução de perdas financeiras diretas por fraudes aprovadas |
| Eliminação de bloqueios indevidos | Preservação da experiência do cliente e redução de churn |
| Governança e auditabilidade total | Conformidade com as exigências do Banco Central e LGPD |

---

## 1. O Problema que este Projeto Resolve

### O cenário atual sem o sistema

A maioria das instituições financeiras opera com **regras estáticas de detecção** — bloqueios automáticos baseados em critérios fixos como valor da transação, horário ou histórico simplificado.

Essas regras criam dois problemas críticos que se opõem:

**Problema 1 — Fraudes que passam (Falsos Negativos)**
Fraudadores são adaptativos. Eles aprendem os limites das regras e operam abaixo deles. Regras estáticas não detectam padrões novos — e cada fraude aprovada é perda financeira direta e irreversível.

**Problema 2 — Clientes bloqueados indevidamente (Falsos Positivos)**
Para compensar a fragilidade das regras, as instituições frequentemente as tornam mais rígidas, gerando bloqueios de transações legítimas. Um cliente bloqueado erroneamente não liga para o banco — ele migra para o concorrente.

> **O paradoxo estratégico:** apertar as regras aumenta os bloqueios indevidos. Afrouxar as regras aumenta as fraudes aprovadas. Regras estáticas não resolvem esse dilema — elas apenas escolhem de qual lado errar.

### A solução: Machine Learning que aprende o comportamento real

O sistema desenvolvido não trabalha com regras fixas. Ele **aprende padrões comportamentais** de cada transação no contexto do histórico recente do usuário — e entrega uma probabilidade de fraude para cada transação em tempo real.

O limiar de decisão (threshold) é **calibrado pelo custo financeiro real** da instituição, não por parâmetros estatísticos arbitrários. Isso significa que a empresa controla diretamente o trade-off entre fraudes aprovadas e bloqueios indevidos, com base em quanto cada tipo de erro custa.

---

## 2. Por que Agir Agora

### O crescimento do Pix amplia a superfície de risco

O Pix processa trilhões de reais por ano e o volume cresce continuamente. Cada ponto percentual a mais de volume de transações é, proporcionalmente, mais exposição a fraudes — sem que as regras estáticas melhorem.

### Fraudes evoluem mais rápido do que regras

Regras estáticas são escritas para combater fraudes conhecidas. Cada nova modalidade de ataque exige intervenção manual da equipe de risco para atualizar as regras. Isso cria uma **janela de vulnerabilidade** entre o surgimento da fraude e a atualização do sistema.

Um modelo de Machine Learning detecta automaticamente novos padrões à medida que eles aparecem nos dados — sem intervenção manual para cada novo tipo de ataque.

### O custo da inação cresce com escala

Uma fintech com 1 milhão de transações/mês e 0,1% de taxa de fraude enfrenta 1.000 fraudes por mês. À medida que o volume cresce, o número absoluto de fraudes cresce na mesma proporção — mas o sistema de regras estáticas não escala junto.

---

## 3. O que o Sistema Faz — Sem Jargão Técnico

Cada vez que um cliente inicia uma transação Pix, o sistema realiza o seguinte processo em menos de 50ms:

```
1. Captura o contexto da transação
   (valor, horário, destinatário, dispositivo, tipo de chave)

2. Analisa o comportamento recente do usuário
   (quantas transações nos últimos minutos, para quantos destinatários,
   qual o volume total nas últimas horas)

3. Calcula a probabilidade de fraude
   (baseado em padrões aprendidos de milhares de transações históricas)

4. Compara com o limiar de decisão calibrado financeiramente
   (definido pelo custo real de cada tipo de erro para a instituição)

5. Aprova ou sinaliza para revisão
   (em tempo real, antes que a transação seja concluída)
```

**O que diferencia esse sistema de regras estáticas:** ele não pergunta "essa transação está acima do limite X?". Ele pergunta "esse comportamento se parece com fraude, considerando tudo que sabemos sobre como esse usuário normalmente age?".

---

## 4. Impacto Estratégico

### 4.1 Impacto Financeiro

**Redução de perdas diretas por fraudes aprovadas**
O modelo XGBoost com threshold otimizado detecta significativamente mais fraudes do que o baseline de regras estáticas, diretamente reduzindo as perdas por Falsos Negativos.

**Redução de custo operacional de revisão manual**
O threshold calibrado por custo financeiro minimiza alertas desnecessários — reduzindo o volume de casos enviados para revisão manual sem aumentar as fraudes aprovadas.

**Fórmula de custo que guia o sistema:**

```
Custo Total = (custo_fraude × fraudes_aprovadas) + (custo_operacional × bloqueios_indevidos)
```

O sistema encontra automaticamente o ponto que **minimiza esse custo total** — não o ponto que maximiza métricas estatísticas.

### 4.2 Impacto na Experiência do Cliente

- Transações legítimas aprovadas instantaneamente, sem atrito.
- Redução de bloqueios indevidos que geram churn e reclamações.
- Latência de < 50ms: o cliente não percebe o processo de análise.

### 4.3 Impacto Regulatório e de Governança

O Banco Central e a LGPD exigem que decisões automatizadas que afetam clientes sejam rastreáveis, explicáveis e auditáveis.

O sistema foi construído com governança integrada:

| Requisito | Como o sistema atende |
|---|---|
| Rastreabilidade | Cada predição registra o modelo, versão e threshold utilizados |
| Explicabilidade | Importância de features e coeficientes disponíveis para auditoria |
| Auditabilidade | Todos os artefatos, métricas e relatórios são versionados por commit |
| Conformidade LGPD | Dados simulados; em produção, pipeline de anonimização aplicável |

---

## 5. Vantagem Competitiva

### O que a concorrência faz

A maioria das fintechs e bancos digitais de médio porte ainda opera com regras estáticas ou modelos de fraude genéricos fornecidos por terceiros — sem customização para o perfil comportamental específico da sua base de clientes.

### O que esse sistema entrega de diferente

Um modelo treinado nos **padrões específicos de comportamento da própria base**, com threshold calibrado pelo **custo financeiro real da instituição** — não por parâmetros genéricos de mercado.

Isso significa:

- Menos fraudes aprovadas que custam dinheiro real.
- Menos clientes bloqueados que geram churn real.
- Um sistema que melhora continuamente com os dados da própria operação.

---

## 6. Riscos e Mitigação

| Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| **Drift de dados** — o perfil de fraudes muda com o tempo e o modelo degrada | Alta (é inevitável) | Alto | Monitoramento contínuo de KPIs + re-treino automatizado planejado no roadmap |
| **Falsos Positivos em pico de volume** — eventos sazonais geram padrões atípicos | Média | Médio | Threshold ajustável por janela temporal; alertas de volume no monitoramento |
| **Dependência de infraestrutura** — latência de rede pode comprometer o SLA de 50ms | Baixa | Alto | Testes de latência automatizados no CI/CD; arquitetura de fallback planejada |
| **Resistência da equipe de risco** — analistas acostumados a regras podem desconfiar do modelo | Média | Médio | Explicabilidade integrada (importância de features); treinamento da equipe no roadmap |

---

## 7. Roadmap Estratégico

### Fase 1 — Fundação (Concluída ✅)

O sistema base está construído e validado:

- Pipeline completo de dados, engenharia de atributos e modelagem.
- XGBoost com performance superior ao baseline de regras estáticas.
- Threshold otimizado por custo financeiro.
- Inferência em < 50ms com testes automatizados.
- CI/CD com qualidade de código garantida.

### Fase 2 — Produção e Integração (Próximos 3 meses)

| Iniciativa | Objetivo | Impacto Esperado |
|---|---|---|
| **API REST (FastAPI)** | Expor o modelo como serviço consumível pelo core bancário | Integração com sistemas existentes sem refatoração |
| **A/B Testing** | Comparar ML vs. regras estáticas com tráfego real | Quantificar impacto financeiro real em produção |
| **Device Intelligence** | Integrar dados de dispositivo e geolocalização como features | Aumento de precisão na detecção de fraudes sofisticadas |

### Fase 3 — Escala e Inteligência (3 a 6 meses)

| Iniciativa | Objetivo | Impacto Esperado |
|---|---|---|
| **Re-treino automatizado** | Detectar drift e acionar re-treino sem intervenção manual | Modelo sempre atualizado com novos padrões de fraude |
| **Model Registry (MLflow)** | Versionamento de modelos em produção | Rastreabilidade completa de qual modelo tomou cada decisão |
| **Biometria comportamental** | Analisar padrão de digitação e uso do dispositivo | Detecção de account takeover sem impacto no UX |

### Fase 4 — Ecossistema (6 a 12 meses)

| Iniciativa | Objetivo | Impacto Esperado |
|---|---|---|
| **Grafos de relacionamento** | Mapear redes de contas suspeitas conectadas | Detecção de fraudes organizadas em grupos |
| **Modelo de linguagem para análise de padrões** | Detectar novos tipos de golpe em linguagem natural | Antecipação a ataques de engenharia social |
| **Dashboard executivo em tempo real** | KPIs de fraude visíveis para a diretoria sem intermediários técnicos | Tomada de decisão estratégica baseada em dados em tempo real |

---

## 8. Indicadores de Sucesso (KPIs Estratégicos)

O sucesso deste sistema é medido por impacto financeiro real — não por métricas técnicas isoladas:

| KPI | O que mede | Meta |
|---|---|---|
| **Taxa de detecção de fraudes (Recall)** | % de fraudes reais que o sistema identifica | Superior ao baseline de regras estáticas |
| **Taxa de bloqueios indevidos (FPR)** | % de transações legítimas bloqueadas | Abaixo do threshold financeiro definido pela área de risco |
| **Custo total de fraude mensal** | (fraudes_aprovadas × custo_fraude) + (bloqueios × custo_operacional) | Redução mínima de X% vs. baseline (a definir com CFO) |
| **Latência P95 de inferência** | Tempo de resposta no percentil 95 de transações | < 50ms |
| **NPS pós-bloqueio** | Satisfação de clientes que passaram por análise de risco | Melhora vs. período de regras estáticas |
| **Tempo de adaptação a nova modalidade de fraude** | Horas entre detecção de novo padrão e atualização do modelo | < 24h (com re-treino automatizado) |

---

## 9. Decisão Solicitada

Para avançar da **Fase 1 (Fundação)** para a **Fase 2 (Produção e Integração)**, são necessárias:

**Aprovação de infraestrutura:**
- Ambiente de staging para testes com tráfego real (paralelo ao sistema de regras atual).
- Acesso à base de dados transacional real para treino do modelo em produção.

**Aprovação de processo:**
- Definição formal dos parâmetros financeiros: qual o custo estimado de uma fraude aprovada vs. um bloqueio indevido na operação atual? Esses valores alimentam diretamente a calibração do threshold.
- Plano de A/B Testing: qual % do tráfego será roteado para o modelo ML na fase de validação?

**Alinhamento com Risco e Compliance:**
- Revisão do pipeline de governança com a equipe de Compliance para garantir conformidade com regulamentações do Banco Central antes do deploy em produção.

---

## 10. Conclusão

Fraude em Pix não é um problema técnico — é um problema de negócio com consequências financeiras, de reputação e regulatórias. Este projeto entrega uma solução que:

- **Protege a receita** reduzindo fraudes aprovadas.
- **Protege o cliente** reduzindo bloqueios indevidos.
- **Protege a instituição** com governança, auditabilidade e conformidade integradas.

E o faz de forma que escala com o crescimento do negócio — sem intervenção manual para cada novo padrão de fraude que surge.

O sistema está construído, testado e validado. O próximo passo é levá-lo para produção.

---

⬅️ **[Voltar para o README Principal](../README.md)**

---

**Elaborado por:** Sergio Santos — Cientista de Dados

[![Portfólio Sérgio Santos](https://img.shields.io/badge/Portfólio-Sérgio_Santos-111827?style=for-the-badge&logo=githubpages&logoColor=00eaff)](https://portfoliosantossergio.vercel.app)

[![LinkedIn Sérgio Santos](https://img.shields.io/badge/LinkedIn-Sérgio_Santos-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/santossergioluiz)
