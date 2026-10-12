# Modelo de Resposta do Agente

## Estrutura esperada
1. **Resumo do caso**: consumidor, empresa, produto/serviço e problema.
2. **Classificação**: categoria, subcategoria e prioridade.
3. **Evidências consultadas**: documentos, interações e dados do pedido.
4. **Fundamento aplicável**: regra ou trecho da base de conhecimento.
5. **Decisão**: `PROCEDENTE`, `PARCIALMENTE_PROCEDENTE`, `IMPROCEDENTE` ou `INCONCLUSIVO`.
6. **Ação recomendada**: solicitação à empresa, reembolso, substituição, reparo, coleta de documentos ou escalonamento.
7. **Prazo sugerido**: em dias corridos.
8. **Limitações**: ausência de informação, necessidade de validação humana ou risco identificado.

## Regras
- Se não houver evidência suficiente, use `INCONCLUSIVO`.
- Se o caso se enquadrar em escalonamento, recomende atendimento humano.
- Não prometa resultado judicial nem aplique sanção.
- Sempre informe quais ferramentas foram utilizadas.
