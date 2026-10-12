# Manual de Análise de Reclamações

## 1. Fluxo recomendado
1. Validar os dados da reclamação.
2. Identificar consumidor, empresa, produto/serviço e pedido.
3. Consultar o histórico de interações.
4. Recuperar as regras aplicáveis na base de conhecimento.
5. Avaliar evidências e prazos.
6. Definir uma das decisões possíveis.
7. Registrar a recomendação e o prazo.

## 2. Evidências aceitas
- Nota fiscal ou comprovante de compra.
- Contrato, termo de adesão ou comprovante de contratação.
- Registro de entrega e data prevista.
- Fotos, vídeos ou laudo técnico.
- Histórico de contato com o SAC.
- Comprovante de cancelamento ou cobrança.

## 3. Decisões possíveis
| Decisão | Quando usar |
|---|---|
| `PROCEDENTE` | Há evidência suficiente de violação do direito do consumidor. |
| `PARCIALMENTE_PROCEDENTE` | Parte do pedido é válida, mas há mitigação, uso indevido ou responsabilidade compartilhada. |
| `IMPROCEDENTE` | As evidências demonstram que não houve violação. |
| `INCONCLUSIVO` | Faltam informações ou documentos essenciais. |

## 4. Regra de ausência de informação
Se não houver evidência suficiente, o agente deve:
- solicitar os documentos ausentes;
- não atribuir culpa;
- manter a reclamação como `AGUARDANDO_DOCUMENTOS`;
- informar ao consumidor quais documentos são necessários.

## 5. Boas práticas
- Cite apenas trechos relevantes da base de conhecimento.
- Diferencie fato comprovado de hipótese.
- Registre a ferramenta usada e a evidência obtida.
- Nunca invente prazo, valor, cláusula ou resposta da empresa.
