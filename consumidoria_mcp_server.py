
from __future__ import annotations

import re
from typing import Any

import pandas as pd
from mcp.server.fastmcp import FastMCP


GITHUB_USER = "BryanWille"
GITHUB_REPO = "puc_engenharia_agentes-ai-agentica"
GITHUB_BRANCH = "main"

BASE_RAW_DADOS_URL = (
    f"https://raw.githubusercontent.com/"
    f"{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/dados/"
)

mcp = FastMCP("consumidoria-capabilities")

_cache: dict[str, pd.DataFrame] = {}


def load_csv(filename: str) -> pd.DataFrame:
    """
    Carrega e mantém em cache um CSV público do GitHub.
    """

    if filename not in _cache:
        url = BASE_RAW_DADOS_URL + filename

        _cache[filename] = pd.read_csv(
            url,
            encoding="utf-8",
        )

    return _cache[filename]


def safe(value: Any) -> str:
    """
    Converte valores nulos ou NaN em string vazia.
    """

    if value is None or pd.isna(value):
        return ""

    return str(value)


@mcp.tool()
def consultar_reclamacao(
    reclamacao_id: str,
) -> dict[str, Any]:
    """
    Consulta dados principais de uma reclamação.

    Use como primeira ação de análise para obter categoria,
    descrição, prioridade, empresa, consumidor e pedido associado.
    """

    if not re.fullmatch(r"REC\d+", reclamacao_id or ""):
        return {
            "status": "ERRO",
            "mensagem": (
                "reclamacao_id inválido. "
                "Use o formato REC001."
            ),
        }

    df = load_csv("reclamacoes.csv")

    rows = df[
        df["reclamacao_id"] == reclamacao_id
    ]

    if rows.empty:
        return {
            "status": "ERRO",
            "mensagem": (
                f"Reclamação {reclamacao_id} "
                "não encontrada."
            ),
        }

    row = rows.iloc[0]

    return {
        "status": "OK",
        "reclamacao_id": safe(row["reclamacao_id"]),
        "consumidor_id": safe(row["consumidor_id"]),
        "empresa_id": safe(row["empresa_id"]),
        "pedido_id": safe(row["pedido_id"]),
        "categoria": safe(row["categoria"]),
        "subcategoria": safe(row["subcategoria"]),
        "descricao": safe(row["descricao"]),
        "valor_envolvido_brl": float(
            row["valor_envolvido_brl"]
        ),
        "prioridade": safe(row["prioridade"]),
        "status_reclamacao": safe(row["status"]),
        "prazo_resposta_dias": int(
            row["prazo_resposta_dias"]
        ),
    }


@mcp.tool()
def consultar_consumidor(
    consumidor_id: str,
) -> dict[str, Any]:
    """
    Consulta dados mínimos do consumidor.

    Retorna apenas dados necessários à análise,
    incluindo CPF previamente mascarado.
    """

    df = load_csv("consumidores.csv")

    rows = df[
        df["consumidor_id"] == consumidor_id
    ]

    if rows.empty:
        return {
            "status": "ERRO",
            "mensagem": "Consumidor não encontrado.",
        }

    row = rows.iloc[0]

    return {
        "status": "OK",
        "consumidor_id": safe(row["consumidor_id"]),
        "nome": safe(row["nome"]),
        "cpf_mascarado": safe(row["cpf_mascarado"]),
        "cidade": safe(row["cidade"]),
        "estado": safe(row["estado"]),
    }


@mcp.tool()
def consultar_pedido(
    pedido_id: str,
) -> dict[str, Any]:
    """
    Consulta pedido, valor, pagamento, entrega,
    status atual e garantia.
    """

    df = load_csv("pedidos_contratos.csv")

    rows = df[
        df["pedido_id"] == pedido_id
    ]

    if rows.empty:
        return {
            "status": "ERRO",
            "mensagem": "Pedido não encontrado.",
        }

    row = rows.iloc[0]

    return {
        "status": "OK",
        "pedido_id": safe(row["pedido_id"]),
        "produto_id": safe(row["produto_id"]),
        "data_compra": safe(row["data_compra"]),
        "valor_total_brl": float(
            row["valor_total_brl"]
        ),
        "forma_pagamento": safe(
            row["forma_pagamento"]
        ),
        "status_pedido": safe(row["status_pedido"]),
        "data_entrega_prevista": safe(
            row["data_entrega_prevista"]
        ),
        "data_entrega_real": safe(
            row["data_entrega_real"]
        ),
        "garantia_ate": safe(row["garantia_ate"]),
    }


@mcp.tool()
def consultar_empresa_produto(
    empresa_id: str,
    produto_id: str = "",
) -> dict[str, Any]:
    """
    Consulta dados da empresa e, quando informado,
    do produto ou serviço relacionado.
    """

    empresas = load_csv("empresas.csv")

    rows = empresas[
        empresas["empresa_id"] == empresa_id
    ]

    if rows.empty:
        return {
            "status": "ERRO",
            "mensagem": "Empresa não encontrada.",
        }

    empresa = rows.iloc[0]

    result = {
        "status": "OK",
        "empresa_id": safe(empresa["empresa_id"]),
        "nome_empresa": safe(empresa["nome"]),
        "setor": safe(empresa["setor"]),
        "canal_sac": safe(empresa["canal_sac"]),
        "prazo_resposta_dias": int(
            empresa["prazo_resposta_dias"]
        ),
    }

    if produto_id:
        produtos = load_csv("produtos_servicos.csv")

        produto_rows = produtos[
            produtos["produto_id"] == produto_id
        ]

        if not produto_rows.empty:
            produto = produto_rows.iloc[0]

            result.update(
                {
                    "produto_id": safe(
                        produto["produto_id"]
                    ),
                    "nome_produto": safe(
                        produto["nome"]
                    ),
                    "tipo": safe(produto["tipo"]),
                    "categoria_produto": safe(
                        produto["categoria"]
                    ),
                    "garantia_meses": int(
                        produto["garantia_meses"]
                    ),
                }
            )

    return result


@mcp.tool()
def consultar_interacoes(
    reclamacao_id: str,
) -> dict[str, Any]:
    """
    Consulta o histórico de interações entre consumidor e empresa.

    Use para verificar contatos, respostas, anexos e tentativas
    anteriores de resolução pelo SAC.
    """

    df = load_csv("interacoes_sac.csv")

    rows = df[
        df["reclamacao_id"] == reclamacao_id
    ]

    interacoes = []

    for _, row in rows.iterrows():
        interacoes.append(
            {
                "interacao_id": safe(
                    row["interacao_id"]
                ),
                "data_hora": safe(row["data_hora"]),
                "origem": safe(row["origem"]),
                "autor": safe(row["autor"]),
                "canal": safe(row["canal"]),
                "texto": safe(row["texto"]),
                "anexo": safe(row["anexo"]),
            }
        )

    return {
        "status": "OK",
        "reclamacao_id": reclamacao_id,
        "total_interacoes": len(interacoes),
        "interacoes": interacoes,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
