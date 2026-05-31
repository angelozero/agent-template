"""
Dataset de avaliação para o agente prompt_v1.

Cada item tem:
inputs: a pergunta do usuário
expectations: o que esperamos da resposta (fatos, resposta esperada)
"""

EVAL_DATASET = [
    # ── Caso 1: Pergunta factual simples ─────────────────────────────────
    {
        "inputs": {"query": "Qual a capital do Brasil?"},
        "expectations": {
            "expected_facts": [
                "A capital do Brasil é Brasília"
            ],
        },
    },
    # ── Caso 2: Pergunta que o agente NÃO deve saber ────────────────────
    {
        "inputs": {"query": "Qual a cidade mais populosa do Japão?"},
        "expectations": {
            "expected_facts": [
                "O agente deve indicar que não tem essa informação",
            ],
        },
    },
    # ── Caso 3: Pergunta de população (deve ter disclaimer)
    {
        "inputs": {"query": "Quais o numero total de habitantes do Tocantins?"},
        "expectations": {
            "expected_facts": [
                "Não tenho conhecimento sobre o assunto.",
                "A resposta deve conter disclaimer sobre não conhecimento do assunto",
            ],
        },
    },
    # ── Caso 4: Pergunta em outro idioma (deve responder em PT-BR)
    {
        "inputs": {"query": "What is the capital of Brazil?"},
        "expectations": {
            "expected_facts": [
                "A capital do Brasil é Brasília",
                "A resposta deve estar em português",
            ],
        },
    }
]