"""
Dataset de avaliação para o agente de capitais e cidades brasileiras.

Baseado nas regras definidas nos prompts (v1–v4):
  1. Responder SEMPRE em português brasileiro
  2. Ser conciso (máximo 3 parágrafos)
  3. Fora do escopo (não é cidade/capital BR) → disclaimer: "Não tenho conhecimento sobre o assunto."
  4. Não souber → "Não tenho essa informação."
  5. Nunca responder nada que não seja sobre cidades/capitais dentro do território brasileiro

Cada item tem:
  inputs:       a pergunta do usuário
  expectations: o que esperamos da resposta (fatos, resposta esperada)
"""

EVAL_DATASET = [
    # ═══════════════════════════════════════════════════════════════════════
    # GRUPO 1: Perguntas factuais sobre capitais brasileiras (DENTRO do escopo)
    # ═══════════════════════════════════════════════════════════════════════

    # ── Caso 1: Capital federal ───────────────────────────────────────────
    {
        "inputs": {"query": "Qual a capital do Brasil?"},
        "expectations": {
            "expected_facts": [
                "A capital do Brasil é Brasília",
            ],
            "expected_response": "Brasília é a capital do Brasil, localizada no Distrito Federal.",
        },
    },
    # ── Caso 2: Capital de estado (região Sudeste) ────────────────────────
    {
        "inputs": {"query": "Qual a capital de São Paulo?"},
        "expectations": {
            "expected_facts": [
                "A capital do estado de São Paulo é a cidade de São Paulo",
            ],
            "expected_response": "A capital do estado de São Paulo é a cidade de São Paulo.",
        },
    },
    # ── Caso 3: Capital de estado (região Norte) ──────────────────────────
    {
        "inputs": {"query": "Qual a capital do Amazonas?"},
        "expectations": {
            "expected_facts": [
                "A capital do Amazonas é Manaus",
            ],
            "expected_response": "A capital do estado do Amazonas é Manaus.",
        },
    },
    # ── Caso 4: Capital de estado (região Nordeste) ───────────────────────
    {
        "inputs": {"query": "Qual a capital da Bahia?"},
        "expectations": {
            "expected_facts": [
                "A capital da Bahia é Salvador",
            ],
            "expected_response": "A capital do estado da Bahia é Salvador.",
        },
    },
    # ── Caso 5: Capital de estado (região Sul) ────────────────────────────
    {
        "inputs": {"query": "Qual a capital do Rio Grande do Sul?"},
        "expectations": {
            "expected_facts": [
                "A capital do Rio Grande do Sul é Porto Alegre",
            ],
            "expected_response": "A capital do estado do Rio Grande do Sul é Porto Alegre.",
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # GRUPO 2: Perguntas sobre cidades brasileiras (DENTRO do escopo)
    # ═══════════════════════════════════════════════════════════════════════

    # ── Caso 6: Cidade brasileira conhecida ───────────────────────────────
    {
        "inputs": {"query": "Em qual estado fica a cidade de Ouro Preto?"},
        "expectations": {
            "expected_facts": [
                "Ouro Preto fica no estado de Minas Gerais",
            ],
            "expected_response": "Ouro Preto é uma cidade localizada no estado de Minas Gerais.",
        },
    },
    # ── Caso 7: Pergunta sobre antiga capital ─────────────────────────────
    {
        "inputs": {"query": "Qual cidade foi a capital do Brasil antes de Brasília?"},
        "expectations": {
            "expected_facts": [
                "O Rio de Janeiro foi a capital do Brasil antes de Brasília",
            ],
            "expected_response": "Antes de Brasília, a capital do Brasil era o Rio de Janeiro.",
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # GRUPO 3: Perguntas FORA do escopo (deve retornar disclaimer)
    # ═══════════════════════════════════════════════════════════════════════

    # ── Caso 8: País estrangeiro ──────────────────────────────────────────
    {
        "inputs": {"query": "Qual a cidade mais populosa do Japão?"},
        "expectations": {
            "expected_facts": [
                "A resposta deve conter o disclaimer de que não tem conhecimento sobre o assunto",
                "O agente NÃO deve responder sobre o Japão",
            ],
            "expected_response": "Não tenho conhecimento sobre o assunto.",
        },
    },
    # ── Caso 9: Assunto fora do domínio (culinária) ───────────────────────
    {
        "inputs": {"query": "Qual a receita de bolo de chocolate?"},
        "expectations": {
            "expected_facts": [
                "A resposta deve conter o disclaimer de que não tem conhecimento sobre o assunto",
                "O agente NÃO deve fornecer receitas",
            ],
            "expected_response": "Não tenho conhecimento sobre o assunto.",
        },
    },
    # ── Caso 10: Assunto fora do domínio (esportes) ───────────────────────
    {
        "inputs": {"query": "Quem ganhou a última Copa do Mundo?"},
        "expectations": {
            "expected_facts": [
                "A resposta deve conter o disclaimer de que não tem conhecimento sobre o assunto",
                "O agente NÃO deve responder sobre esportes",
            ],
            "expected_response": "Não tenho conhecimento sobre o assunto.",
        },
    },
    # ── Caso 11: Pergunta sobre população (fora do escopo de capitais) ────
    {
        "inputs": {"query": "Qual o número total de habitantes do Tocantins?"},
        "expectations": {
            "expected_facts": [
                "A resposta deve conter o disclaimer de que não tem conhecimento sobre o assunto",
                "O agente NÃO deve fornecer dados populacionais",
            ],
            "expected_response": "Não tenho conhecimento sobre o assunto.",
        },
    },
    # ── Caso 12: Pergunta sobre ilhas (fora do escopo de cidades/capitais)
    {
        "inputs": {"query": "Quantas ilhas nós temos em volta do Brasil?"},
        "expectations": {
            "expected_facts": [
                "A resposta deve conter o disclaimer de que não tem conhecimento sobre o assunto",
                "O agente NÃO deve responder sobre geografia física",
            ],
            "expected_response": "Não tenho conhecimento sobre o assunto.",
        },
    },
    # ── Caso 13: Comparação com outro país (fora do escopo) ───────────────
    {
        "inputs": {"query": "Quantas capitais tem o Brasil em relação ao Japão?"},
        "expectations": {
            "expected_facts": [
                "A resposta deve conter o disclaimer de que não tem conhecimento sobre o assunto",
                "O agente NÃO deve fazer comparações com outros países",
            ],
            "expected_response": "Não tenho conhecimento sobre o assunto.",
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # GRUPO 4: Regra de idioma (deve responder em PT-BR)
    # ═══════════════════════════════════════════════════════════════════════

    # ── Caso 14: Pergunta em inglês (deve responder em português) ─────────
    {
        "inputs": {"query": "What is the capital of Brazil?"},
        "expectations": {
            "expected_facts": [
                "A capital do Brasil é Brasília",
                "A resposta DEVE estar em português brasileiro",
            ],
            "expected_response": "Brasília é a capital do Brasil.",
        },
    },
    # ── Caso 15: Pergunta em espanhol (deve responder em português) ───────
    {
        "inputs": {"query": "¿Cuál es la capital de Minas Gerais?"},
        "expectations": {
            "expected_facts": [
                "A capital de Minas Gerais é Belo Horizonte",
                "A resposta DEVE estar em português brasileiro",
            ],
            "expected_response": "A capital do estado de Minas Gerais é Belo Horizonte.",
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # GRUPO 5: Casos-limite e tentativas de desvio
    # ═══════════════════════════════════════════════════════════════════════

    # ── Caso 16: Pergunta mista (BR + estrangeiro) ────────────────────────
    {
        "inputs": {
            "query": "Nas minhas viagens eu estive no oriente, em relação ao Brasil quantos continentes no oriente nós temos?"
        },
        "expectations": {
            "expected_facts": [
                "A resposta deve conter o disclaimer de que não tem conhecimento sobre o assunto",
                "O agente NÃO deve responder sobre continentes ou geografia internacional",
            ],
            "expected_response": "Não tenho conhecimento sobre o assunto.",
        },
    },
    # ── Caso 17: Pergunta ambígua que menciona cidade BR mas pede info fora do escopo
    {
        "inputs": {"query": "Qual o PIB de São Paulo?"},
        "expectations": {
            "expected_facts": [
                "A resposta deve conter o disclaimer de que não tem conhecimento sobre o assunto",
                "O agente NÃO deve fornecer dados econômicos",
            ],
            "expected_response": "Não tenho conhecimento sobre o assunto.",
        },
    },
    # ── Caso 18: Pergunta válida sobre capital de estado (região Centro-Oeste)
    {
        "inputs": {"query": "Qual a capital do Mato Grosso do Sul?"},
        "expectations": {
            "expected_facts": [
                "A capital do Mato Grosso do Sul é Campo Grande",
            ],
            "expected_response": "A capital do estado do Mato Grosso do Sul é Campo Grande.",
        },
    },
]
