# -*- coding: utf-8 -*-
"""
TCC – Experimento de Recomendação
Comparando: Popularidade vs Similaridade vs Diversidade (MMR)
Métricas usadas: ILD (diversidade dentro da lista) e Cobertura (quanto do catálogo foi exposto)

Como rodar:
    - pip install pandas numpy scikit-learn matplotlib scipy tqdm
    - MovieLens: data/ml-latest-small/movies.csv e data/ml-latest-small/ratings.csv
    - python main.py

Saídas:
    outputs/recommendations_topN.csv  -> recomendações por usuário (com títulos)
    outputs/metrics.csv               -> tabela de métricas por modelo
    outputs/metrics.png               -> gráfico comparando ILD e Cobertura
"""

import os
import json
import random
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# =========================
# CONFIGURAÇÕES
# =========================
PASTA_DADOS = "data/ml-latest-small"   # onde estão movies.csv e ratings.csv
PASTA_SAIDA = "outputs"

TAMANHO_LISTA = 10                     # tamanho da lista de recomendação (top-N)
MIN_INTERACOES_USUARIO = 5             # mínimo de filmes no treino para considerar o usuário
FRACAO_TESTE_USUARIO = 0.2             # fração das interações enviada para teste
SEMENTE_ALEATORIA = 42                 # para resultados reproduzíveis
MAX_USUARIOS = None                    # exemplo: 100. None usa todos.

# Diversidade (MMR): score = λ * relevância - (1-λ) * máxima similaridade com itens já escolhidos
LAMBDA_MMR = 0.7                       # 0.7 = mais peso pra relevância; 0.5 = meio a meio; 0.3 = mais diversidade


# =========================
# FUNÇÕES DE APOIO
# =========================
def garantir_pastas():
    """Garante que a pasta de saída exista."""
    os.makedirs(PASTA_SAIDA, exist_ok=True)

def fixar_sementes(semente=SEMENTE_ALEATORIA):
    """Fixa sementes para resultados repetíveis."""
    random.seed(semente)
    np.random.seed(semente)

def ler_movielens(pasta: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Lê os CSVs do MovieLens (filmes e avaliações)."""
    filmes = pd.read_csv(os.path.join(pasta, "movies.csv"))
    avals  = pd.read_csv(os.path.join(pasta, "ratings.csv"))
    return filmes, avals

def extrair_generos(filmes: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], np.ndarray]:
    """
    Cria vetores “multi-hot” de gêneros para cada filme.
    Retorna:
      - DataFrame com coluna item_idx,
      - lista de gêneros,
      - matriz item_features [num_itens x num_generos], normalizada (L2).
    """
    filmes = filmes.copy()
    filmes['genres'] = filmes['genres'].fillna('(no genres listed)')

    # mapeia movieId -> índice contínuo
    ids_unicos = filmes['movieId'].unique()
    movieid_para_idx = {mid: i for i, mid in enumerate(ids_unicos)}
    filmes['item_idx'] = filmes['movieId'].map(movieid_para_idx)

    # lista de gêneros
    lista_generos = sorted(
        set(g for row in filmes['genres'].str.split('|') for g in row if g and g != '(no genres listed)')
    )
    if not lista_generos:
        lista_generos = ['GENERO_DESCONHECIDO']

    genero_para_idx = {g: i for i, g in enumerate(lista_generos)}
    qtd_itens = len(filmes)
    qtd_generos = len(lista_generos)
    matriz_itens = np.zeros((qtd_itens, qtd_generos), dtype=np.float32)

    for _, linha in filmes.iterrows():
        idx_item = int(linha['item_idx'])
        for g in linha['genres'].split('|'):
            if g in genero_para_idx:
                matriz_itens[idx_item, genero_para_idx[g]] = 1.0

    normalize(matriz_itens, norm='l2', axis=1, copy=False)  # normaliza para usar cosseno
    return filmes, lista_generos, matriz_itens

def separar_por_usuario(
    avals: pd.DataFrame,
    fracao_teste=FRACAO_TESTE_USUARIO,
    minimo_interacoes=MIN_INTERACOES_USUARIO
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa, por usuário, uma parte das interações para teste.
    Garante ao menos uma interação no treino.
    """
    agrupado = avals.groupby('userId')
    linhas_treino, linhas_teste = [], []

    for uid, grp in agrupado:
        itens = grp.sample(frac=1.0, random_state=SEMENTE_ALEATORIA)  # embaralha
        n = len(itens)
        if n < minimo_interacoes:
            linhas_treino.append(itens)
            continue

        tam_teste = max(1, int(round(n * fracao_teste)))
        parte_teste = itens.iloc[:tam_teste]
        parte_treino = itens.iloc[tam_teste:]
        if parte_treino.empty:
            parte_treino = parte_teste.iloc[:1]
            parte_teste = parte_teste.iloc[1:]

        linhas_treino.append(parte_treino)
        linhas_teste.append(parte_teste)

    treino = pd.concat(linhas_treino, axis=0).reset_index(drop=True)
    teste  = pd.concat(linhas_teste,  axis=0).reset_index(drop=True) if linhas_teste else pd.DataFrame(columns=avals.columns)
    return treino, teste

def dicionario_usuario_itens(avals_treino_idx: pd.DataFrame) -> Dict[int, set]:
    """Cria um dicionário: user_idx -> {itens do treino}."""
    user_para_itens = defaultdict(set)
    for _, r in avals_treino_idx.iterrows():
        user_para_itens[int(r['user_idx'])].add(int(r['item_idx']))
    return user_para_itens

def ild_lista_indices(indices_itens: List[int], matriz_itens: np.ndarray) -> float:
    """ILD = média de (1 - cosseno) entre pares de itens da lista."""
    if len(indices_itens) <= 1:
        return 0.0
    feats = matriz_itens[indices_itens]
    sim = cosine_similarity(feats)
    k = sim.shape[0]
    tri_sup = np.triu_indices(k, k=1)
    distancias = 1.0 - sim[tri_sup]
    return float(np.mean(distancias)) if distancias.size > 0 else 0.0


# =========================
# MODELOS DE RECOMENDAÇÃO
# =========================
def recomendar_popularidade(vistos_usuario: set, todos_itens: List[int],
                            ranking_popularidade: List[int], topn=TAMANHO_LISTA) -> List[int]:
    """Pega do ranking global os itens que o usuário ainda não viu."""
    recs = []
    for it in ranking_popularidade:
        if it not in vistos_usuario:
            recs.append(it)
            if len(recs) >= topn:
                break
    return recs

def recomendar_similaridade_conteudo(vistos_usuario: set, matriz_itens: np.ndarray,
                                     topn=TAMANHO_LISTA) -> List[int]:
    """Cria um perfil do usuário (média dos gêneros vistos) e recomenda pelo cosseno."""
    todos = np.arange(matriz_itens.shape[0])
    candidatos = [i for i in todos if i not in vistos_usuario]
    if not candidatos or not vistos_usuario:
        return candidatos[:topn]

    feats_vistos = matriz_itens[list(vistos_usuario)]
    perfil = np.mean(feats_vistos, axis=0)
    norma = np.linalg.norm(perfil)
    if norma > 0:
        perfil /= norma

    feats_cand = matriz_itens[candidatos]
    sims = feats_cand.dot(perfil)  # cosseno (matriz normalizada)
    ordem = np.argsort(-sims)
    return [candidatos[i] for i in ordem[:topn]]

def mmr_reordenar(candidatos_base: List[int], pontuacoes_base: np.ndarray,
                  matriz_itens: np.ndarray, topn=TAMANHO_LISTA, lambda_mmr=LAMBDA_MMR) -> List[int]:
    """
    MMR: mistura relevância com “não-repetição”.
    Quanto menor a similaridade com o que já foi escolhido, melhor.
    """
    if not candidatos_base:
        return []

    escolhidos = []
    restantes = set(candidatos_base)

    while len(escolhidos) < min(topn, len(restantes)):
        melhor_item, melhor_score = None, -1e9
        for it in list(restantes):
            relevancia = float(pontuacoes_base[it])
            if not escolhidos:
                score = relevancia
            else:
                it_vec = matriz_itens[it]
                max_sim = max((float(np.dot(it_vec, matriz_itens[s])) for s in escolhidos), default=0.0)
                score = lambda_mmr * relevancia - (1 - lambda_mmr) * max_sim
            if score > melhor_score:
                melhor_item, melhor_score = it, score
        escolhidos.append(melhor_item)
        restantes.remove(melhor_item)

    return escolhidos

def recomendar_diversidade_mmr(vistos_usuario: set, matriz_itens: np.ndarray,
                               topn=TAMANHO_LISTA, lambda_mmr=LAMBDA_MMR) -> List[int]:
    """Começa como similaridade por conteúdo e reordena com MMR para variar mais."""
    todos = np.arange(matriz_itens.shape[0])
    candidatos = [i for i in todos if i not in vistos_usuario]
    if not candidatos or not vistos_usuario:
        return candidatos[:topn]

    feats_vistos = matriz_itens[list(vistos_usuario)]
    perfil = np.mean(feats_vistos, axis=0)
    norma = np.linalg.norm(perfil)
    if norma > 0:
        perfil /= norma

    pontuacoes = matriz_itens.dot(perfil)        # score para TODOS os itens
    ordem_base = sorted(candidatos, key=lambda i: -pontuacoes[i])
    return mmr_reordenar(ordem_base, pontuacoes, matriz_itens, topn=topn, lambda_mmr=lambda_mmr)


# =========================
# AVALIAÇÃO (ILD e Cobertura)
# =========================
def avaliar_modelos(user_para_itens_treino: Dict[int, set],
                    matriz_itens: np.ndarray,
                    ranking_popularidade: List[int],
                    usuarios_avaliacao: List[int],
                    topn=TAMANHO_LISTA,
                    lambda_mmr=LAMBDA_MMR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Para cada usuário:
      - gera listas com os três modelos
      - calcula ILD e atualiza a cobertura de cada modelo
    Retorna:
      - DataFrame com recomendações por usuário
      - DataFrame com métricas agregadas por modelo
    """
    registros = []
    nomes_modelos = ["Popularidade", "Similaridade", "DiversidadeMMR"]
    ild_por_modelo = {m: [] for m in nomes_modelos}
    cobertura_por_modelo = {m: set() for m in nomes_modelos}

    for uid in tqdm(usuarios_avaliacao, desc="Gerando recomendações"):
        vistos = user_para_itens_treino.get(uid, set())

        # A) Popularidade
        rec_pop = recomendar_popularidade(vistos, list(range(matriz_itens.shape[0])),
                                          ranking_popularidade, topn=topn)
        ild_por_modelo["Popularidade"].append(ild_lista_indices(rec_pop, matriz_itens))
        cobertura_por_modelo["Popularidade"].update(rec_pop)
        registros.append({"user_idx": uid, "model": "Popularidade", "items": json.dumps([int(x) for x in rec_pop])})

        # B) Similaridade
        rec_sim = recomendar_similaridade_conteudo(vistos, matriz_itens, topn=topn)
        ild_por_modelo["Similaridade"].append(ild_lista_indices(rec_sim, matriz_itens))
        cobertura_por_modelo["Similaridade"].update(rec_sim)
        registros.append({"user_idx": uid, "model": "Similaridade", "items": json.dumps([int(x) for x in rec_sim])})

        # C) Diversidade (MMR)
        rec_div = recomendar_diversidade_mmr(vistos, matriz_itens, topn=topn, lambda_mmr=lambda_mmr)
        ild_por_modelo["DiversidadeMMR"].append(ild_lista_indices(rec_div, matriz_itens))
        cobertura_por_modelo["DiversidadeMMR"].update(rec_div)
        registros.append({"user_idx": uid, "model": "DiversidadeMMR", "items": json.dumps([int(x) for x in rec_div])})

    # agrega resultados
    total_itens = matriz_itens.shape[0]
    linhas = []
    for nome in nomes_modelos:
        ild_medio = float(np.mean(ild_por_modelo[nome])) if ild_por_modelo[nome] else 0.0
        cobertura = len(cobertura_por_modelo[nome]) / total_itens if total_itens > 0 else 0.0
        linhas.append({"modelo": nome, "ILD_medio": ild_medio, "Cobertura": cobertura})

    recs_df = pd.DataFrame.from_records(registros)
    metricas_df = pd.DataFrame(linhas)
    return recs_df, metricas_df

def plot_metrics(metrics_df: pd.DataFrame, outpath: str):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = np.arange(len(metrics_df))
    width = 0.35

    ax1.bar(x - width/2, metrics_df["ILD_medio"], width, label="ILD médio", color="#1f77b4")
    ax1.set_ylabel("ILD médio", fontsize=11)
    ax1.set_xlabel("Modelo", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_df["modelo"], fontsize=10)
    ax1.tick_params(axis='y', labelsize=9)

    # Barras de Cobertura no eixo secundário
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, metrics_df["Cobertura"], width, label="Cobertura", color="#ff7f0e")
    ax2.set_ylabel("Cobertura (proporção)", fontsize=11)
    ax2.tick_params(axis='y', labelsize=9)

    # Título e legenda
    plt.title("Comparação de Diversidade (ILD) e Cobertura por Modelo", fontsize=13, weight='bold', pad=15)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
    fig.tight_layout()

    # Fundo branco
    fig.patch.set_facecolor("white")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

# =========================
# PROGRAMA PRINCIPAL
# =========================
def main():
    print(">> Preparando ambiente...")
    fixar_sementes()
    garantir_pastas()

    # 1) Ler dados
    if not os.path.exists(PASTA_DADOS):
        raise FileNotFoundError(
            f"Pasta de dados '{PASTA_DADOS}' não encontrada.\n"
            f"Coloque movies.csv e ratings.csv em {PASTA_DADOS}"
        )
    print(">> Lendo MovieLens...")
    filmes, avals = ler_movielens(PASTA_DADOS)

    # 2) Mapas de ids -> índices numéricos
    ids_usuarios = sorted(avals['userId'].unique())
    userId_para_idx = {uid: i for i, uid in enumerate(ids_usuarios)}
    avals['user_idx'] = avals['userId'].map(userId_para_idx)

    # Filmes -> item_idx + vetores de gênero
    filmes, generos, matriz_itens = extrair_generos(filmes)
    movieId_para_idx = dict(zip(filmes['movieId'].values, filmes['item_idx'].values))
    avals = avals[avals['movieId'].isin(movieId_para_idx)]
    avals['item_idx'] = avals['movieId'].map(movieId_para_idx)

    # 3) Split por usuário
    print(">> Separando treino e teste por usuário...")
    treino, teste = separar_por_usuario(
        avals,
        fracao_teste=FRACAO_TESTE_USUARIO,
        minimo_interacoes=MIN_INTERACOES_USUARIO
    )

    # 4) Filtra quem tem interações mínimas no treino
    contagens_treino = treino.groupby('user_idx')['item_idx'].nunique()
    usuarios_validos = contagens_treino[contagens_treino >= MIN_INTERACOES_USUARIO].index.tolist()
    if MAX_USUARIOS is not None and MAX_USUARIOS > 0:
        random.shuffle(usuarios_validos)
        usuarios_validos = usuarios_validos[:MAX_USUARIOS]

    if not usuarios_validos:
        raise RuntimeError(
            "Nenhum usuário com interações suficientes após o split.\n"
            "Dica: reduza MIN_INTERACOES_USUARIO ou confira o dataset."
        )

    # 5) user -> itens do treino
    treino_idx = treino[['user_idx', 'item_idx']]
    user_para_itens_treino = dicionario_usuario_itens(treino_idx)

    # 6) Ranking de popularidade (base treino)
    contagens_pop = treino.groupby('item_idx')['user_idx'].nunique().sort_values(ascending=False)
    ranking_popularidade = contagens_pop.index.tolist()

    # 7) Recomendações + métricas
    print(">> Gerando recomendações e calculando métricas...")
    recs_df, metricas_df = avaliar_modelos(
        user_para_itens_treino=user_para_itens_treino,
        matriz_itens=matriz_itens,
        ranking_popularidade=ranking_popularidade,
        usuarios_avaliacao=usuarios_validos,
        topn=TAMANHO_LISTA,
        lambda_mmr=LAMBDA_MMR
    )

    # 8) Traduz item_idx -> títulos (pra ficar legível)
    idx_para_titulo = dict(zip(filmes['item_idx'].values, filmes['title'].values))

    def traduzir_listas(col):
        listas = []
        for bruto in col:
            try:
                idxs = json.loads(bruto)
                titulos = [idx_para_titulo.get(int(i), f"item_{i}") for i in idxs]
                listas.append("|".join(titulos))
            except Exception:
                listas.append("")
        return listas

    recs_df_out = recs_df.copy()
    recs_df_out['titles'] = traduzir_listas(recs_df_out['items'])

    # 9) Salvar saídas
    caminho_recs = os.path.join(PASTA_SAIDA, "recommendations_topN.csv")
    caminho_metricas = os.path.join(PASTA_SAIDA, "metrics.csv")
    caminho_grafico = os.path.join(PASTA_SAIDA, "metrics.png")

    recs_df_out.to_csv(caminho_recs, index=False, encoding="utf-8")
    metricas_df.to_csv(caminho_metricas, index=False, encoding="utf-8")
    plot_metrics(metricas_df, caminho_grafico)


    print("\n==> Métricas agregadas")
    print(metricas_df.to_string(index=False))
    print(f"\nArquivos salvos em '{PASTA_SAIDA}':")
    print(f"- {os.path.basename(caminho_recs)}")
    print(f"- {os.path.basename(caminho_metricas)}")
    print(f"- {os.path.basename(caminho_grafico)}")
    print("\nPronto! Abra o CSV no Excel/Sheets e o PNG para o gráfico. ")

if __name__ == "__main__":
    main()
