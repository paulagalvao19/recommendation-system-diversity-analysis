TCC Diversidade – Execução rápida

1) Python 3.10+ instalado. No Windows, abra "Prompt de Comando" dentro da pasta do projeto.

2) Crie e ative um ambiente virtual:
   Windows:
     python -m venv .venv
     .venv\Scripts\activate

   macOS / Linux:
     python3 -m venv .venv
     source .venv/bin/activate

3) Instale as dependências:
     pip install -r requirements.txt

4) Execute:
     python main.py

Saídas esperadas (pasta outputs/):
  - recommendations_topN.csv   -> Recomendações por usuário e modelo
  - metrics.csv                -> ILD médio e cobertura por modelo
  - metrics.png                -> Gráfico comparando as métricas

Observações importantes:
- O erro "Object of type int64 is not JSON serializable" foi corrigido convertendo
  os índices de itens para int nativo antes do json.dumps.
- Se aparecer erro de dados, confirme que a pasta data/ml-latest-small contém
  movies.csv e ratings.csv (já incluso aqui).
