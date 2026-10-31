# Projeto 1 — Classificação de sessões anómalas de carregamento de veículos elétricos

Mineração de Dados · Mestrado em Engenharia de Dados · ISEP · 2026/2027

Ellah Fleischmann (1231478) · Inês Martinho (1230806)

Previsão, no início de cada sessão de carregamento, de que a sessão vai terminar de forma anómala, com os dados do *China EV Charging Dataset* (versão educativa). O relatório final descreve o trabalho; este repositório contém o código para o reproduzir.

## Estrutura

| Ficheiro | Conteúdo |
|---|---|
| `task1.ipynb` | Task 1: auditoria, análise exploratória e preparação dos dados |
| `task2.ipynb` | Task 2: definição da previsão, métricas, validação temporal e baselines |
| `task3.ipynb` | Task 3: ablação por grupos de variáveis, modelos individuais e ensembles |
| `task4.ipynb` | Task 4: otimização, limiar de decisão, calibração, avaliação no teste e interpretação |
| `task5.ipynb` | Task 5: avaliação detalhada, seleção e ficha do modelo final, utilidade operacional e reprodutibilidade |
| `avaliacao.py` | Métricas, folds temporais, baselines e limiar de decisão (Tasks 2 a 5) |
| `preparacao.py` | Preparação dos dados como pipeline do scikit-learn (Task 2) |
| `ablacao.py` | Grupos de variáveis, pré-processamento e modelos (Tasks 3 a 5) |
| `otimizacao.py` | Pesquisa de hiperparâmetros, regra de escolha e ficheiros guardados (Tasks 4 e 5) |
| `requirements.txt` | Versões das bibliotecas |
| `Resultados/` | Ficheiros gerados pelas Tasks 4 e 5 |

## Dados

Colocar o ficheiro `Charging_Data_educational.csv` (441 077 sessões, 179,1 MB) na mesma pasta dos cadernos.

SHA-256: `f6109dc066162bc770417c39294cdecc7519646c3cfa8c1d95671ea30f0e0a9b`

## Instalação (Windows)

Requer Python 3.14.2.

```
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Abrir a pasta no VS Code e escolher o ambiente `.venv` como kernel dos cadernos.

## Ordem de execução

Correr cada caderno de cima a baixo, num kernel limpo, por esta ordem:

| Ordem | Caderno | Gera | Tempo aproximado |
|---|---|---|---|
| 1 | `task1.ipynb` | — | 1 min |
| 2 | `task2.ipynb` | — | 1 min |
| 3 | `task3.ipynb` | — | 38 min |
| 4 | `task4.ipynb` | `Resultados/pesquisa_*`, `Resultados/task4_*` | 24 min (64 min de raiz) |
| 5 | `task5.ipynb` | `Resultados/task5_*` | 2 min |

A Task 5 usa os ficheiros gerados pela Task 4 e não treina nenhum modelo.

**Pontos de controlo.** Os ficheiros `Resultados/pesquisa_*.csv` guardam a pesquisa de hiperparâmetros da Task 4. Se existirem, a pesquisa não é recalculada. Para reproduzir tudo de raiz, apagar a pasta `Resultados/`.

## Reprodutibilidade

- Sementes fixas: `random_state = 0` nos modelos, `seed = 0` no desempate dos alertas, `SEED = 0` na pesquisa aleatória e `default_rng(0)` na importância por permutação.
- Tudo o que aprende com os dados é ajustado dentro de cada fold, só com o treino desse fold.
- O período de teste (outubro a dezembro de 2021) só é avaliado na secção 11 do caderno da Task 4, com o modelo, as variáveis, os hiperparâmetros e o limiar fixados antes (`Resultados/task4_modelos_escolhidos.json` e `Resultados/task4_limiar.json`). Voltar a correr o caderno repete a mesma avaliação, com os mesmos resultados.
- Os tempos foram medidos em Windows 11 e variam ligeiramente entre execuções.

## Referência

Zhang, Y., Xu, T., Chen, T., et al. (2025). A high-resolution electric vehicle charging transaction dataset with multidimensional features in China. *Scientific Data*, 12, 643. https://doi.org/10.1038/s41597-025-04982-1