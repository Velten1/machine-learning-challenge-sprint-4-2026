# AgroGuard · Machine Learning & Modelling

FIAP · Challenge Sprint 4 · 1TIAPZ · 2026

| Integrante | RM |
|---|---|
| Caio Ryan Velten Alves | 572913 |
| Carlos Alberto Oliveira Nascimento | 565694 |
| Davi França Aranda | 572751 |
| João Vitor Fernandes Gomes | 573882 |
| Murilo de Oliveira Moreira | 573534 |

## Relatório final

O AgroGuard organiza a telemetria de equipamentos agrícolas e apresenta scores por subsistema para apoiar a inspeção de condições fora do padrão. Esta entrega consolida a modelagem existente, seus resultados, sua integração e a evolução desde a Sprint 3.

**[Abrir o relatório técnico final em PDF](AgroGuard%20-%20Machine%20Learning%20-%20Sprint%204.pdf)**

O relatório contempla problema e contribuição, dados e metodologia, comparação dos modelos desenvolvidos, interpretação dos resultados, evolução desde a Sprint 3 e melhorias futuras. O arquivo `AgroGuard - Machine Learning - Sprint 4.zip` contém o PDF para anexar no campo de envio da disciplina.

## Implementação e alcance da avaliação

São 12 pipelines preexistentes, com RobustScaler e Isolation Forest, organizados em três perfis de equipamento e quatro componentes. A saída operacional combina magnitude robusta do desvio, gate de novidade do Isolation Forest e regras de limites físicos configurados.

Os dados são sintéticos. O score de 0 a 100 é um indicador operacional. Sua calibração como probabilidade de sinistro e o desempenho em máquinas reais ainda dependem de avaliação própria.

O notebook histórico utiliza a base de desenvolvimento para escolher o limiar e calcular métricas. As contagens de anomalias nas suas saídas salvas diferem dos CSV atuais. O relatório documenta essa diferença e apresenta as duas versões com suas respectivas origens. As métricas atuais são uma checagem retrospectiva dos artefatos existentes na base de desenvolvimento e calibração, sem holdout independente.

## Reproduzir a verificação

Com Python 3.13, execute na raiz do repositório.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest ia/ml-service/test_infer_buffer.py -q
.\.venv\Scripts\python.exe scripts/verificar_entrega.py
.\.venv\Scripts\python.exe scripts/verificar_http.py
```

O primeiro teste reproduz os cinco casos da suíte original. A verificação seguinte inspeciona os dados, a equivalência de atributos batch/streaming, os scores dos artefatos e as respostas da aplicação. O último comando inicia um servidor Uvicorn local na porta 8885, envia seis requisições por HTTP e confere as respostas com os exemplos registrados. A porta precisa estar disponível.

Os comandos de verificação reutilizam os modelos existentes. Eles atualizam apenas os arquivos de evidências. Uma limitação observada é a aceitação de timestamp malformado como string, registrada no relatório e no JSON de verificação.

## Executar a API

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --app-dir ia/ml-service --host 127.0.0.1 --port 8001
```

Abra `http://127.0.0.1:8001/docs`. A rota `GET /health` informa os perfis carregados e `POST /infer` recebe `equipamento_id` e um `buffer` de uma a seis leituras. Requisições completas estão no campo `requisicao` dos exemplos em `evidencias/casos-inferencia.json`.

O serviço é um recorte executável da camada de ML. O painel, o banco, o assistente e o hardware pertencem à plataforma integrada de origem.

## Estrutura dos materiais

| Caminho | Conteúdo |
|---|---|
| `ia/ml-service/` | Serviço de inferência, validação de entrada e suíte original de testes. |
| `ia/models/` | Doze pipelines treinados e metadados de calibração. |
| `ia/data/` | Três séries sintéticas e sua união consolidada. |
| `ia/notebooks/Sprint2_EDA.ipynb` | Preparação e exploração da etapa anterior. |
| `ia/notebooks/treino_isolation_forest.ipynb` | Registro histórico de modelagem com saídas salvas e nota sobre o alcance das métricas. |
| `ia/scripts/treinar_streaming.py` | Script original de treinamento da composição operacional. |
| `evidencias/metricas-legadas.csv` | Transcrição das métricas salvas no notebook histórico. |
| `evidencias/metricas-artefatos-atuais.csv` | Checagem retrospectiva dos artefatos incluídos. |
| `evidencias/casos-inferencia.json` | Seis requisições e respostas, com exemplos normais e anômalos por perfil. |
| `evidencias/teste-http.json` | Resultado da reprodução com servidor HTTP real. |
| `evidencias/verificacao.json` | Qualidade dos CSV, verificações funcionais e limitações observadas. |
| `evidencias/proveniencia.json` | Origem, commit e hashes SHA-256 dos arquivos. |

Para abrir os notebooks, execute `jupyter lab` no ambiente e navegue até `ia/notebooks`. O notebook histórico foi organizado para exportar seu experimento legado em `tmp/modelos-legados`, preservando os artefatos operacionais de `ia/models`. O algoritmo e as saídas históricas foram mantidos. A identificação, a nomenclatura e o destino de exportação foram revisados para esta entrega.

## Origem e referências

Os CSV, os pipelines e a implementação operacional foram extraídos do [AgroGuard, commit f7175434f2dab4f3109f832a83515b0d9a54e599](https://github.com/Velten1/Agroguard/tree/f7175434f2dab4f3109f832a83515b0d9a54e599). O manifesto registra quais arquivos são cópias sem alteração. O relatório referencia os materiais das sprints anteriores e o histórico das evoluções de hardware e aplicativo.

Consultamos a [documentação do Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html), o [AeroPulse](https://github.com/diegormirhan/aeropulse) e o projeto [Predictive Maintenance de Henrik Axelsson](https://github.com/henrikamnas/predictive-maintenance) para orientar rastreabilidade, discussão da validação e apresentação dos limites. Os resultados desses projetos permanecem atribuídos aos respectivos autores. As tabelas desta entrega utilizam exclusivamente os arquivos do AgroGuard.
