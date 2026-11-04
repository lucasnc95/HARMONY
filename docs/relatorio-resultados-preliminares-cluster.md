# Relatório dos resultados preliminares no cluster

**ERAD** estima quanto trabalho atribuir a cada dispositivo usando a quantidade de elementos da partição e o tempo de execução medido. **HWTOPOLB, nesta implementação, usa essas mesmas medidas e acrescenta uma perturbação de até ±10% à divisão proposta.** A *seed* determina a sequência dessas perturbações; ela não altera o ERAD. Nenhum dos dois benchmarks usou as métricas de topologia coletadas pelo *probe* ([algoritmos](../dcl/algorithms.hpp), [README](../README.md#7-running-on-a-pbs-cluster)).

## 1. Configuração e verificação

| Item | Valor registrado |
|---|---:|
| Nós / ranks MPI | 2 / 2 |
| GPUs identificadas nos logs | 2 × NVIDIA A100 80GB PCIe |
| Elementos por benchmark | 1.000.000 |
| Iterações por execução | 100 |
| Intervalo de rebalanceamento | 10 iterações |
| Rebalanceamentos registrados por execução | 9 |
| Seeds | 42, 123 e 456 |
| Intensidade da perturbação | ±10% |
| Aquecimento antes das medições | 0 iterações |

Os 12 arquivos de execução — **2 cargas × 2 algoritmos × 3 seeds** — registram o modo `gpu_opencl` e 100 iterações cada. O teste de halo assíncrono passou na comparação com uma referência de CPU. O job, contudo, **terminou com erro ao chamar `git` no pós-processamento**, depois de produzir os logs e CSVs; o resumo foi recuperado desses arquivos. Consulte o [log do job](../hwtopolb_eval.out), os [metadados recuperados](../benchmarks/comparison_metadata.json) e o [roteiro de execução](../benchmarks/compare_algos.sh).

## 2. Tempos totais: comparação por seed

**Menor tempo é melhor.** A coluna “ERAD ÷ HWTOPOLB” indica quantas vezes o tempo do ERAD representa o tempo do HWTOPOLB; valores abaixo de 1 indicam HWTOPOLB mais lento.

| Carga | Seed | ERAD (s) | HWTOPOLB (s) | ERAD ÷ HWTOPOLB | Diferença de HWTOPOLB |
|---|---:|---:|---:|---:|---:|
| kNeighbor | 42 | 3,739960 | 0,220817 | 16,94× | 94,10% menos tempo |
| kNeighbor | 123 | 3,579730 | 3,819720 | 0,94× | 6,70% mais tempo |
| kNeighbor | 456 | 3,879730 | 0,214246 | 18,11× | 94,48% menos tempo |
| LeanMD | 42 | 4,300190 | 0,438039 | 9,82× | 89,81% menos tempo |
| LeanMD | 123 | 4,329970 | 4,459910 | 0,97× | 3,00% mais tempo |
| LeanMD | 456 | 4,221200 | 0,361940 | 11,66× | 91,43% menos tempo |

**O padrão observado é o mesmo nas duas cargas:** HWTOPOLB foi muito mais rápido nas execuções com seeds 42 e 456, mas um pouco mais lento com a seed 123. Isso é uma observação dos [logs e CSVs individuais](../benchmarks), não uma estimativa confiável de ganho geral: houve **uma única execução por combinação** de carga, algoritmo e seed.

### Resumo agregado publicado pelo projeto

| Carga | Algoritmo | Mediana dos 3 tempos (s) | Desvio padrão dos 3 tempos (s) |
|---|---|---:|---:|
| kNeighbor | ERAD | 3,739960 | 0,150116 |
| kNeighbor | HWTOPOLB | 0,220817 | 2,079727 |
| LeanMD | ERAD | 4,300190 | 0,056210 |
| LeanMD | HWTOPOLB | 0,438039 | 2,344305 |

Esses desvios medem a dispersão **entre três seeds**, não a variação de repetições da mesma seed. Os desvios grandes de HWTOPOLB refletem a execução lenta com seed 123. Por isso, suas medianas rápidas não devem ser lidas isoladamente como prova de superioridade ([resumo CSV](../benchmarks/comparison_summary.csv)).

## 3. Onde o tempo foi registrado

A tabela separa a soma dos passos comuns, a soma dos passos em que houve rebalanceamento e a diferença entre essas somas e o tempo total. Todos os valores estão em **segundos**.

| Carga | Algoritmo | Seed | Passos comuns | Passos com rebalanceamento | Tempo fora dessas somas | Total |
|---|---|---:|---:|---:|---:|---:|
| kNeighbor | ERAD | 42 | 1,830 | 0,911 | 0,999 | 3,740 |
| kNeighbor | HWTOPOLB | 42 | 0,040 | 0,122 | 0,059 | 0,221 |
| kNeighbor | ERAD | 123 | 1,830 | 0,751 | 0,999 | 3,580 |
| kNeighbor | HWTOPOLB | 123 | 1,830 | 0,971 | 1,019 | 3,820 |
| kNeighbor | ERAD | 456 | 1,829 | 1,060 | 0,990 | 3,880 |
| kNeighbor | HWTOPOLB | 456 | 0,049 | 0,120 | 0,045 | 0,214 |
| LeanMD | ERAD | 42 | 1,830 | 1,431 | 1,040 | 4,300 |
| LeanMD | HWTOPOLB | 42 | 0,047 | 0,308 | 0,083 | 0,438 |
| LeanMD | ERAD | 123 | 1,829 | 1,471 | 1,030 | 4,330 |
| LeanMD | HWTOPOLB | 123 | 1,830 | 1,590 | 1,040 | 4,460 |
| LeanMD | ERAD | 456 | 1,830 | 1,361 | 1,031 | 4,221 |
| LeanMD | HWTOPOLB | 456 | 0,045 | 0,249 | 0,067 | 0,362 |

**O que isso mostra:** nas execuções lentas, até os passos *sem rebalanceamento* somam cerca de **1,83 s**; nas execuções rápidas de HWTOPOLB, somam aproximadamente **0,04–0,05 s**. A diferença, portanto, não aparece apenas nos nove passos marcados como rebalanceamento. O “tempo fora dessas somas” é uma diferença contábil; os arquivos atuais **não permitem atribuí-la com segurança** a MPI, transferência, espera, validação final ou outra fase. Os tempos por passo vêm dos [CSVs](../benchmarks).

Os logs também informam um **“ganho médio estimado por rebalanceamento”**, na faixa de aproximadamente **0,004 a 0,099 milissegundo** entre as 12 execuções. É uma projeção calculada a partir das capacidades estimadas, **não uma redução de tempo total medida**; não explica, por si, as diferenças de vários segundos observadas ([cálculo da estimativa](../benchmarks/kneighbor.cpp)).

## 4. Topologia medida separadamente

| Métrica do probe | Dispositivo/rank 0 | Dispositivo/rank 1 |
|---|---:|---:|
| Latência PCIe | 42.780,25 ns | 31.074,10 ns |
| Largura de banda PCIe | 10,7925 GB/s | 10,3719 GB/s |
| Latência MPI para o outro rank | 4.998.767,43 ns | 4.999.223,57 ns |
| Largura de banda MPI para o outro rank | 0,02786 GB/s | 0,03073 GB/s |

São os valores registrados no [JSON de topologia](../topo_metrics_cluster.json). Os campos de distância NUMA, contenção de memória e potência estão **vazios**. Como os benchmarks não forneceram esse JSON aos algoritmos, **não é possível associar a diferença de desempenho ao uso dessas métricas**.

## Leitura dos resultados

A rodada demonstra execução em GPU e revela uma **variação muito grande e ainda sem causa estabelecida** na comparação de tempos. Ela não cobre repetições da mesma seed, outras configurações de ranks e GPUs, nem uma comparação que incorpore as métricas de topologia. O [plano de investigação](superpowers/plans/2026-09-30-harmony-gpu-variance-investigation.md) propõe medir separadamente as fases de cada rank e repetir as condições em ordem alternada para determinar se o padrão acompanha a seed ou as condições de execução.
