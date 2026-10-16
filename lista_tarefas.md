# Lista de Tarefas Detalhada - Auditoria HARMONY

## Fase 1: Estabilidade MPI e Consenso Distribuído (Críticos e Altos)

### [x] C1 - Prevenir perda de elementos no particionamento hierárquico
* **Descrição:** Ranks sem GPU recebem métrica de tempo fictícia que infla sua capacidade, mas eles descartam partições pois não entram no `MPI_Allgatherv`.
* **Citação da Auditoria:** Item "C1 — Crítico — O modo hierárquico pode descartar todos os elementos globais". 
* **Arquivos e Linhas Afetados:** `dcl/runtime_impl.hpp`, linhas 3411–3439, 3465–3506 e 3572–3603.
* **Critérios de Aceite:**
  - Ranks sem GPU devem ter capacidade/peso calculados estritamente como `0`.
  - Inserir verificação e validação coletiva (assert) garantindo que `soma(element_count) == global_elements` antes de aceitar uma matriz de partições nova.
  - Teste executado com `np > total_devices` roda sem perda de elementos.

### [x] C2 - Consenso global para decisão NUMA
* **Descrição:** A aceitação da migração NUMA ocorre localmente, permitindo que processos adotem branches distintos em funções que utilizam coletivas do MPI (ex: `MPI_Barrier`), o que induz a travamento total.
* **Citação da Auditoria:** Item "C2 — Crítico — Decisão NUMA diferente entre ranks trava a execução MPI".
* **Arquivos e Linhas Afetados:** `dcl/runtime_impl.hpp`, linhas 3142–3267 e 2830–2860.
* **Critérios de Aceite:**
  - Toda avaliação que gere ou não um rebalanceamento e use operações coletivas deve ser resolvida por um consenso global (ex: `MPI_Allreduce` agregando custo máximo e ganho mínimo, e `MPI_Bcast` do resultado booleano).
  - Um teste de divergir simulação de largura de banda PCIe entre ranks passa sem gerar erro `timeout` de coletiva do MPI.

### [x] A3 - Sincronização do balanceamento estocástico
* **Descrição:** Cada rank inicializa seu próprio gerador `std::random_device`, causando divergência nos vetores alocados na estratégia HWTOPOLB.
* **Citação da Auditoria:** Item "A3 — Alto — O balanceamento estocástico produz partições diferentes por rank".
* **Arquivos e Linhas Afetados:** `dcl/algorithms.hpp` (121–145), `dcl/runtime_impl.hpp` (820–899), `benchmarks/kneighbor.cpp` (154–156), `benchmarks/leanmd.cpp` (156–157).
* **Critérios de Aceite:**
  - O cálculo pseudo-aleatório é feito de forma previsível (suporte a *seed*).
  - O rank 0 escolhe o vetor/fronteira propostos e difunde via `MPI_Bcast` para garantir identidade em todos os nodos.

---

## Fase 2: Robustez das Métricas e Parsing de Topologia

### [x] A6 - Impedir publicação de parâmetros sintéticos no probe
* **Descrição:** O probe injeta matrizes com "medidas hardcoded" (ex: `4.500 ns`) caso o `clGetPlatformIDs` falhe, reportando sucesso no script enganosamente. 
* **Citação da Auditoria:** Item "A6 — Alto — O probe publica parâmetros sintéticos como se fossem medições".
* **Arquivos e Linhas Afetados:** `topo_probe.cpp`, linhas 27–40, 135–140, 199–261, 282–306 e 308–336.
* **Critérios de Aceite:**
  - A execução deve abortar explicitamente com código de erro ao invés de usar fallback de constantes (ou exigir *flag* explícita para modo sintético).
  - Medição checa sondagem por par efetiva; ausências devem falhar ou não gerar valores default disfarçados.

### [x] A2 - Corrigir ingestão de dimensões da matriz de topologia no runtime
* **Descrição:** `set_topo_metrics` verifica apenas os dispositivos locais `devices_.size()` no lugar da matriz global de todas as instâncias (derivável por `all_device_counts_`), rejeitando configurações válidas.
* **Citação da Auditoria:** Item "A2 — Alto — O runtime rejeita matrizes globais produzidas pelo probe".
* **Arquivos e Linhas Afetados:** `dcl/runtime_impl.hpp` (4495–4508) e `topo_probe.cpp` (263–270).
* **Critérios de Aceite:**
  - Tamanho da validação de ingestão obedece `soma(all_device_counts_)`.
  - Teste unitário/integrado de `np=2` consome com êxito matriz proveniente do JSON do `topo_probe`.

### [x] M3 - Leitor JSON robusto
* **Descrição:** Parser customizado falha miseravelmente aceitando chaves truncadas, arrays quebrados, `NaN` ou floats por ints, pois usa `std::istringstream` de forma relaxada.
* **Citação da Auditoria:** Item "M3 — Médio — O leitor de topologia aceita JSON inválido e valores truncados".
* **Arquivos e Linhas Afetados:** `dcl/topo_metrics_io.hpp`, linhas 55–103 e 150–208.
* **Critérios de Aceite:**
  - Inclusão de um parser estrito ou rotina explícita que exija tokens completos e encerramento em `}`.
  - Campos numéricos sofrem asserção de `std::isfinite` não permitindo infinito/NaN.
  - JSON truncado ou chave sem separador disparam `runtime_error` claro.

---

## Fase 3: Algoritmos Físicos e Rebalanceamento

### [x] A4 - Corrigir aplicação estrita do limite de energia (Power Cap)
* **Descrição:** Se todo o sistema satura o limite, a função os re-escala pra cobrir 100% da métrica (anulando o limite). Além disso, a potência é inferida misturando proposição sem verificação em hardware (que acaba extrapolando TDP nominal).
* **Citação da Auditoria:** Item "A4 — Alto — O limite de energia não é garantido pelo algoritmo".
* **Arquivos e Linhas Afetados:** `dcl/algorithms.hpp` (204–287) e `dcl/runtime_impl.hpp` (3037–3039).
* **Critérios de Aceite:**
  - Semântica de normalização não pode devolver uma distribuição que matematicamente viola os TDP nominais recebidos por input.
  - Quando a capacidade limite for inviável/estourada localmente, o processo deve barrar e aplicar *rate-limit* (ou corte viável). Teste unitário certifica cenários saturados.

### [x] A5 - Custo real de rede (MPI) vs Barramento (PCIe)
* **Descrição:** A decisão de migração checa somente PCIe e distância NUMA local ignorando `mpi_latency_ns` ou destino, aprova migrações caras para destinos lentos na rede.
* **Citação da Auditoria:** Item "A5 — Alto — O custo de migração entre ranks ignora as métricas MPI".
* **Arquivos e Linhas Afetados:** `dcl/topo_metrics_io.hpp` (257–291) e `dcl/runtime_impl.hpp` (3180–3187).
* **Critérios de Aceite:**
  - Composição do peso modelado deve englobar origem PCIe + rede MPI `mpi_latency_ns`/`mpi_bandwidth_gbps` + destino PCIe em chamadas `inter-rank`.
  - Simulando latência enorme da rede (ex 1s), o `gain_threshold` deve refugar as propostas ativamente.

### [x] A1 - Combate a oscilações via Histerese
* **Descrição:** Usa cálculo `1/tempo` em vez de produtividade base, ignorando o montante do tempo. Isso provoca partições vai-e-vem (oscilação contínua 800-400 -> 600-600) desnecessárias.
* **Citação da Auditoria:** Item "A1 — Alto — O modo hierárquico oscila mesmo com velocidades fixas".
* **Arquivos e Linhas Afetados:** `dcl/runtime_impl.hpp`, linhas 3345–3388 e 3411–3439.
* **Critérios de Aceite:**
  - Cálculo de capacidade passa a ser `element_count/tempo` real de processamento.
  - Inclui mecanismo de histerese que rejeita variações menores que a margem (threshold) nas migrações.

### [x] M4 - Limitar MPI Overhead no balanceamento hierárquico
* **Descrição:** A fase dita "local", cujo propósito listado no README era diminuir overhead inter-rank, acaba executando sempre os limites de `MPI_Barrier` e `MPI_Allgatherv`.
* **Citação da Auditoria:** Item "M4 — Médio — A fase local do balanceamento hierárquico é apenas cálculo".
* **Arquivos e Linhas Afetados:** `dcl/runtime_impl.hpp` (3372-3407, 3486-3506, 3586-3603); `README.md` R3.
* **Critérios de Aceite:**
  - Evitar invocar coletivas `AllGatherv/Barrier` caso as fronteiras numéricas recém-estabelecidas entre nós MPI distintos sejam rigorosamente idênticas ao passo anterior, processando a troca puramente nos buffers GPU internos.

---

## Fase 4: Fidelidade dos Benchmarks e Integração

### [x] A7 - Integridade da Execução dos Benchmarks 
* **Descrição:** Os benchmarks simulam execução correndo um `for` local para toda as partições, sequencialmente (no CPU), nunca acionando buffers nem kernels de verdade. O cálculo de relógio falha na percepção do wall-time absoluto.
* **Citação da Auditoria:** Item "A7 — Alto — Os benchmarks não medem execução distribuída ou kernels OpenCL".
* **Arquivos e Linhas Afetados:** `benchmarks/kneighbor.cpp` (68-96, 179-217), `benchmarks/leanmd.cpp` (68-96, 181-231), `benchmarks/compare_algos.sh` (58-71).
* **Critérios de Aceite:**
  - Trocar chamadas sequenciais para executar a API integral (`execute`, `create_kernel`, `create_field`) focado exclusivamente em fatias com permissão local de owning.
  - Tempos consolidados utilizam chamadas tipo `MPI_Reduce(MPI_MAX)` calculando Wall Time total real do cluster e não a soma simples do vetor rank por rank.

### [x] A8 - Enforce no Pipeline do PBS Job
* **Descrição:** Um erro de `mpirun` pode ocorrer e o job atual ainda printará _"All Evaluations Completed Successfully"_, corrompendo a matriz de análise do CI/PBS.
* **Citação da Auditoria:** Item "A8 — Alto — Falhas no job PBS são anunciadas como sucesso".
* **Arquivos e Linhas Afetados:** `job_hwtopolb.pbs`, linhas 24–27 e 36–67.
* **Critérios de Aceite:**
  - Arquivo PBS passa a usar `set -euo pipefail`.
  - Checar as variáveis vitais (`PBS_NODEFILE`, `PBS_O_WORKDIR`) e abortar via `$?` estrito em falha nas chamadas.

### [x] M1 - Divergência Stencil CPU/GPU
* **Descrição:** Borda do alg CPU divide forçado por `2*k+1`, enquanto o Kernel adequadamente normaliza pelo vizinho disponível encontrado, falhando em bordas `0`.
* **Citação da Auditoria:** Item "M1 — Médio — O stencil CPU diverge do kernel nas bordas".
* **Arquivos e Linhas Afetados:** `benchmarks/kneighbor.cpp` (190–202) e `benchmarks/kneighbor.cl` (10–18).
* **Critérios de Aceite:**
  - Versão CPU deve espelhar kernel somando um contador limítrofe (`valid_neighbors`).
  - Toleração estrita a diferenças (`tolerance = 1e-5`) introduzida após o passo CPU comparar com Output GPU.

### [x] M2 - Estatística Reprodutível
* **Descrição:** A comparação só gera 1 seed volátil reportando erro decimal flutuante sem base amostral para relatório de performance final.
* **Citação da Auditoria:** Item "M2 — Médio — A comparação estocástica não é reprodutível".
* **Arquivos e Linhas Afetados:** `dcl/algorithms.hpp` (121-145), `compare_algos.sh` (27-38), `comparison_report.md` (15-25).
* **Critérios de Aceite:**
  - Script passa sementes fixas para os executáveis por parãmetro ou env.
  - Roda `N` execuções computando mediana ou desvio padrão básico, com publicação na saída CSV.

### [x] P1 - Correção da Documentação Herdada
* **Descrição:** O README mente sobre arquivos `.out` removidos afirmando que "nunca foram trackeados", prejudicando resgate de hashes comparativos antigos.
* **Citação da Auditoria:** Item "P1 — Atenção — Dados históricos foram retirados e a descrição da remoção está incorreta".
* **Arquivos e Linhas Afetados:** `.gitignore` (28-40) e `README.md` (251-253).
* **Critérios de Aceite:**
  - Documentação atualizada para dizer quando/como foram ignorados e instrução base de como consultar os commits pré-limpeza caso preciso (baseado em `b3206cd`).

---

## Fase 5: Estabilidade OpenCL e Riscos Herdados

### [x] H1 - Vazamento de Recursos OpenCL
* **Descrição:** Se houver erro num construtor de buffer OpenCL, o runtime sai por Throw e não deleta a Command Queue, Context e programas anteriores no vetor de recursos.
* **Citação da Auditoria:** Item "H1 — Alto — Objetos OpenCL podem vazar em redescoberta ou falha parcial".
* **Arquivos e Linhas Afetados:** `dcl/runtime_impl.hpp`, linhas 346–347, 500–525, 527–580 e 1301–1310.
* **Critérios de Aceite:**
  - Utilização rigorosa do conceito RAII (como wrappers de release inteligente `unique_ptr` acoplados ao destrutor OpenCL ou rollbacks na função catch de erros) no `create_field`/`discover_devices`.

### [x] H2 - Incerteza de Execução de Halo Async
* **Descrição:** A leitura do campo para a `transfer_queue` usa evento retido vindo da `kernel_queue` sem forçar comando `clFlush`, podendo pausar perenemente se a API empilhar a execução.
* **Citação da Auditoria:** Item "H2 — Alto — Dependências de eventos entre filas OpenCL podem ficar sem progresso".
* **Arquivos e Linhas Afetados:** `dcl/runtime_impl.hpp` (1868-1905, 2112-2146, 2152-2168, 2235-2246).
* **Critérios de Aceite:**
  - Adição garantida da instrução `clFlush()` após agendar enfileiramentos cujos marcadores servem para interdependência em filas separadas (Transfer <> Execução).

### [x] H3 - Fatiamento Mínimo e Halo Border
* **Descrição:** O modo hierárquico pode ceder fatias de 1 item, que acabam podadas pelo filtro simplista de halo (`element_count < halo`), pulando-o e fundindo matrizes vizinhas que fisicamente não eram limítrofes.
* **Citação da Auditoria:** Item "H3 — Médio — Partições menores que o halo saem da troca de fronteira".
* **Arquivos e Linhas Afetados:** `dcl/runtime_impl.hpp` (2084-2094).
* **Critérios de Aceite:**
  - Se a partição estipulada pelo modelador cair abaixo do tamanho do Halo, forçar rebalance para transferir esse minúsculo resíduo, extirpando a micropartição em vez de ignorá-la silenciosamente na barreira Halo.
