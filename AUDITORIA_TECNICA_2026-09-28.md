# Auditoria técnica das alterações do HARMONY

**Data:** 28 de setembro de 2026

**Estado auditado:** `037fd1b` (`master`)

**Intervalo:** `b3206cd..037fd1b` (seis commits feitos após o estado que aparece como base do trabalho local)

**Escopo:** C++20, MPI, OpenCL, particionamento, migração de campos, métricas de topologia, modelos de custo e energia, testes, benchmarks e job PBS.

## Resumo executivo

A implementação adicionou métricas de topologia, balanceamento hierárquico, restrições NUMA/energia e benchmarks, mas ainda há falhas que impedem considerar os resultados de desempenho e algumas garantias de segurança operacional como validados. Os testes declarados no README passam, porém exercitam sobretudo dispositivos simulados e não cobrem transferência real de campos entre ranks nem execução OpenCL em GPU. Esta auditoria encontrou problemas no comportamento distribuído e na fidelidade das medições; os achados detalhados abaixo separam defeitos reproduzidos, conclusões sustentadas pelo fluxo do código e pontos que exigem validação em cluster.

No intervalo auditado, foram classificados **2 achados críticos, 8 altos, 4 médios e 1 ponto de atenção**. Outros **3 riscos** relevantes já existiam no ancestral `b3206cd` e são identificados separadamente para não atribuí-los aos commits recentes.

**Prioridade proposta:** corrigir primeiro as condições de divergência ou travamento MPI e a incompatibilidade das métricas globais; depois tornar os limites de energia e o custo de migração verificáveis; por fim reconstruir benchmarks que executem a biblioteca de fato antes de usar suas tabelas para decisões de desempenho.

## Como o intervalo foi identificado

O histórico atual é linear: `b3206cd` → `14dfb9d` → `0f8b77c` → `5e97bb5` → `087a2f7` → `2f0b0dc` → `037fd1b`. O reflog registra a criação e reaplicação dos commits de setembro de 2026 após um rebase. `b3206cd` é o ancestral imediatamente anterior a essas alterações. O histórico disponível não comprova literalmente o instante do clone; por isso, esse ancestral é a melhor base verificável para o pedido. Commits anteriores a `b3206cd` foram consultados como contexto para separar regressões de problemas herdados.

| Commit | Mudança auditada |
|---|---|
| `14dfb9d` | Implementação principal R1–R5; novos testes, kernels, probe e benchmarks; remoção de artefatos rastreados |
| `0f8b77c` | Job PBS e lançamento MPI dos benchmarks |
| `5e97bb5` | Documentação das funcionalidades e resultados |
| `087a2f7` | Ajuste dos testes de logs hierárquicos e comandos de compilação |
| `2f0b0dc` | Nota sobre origem local dos números dos benchmarks |
| `037fd1b` | Execução automática dos testes no job PBS |

`git diff b3206cd..037fd1b` registra 834 arquivos modificados, 4.669 linhas adicionadas e 96.118 removidas. Desse total, 804 arquivos e 95.370 linhas removidas pertencem a `Results/`, onde estavam logs e CSVs de execuções anteriores. Esses dados continuam recuperáveis no histórico Git, mas não estão mais na árvore de trabalho atual.

## Validação executada

| Verificação | Resultado | Alcance |
|---|---|---|
| Recompilação independente das cinco suítes `tests/test_*.cpp` com `mpic++ -std=c++20 -Wall -Wextra -Wpedantic -Wno-unused-parameter -O2 -lOpenCL` | Cinco compilações concluídas, sem avisos do compilador | Compilação e vínculo locais |
| Execução das suítes oficiais | 24/24 testes declarados passaram; suíte hierárquica com dois ranks | Dispositivos simulados; sem GPU |
| Benchmarks curtos kNeighbor e LeanMD, `N=4000`, quatro iterações, um e dois ranks | Concluídos; inspeção do código e dos CSVs confirmou diferença entre tempo registrado e tempo integral | CPU local, sem avaliação estatística de desempenho |
| Falhas simuladas dos comandos do job PBS, mantendo o script original | Script retornou `0` e imprimiu `All Evaluations Completed Successfully` | Reproduz a detecção inadequada de falhas |
| Consulta OpenCL local (`clGetPlatformIDs`) | Código `-1001`, zero plataformas | Não permite medir execução de kernels/dispositivos reais neste ambiente |
| `cppcheck --enable=all` no código central e novos executáveis | Avisos de estilo em `topo_probe.cpp`; nenhum dos achados semânticos abaixo foi detectado | Análise estática complementar |

O lançamento MPI inicialmente falhou no isolamento local porque PMIx não conseguiu abrir sockets. Os testes foram repetidos com a permissão apropriada. A máquina oferece apenas um slot MPI por padrão; o teste com dois ranks foi executado com `--oversubscribe`. Isto é uma característica do ambiente de auditoria, não um defeito do HARMONY.

## Achados detalhados

### C1 — Crítico — O modo hierárquico pode descartar todos os elementos globais

**Origem:** `14dfb9d`. **Código:** [runtime_impl.hpp](/home/agent/HARMONY/dcl/runtime_impl.hpp:3411), linhas 3411–3439, 3465–3506 e 3572–3603.

O tempo do rank é tomado como o máximo dos tempos de seus dispositivos. Quando um rank não possui dispositivo, `local_max_time` recebe `1e-9` (linhas 3411-3414), interpretado pelo cálculo inverso como capacidade extremamente alta. O algoritmo então destina quase toda a carga a esse rank. Como ele não possui dispositivo, não publica partição alguma no `MPI_Allgatherv`; os intervalos que lhe foram atribuídos desaparecem da lista final. O método atualiza `partitions_` sem verificar a invariável `soma(element_count)==global_elements`.

**Reprodução real em MPI:** quatro ranks, três dispositivos simulados, `1.200.000` elementos. Após o primeiro `maybe_rebalance_hierarchical()`, os logs mostram `inter-node loads: [2.5e-08, 2.5e-08, 2.5e-08, 1]` e as três partições com `count=0`; `covered=0 expected=1200000`. Os passos seguintes permanecem sem carga. Isto é perda de cobertura lógica e pode produzir resultados numéricos completamente incorretos sem exceção.

**Correção:** atribuir capacidade zero aos ranks sem dispositivos e peso zero no corte inter-rank; não criar intervalos para eles. Antes da troca de partições, validar coletivamente que os intervalos são contíguos, sem sobreposição e cobrem exatamente `global_elements`. Acrescentar teste com `np > total_devices`, além de topologias heterogêneas reais.

### C2 — Crítico — Decisão NUMA diferente entre ranks trava a execução MPI

**Origem:** `14dfb9d`. **Código:** [decisão NUMA](/home/agent/HARMONY/dcl/runtime_impl.hpp:3142), linhas 3142–3267; [coleta das métricas](/home/agent/HARMONY/dcl/runtime_impl.hpp:2830), linhas 2830–2860.

A aceitação da migração é decidida localmente a partir de `topo_metrics_` e do limiar de cada rank. Se um rank julga o custo alto, retorna após `print_balance_interval_metrics`, que chama três `MPI_Allreduce`. Se outro rank aceita, entra em `synchronize(true)`, que faz `MPI_Barrier` antes de redistribuir. As chamadas coletivas deixam de corresponder entre membros do mesmo comunicador, violando o contrato MPI. A condição pode ocorrer se métricas, limiares ou flags de política diferirem entre ranks; a API não impõe um consenso global antes do ramo.

**Reprodução MPI:** dois ranks simulados, mesmo campo e mesmos tempos `{0,001, 0,010}` s. Apenas a largura de banda PCIe informada diferia (`0,001` versus `1000 GB/s`). O rank 0 registrou `[NUMA] migration cost exceeds gain threshold, skipping rebalance`; o rank 1 seguiu o ramo de aplicação. Nenhum imprimiu a saída após a chamada e `timeout 8s` encerrou o processo com código `124`. Essa prova isola a configuração divergente; não afirma que métricas idênticas provocam o mesmo travamento.

**Correção:** validar a consistência das políticas/métricas relevantes entre ranks e decidir aceitar/rejeitar coletivamente, por exemplo com redução do custo máximo e ganho mínimo seguida de broadcast de um único booleano. Garantir a mesma sequência de coletivas e tratamento de erros em todos os ranks, inclusive quando uma chamada OpenCL local falha.

### A1 — Alto — O modo hierárquico oscila mesmo com velocidades fixas

**Origem:** `14dfb9d`. **Código:** [runtime_impl.hpp](/home/agent/HARMONY/dcl/runtime_impl.hpp:3345), linhas 3345–3388 e 3411–3439.

O peso de cada rank/dispositivo é calculado como `1/tempo`, ignorando quantos elementos foram processados para obter esse tempo. Com velocidades constantes de 60 e 30 milhões de elementos/s e `1.200.000` elementos, uma reprodução MPI de dois ranks alternou as partições `800.000/400.000` → `600.000/600.000` → `800.000/400.000` → `600.000/600.000`. O resultado correto para throughput proporcional seria manter aproximadamente `800.000/400.000` depois da primeira adaptação. A oscilação provoca migrações e sincronizações repetidas sem mudança de hardware.

**Correção:** usar capacidade estimada como `element_count/tempo` por dispositivo, agregar capacidades por rank e aplicar histerese/custo de migração antes de efetuar a próxima redistribuição. Testar convergência em vários passos, não apenas a primeira chamada.

### A2 — Alto — O runtime rejeita matrizes globais produzidas pelo probe

**Origem:** `14dfb9d`. **Código:** [validação no runtime](/home/agent/HARMONY/dcl/runtime_impl.hpp:4495), linhas 4495–4508; [matriz gerada pelo probe](/home/agent/HARMONY/topo_probe.cpp:263), linhas 263–270.

O probe grava matrizes MPI de tamanho `total_devices × total_devices` somando os dispositivos de todos os ranks. `set_topo_metrics`, porém, calcula `total_devices` a partir de `devices_.size()`, que contém apenas os dispositivos **locais** quando houve descoberta real de OpenCL. Uma matriz válida para o cluster é rejeitada em qualquer rank onde o total global difere do total local. Em reprodução MPI com duas instâncias de OpenCL simuladas por rank, ambos tinham `local_devices=2`, `global_devices=4` e receberam `Invalid matrix dimensions` ao aplicar matrizes 4×4.

**Correção:** derivar o tamanho esperado da soma de `all_device_counts_`, validar também o comprimento de todos os vetores por dispositivo e usar a mesma ordem de índices globais na descoberta e no probe. Adicionar teste de integração `np=2` que carrega o JSON gerado por `topo_probe`.

### A3 — Alto — O balanceamento estocástico produz partições diferentes por rank

**Origem:** `14dfb9d`. **Código:** [algorithms.hpp](/home/agent/HARMONY/dcl/algorithms.hpp:121), linhas 121–145; [runtime_impl.hpp](/home/agent/HARMONY/dcl/runtime_impl.hpp:820), linhas 820–899; [kneighbor.cpp](/home/agent/HARMONY/benchmarks/kneighbor.cpp:154), linhas 154–156; [leanmd.cpp](/home/agent/HARMONY/benchmarks/leanmd.cpp:156), linhas 156–157.

Cada processo instancia seu próprio gerador com `std::random_device`, calcula `hwtopolb_loads` e chama `rebalance_to` localmente. Essa API aplica os limites recebidos sem uma decisão coletiva ou broadcast. Em reprodução de dois ranks e `1.000.000` elementos, o primeiro limite ficou em `635.008` no rank 0 e `648.720` no rank 1 após **a mesma chamada**; os limites continuaram divergentes em todos os quatro passos observados. Os benchmarks não registram essa discrepância, pois apenas o rank 0 escreve CSV e não há campos registrados para migrar.

Em uma aplicação com campos distribuídos, ranks que discordam sobre a origem e o destino dos intervalos podem ler dados errados, trocar quantidades incompatíveis ou permanecer bloqueados na comunicação. Essa consequência é inferida do contrato de migração; a reprodução comprova diretamente a divergência de partições.

**Correção:** escolher uma única proposta no rank 0, distribuir a decisão por `MPI_Bcast` e verificar por hash ou contagens que todos os ranks aplicam o mesmo vetor antes de qualquer migração. Permitir seed controlada para repetir o experimento.

### A4 — Alto — O limite de energia não é garantido pelo algoritmo

**Origem:** `14dfb9d`. **Código:** [algorithms.hpp](/home/agent/HARMONY/dcl/algorithms.hpp:204), linhas 204–287; [chamada no runtime](/home/agent/HARMONY/dcl/runtime_impl.hpp:3037), linhas 3037–3039.

`apply_power_cap` reduz as frações de carga de dispositivos acima do limite, mas renormaliza para que a soma seja sempre 1. Se **todos** estão acima do limite, a normalização anula integralmente a redução. Em prova com duas cargas de 50%, potência observada `200 W` em cada dispositivo e orçamento `300 W`, a saída continua `50%/50%`, embora o total observado seja `400 W`. Quando apenas um dispositivo está acima do limite, o excesso é redistribuído sem verificar se o receptor também passará do limite; com TDP `200 W` e potência `{300,199} W`, o resultado `33,3%/66,7%` levaria o segundo a aproximadamente `265 W` sob escalonamento linear. Pior: a função aplica a potência medida sob a distribuição **atual** a uma distribuição **proposta** sem receber a primeira. Com cargas atuais `10%/90%`, proposta `90%/10%`, potência atual `{300,100} W` e TDP `{200,200} W`, a saída calculada é `60%/40%`: a carga do dispositivo já quente aumenta seis vezes. A potência projetada é inferência do modelo, não medição em hardware. Usar somente a potência atual não demonstra respeito ao orçamento futuro.

**Correção:** retornar inviabilidade quando a capacidade energética agregada não suporta a carga atual, reduzir ritmo/trabalho se esse for o contrato desejado e reavaliar limites após cada redistribuição. Incluir testes de todos saturados e de receptor perto do TDP. Evitar afirmar que `power_budget_watts` é um teto rígido até que seja imposto no hardware ou por um mecanismo de execução.

### A5 — Alto — O custo de migração entre ranks ignora as métricas MPI

**Origem:** `14dfb9d`. **Código:** [topo_metrics_io.hpp](/home/agent/HARMONY/dcl/topo_metrics_io.hpp:257), linhas 257–291; [chamada no runtime](/home/agent/HARMONY/dcl/runtime_impl.hpp:3180), linhas 3180–3187.

A função usada para decidir se a migração compensa lê apenas latência/largura de banda PCIe da origem e a distância NUMA. Ela não consulta `mpi_latency_ns` nem `mpi_bandwidth_gbps`, nem o PCIe do destino. Em prova com migração de `1 GB`, variar a latência MPI de `1 µs` para `1 s` e a largura de banda de `10 GB/s` para `0,001 GB/s` manteve o custo calculado em `0,100005 s` em ambos os casos. Alterar também a banda PCIe do destino para `0,001 GB/s` não modificou o resultado. Em clusters, isso pode aprovar uma migração muito mais lenta que o ganho previsto, anulando o objetivo de R1/R2.

**Correção:** distinguir migração no mesmo dispositivo, entre dispositivos no mesmo rank e entre ranks; compor os estágios PCIe origem, rede MPI e PCIe destino com parâmetros globais válidos e calibrar o modelo com transferências reais.

### A6 — Alto — O probe publica parâmetros sintéticos como se fossem medições

**Origem:** `14dfb9d`. **Código:** [topo_probe.cpp](/home/agent/HARMONY/topo_probe.cpp:27), linhas 27–40, 135–140, 199–261, 282–306 e 308–336.

Quando não encontra plataforma OpenCL, o probe adiciona silenciosamente um dispositivo fictício com PCIe `4.500 ns` e `15,75 GB/s`. Os campos NUMA e energia são preenchidos com constantes (`10`, `250 W`, `120 W`), independentemente do hardware. No ensaio local, `clGetPlatformIDs` retornou `-1001` e zero plataformas; ainda assim o probe com dois ranks finalizou com sucesso, salvou JSON e apresentou dois dispositivos com esses valores PCIe. Com quatro ranks, a matriz MPI incluiu `0→1` medido em `4.192,5 ns` e pares não sondados como `0→2` e `0→3` com fallback `1.500 ns / 3,5 GB/s`. O código mede somente o anel `rank→next_rank`, mas nomeia a matriz completa como topologia medida.

Esses valores entram no mesmo formato que medições reais, sem flag de origem por campo ou par. Aplicá-los à política de balanceamento pode induzir decisões incorretas, e o resultado não satisfaz a promessa de métricas PCIe/MPI exatas descrita no README.

**Correção:** falhar explicitamente quando o hardware exigido não existe, salvo se o usuário optar por modo sintético; marcar cada medida com origem/confiança, sondar todos os pares necessários ou deixar entradas ausentes com erro explícito. Medir NUMA e potência por ferramentas adequadas ou não preencher esses campos. Verificar códigos de retorno de todas as operações OpenCL e MPI do probe.

### A7 — Alto — Os benchmarks não medem execução distribuída ou kernels OpenCL

**Origem:** `14dfb9d`. **Código:** [kneighbor.cpp](/home/agent/HARMONY/benchmarks/kneighbor.cpp:68), linhas 68–96 e 179–217; [leanmd.cpp](/home/agent/HARMONY/benchmarks/leanmd.cpp:68), linhas 68–96 e 181–231; [compare_algos.sh](/home/agent/HARMONY/benchmarks/compare_algos.sh:58), linhas 58–71.

Os programas descobrem dispositivos OpenCL e criam partições no `Runtime`, mas nunca chamam `create_field`, `create_kernel` nem `execute`; os arquivos `.cl` adicionados tampouco são carregados. O cálculo ocorre em vetores de CPU. Em cada rank, o laço percorre **todas** as partições globais sem filtrar `owning_rank`, de modo que dois ranks repetem o mesmo trabalho. Como os dispositivos são processados em sequência em cada processo, `max_element(device_times)` também não é o tempo total daquele passo. O script soma esse máximo, enquanto o programa mede separadamente um `total_duration_s` mais amplo e não o usa na tabela.

Na execução curta auditada, kNeighbor com um rank informou `total_duration_s=0,00023607`, mas a soma de `step_time_s` no CSV foi `0,000093`; com dois ranks, `0,000441623` e `0,000182`, respectivamente. A precisão de seis casas decimais dos CSVs ainda perde resolução de passos curtos. Esses valores são apenas provas da divergência da métrica, não resultados comparativos de desempenho. As tabelas em `README.md` e `benchmarks/comparison_report.md` devem ser tratadas como resultados de uma simulação CPU local até existir execução com campos, kernels, migração e sincronização reais.

**Correção:** executar apenas as partições pertencentes ao rank e usar os kernels/fields da biblioteca, ou rotular explicitamente o programa como simulação CPU. Medir o tempo de parede por rank e reduzir com `MPI_MAX`; registrar separadamente computação, comunicação, migração e balanceamento. Verificar resultados numéricos, repetir ensaios e publicar hardware, número de ranks e dispersão.

### A8 — Alto — Falhas no job PBS são anunciadas como sucesso

**Origem:** `0f8b77c` e `037fd1b`. **Código:** [job_hwtopolb.pbs](/home/agent/HARMONY/job_hwtopolb.pbs:24), linhas 24–27 e 36–67.

O script não ativa encerramento em caso de falha e não verifica os códigos de retorno de `module`, `mpic++`, `mpirun` ou `bash benchmarks/compare_algos.sh`. Executei o job original em uma árvore temporária, com substitutos desses comandos que retornam erro, e um benchmark que também retorna erro. O job mesmo assim terminou com código `0` e a mensagem `All Evaluations Completed Successfully`. Isso pode fazer uma rodada de cluster parecer válida após falha de compilação, teste, probe ou benchmark.

**Correção:** usar `set -euo pipefail`, validar arquivos/variáveis essenciais (`PBS_O_WORKDIR`, `PBS_NODEFILE`) e fazer o resultado de cada etapa controlar o status final. Incluir no log o commit, ambiente e comando que falhou.

### M1 — Médio — O stencil CPU diverge do kernel nas bordas

**Origem:** `14dfb9d`. **Código:** [kneighbor.cpp](/home/agent/HARMONY/benchmarks/kneighbor.cpp:190), linhas 190–202, e [kneighbor.cl](/home/agent/HARMONY/benchmarks/kneighbor.cl:10), linhas 10–18.

O kernel OpenCL divide a soma pelo número real de vizinhos encontrados; o laço CPU sempre divide por `2*k+1`. Com entrada constante igual a `1`, `k=5` e posição `0`, o kernel produziria `1`, enquanto a versão CPU produz `6/11`. Os resultados do benchmark tampouco são conferidos contra uma referência, então essa divergência passa despercebida.

**Correção:** contar vizinhos no caminho CPU, comparar as saídas CPU/OpenCL em bordas e no interior, e falhar o benchmark quando a diferença ultrapassar uma tolerância definida.

### M2 — Médio — A comparação estocástica não é reprodutível

**Origem:** `14dfb9d`. **Código:** [algorithms.hpp](/home/agent/HARMONY/dcl/algorithms.hpp:121), linhas 121–145; [compare_algos.sh](/home/agent/HARMONY/benchmarks/compare_algos.sh:27), linhas 27–38; [comparison_report.md](/home/agent/HARMONY/benchmarks/comparison_report.md:15), linhas 15–25.

`hwtopolb_loads` inicializa o gerador com `std::random_device`, sem opção de seed nos programas. O script faz apenas uma execução por combinação de programa e algoritmo e o relatório apresenta números com seis casas decimais, sem intervalo de variação. Com isso, a diferença de alguns por cento entre ERAD e HWTOPOLB não sustenta uma conclusão estável. O texto do relatório também atribui efeitos de comunicação, barreiras e migração aos números, embora os benchmarks não movimentem campos nem executem seus kernels OpenCL.

**Correção:** aceitar e registrar uma seed, executar séries repetidas com os mesmos dados e configuração, apresentar mediana e dispersão, e fundamentar afirmações de comunicação com métricas coletadas da execução distribuída real.

### M3 — Médio — O leitor de topologia aceita JSON inválido e valores truncados

**Origem:** `14dfb9d`. **Código:** [topo_metrics_io.hpp](/home/agent/HARMONY/dcl/topo_metrics_io.hpp:55), linhas 55–103 e 150–208.

O parser lê números com `std::istringstream` sem verificar o consumo completo do token e encerra a leitura sem exigir fechamento do objeto ou fim do arquivo. Provas pequenas mostraram que aceitou `{` sem `}`, `{ "pcie_latency_ns":[1.2.3] }` (valor truncado), arrays com inteiros fracionários convertidos para `int`, chave sem vírgula separadora, vírgula final e `{}junk`. Isso permite que uma topologia corrompida seja carregada silenciosamente. `set_topo_metrics` também não exige finitude de todos os números: um fator de contenção `NaN` passa pela verificação `factor < 1.0`, e `adjusted_capacity` devolve `NaN`. O particionador evita aplicar cargas não finitas em algumas rotas, mas a medição torna-se inválida sem diagnóstico na entrada.

**Correção:** usar um parser JSON testado ou validar integralmente sintaxe, consumo do token, tipos e fim do documento; depois validar dimensões, faixas e `std::isfinite` para cada métrica antes de aceitar o objeto.

### M4 — Médio — A fase local do balanceamento hierárquico é apenas cálculo

**Origem:** `14dfb9d`. **Código:** [runtime_impl.hpp](/home/agent/HARMONY/dcl/runtime_impl.hpp:3372), linhas 3372–3407, 3486–3506 e 3586–3603; descrição no [README.md](/home/agent/HARMONY/README.md) na seção R3.

O modo calcula `p_local`, sincroniza todos os ranks por `MPI_Barrier`, calcula a divisão entre ranks, reúne as partições por `MPI_Allgatherv` e só então atualiza/migra os campos. Assim, uma mudança de carga **dentro do mesmo rank** também passa pelas coletivas globais, e não há primeira redistribuição local independente. A implementação pode ainda ser útil como construção de uma proposta em dois níveis, mas a redução de frequência de comunicação anunciada não decorre do fluxo atual.

**Correção:** caso a meta seja reduzir coletivas, executar a migração local quando o corte entre ranks não muda e escalar para coletivas apenas quando uma condição global sincronizada exigir troca inter-rank. Medir contagem e tempo das coletivas antes de afirmar ganho.

### P1 — Atenção — Dados históricos foram retirados e a descrição da remoção está incorreta

**Origem:** `14dfb9d`. **Código:** [.gitignore](/home/agent/HARMONY/.gitignore:28), linhas 28–40; [README.md](/home/agent/HARMONY/README.md:251), linhas 251–253.

O commit removeu 804 arquivos rastreados de `Results/`, além de artefatos binários rastreados como `HIS.out`, `job_mpiocl.out` e `profiling.out`. O README afirma que os binários nunca foram rastreados, o que contradiz o diff. As medições antigas podem ser recuperadas de `b3206cd`, mas a árvore atual não traz os dados brutos associados a tabelas anteriores. Isto reduz a rastreabilidade de desempenho e pode ocultar regressões até que os ensaios sejam refeitos.

**Correção:** documentar a remoção com precisão, preservar resultados de referência versionados em local definido ou arquivá-los com hash/commit, parâmetros e ambiente, e gerar novos dados com os benchmarks corrigidos.

## Riscos herdados encontrados ao auditar as rotas alteradas

### H1 — Alto — Objetos OpenCL podem vazar em redescoberta ou falha parcial

**Existia em `b3206cd` e permanece no estado auditado. Código atual:** [runtime_impl.hpp](/home/agent/HARMONY/dcl/runtime_impl.hpp:346), linhas 346–347, 500–525, 527–580 e 1301–1310.

`discover_devices()` inicia com `clear_runtime_state()`, que apaga vetores e mapas sem liberar os `cl_context`, `cl_command_queue`, `cl_mem`, `cl_program` e `cl_kernel` que o destrutor normalmente liberaria. Se a descoberta for repetida, o runtime perde as referências aos objetos anteriores. `create_field()` cria um buffer por dispositivo; se a segunda alocação falha, o primeiro buffer ainda está apenas em uma estrutura local que sai por exceção, sem liberação. `create_kernel()` tem o mesmo padrão para programas e kernels quando uma etapa posterior falha. Esses defeitos não foram introduzidos pelos seis commits, mas afetam diretamente fluxos que passaram a usar descoberta e métricas de topologia.

Uma reprodução com OpenCL simulado instrumentou criação e liberação em dois ranks. Após duas descobertas e três falhas forçadas na segunda alocação de campo, **cada rank** terminou com `1` contexto, `4` filas e `3` buffers sem liberação. A prova usa funções OpenCL substitutas, portanto demonstra a lógica de ownership e não quantifica vazamento em um driver real.

**Correção:** adotar RAII para todos os handles OpenCL; antes de limpar o estado, finalizar filas e liberar objetos em ordem; criar campos/kernels em temporários com guardas que desfazem alocação parcial; adicionar testes de injeção de falhas em cada chamada OpenCL relevante.

### H2 — Alto — Dependências de eventos entre filas OpenCL podem ficar sem progresso

**Existia em `b3206cd` e permanece no estado auditado. Código atual:** [troca de halo](/home/agent/HARMONY/dcl/runtime_impl.hpp:2112), linhas 2112–2146, 2152–2168 e 2235–2246; [execução assíncrona](/home/agent/HARMONY/dcl/runtime_impl.hpp:1868), linhas 1868–1905.

As leituras de halo são enfileiradas em `transfer_queue` com `event_wait_list` que pode conter eventos de kernels enfileirados em `kernel_queue`. Não há `clFlush(kernel_queue)` no runtime. Quando o passo anterior foi assíncrono (`synchronize_at_end=false`), uma implementação OpenCL pode reter comandos nessa fila produtora; esperar pela leitura na outra fila não garante que o kernel anterior tenha sido submetido. A [referência Khronos de `clFlush`](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clFlush.html) exige flush da fila produtora antes de usar seu evento como dependência em outra fila. O efeito possível é espera indefinida no halo ou ausência de progresso. Este é um risco demonstrado por inspeção e pelo contrato da API, **não** um deadlock reproduzido em GPU nesta máquina.

**Correção:** fazer `clFlush` e verificar seu retorno ao publicar dependências entre filas, ou aguardar explicitamente a conclusão na fila produtora antes do uso cruzado. Incluir teste de passos assíncronos com halo em duas filas em um driver real, com timeout e verificação de resultado.

### H3 — Médio — Partições menores que o halo saem da troca de fronteira

**Existia em `b3206cd`; a nova política hierárquica pode alcançá-lo com mais frequência. Código atual:** [runtime_impl.hpp](/home/agent/HARMONY/dcl/runtime_impl.hpp:2084), linhas 2084–2094.

`exchange_halos_for_field` descarta qualquer partição cujo `element_count < halo`, embora ela ainda contenha elementos e possa ser processada em `run_border_phase`. Ao rebalancear para dispositivos muito rápidos/lentos ou quando há mais dispositivos que unidades úteis, o modo hierárquico pode produzir uma fatia positiva menor que a largura do halo. A troca então liga partições não adjacentes como se a fatia pequena não existisse, com potencial de valores de fronteira incorretos. Este cenário foi identificado por inspeção; não houve execução de kernel OpenCL para verificar a magnitude do erro.

**Correção:** garantir na construção das partições um tamanho mínimo compatível com o halo, ou implementar troca que atravessa múltiplas partições curtas sem perder dados e sem falsificar adjacência. Acrescentar teste numérico de stencil com `element_count` menor, igual e maior que `halo`.

## Cobertura e limites

Os testes oficiais usam `set_simulated_devices_count` e `set_simulated_times` em rotas importantes. O teste hierárquico com dois ranks passou, mas não valida tráfego de dados entre GPUs nem leitura de resultados após rebalanceamento real. A execução OpenCL real e as propriedades de desempenho de PCIe, MPI entre nós e NUMA exigem um cluster com dispositivos disponíveis. A ausência de um travamento durante os testes atuais não exclui deadlocks em topologias assimétricas, falhas em ranks individuais ou campos que precisem migrar.

| Risco HPC solicitado | Conclusão desta auditoria |
|---|---|
| Deadlock MPI | C2 reproduz travamento condicionado à divergência de configuração. A3 comprova partições diferentes entre ranks e é uma condição perigosa para envios/recebimentos bloqueantes; a prova adicional com campos reais ainda falta. |
| Race conditions | Nenhuma race de memória entre threads foi demonstrada. Os benchmarks novos são sequenciais em cada rank, e esta máquina não oferece dispositivo OpenCL para analisar concorrência de filas/kernels em execução. H2 aponta uma dependência entre filas sem submissão garantida, que é risco de progresso, não prova de race de dados. |
| Memory leak | H1 reproduz vazamento de handles OpenCL em rotas herdadas por redescoberta e falha parcial de criação de campo. Não há medição de consumo de GPU/driver real nesta máquina. |
| Integridade numérica | C1 reproduz perda integral de cobertura global; M1 mostra divergência numérica entre CPU e kernel nas bordas. Nenhum teste oficial compara resultado final após migração. |
| Escalabilidade e validade de tempo | A7, A5 e A6 mostram que os ensaios atuais não medem migração distribuída nem usam completamente as métricas de rede prometidas. |

## Ordem sugerida de correção e validação

1. Corrigir invariantes globais de partições e garantir que decisões de balanceamento e sequência de coletivas sejam iguais em todos os ranks; adicionar testes com ranks sem dispositivo e contagens assimétricas.
2. Corrigir a ingestão e a medição das métricas globais de topologia; impedir que valores sintéticos sejam aceitos como medições reais.
3. Rever a semântica do limite de energia e o modelo de custo para migração inter-rank; testar casos em que todos os dispositivos estão acima do limite e casos dominados por rede.
4. Fazer os benchmarks exercitarem kernel, campos, troca de dados e MPI reais, com validação numérica e medição de tempo de parede global.
5. Fazer o job falhar com o primeiro comando malsucedido e arquivar resultados com origem verificável.

## Referências normativas usadas para interpretar riscos

- [MPI Forum, MPI 4.1, correção de operações coletivas](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node172.htm): coletivas do mesmo comunicador precisam ser chamadas na mesma ordem por todos os membros; retornos divergentes em rotas com coletivas podem causar travamento ou comportamento indefinido.
- [Khronos, especificação OpenCL](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html): contrato de filas, eventos, transferências de buffers e liberação de recursos usado na análise das rotas OpenCL.
