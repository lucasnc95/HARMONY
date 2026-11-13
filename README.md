# HARMONY

Camada de abstração sobre **OpenCL + MPI** para executar kernels OpenCL em
um cluster heterogêneo (CPUs, GPUs e aceleradores em vários nós) com
**balanceamento de carga automático**.

O usuário escreve kernels OpenCL comuns e descreve os dados e o que cada
kernel lê e escreve; o runtime (namespace `harmony`) cuida de:

- descobrir os dispositivos OpenCL de todos os ranks MPI;
- particionar o domínio 1D entre todos os dispositivos;
- executar vários kernels por passo, em paralelo quando não dependem uns dos outros;
- trocar halos entre partições vizinhas só quando necessário, com todas as
  mensagens em paralelo e sobrepostas ao cálculo do interior;
- medir o tempo de cada dispositivo e redistribuir os dados quando compensa;
- reunir o resultado no rank 0;
- medir o tempo de cada fase e imprimir um relatório legível.

## Estrutura do repositório

```
build.sh               compila a biblioteca e as ferramentas
bsbash/                funções do build (compilação incremental, detecção de MPI/OpenCL)
include/harmony/       API pública
  types.hpp              tipos: FieldSpec, KernelBinding, HaloSpec, ExecutionStep, AutoBalancePolicy, ...
  runtime.hpp            harmony::Runtime e os builders (bind / step)
src/                   biblioteca (libharmony.a)
  build.sh               lista de fontes da biblioteca
  runtime.cpp            fachada da API pública
  runtime_impl.hpp       declaração de Runtime::Impl (privado)
  impl/                  implementação, um arquivo por responsabilidade
    devices.cpp            MPI, descoberta de dispositivos, filas OpenCL
    resources.cpp          campos (buffers) e kernels
    partitioning.cpp       decomposição do domínio, cargas <-> partições
    execution.cpp          execute(): invocações, filas, fases full/interior/border
    events.cpp             dependências entre comandos, estatísticas de kernel, sincronização
    communication.cpp      transferências MPI em blocos e gather()
    halo.cpp               troca de halos entre partições vizinhas
    redistribution.cpp     migração de dados após reparticionar, rebalance_to()
    balancing.cpp          políticas de balanceamento (threshold e profiled)
    timing.cpp             medição por fase e relatório de tempos
  detail/                utilitários internos (erros, matemática de cargas, tabelas)
tools/migration_profiler/  mede o custo de migração (modo "profiled")
examples/              aplicações independentes, cada uma com seu build.sh
  quickstart_stencil1d/  exemplo mínimo, verificado contra a CPU
  multi_kernel/          três kernels por passo, verificado contra a CPU
  cardiac_tissue/        tecido cardíaco com modelo do tipo ToR-ORd, 2 kernels por passo + snapshots VTK
  his/                   simulação do Sistema Imune Humano (malha 3D) + snapshots VTK
  stencil2d/             stencils 2D: HARMONY vs. MPI+OpenCL escrito à mão
  run_checks.sh          compila e verifica todos os exemplos
scripts/pbs/           jobs PBS para o cluster (HIS e tecido cardíaco)
```

## Requisitos

- Compilador C++17 com wrapper MPI (`mpicxx`, `mpic++`; MPICH ou Open MPI)
- OpenCL 1.2+ (headers e ICD loader) e ao menos um driver/dispositivo
- bash

## Compilação

```bash
./build.sh
```

Gera `lib/libharmony.a` e `bin/migration_profiler`. A compilação é
incremental: só recompila o que mudou (fonte, headers incluídos ou flags).

```
Usage: ./build.sh [options] [targets]

Targets (default: lib tools):
  lib        static library lib/libharmony.a
  tools      bin/migration_profiler
  examples   builds every example with its own build.sh (needs lib)
  install    copies include/ and lib/ to the prefix given with -p
  clean      removes build/, lib/, bin/ and the examples' build output

Options:
  -r         release build (default): -O3
  -d         debug build: -O0 -g
  -j N       parallel compilations (default: number of cores)
  -p DIR     installation prefix for "install"
  -q         quiet: only errors
  -v         verbose: print every command
  -h         this help
```

Variáveis de ambiente: `MPICXX` (wrapper MPI), `OPENCL_INCLUDE_DIR` (pasta com
`CL/cl.h`), `OPENCL_LIBRARY_DIR` (pasta com `libOpenCL.so`),
`CL_TARGET_OPENCL_VERSION` (padrão 300) e `EXTRA_CXXFLAGS`.

O OpenCL é procurado no caminho padrão e em locais comuns (`/usr/local/cuda`,
`/opt/rocm`, ...). As flags encontradas ficam gravadas em
`lib/harmony_config.sh`, que o `build.sh` de cada exemplo (ou de uma
aplicação sua) lê para compilar com o mesmo compilador e o mesmo OpenCL. Se a
biblioteca do OpenCL estiver fora do caminho padrão, o diretório é gravado no
executável (rpath), sem precisar de `LD_LIBRARY_PATH`.

No cluster, por exemplo:

```bash
MPICXX=/opt/ohpc/pub/mpi/mpich-ofi-gnu13-ohpc/3.4.3/bin/mpic++ \
OPENCL_INCLUDE_DIR=/usr/local/cuda-12.6/targets/x86_64-linux/include \
OPENCL_LIBRARY_DIR=/usr/lib64 \
./build.sh
```

Para instalar em outro lugar (headers + biblioteca):

```bash
./build.sh -p $HOME/opt/harmony install
```

## Uso da API

A forma de chamar é: `bind(...).arg(...)` liga os
argumentos de um kernel e `step(...).invoke(...).with_halo_exchange(...)
.with_balance(...)` monta uma iteração.

```cpp
#include <harmony/runtime.hpp>

int main(int argc, char** argv) {
    auto runtime = harmony::Runtime::create(argc, argv);       // inicializa o MPI
    runtime.discover_devices({harmony::DeviceKind::all, 0});   // todos os dispositivos de todos os ranks
    runtime.set_partition({n, 1, sizeof(float), 1});           // n elementos de 1 float

    auto u = runtime.create_field({"u", n, 1, sizeof(float), harmony::BufferUsage::read_write, host_u.data(),
                                   harmony::RedistributionDependency::proportional});
    // ... v, g, e da mesma forma
    auto gradient = runtime.create_kernel({"kernels.cl", "gradient", ""});
    auto update   = runtime.create_kernel({"kernels.cl", "update", ""});
    auto energy   = runtime.create_kernel({"kernels.cl", "energy", ""});

    auto bind_gradient = runtime.bind(gradient)
        .arg(0, u, harmony::Access::read)
        .arg(1, g, harmony::Access::write)
        .arg(2, n)
        .build();
    auto bind_update = runtime.bind(update)
        .arg(0, u, harmony::Access::read)
        .arg(1, g, harmony::Access::read)
        .arg(2, v, harmony::Access::write)
        .arg(3, c)
        .arg(4, n)
        .build();
    auto bind_energy = runtime.bind(energy)
        .arg(0, u, harmony::Access::read)
        .arg(1, e, harmony::Access::write)
        .arg(2, n)
        .build();

    harmony::ExecutionStep step = runtime.step("advance")
        .invoke(bind_gradient)
        .invoke(bind_update)
        .invoke(bind_energy)
        .with_halo_exchange(harmony::HaloSpec{1, {u, g}})      // largura 1, campos u e g
        .with_halo_sync(harmony::HaloSync::overlap)
        .with_balance(harmony::AutoBalancePolicy{
            harmony::BalanceMode::dynamic_profiled,
            100,                     // interval
            0.0f,                    // threshold (modos threshold)
            iterations,              // total_iterations (modos profiled)
            "profiling_results.txt", // profiling_file (modos profiled)
            500                      // warmup: primeira tentativa na iteração 500
        })
        .tag_field(v, harmony::StepFieldRole::rebalance_source)  // opcional
        .synchronize_at_end(false)
        .build();

    runtime.start_timing();
    for (int it = 0; it < iterations; ++it) runtime.execute(step);   // normalmente alterna u<->v
    runtime.stop_timing();
    runtime.print_timing_report();                                    // tabelas no rank 0

    runtime.gather(v, result.data(), n * sizeof(float));              // resultado completo no rank 0
}
```

Conceitos:

- **Field**: buffer distribuído de `global_elements` elementos. Cada dispositivo
  mantém uma réplica do tamanho total, mas só atualiza a própria partição (+ halos).
  `RedistributionDependency::proportional` indica que o campo acompanha a partição;
  `none` indica dados replicados (ex.: parâmetros), que só podem ser lidos.
- **Partition**: as fronteiras são sempre múltiplas de `granularity`
  (ex.: um plano XY de uma malha 3D).
- **KernelBinding** (`runtime.bind(kernel)`): um kernel com seus argumentos.
  - `arg(i, campo, acesso)`: campo e como o kernel o usa: `Access::read`,
    `Access::write` (sobrescreve toda a partição) ou `Access::read_write`;
  - `arg(i, campo)`: sem acesso explícito o runtime usa `read` para campos
    replicados e para campos com troca de halo no passo, e `read_write` (o mais
    conservador) para os demais;
  - `arg(i, valor)`: escalar (int, float, struct trivial, ...).
- **ExecutionStep** (`runtime.step(nome)`): uma iteração.
  - `invoke(binding, geometria)`: um kernel; vários `invoke` = vários kernels,
    em ordem de programa. A `LaunchGeometry{global_offset, global_size,
    local_size}` é opcional; `global_size` deve ser 0 ou o total de elementos
    da partição (cada dispositivo roda só a sua parte; antes o valor era
    ignorado sem aviso);
  - `with_halo_exchange(HaloSpec{largura, {campos}})`: campos cujas bordas os
    kernels leem. Pode ser chamado mais de uma vez para larguras diferentes;
  - `with_halo_sync(HaloSync)`: o que acontece antes da troca de halos;
  - `with_balance(AutoBalancePolicy{...})`: política de balanceamento;
  - `tag_field(campo, StepFieldRole::rebalance_source)`: campos movidos quando
    a divisão muda. Sem nenhum `tag_field`, todos os campos proporcionais do
    passo são movidos;
  - `synchronize_at_end(bool)`: espera os dispositivos no fim do passo
    (padrão `false`, para não perder a sobreposição).

### Vários kernels por passo

Um passo é uma **lista ordenada de `invoke`**, e cada argumento de campo
declara o acesso. O resultado é sempre o mesmo de rodar os kernels um depois
do outro, na ordem escrita; o paralelismo é otimização do runtime. A partir
dos acessos e do `HaloSpec` o runtime deduz:

| O que | Como |
|---|---|
| **Dependências** | leitura depois de escrita, escrita depois de leitura e escrita depois de escrita, por campo e por dispositivo, impostas com eventos OpenCL |
| **Paralelismo** | cada dispositivo tem várias filas de kernel (`DeviceSelection::kernel_queues`, padrão 4). Um kernel que depende de outro do mesmo passo vai para a fila dele; um independente vai para a próxima fila e roda ao mesmo tempo. No exemplo acima, `update` espera `gradient` (lê `g`), e `energy` roda em paralelo aos dois |
| **Quando trocar halos** | o halo de um campo fica desatualizado quando o campo é escrito e só é trocado quando um kernel posterior o lê (campo listado num `HaloSpec`), uma única vez. No exemplo, `u` é trocado antes de `gradient` e `g` antes de `update`, sem nenhuma chamada explícita |
| **O que migrar** | os campos com `tag_field(..., rebalance_source)`; sem tags, todos os campos proporcionais usados pelo passo |

Regras verificadas em `execute()` (erro com mensagem clara):

- campos replicados (`none`) não podem ser escritos;
- um campo com troca de halo não pode ser `read_write` no mesmo kernel, nem
  lido com halo por um kernel que também o escreve (os vizinhos leriam
  valores já alterados); use dois buffers;
- `LaunchGeometry::global_size` deve ser 0 ou o total de elementos.

Limitações: todos os kernels de um passo percorrem o mesmo domínio
particionado (um work-item por elemento), e um `write` deve escrever todos os
elementos da partição.

## Troca de bordas (halo)

Um kernel que precisa de halo desatualizado roda em três fases por
partição (`run_interior_phase`, `exchange_halo_set`, `run_border_phase`):
**interior** (não depende dos vizinhos), **troca de halos** e
**bordas**. Sem halo pendente, roda de uma vez. A troca processa todas as
mensagens de todos os campos e de todos os pares de vizinhos ao mesmo tempo:

1. todos os `MPI_Irecv` são postados primeiro;
2. todas as leituras dos dispositivos são enfileiradas de uma vez; entre
   dispositivos do mesmo rank e do mesmo contexto OpenCL a cópia é feita
   direto no dispositivo (`clEnqueueCopyBuffer`, sem passar pelo host);
3. cada `MPI_Isend` sai assim que a sua leitura termina;
4. cada halo recebido é escrito no dispositivo assim que chega (`MPI_Waitany`);
5. no fim, espera os envios e as escritas.

O interior nunca espera a troca (não lê a região de halo); as bordas dependem
das escritas do halo por eventos OpenCL. As mensagens de halo usam um
comunicador MPI próprio.

A sincronização antes da troca é escolhida por passo (`with_halo_sync(...)`):

| `HaloSync` | Comportamento |
|---|---|
| `overlap` (padrão) | a troca acontece enquanto o interior é calculado |
| `local` | espera os dispositivos do rank terminarem o interior antes de trocar |
| `global` | `local` + `MPI_Barrier`: todos os ranks começam a troca juntos |

Nos exemplos: `--halo-sync overlap|local|global`.

## Medição de tempos e logs

A medição não adiciona nenhuma sincronização: os tempos da CPU são lidos com
`MPI_Wtime` e os dos dispositivos vêm dos eventos de profiling do OpenCL. A
sobreposição entre cálculo e troca de bordas fica igual, sem custo mensurável
nos testes.

- `start_timing()` e `stop_timing()` são coletivos e delimitam o intervalo
  medido. `stop_timing()` devolve o tempo de parede do processo mais lento e
  congela os contadores: o que vier depois, como `gather()`, não entra no resumo.
- `print_timing_report()` (coletivo) imprime o resumo no rank 0;
  `timing_report()` devolve os mesmos dados numa struct (`TimingReport`).
- `write_timing_report(arquivo)` (coletivo) grava o mesmo relatório em CSV,
  para scripts.
- `set_verbose(false)` desliga o log das tentativas de balanceamento (ligado
  por padrão; nos exemplos, `--verbose 0`).
- `DeviceSelection::cpu_host_cores`: núcleos de cada dispositivo de CPU
  deixados livres para o processo MPI (o dispositivo vira um sub-dispositivo
  OpenCL com os demais; se o driver não suportar, um aviso é impresso).

### Log do balanceamento

Impresso pelo rank 0 durante a execução. Na primeira tentativa, a política
é descrita (warm-up, intervalo e critério). Cada tentativa tem três linhas:

1. `Balance #N | iteration I | política | CHANGED/kept the division`;
2. `Why:` o motivo da decisão (no `threshold`, a mudança da divisão comparada
   ao limiar; no `profiled`, quanto tempo a nova divisão economiza nas
   iterações que faltam comparado ao custo de mover os dados);
3. `Time per iteration:` o tempo por iteração do dispositivo mais lento (é ele
   que define o tempo do passo), medido agora, e o esperado com a nova
   divisão, supondo que cada dispositivo mantém a velocidade medida.

Quando a divisão muda, uma tabela mostra por dispositivo o tempo de kernel
por iteração na janela medida, os elementos e a fatia antes e depois, e marca
o mais lento. Exemplo (HIS pequeno, 2 processos, warm-up 30, intervalo 20):

```
[HARMONY] Load balancing: dynamic threshold
          Warm-up: 30 iterations with the initial division, then an attempt every 20 iterations.
          Each attempt measures the last 20 iterations and proposes shares proportional to the speed of each device;
          the new division is applied when it changes by at least 0.1 (L2 distance of the cumulative loads).
[HARMONY] Balance #1 | iteration 30 | dynamic threshold | CHANGED the division
          Why: division change 0.125 >= threshold 0.1
          Time per iteration: 497.3 us now (slowest: CPU dev 0) -> 380.0 us expected | measured over 20 iterations
          Dev  Rank  Type  Name                                  Kernel time/iteration  Elements before  Elements after  Change  Share before  Share after  Note
          ---  ----  ----  ------------------------------------  ---------------------  ---------------  --------------  ------  ------------  -----------  ----------
          0    0     CPU   cpu-skylake-avx512-Intel(R) Xeon(...               497.3 us            1,280             960    -320        50.0 %       37.5 %  <- slowest
          1    1     CPU   cpu-skylake-avx512-Intel(R) Xeon(...               304.0 us            1,280           1,600    +320        50.0 %       62.5 %
          Data moved: 10.0 KiB in 194.2 us | decision took 158.1 us
[HARMONY] Balance #2 | iteration 50 | dynamic threshold | CHANGED the division
          Why: division change 0.375 >= threshold 0.1
          Time per iteration: 761.3 us now (slowest: CPU dev 1) -> 309.9 us expected | measured over 20 iterations
          Dev  Rank  Type  Name                                  Kernel time/iteration  Elements before  Elements after  Change  Share before  Share after  Note
          ---  ----  ----  ------------------------------------  ---------------------  ---------------  --------------  ------  ------------  -----------  ----------
          0    0     CPU   cpu-skylake-avx512-Intel(R) Xeon(...               154.9 us              960           1,920    +960        37.5 %       75.0 %
          1    1     CPU   cpu-skylake-avx512-Intel(R) Xeon(...               761.3 us            1,600             640    -960        62.5 %       25.0 %  <- slowest
          Data moved: 30.0 KiB in 187.8 us | decision took 53.4 us
...
```

### Resumo no fim

Logo no início, **uma linha com o caminho crítico** (TIMELINE OF THE CRITICAL
PATH): o processo MPI que controla o dispositivo mais lento (o de maior tempo
de kernel), com o tempo total dividido em partes exclusivas que somam o tempo
de parede: computação OpenCL (tempo de kernel desse dispositivo), sobrecarga
OpenCL (o resto do tempo bloqueado no OpenCL: lançamentos, cópias, latência do
driver), preparação de halos, MPI de halos, de migração, de balanceamento e
barreiras, migração de dados, balanceamento e tempo não medido. É a linha
para os gráficos de barras empilhadas do artigo (também sai no CSV, linha
`timeline`, e em `TimingReport::timeline`). As tabelas abaixo dela continuam.

Os tempos de cada processo MPI são **exclusivos**: quando uma etapa começa
dentro de outra, a de fora é pausada. Por isso as colunas somam exatamente o
tempo total, e "Not measured" é o tempo fora do HARMONY (código da aplicação
entre as chamadas). Para cada dispositivo, Compute + Transfers + Idle também
somam o tempo total; kernels concorrentes contam uma vez. A seção LOAD
BALANCING lista as tentativas (com o tempo por iteração antes → depois) e a
fatia de cada dispositivo no início e depois de cada mudança.

```
====================================================================================================
HARMONY - TIMING SUMMARY
====================================================================================================
Wall time: 950.52 ms    Iterations: 101 (9.41 ms per iteration)
MPI processes: 2    Devices: 2
Load balancing: dynamic threshold, warm-up 30, interval 20, threshold 0.1

TIME OF EACH DEVICE (Compute + Transfers + Idle = wall time)
  Compute:    OpenCL kernels running
  Transfers:  copies between host and device memory (halos and data migration)
  Idle:       no work (waiting for other devices, MPI or the CPU)

  Dev  Rank  Type  Name                                  Elements   Share           Compute        Transfers                Idle
  ---  ----  ----  ------------------------------------  --------  ------  ----------------  ---------------  ------------------
  0    0     CPU   cpu-skylake-avx512-Intel(R) Xeon(...     2,048  80.0 %  44.59 ms (4.7 %)  69.1 us (0.0 %)  905.87 ms (95.3 %)
  1    1     CPU   cpu-skylake-avx512-Intel(R) Xeon(...       512  20.0 %  49.38 ms (5.2 %)  99.4 us (0.0 %)  901.04 ms (94.8 %)
  Busiest device: CPU dev 1, 49.38 ms of kernels = 488.9 us per iteration. Least busy: CPU dev 0, 1.11x less.
  A large Idle on the fast devices means they wait for the busiest one.

TIME OF EACH MPI PROCESS (the CPU that drives the devices; the columns add up to the wall time)
  OpenCL launch:   preparing and enqueueing kernels
  OpenCL wait:     CPU waiting for the devices to finish kernels and copies
  Halo exchange:   preparing and copying borders between devices (without the waits)
  MPI:             waiting for MPI messages and collective operations
  Data migration:  moving data between devices after a rebalance (without MPI)
  Balancing:       measuring times and deciding the new division (without MPI)
  Not measured:    outside HARMONY (application code between calls)
  High OpenCL wait or MPI usually means waiting for a slower device or process.

  Rank  Host     OpenCL launch         OpenCL wait    Halo exchange                 MPI    Data migration         Balancing     Not measured
  ----  ----  ----------------  ------------------  ---------------  ------------------  ----------------  ----------------  ---------------
  0     vm    21.76 ms (2.3 %)  849.92 ms (89.4 %)  2.38 ms (0.3 %)    75.78 ms (8.0 %)  345.3 us (0.0 %)  326.0 us (0.0 %)  20.6 us (0.0 %)
  1     vm    38.63 ms (4.1 %)  682.41 ms (71.8 %)  4.08 ms (0.4 %)  224.92 ms (23.7 %)  402.6 us (0.0 %)   56.9 us (0.0 %)  19.5 us (0.0 %)

  MPI time by reason
  Rank  Halo exchange  Data migration  Balancing  Barriers      Total
  ----  -------------  --------------  ---------  --------  ---------
  0          75.01 ms        413.2 us   131.1 us  226.2 us   75.78 ms
  1         224.44 ms        349.8 us   114.2 us   23.1 us  224.92 ms

LOAD BALANCING
  4 attempts, 4 changed the division. Data moved: 128.0 KiB in 801.0 us. Decisions: 421.9 us.
  Time/iteration = kernel time of the slowest device per iteration, measured before the attempt
  and expected after it (for a kept division: expected with the best division).

  #  Iteration  Result         Time/iteration  Data moved  Moved in  Why
  -  ---------  -------  --------------------  ----------  --------  --------------------------------------
  1         30  changed  497.3 us -> 380.0 us    10.0 KiB  194.2 us  division change 0.125 >= threshold 0.1
  2         50  changed  761.3 us -> 309.9 us    30.0 KiB  187.8 us  division change 0.375 >= threshold 0.1
  3         70  changed  888.6 us -> 266.6 us    42.0 KiB  221.8 us  division change 0.525 >= threshold 0.1
  4         90  changed   1.15 ms -> 296.0 us    46.0 KiB  197.2 us  division change 0.575 >= threshold 0.1

  Share of the domain per device: at the start and after each change (#attempt @iteration)
  Dev  Type   Start  #1 @30  #2 @50  #3 @70  #4 @90
  ---  ----  ------  ------  ------  ------  ------
  0    CPU   50.0 %  37.5 %  75.0 %  22.5 %  80.0 %
  1    CPU   50.0 %  62.5 %  25.0 %  77.5 %  20.0 %
====================================================================================================
```

## Políticas de balanceamento

`AutoBalancePolicy{mode, interval, threshold, total_iterations, profiling_file, warmup}`

| Modo | Quando tenta | Como decide |
|---|---|---|
| `off` | nunca | — |
| `static_threshold` | uma vez, na iteração `warmup` | distância L2 entre cargas atual e proposta ≥ `threshold` |
| `dynamic_threshold` | na iteração `warmup` e depois a cada `interval` | idem |
| `static_profiled` | uma vez, na iteração `warmup` | ganho estimado até o fim da execução > custo estimado de migração |
| `dynamic_profiled` | na iteração `warmup` e depois a cada `interval` | idem |

**Warm-up** (`warmup`): iterações executadas com a divisão inicial antes da
primeira tentativa. `0` = `interval` (comportamento anterior). Toda tentativa
mede no máximo as últimas `interval` iterações; com `warmup > interval`, o
início da execução (compilação JIT, transitório inicial, estímulo) não entra
na primeira medição. Nos exemplos: `--balance-warmup N` (HIS e cardíaco) ou
`--warmup N` (quickstart e multi_kernel); nos jobs: `WARMUPS="0 1000 3000"`.

As cargas propostas são proporcionais à vazão medida de cada dispositivo
(elementos / segundo de kernel na janela). Os modos `profiled` usam o arquivo
gerado pelo `migration_profiler`, com segmentos lineares
`custo = m * bytes + b`. Sem o arquivo, um modelo conservador é usado. O
ganho por iteração é calculado com o número real de iterações da janela.

Também é possível reparticionar manualmente com `Runtime::rebalance_to(cargas_acumuladas)`.

## Exemplos

Os exemplos **não fazem parte da biblioteca**: cada pasta é uma aplicação
independente, com seu próprio `main`, parser de argumentos, kernels e
`build.sh`. Eles usam só a API pública (`include/harmony`) e
`lib/libharmony.a`, e podem ser copiados para fora do repositório.

```bash
./build.sh                         # biblioteca
./examples/his/build.sh            # um exemplo -> examples/his/bin/his
./build.sh examples                # ou todos de uma vez
```

O `build.sh` de cada exemplo usa `HARMONY_ROOT` (padrão: a raiz do
repositório; também aceita o prefixo de `./build.sh -p DIR install`) e lê
`lib/harmony_config.sh` para usar o mesmo compilador e o mesmo OpenCL da
biblioteca; `MPICXX`, `OPENCL_INCLUDE_DIR` e `OPENCL_LIBRARY_DIR` têm
prioridade sobre ele. Todos aceitam
`--kernel-file` para indicar outro caminho do `.cl`.

### quickstart_stencil1d

```bash
mpirun -np 2 examples/quickstart_stencil1d/bin/quickstart_stencil1d --n 100000 --iterations 200 --interval 50
```

Um kernel por passo. Imprime `max |device - cpu|` e retorna código diferente
de zero se divergir da versão sequencial.

### multi_kernel

```bash
mpirun -np 2 examples/multi_kernel/bin/multi_kernel --n 200000 --iterations 100
```

Três kernels por passo (`gradient` → `update`, e `energy` em paralelo), com
o halo do campo intermediário `g` trocado automaticamente. Verificado contra a
CPU.

### cardiac_tissue — tecido cardíaco (modelo do tipo ToR-ORd)

```bash
mpirun -np 2 examples/cardiac_tissue/bin/cardiac_tissue --nx 24 --ny 24 --nz 384 --time 100
```

Monodomínio com *operator splitting*, **dois kernels por passo** (`tor_ord.cl`):

| Kernel | Acessos | O que faz |
|---|---|---|
| `diffusion` | `vm`: read (halo de 1 plano), `vm_tmp`: write | Laplaciano de 7 pontos, bordas sem fluxo |
| `reaction` | `vm_tmp`: read, `state`: read_write, `vm`: write | integra o modelo celular por `dt` |

O `reaction` espera o `diffusion` (lê `vm_tmp`), e o halo de `vm` é trocado
automaticamente antes de cada `diffusion`, porque `vm` foi escrito pelo
`reaction` do passo anterior. O tecido 3D (`nx × ny × nz`) é linearizado em
1D com `z` como eixo mais lento, então o HARMONY corta o tecido em planos `z`
e o halo é um plano (`nx·ny` elementos).

**Modelo celular** (`torord_like.h`, o mesmo arquivo compilado no kernel e na
verificação na CPU): família O'Hara-Rudy (ORd 2011, a base do ToR-ORd), com
40 variáveis de estado + V, ~15 correntes, três tipos celulares (endo/mid/epi
ao longo de `x`, como no `ToRORd_fkatp_mixed_endo_mid_epi` do MonoAlg3D) e
duas extensões no estilo do ToR-ORd: ICaL dividida entre subespaço (80%) e
citosol (20%), e correntes de cloreto (ICaCl, IClb). **Não é a calibração
publicada do ToR-ORd**: é um modelo de benchmark com a mesma estrutura e o
mesmo custo. Para uso científico, substitua `torord_like.h` pelas equações
oficiais. Célula isolada a 1 Hz: repouso −87,5 mV, pico ~39 mV,
dV/dt máx. ~250 V/s, APD90 de 248 ms (endo), 229 ms (epi) e 341 ms (mid).
No tecido, a velocidade de condução é ~0,52 m/s.

**Por que é um bom teste de balanceamento:** cada célula divide o `dt` em
subpassos conforme a rigidez da fase do potencial de ação (limites de
variação de V, `--dv-max`, e de Ca²⁺ citosólico, `--ca-tol`). Com os valores
padrão, o custo por célula é de ~120 subpassos no upstroke, 10–13 na liberação
de Ca²⁺ e no início do platô, ~2,5 no platô, ~1,5 na repolarização e 1 no
repouso. A onda parte de `z = 0` e anda ao longo de `z`, o eixo particionado,
então a região cara passa de um dispositivo para o seguinte; depois o tecido
inteiro atravessa platô, repolarização e repouso, e o custo total muda de novo.
Para deixar o kernel ainda mais pesado: diminua `--dv-max`/`--ca-tol`, aumente
`--min-substeps` ou use `--precision double` (caro em GPUs de consumo).

No fim, a saída mostra a velocidade de condução, o trabalho do modelo celular
por faixa de `z` (onde a carga esteve) e o relatório de tempos. `--check 1`
compara com uma execução sequencial na CPU (só para tecidos pequenos).

| Opção | Padrão | Descrição |
|---|---|---|
| `--nx --ny --nz` | `24 24 384` | tamanho do tecido em células; a onda anda em `z` |
| `--dx` / `--dt` | `0.01` cm / `0.01` ms | discretização (estabilidade: D·dt/dx² ≤ 1/6) |
| `--time` | `100` | ms simulados |
| `--diffusivity` | `0.0012` | cm²/ms |
| `--bcl` / `--stim-planes` | `500` / `3` | intervalo entre estímulos e planos estimulados |
| `--dv-max` / `--ca-tol` | `0.005` / `2e-5` | tolerâncias dos subpassos (menor = mais pesado) |
| `--min-substeps` / `--max-substeps` | `1` / `1000` | limites dos subpassos |
| `--precision` | `float` | `float` ou `double` |
| `--balance-mode` / `--balance-strategy` | `dynamic` / `threshold` | como no HIS |
| `--rebalance-interval` / `--rebalance-threshold` | `200` passos / `0.01` | |
| `--balance-warmup` | `0` (= intervalo) | passos antes da primeira tentativa |
| `--device` | `all` | `all`, `gpu` ou `cpu`: dispositivos OpenCL usados |
| `--progress` | `60` | segundos entre linhas de progresso (com estimativa do tempo total) |
| `--cpu-host-cores` | `0` | núcleos de cada dispositivo de CPU deixados para o processo MPI |
| `--report-csv` | — | relatório de tempos em CSV |
| `--halo-sync`, `--verbose`, `--timing-file`, `--check` | | como nos outros exemplos |

O que já se vê nesta máquina (dispositivos POCL idênticos, sem GPU):

- com a onda atravessando o tecido, a frente anda ~5 planos por milissegundo,
  mais rápido que uma janela de medição de 1 ms: o balanceamento dinâmico
  oscila (fatias de 12/88, 2/98, 36/64, 4/96 %...), porque decide com base na
  janela anterior e supõe custo uniforme dentro de cada partição;
- o balanceamento estático decidiu durante o estímulo (região pesada no rank 0)
  e deixou o tempo pior (54 s → 73 s em 60 ms simulados);
- quando a heterogeneidade dura mais que o intervalo (fase do estímulo), o
  dinâmico reduz o tempo (~10% num teste curto).

O ganho com dispositivos de velocidades diferentes (GPU + CPU) precisa ser
medido no cluster: `scripts/pbs/cardiac_benchmark.pbs` roda o caso sem
balanceamento e todas as políticas para vários intervalos (veja
[Cluster (PBS)](#cluster-pbs)).

### his — Sistema Imune Humano

```bash
mpirun -np 2 examples/his/bin/his --x 50 --y 50 --z 3200 --iterations 10000 \
  --balance-mode dynamic --balance-strategy threshold \
  --rebalance-interval 1000 --rebalance-threshold 0.0003125
```

| Opção | Padrão | Descrição |
|---|---|---|
| `--x --y --z` | `50 50 3200` | dimensões da malha |
| `--iterations` | `10000` | passos de tempo |
| `--balance-mode` | `dynamic` | `off`, `static` ou `dynamic` |
| `--balance-strategy` | `threshold` | `threshold` ou `profiled` |
| `--rebalance-interval` | `1000` | iterações entre tentativas |
| `--balance-warmup` | `0` (= intervalo) | iterações antes da primeira tentativa |
| `--rebalance-threshold` | `0.0003125` | limiar do modo `threshold` |
| `--profiling-file` | `profiling_results.txt` | perfil do modo `profiled` |
| `--timing-file` | — | acrescenta o tempo de parede (uma linha por execução) |
| `--gather-final` | `1` | reúne a malha final no rank 0 e imprime um checksum |
| `--print-mesh` | `0` | imprime a concentração de antígeno da malha final |
| `--halo-sync` | `overlap` | `overlap`, `local` ou `global` |
| `--verbose` | `1` | `0` desliga o log de cada tentativa de balanceamento |
| `--device` | `all` | `all`, `gpu` ou `cpu` |
| `--cpu-host-cores` | `0` | núcleos de cada dispositivo de CPU deixados para o processo MPI |
| `--report-csv` | — | relatório de tempos em CSV |
| `--precision` | `float` | `float` ou `double` (o kernel usa `REAL`; em float o resultado é o mesmo de antes) |

Observação sobre o modelo (herdada do kernel original, não alterada): na
equação da citocina anti-inflamatória (CA), o termo de difusão usa
`Laplaciano(CELULAS_A_OFFSET, ...)`, isto é, a difusão do **antígeno**, e não
a da própria CA. Por isso a CA passa do teto `caInf = 3.6` perto da frente do
antígeno. Corrigir muda os resultados de referência (checksum).

### stencil2d — HARMONY vs. MPI+OpenCL

```bash
mpirun -np 2 examples/stencil2d/bin/stencil2d --impl both --kernel lu --nx 1024 --ny 1024 --iters 100
```

`--impl harmony|mpiocl|both`, `--kernel ocean|lu|map`, `--device cpu|gpu|all`,
`--nx`, `--ny`, `--n` (tamanho do `map`), `--flop-repeat`, `--iters`,
`--halo-sync overlap|local|global`, `--report 0|1`. Com `--impl both` as duas
versões rodam, os tempos são mostrados lado a lado (com speedup) e os
resultados são comparados (devem ser idênticos bit a bit).

Opções para os benchmarks:

| Opção | Padrão | Descrição |
|---|---|---|
| `--precision` | `float` | `float` ou `double` (as duas versões) |
| `--chains` | `1` | cadeias independentes no laço de flops do `lu`/`map`: 4 × `flop-repeat` × `chains` flops por elemento. Com 1, cada work-item só faz operações dependentes, o que favorece a GPU; com 4, a CPU consegue usar as unidades vetoriais (paralelismo de instrução) |
| `--balance-mode` / `--balance-strategy` | `off` / `profiled` | balanceamento (só HARMONY) |
| `--rebalance-interval` / `--balance-warmup` / `--rebalance-threshold` / `--profiling-file` | | como nos outros exemplos |
| `--timing-file` | — | acrescenta o tempo de parede (com `--impl both`: HARMONY e MPI+OpenCL) |
| `--report-csv` | — | relatório de tempos em CSV (`write_timing_report`) |
| `--cpu-host-cores` | `0` | núcleos de cada dispositivo de CPU deixados para o processo MPI |

Com `--precision float --chains 1` os kernels são os mesmos de antes (mesmos resultados).

## Benchmarks para o artigo

Cinco benchmarks (`his`, `ocean`, `lu`, `map`, `cardiac`), por enquanto só em
`double` (`PRECISIONS="float double"` roda os dois), em quatro cenários, com 10 execuções por configuração:

| Cenário | Nós (1 processo MPI por nó) | Configurações |
|---|---|---|
| `gpu1` | compute-0-2, só a GPU | `gpu` |
| `n1` | compute-0-2: GPU + CPU | `off`, `static_threshold`, `static_profiled`, `dynamic_threshold`, `dynamic_profiled` |
| `n2` | compute-0-2 + compute-3-6 | as mesmas 5 |
| `n3` | compute-0-2 + compute-3-6 + compute-1-0 (128 núcleos e 1 GPU) | as mesmas 5 |

`off` = fatias iguais, sem balanceamento; `*_threshold` com limiar fixo de
0,01; `*_profiled` com o perfil de migração medido nos nós do cenário (uma
vez por cenário, em `results/paper/profiles/`). Estático = uma tentativa
depois do warm-up; dinâmico = depois do warm-up, uma a cada intervalo.

| Script (`scripts/pbs/`) | O que faz |
|---|---|
| `paper_cases.sh` | tamanho, duração e agenda de balanceamento de cada benchmark em cada precisão |
| `paper_job.pbs` | um benchmark, uma precisão, um cenário; todas as configurações, `RUNS` vezes |
| `submit_paper_tests.sh` | submete a matriz (5 benchmarks × 4 cenários = 20 jobs em double) com nós, núcleos, GPUs e **RAM** (`mem=`) de cada nó |
| `rank_wrapper.sh` | roda em cada processo: limita a GPU usada na compute-1-0 (8 GPUs) |
| `paper_stats.sh` | média, desvio padrão, IC de 95 % (t de Student), CV, tempo por iteração e speedup sobre `gpu1` |

```bash
DRY_RUN=1 scripts/pbs/submit_paper_tests.sh               # mostra os qsub
RUNS=1 OUT_DIR=results/pilot scripts/pbs/submit_paper_tests.sh   # piloto: mede os tempos
scripts/pbs/submit_paper_tests.sh                         # tudo
scripts/pbs/paper_stats.sh results/paper                  # estatística -> results/paper/stats.csv
```

Nós e RAM pedidos (sem `mem=` o PBS reserva só 6 GB):

| Nó | Chunk |
|---|---|
| compute-0-2 | `ncpus=32:ngpus=1:mem=28gb:mpiprocs=1` |
| compute-3-6 | `ncpus=24:ngpus=1:mem=20gb:mpiprocs=1` |
| compute-1-0 | `ncpus=128:ngpus=1:mem=64gb:mpiprocs=1` |

Na compute-1-0 o `rank_wrapper.sh` mostra ao programa só a GPU 0
(`CUDA_VISIBLE_DEVICES=0`) se o PBS não tiver restringido as GPUs. Para o
PBS reservar exatamente a GPU do vnode 0, troque o chunk por um com
`vnode=compute-1-0[0]` (exemplo no cabeçalho do `submit_paper_tests.sh`;
confira os vnodes com `pbsnodes -av`).

**Tamanhos.** Cada dispositivo guarda o domínio inteiro de cada campo em um
único buffer OpenCL, então cada campo tem que caber no maior buffer do menor
dispositivo: na NVIDIA isso é 1/4 da memória, ~2,9 GiB na RTX A2000 12 GB
(a tabela de dispositivos mostra "Largest buffer" e o HARMONY avisa se um
campo não couber). Todos os campos ficam ≤ 2,75 GiB:

| Benchmark | float | double | por dispositivo |
|---|---|---|---|
| `his` | 128 × 128 × 5376 pontos | 128 × 128 × 2688 | 5,3 GiB |
| `ocean`, `lu` | 26624² | 18944² | 5,3 GiB |
| `map` | 700 M elementos | 350 M | 5,2 GiB |
| `cardiac` | 256 × 256 × 280 células (5,1 × 5,1 × 5,6 cm) | 256 × 256 × 140 (2,8 cm em z) | 3,0 GiB |

No cardíaco, `dx = 0,02 cm` e `dt = 0,05 ms` (estabilidade da difusão:
`D·dt/dx² = 0,15 ≤ 1/6`; o modelo celular subdivide o `dt` onde é rígido).
Usar mais que ~5 GiB dos 12 GB exigiria dividir cada campo em vários buffers.

**Duração.** Escolhida para uma execução de ~0,5–2 min numa GPU, a partir dos
tempos antigos do HIS e do cardíaco (estimativa: confirme com o piloto e
ajuste com `LENGTH`). A configuração `off` com CPUs é de longe a mais lenta:
a CPU recebe a mesma fatia da GPU, e em float uma GPU dessas é ~50–100×
mais rápida que a CPU nos stencils. Em float, só as execuções `off` podem
somar dezenas de horas; `RUNS_OFF` permite menos repetições para elas.

Parâmetros (variáveis de ambiente do `submit_paper_tests.sh`): `BENCHES`,
`PRECISIONS` (padrão `double`), `SCENARIOS`, `RUNS` (10), `RUNS_OFF`, `WARMUP` e `INTERVAL`
(padrão: 10 % da execução), `LENGTH`, `THRESHOLD` (0,01), `CPU_HOST_CORES`
(1), `PER_CONFIG=1` (um job por configuração, para limites de walltime),
`WALLTIME`, `CHUNK_0_2`, `CHUNK_3_6`, `CHUNK_1_0`, `GPU_PIN`, `QUEUE`.

Resultados em `results/paper/<benchmark>/<precisão>/<cenário>/<configuração>/`:
`timings.txt` (um tempo por execução), `run_<N>.log`, `report_run<N>.csv`
(com a linha do caminho crítico) e `info.txt` (comando exato).

## Figuras: snapshots em VTK

O HIS e o cardíaco têm um segundo programa, `his_snapshots` e
`cardiac_snapshots` (gerados pelo `build.sh` de cada exemplo), que roda a
**mesma simulação** (mesmos kernels e passos do HARMONY, com balanceamento) e
grava a malha em arquivos VTK legados (`STRUCTURED_POINTS`, binário) a cada
intervalo, **sem medir tempos**. É para rodar uma vez, com uma malha pequena,
e fazer as figuras das fases da computação no ParaView (a série
`nome_0000.vtk`, `nome_0001.vtk`, ... abre como animação).

```bash
mpirun -np 2 examples/cardiac_tissue/bin/cardiac_snapshots --nx 24 --ny 24 --nz 256 --time 300 --every 2 --output vtk/cardiac
mpirun -np 2 examples/his/bin/his_snapshots --x 32 --y 32 --z 256 --iterations 20000 --every 1000 --output vtk/his
```

| Programa | Campos (um valor por ponto) |
|---|---|
| `cardiac_snapshots` | `Vm` (mV), `Cai` (mM), `activation_time` (ms; −1 = ainda não ativou), `substeps_per_step` (subpassos do modelo celular por passo desde o snapshot anterior: o **custo** de cada fase), `celltype` (0 endo, 1 epi, 2 mid), `device`, `rank` |
| `his_snapshots` | as 8 populações (`A_antigen`, `MR_resting_macrophage`, `MA_activated_macrophage`, `N_neutrophil`, `CH_proinflammatory_cytokine`, `ND_apoptotic_neutrophil`, `G_protein_granules`, `CA_antiinflammatory_cytokine`), `device`, `rank` |

`device` é o dispositivo dono de cada ponto naquele instante: colorir por ele
mostra a divisão do domínio acompanhando a onda (cardíaco) quando o
balanceamento está ligado. Opções: tamanho, `--time`/`--iterations`,
`--every`, `--output`, `--precision`, `--device` e as de balanceamento
(`--help` lista todas). O espaçamento no arquivo é em mm (cardíaco: `dx`;
HIS: 0,1, o `deltaX` do kernel).

## Perfil de migração

```bash
mpirun -np 2 bin/migration_profiler profiling_results.txt
```

Mede o tempo de migrar volumes crescentes entre dispositivos vizinhos
(OpenCL + MPI) e grava os segmentos lineares usados pelos modos `profiled`.
Deve ser gerado com o mesmo número de ranks/nós da execução.

## Cluster (PBS)

Compile no nó de login (com os mesmos módulos usados nos jobs):

```bash
module load gnu13 openmpi5
```

```bash
./build.sh && ./build.sh examples
```

Os jobs ficam em `scripts/pbs/` e são submetidos da raiz do repositório. Os
processos MPI são iniciados com o `prun` do OpenHPC, que lança um rank por
linha do node file do PBS. Para ter **um rank por nó**, peça
`mpiprocs=1` em cada chunk:

```bash
qsub -l select=2:ncpus=16:mpiprocs=1 scripts/pbs/cardiac_benchmark.pbs
```

Escolhendo os nós:

```bash
qsub -l select=1:host=compute-3-3:ncpus=16:mpiprocs=1+1:host=compute-3-4:ncpus=16:mpiprocs=1 scripts/pbs/his_benchmark.pbs
```

Com a sintaxe antiga `-l nodes=N:ppn=16` o node file tem 16 linhas por nó e o
`prun` iniciaria 16 ranks por nó.

| Job | O que faz |
|---|---|
| `example_job.pbs` | perfil de migração + uma execução do HIS |
| `his_benchmark.pbs` | HIS com todas as políticas (10 execuções cada), média e desvio padrão em `results/np<N>/summary/` |
| `cardiac_benchmark.pbs` | tecido cardíaco sem balanceamento (referência) e com todas as políticas, para vários intervalos, em `results/cardiac_np<N>/summary/` |
| `paper_job.pbs` (via `submit_paper_tests.sh`) | testes do artigo (veja [Benchmarks para o artigo](#benchmarks-para-o-artigo)) |

Configuração comum (`scripts/pbs/job_env.sh`), alterável com `qsub -v`:

| Variável | Padrão | Descrição |
|---|---|---|
| `MODULES` | `gnu13 openmpi5` | módulos carregados no job (vazio = usar o ambiente atual) |
| `LAUNCHER` | `prun` | comando que inicia os ranks |
| `BIND` | `none` | binding de CPU dos ranks; vazio mantém o padrão do MPI |
| `HARMONY_ROOT` | diretório da submissão | onde estão `bin/`, `lib/` e `examples/` |

Por padrão o Open MPI prende cada rank a **um único núcleo**, e os
dispositivos OpenCL de CPU passam a ter 1 unidade de computação (aparece como
`CUs 1` na tabela de dispositivos). Por isso os jobs desligam o binding
(`BIND=none`). Antes das execuções, o log mostra os CPUs que cada rank pode
usar e a RAM de cada nó.

Exemplo com 3 máquinas escolhidas, cada uma com 1 GPU e 8 CPUs:

```bash
qsub -l select=1:host=compute-3-3:ncpus=8:ngpus=1:mpiprocs=1+1:host=compute-3-4:ncpus=8:ngpus=1:mpiprocs=1+1:host=compute-3-7:ncpus=8:ngpus=1:mpiprocs=1 scripts/pbs/cardiac_benchmark.pbs
```

No `cardiac_benchmark.pbs`, `BASELINES` (padrão `"gpu all"`) define as
execuções de referência sem balanceamento: só as GPUs, e todos os dispositivos
com fatias iguais. `DEVICE` define os dispositivos das execuções com
balanceamento. Sem balanceamento, um dispositivo de CPU recebe a mesma fatia
que uma GPU e todos esperam por ele a cada passo: esse caso pode ser dezenas de
vezes mais lento. Ajuste `NX`, `NY`, `NZ` e `TIME` ao walltime (as linhas de
progresso no log trazem a estimativa do tempo total).

Cada job também aceita os próprios parâmetros com `-v` (tamanho, tempo,
precisão, tolerâncias, intervalos, `WARMUPS`, `HALO_SYNC`, ...; veja o
cabeçalho de cada `.pbs`). Exemplo variando o warm-up:

```bash
qsub -v WARMUPS="0 1000 3000",INTERVALS="100 500" scripts/pbs/cardiac_benchmark.pbs
```
 O log de cada execução termina com o relatório de tempos.

## Testes

```bash
./examples/run_checks.sh
```

Compila os exemplos e roda as verificações: quickstart, multi_kernel e
cardiac_tissue contra a CPU, HIS pequeno com balanceamento (float e double),
stencil2d contra a versão MPI+OpenCL (float e double) e os dois programas de
snapshots VTK.
`RANKS=4` muda o número de ranks e `MPIRUN="mpirun --oversubscribe"` o comando.
Precisa de pelo menos um dispositivo OpenCL; sem GPU, o
[POCL](https://portablecl.org) fornece um dispositivo de CPU.

Observação: o POCL 5.0 tem um bug interno (`pocl_release_dlhandle_cache:
Assertion 'found->ref_count > 0'`) quando expõe mais de um dispositivo
(`POCL_DEVICES="cpu cpu"`). Para simular vários dispositivos por rank, use
também `POCL_KERNEL_CACHE=0`.
