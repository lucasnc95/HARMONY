# Trabalho futuro: validação do HARMONY em GPU com Colab

**Atualizado em:** 28 de setembro de 2026

**Estado de referência do código:** `037fd1b`

**Objetivo:** executar e verificar C++/MPI/OpenCL em GPU real, guardar evidências reproduzíveis e consultar o resultado remotamente sem precisar manter o notebook aberto.

## 1. O que já sabemos

- Na auditoria local do HARMONY, cinco suítes oficiais passaram (24/24 testes), mas as rotas principais usavam dispositivos/tempos simulados. A máquina de auditoria não tinha plataforma OpenCL. Os defeitos e limites estão no [relatório técnico](/home/agent/HARMONY/AUDITORIA_TECNICA_2026-09-28.md).
- O usuário executou `nvidia-smi` e `clinfo -l` em uma sessão **Colab Pro**. O resultado foi **Tesla T4**, driver **580.82.07**, plataforma **NVIDIA CUDA** e dispositivo **Tesla T4**. Portanto, **essa sessão específica oferece um dispositivo OpenCL de GPU**. Ainda não houve execução de um kernel do HARMONY nela.
- Os avisos do `apt` sobre `trusted.gpg` legado e o repositório `r2u` não impediram a enumeração OpenCL; são avisos de gerenciamento de pacotes, não falhas do HARMONY.
- O código roda na VM remota; a conexão do navegador ao notebook não participa de cada transferência CPU–GPU. Medidas PCIe, NUMA, energia e MPI nessa VM, porém, descrevem aquela VM e não substituem medições do cluster de destino. [Colab: onde o código é executado](https://research.google.com/colaboratory/faq.html).
- O [executável de residual de alta ordem](/home/agent/HARMONY/main_high_order_residual.cpp:147) cria campos, compila o kernel `high_order_residual_halo.cl`, executa passos com halo e pode reunir o resultado. Ele **não compara** a saída com uma referência numérica. O comando de compilação abaixo foi confirmado na máquina de auditoria; sua execução na T4 ainda está pendente.
- Com **um rank e um dispositivo**, o HARMONY cria uma partição e a [troca de halo retorna sem agir](/home/agent/HARMONY/dcl/runtime_impl.hpp:2065). Com **dois ranks na mesma VM**, ambos podem ver a mesma T4 física e criar duas partições lógicas; isso permite testar o caminho MPI/OpenCL entre ranks, sem reproduzir duas GPUs físicas ou dois nós.
- Os [benchmarks atuais](/home/agent/HARMONY/benchmarks/kneighbor.cpp:179) calculam em vetores CPU e não chamam `create_field`, `create_kernel` ou `execute`. Seus números não devem ser usados como prova de desempenho GPU ou de migração MPI.

**Interpretação:** a T4 remove o bloqueio de disponibilidade de OpenCL para um teste funcional. A correção dos kernels, dos halos e do gerenciamento de recursos continua por demonstrar por meio de resultados numéricos e execuções repetidas.

## 2. Escolha do ambiente e nível de automação

| Caminho | O que permite | Limite relevante |
|---|---|---|
| **Colab Pro já contratado** | Piloto barato em esforço: abrir uma sessão GPU, executar um notebook que prepara o ambiente e grava resultados em armazenamento externo. | Não foi encontrada API oficial documentada para provisionar e iniciar, de fora, uma sessão clássica do Colab Pro de ponta a ponta. GPU e duração variam. A abertura/partida inicial permanece manual. |
| **Colab Enterprise** | Criar um modelo de runtime com GPU, lançar ou agendar notebooks pela CLI/API, provisionar a VM da execução e salvar resultados no Cloud Storage; acompanhar sem abrir o Colab. | Exige projeto Google Cloud, faturamento, quota de GPU, permissões IAM e custos próprios. A assinatura Pro não equivale a esses recursos. |
| **VM GPU dedicada no Compute Engine** | Automatizar por script de inicialização, executar binários sem notebook, enviar logs/resultados ao Cloud Storage; ambiente mais controlável. | É um fluxo Google Cloud, não uma sessão Colab Pro; também exige faturamento e administração da VM. |

**Decisão recomendada:** começar pelo Colab Pro para obter a primeira prova em T4. Se a exigência passar a ser **criar a instância, executar e consultar tudo sem abrir notebook**, migrar o mesmo teste para **Colab Enterprise**. Para desempenho PCIe/NUMA e vários nós, usar VMs/cluster com topologia conhecida. A [FAQ do Colab](https://research.google.com/colaboratory/faq.html) documenta a variabilidade dos recursos Pro; o [Colab Enterprise](https://docs.cloud.google.com/colab/docs/schedule-notebook-run) documenta execução agendada e resultados em Cloud Storage.

## 3. Artefatos a acrescentar ao repositório antes da primeira rodada

Criar estes arquivos versionados em um commit específico, sem misturar resultados brutos com o código-fonte:

1. `tests/test_gpu_kernel_halo.cpp`: teste de integração que usa **dispositivo OpenCL real**, inicializa dados com padrão que distingue cada índice global, executa kernel, chama `gather` e compara com uma referência CPU. Deve imprimir erro máximo absoluto/relativo, quantidade de divergências e índices divergentes, em especial nas bordas de partição. Deve sair com código diferente de zero ao falhar.
2. `colab/bootstrap.sh`: preparação idempotente de C++20, Open MPI, cabeçalhos/biblioteca OpenCL e `clinfo`; registra versões, verifica `clinfo -l` e falha se não houver dispositivo GPU OpenCL. Não instalar uma versão arbitrária do driver NVIDIA por cima da fornecida pelo runtime.
3. `colab/run_suite.sh`: compila o teste de integração e o exemplo, executa casos com timeout, preserva cada código de saída e gera um sumário estruturado. Usar `set -euo pipefail`; falhas não podem ser anunciadas como sucesso, conforme o defeito encontrado no job PBS.
4. `colab/run_harmony.ipynb`: notebook pequeno que obtém o código em um **commit fixo**, chama `bootstrap.sh`/`run_suite.sh` e envia artefatos ao armazenamento. O notebook é um invólucro de execução; a lógica de teste fica em scripts/C++ versionados para também funcionar fora do Colab.
5. `colab/result_schema.md` ou esquema JSON: contrato de `manifest.json`, `status.json`, resultados numéricos, logs e versões. Cada rodada recebe um `run_id` único e o SHA completo do commit.

Estrutura sugerida para cada rodada:

```text
runs/<commit_sha>/<run_id>/
  manifest.json             # SHA, parâmetros, GPU, driver, OpenCL, MPI, compilador
  status.json               # queued/running/pass/fail/timeout e etapa atual
  environment.txt           # nvidia-smi, clinfo, versões, CPU/memória
  build.log                 # stdout/stderr da compilação
  tests/kernel_np1.json     # erro numérico contra CPU
  tests/halo_np2.json       # erro perto das fronteiras, comparação np1/np2
  tests/async_repeats.json  # duração, timeout, divergências por repetição
  raw/*.log                 # saída integral de cada processo
  summary.md               # leitura humana e limites do ensaio
```

O notebook deve publicar `status.json` ao começar e ao mudar de etapa. Ao terminar, publicar `pass`, `fail` ou `timeout`; se a VM desaparecer, um `running` antigo denuncia interrupção. Não salvar tokens ou chaves nos logs/notebooks. Preferir identidade da plataforma e permissão de escrita apenas no prefixo/bucket de resultados.

## 4. Roteiro funcional no Colab Pro atual

### Passo 1 — Fixar a origem do código

No Colab Pro, abrir o notebook uma vez, selecionar um runtime **GPU** e executar a célula orquestradora. Essa etapa inicial é manual; depois, os resultados podem ser consultados fora do Colab. Levar o repositório ao runtime e fazer checkout de um SHA exato (inicialmente `037fd1b`, ou o commit posterior com correções e teste). Se o remoto for privado, usar um mecanismo autenticado da conta ou um arquivo versionado no Drive/Cloud Storage; não escrever token no notebook ou em uma URL de Git. Executar a partir da raiz do repositório, pois [o carregamento do kernel](/home/agent/HARMONY/dcl/runtime_impl.hpp:534) usa o caminho relativo `high_order_residual_halo.cl`.

### Passo 2 — Preparar compilação e registrar o ambiente

Em uma célula de shell, verificar/instalar `g++`, `openmpi-bin`, `libopenmpi-dev`, `ocl-icd-opencl-dev`, `opencl-headers` e `clinfo`. Esses pacotes existem no Ubuntu 24.04; confirmar a imagem atribuída antes de fixar versões. Exemplo idempotente para `bootstrap.sh`:

```bash
set -euo pipefail
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  build-essential openmpi-bin libopenmpi-dev \
  ocl-icd-opencl-dev opencl-headers clinfo
```

Registrar, no mínimo:

```bash
git rev-parse HEAD
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
clinfo -l
mpic++ --version
mpirun --version
```

O `clinfo -l` deve mostrar **NVIDIA CUDA / Tesla T4** ou outra GPU OpenCL real; plataforma apenas CPU não satisfaz o teste. O suporte da NVIDIA a OpenCL não implica que toda sessão futura do Colab venha configurada igual à sessão já observada. [Khronos: enumeração de dispositivos](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clGetDeviceIDs.html).

### Passo 3 — Fazer um ensaio de compilação e execução

O exemplo existente pode ser compilado da raiz do repositório com:

```bash
mpic++ -std=c++20 -O0 -DCL_TARGET_OPENCL_VERSION=300 \
  main_high_order_residual.cpp dcl/runtime.cpp -lOpenCL \
  -o high_order_probe
```

Começar com malha pequena, `--balance-mode off`, `--repeat 1` e `--iterations 2`. Executar primeiro com `-np 1`, depois com `-np 2` e timeout (por exemplo, 60 s):

```bash
timeout 60s mpirun --allow-run-as-root --oversubscribe -np 1 \
  ./high_order_probe --x 16 --y 16 --z 32 \
  --iterations 2 --repeat 1 --balance-mode off --gather-final 1
timeout 60s mpirun --allow-run-as-root --oversubscribe -np 2 \
  ./high_order_probe --x 16 --y 16 --z 32 \
  --iterations 2 --repeat 1 --balance-mode off --gather-final 1
```

O exemplo assume que o usuário do runtime é `root`; omitir `--allow-run-as-root` se não for. `--oversubscribe` só é necessário quando faltam slots MPI. O programa atual reúne o resultado, mas não o valida. **Sair com código zero aqui é somente um teste de fumaça.** [Open MPI: `mpirun` e oversubscription](https://docs.open-mpi.org/en/main/man-openmpi/man1/mpirun.1.html).

### Passo 4 — Rodar o teste numérico novo

Executar `test_gpu_kernel_halo` em quatro modos, sempre com o mesmo campo inicial e semente fixa, se houver aleatoriedade:

1. **`np=1`, sem halo entre partições:** compila o kernel na T4, executa e compara cada elemento com a referência CPU dentro de tolerância documentada.
2. **`np=2`, balanceamento desligado:** cada rank usa a T4 compartilhada. Comparar o vetor reunido com a referência CPU e com a execução `np=1`; verificar explicitamente as `halo_width` posições de cada lado da fronteira.
3. **Síncrono versus assíncrono:** repetir com `synchronize_at_end=true` e `false`, múltiplas iterações e timeout. Esta etapa investiga o risco de progresso entre filas OpenCL descrito no relatório; uma passagem não prova ausência de deadlock.
4. **Tamanhos de borda:** testar partições com tamanho maior, igual e menor que o halo. O caso menor corresponde ao risco herdado H3 e pode precisar de correção antes de passar.

Se possível, executar repetições independentes para captar intermitência, preservando o primeiro erro/timeout integral. Evitar usar os benchmarks atuais como critério de GPU: eles não executam seus arquivos `.cl`.

### Passo 5 — Avaliar recursos e rebalanceamento

Repetir criação/destruição de `Runtime`, redescoberta e falhas controladas de alocação para investigar o vazamento H1. `nvidia-smi` ajuda a observar consumo de VRAM, mas não conta com precisão handles OpenCL; manter instrumentação de criação/liberação para a conclusão estrita.

Só depois de isolar a execução de kernel/halo, habilitar políticas de rebalanceamento. Antes de interpretar resultados com dados reais, corrigir as condições C1/C2 e as divergências de partições/métricas A2/A3 do [relatório](/home/agent/HARMONY/AUDITORIA_TECNICA_2026-09-28.md). Em cada passo, exigir coletivamente cobertura exata de `global_elements`, ausência de sobreposições e concordância dos ranks sobre partições.

### Passo 6 — Publicar e consultar o resultado sem abrir o notebook

Depois de iniciar a sessão Pro manualmente, o notebook pode enviar `status.json`, logs e `summary.md` ao Google Drive ou, preferencialmente para automação, a um bucket Cloud Storage. A autenticação e autorização do destino precisam ser configuradas na sessão; não presumir que montar Drive ou acessar Cloud Storage ocorrerá sem interação inicial. De outra máquina autenticada, consultar com:

```bash
gcloud storage ls gs://BUCKET/harmony/runs/ --recursive
gcloud storage cat gs://BUCKET/harmony/runs/COMMIT/RUN_ID/status.json
gcloud storage cp gs://BUCKET/harmony/runs/COMMIT/RUN_ID/summary.md ./summary.md
```

Essa consulta **não exige abrir o Colab**. Entretanto, no Pro clássico a criação/partida da sessão ainda é manual, e a duração do runtime não é garantida. [Cloud Storage: listar](https://docs.cloud.google.com/storage/docs/listing-objects), [ler](https://docs.cloud.google.com/storage/docs/reading-objects) e [copiar](https://docs.cloud.google.com/sdk/gcloud/reference/storage/cp) objetos.

## 5. Roteiro para execução totalmente remota com Colab Enterprise

Esta rota é um projeto separado do plano Pro. Validar [preços](https://cloud.google.com/colab/pricing), quota de GPU e permissão para ativar faturamento antes da implantação. A documentação oficial permite criar templates de runtime por CLI/API, executar um notebook uma vez ou em agenda e recuperar resultados no Cloud Storage.

1. **Criar projeto e armazenamento.** Escolher `PROJECT_ID`, `REGION`, bucket privado para notebook/artefatos e bucket/prefixo para resultados. Ativar faturamento e as APIs de Agent Platform, Dataform e Compute Engine exigidas pelo [guia oficial](https://docs.cloud.google.com/colab/docs/schedule-notebook-run). Com a CLI autenticada e o projeto selecionado, a ativação pode ser feita com `gcloud services enable aiplatform.googleapis.com dataform.googleapis.com compute.googleapis.com`. Confirmar quota para uma T4 na região escolhida e estabelecer orçamento/alertas de cobrança.
2. **Definir identidade.** Criar uma conta de serviço de execução com acesso de leitura ao notebook/código e escrita apenas nos resultados. Conceder ao operador as permissões para criar template/execução e usar a conta de serviço. Manter bucket privado; não colocar chaves de serviço no repositório.
3. **Versionar o pacote de execução.** Publicar `run_harmony.ipynb` e, conforme o acesso ao repositório, um snapshot imutável do código ou referência a um commit fixo. Por exemplo: `gcloud storage cp colab/run_harmony.ipynb gs://BUCKET/notebooks/run_harmony.ipynb`. O notebook deve falhar se o SHA resolvido diferir do esperado.
4. **Automatizar pré-requisitos.** Criar um template GPU com T4 e tipo de máquina compatível. Para uma execução agendada com conta de serviço, o notebook deve primeiro obter o código fixado, depois chamar `bootstrap.sh` como etapa inicial e verificar OpenCL antes de executar testes. Confirmar na primeira rodada que a imagem permite instalar os pacotes exigidos. O Colab Enterprise também suporta [post-startup scripts](https://docs.cloud.google.com/colab/docs/post-startup-script), mas um script **privado em `gs://` exige end-user credentials** e tem restrições de proprietário. Se optar por bootstrap no template com conta de serviço, usar uma origem HTTPS acessível e sem segredos ou validar previamente a combinação de autenticação suportada; não supor que `gs://` privado e `--no-enable-euc` funcionem juntos.
5. **Criar o template.** Com `gcloud` autenticado, usar o comando documentado como base (ajustar região e máquina à quota disponível):

   ```bash
   gcloud colab runtime-templates create \
     --display-name=harmony-t4 --runtime-template-id=harmony-t4 \
     --project=PROJECT_ID --region=REGION \
     --machine-type=n1-standard-4 \
     --accelerator-type=NVIDIA_TESLA_T4 --accelerator-count=1 \
     --no-enable-euc
   ```

   A [referência da CLI](https://docs.cloud.google.com/sdk/gcloud/reference/colab/runtime-templates/create) lista T4 e os parâmetros de acelerador. O template descreve a VM; a execução do notebook cria o runtime correspondente.
6. **Lançar uma rodada sem abrir o Colab.** Após publicar o notebook no bucket e ajustar a identidade, disparar:

   ```bash
   gcloud colab executions create \
     --display-name=harmony-COMMIT-RUN_ID \
     --project=PROJECT_ID --region=REGION \
     --notebook-runtime-template=harmony-t4 \
     --gcs-notebook-uri=gs://BUCKET/notebooks/run_harmony.ipynb \
     --gcs-output-uri=gs://BUCKET/harmony/executions/ \
     --service-account=SERVICE_ACCOUNT_EMAIL \
     --execution-timeout=2h
   ```

   Os parâmetros de [execução única](https://docs.cloud.google.com/colab/docs/schedule-notebook-run) e da [CLI](https://docs.cloud.google.com/sdk/gcloud/reference/colab/executions/create) são documentados oficialmente. A conta de serviço e o notebook devem ter acesso real aos objetos necessários; validar com uma rodada curta antes de agendar.
7. **Acompanhar remotamente.** Consultar `gcloud colab executions list --project=PROJECT_ID --region=REGION` e `gcloud colab executions describe EXECUTION_ID --project=PROJECT_ID --region=REGION`; ler `status.json` e `summary.md` no bucket como no passo 4. O resultado oficial da execução é salvo em IPYNB no Cloud Storage; os JSON/logs próprios do HARMONY dão uma interface mais simples para alertas e comparação entre rodadas. [Google: resultados de execução](https://docs.cloud.google.com/colab/docs/schedule-notebook-run#view_results).
8. **Agendar somente após a rodada única passar.** Usar `gcloud colab schedules create` com cron, o mesmo template, notebook, conta de serviço e bucket; limitar concorrência a 1 para evitar duas rodadas competindo pela GPU. Exemplo semanal em UTC, ajustável:

   ```bash
   gcloud colab schedules create \
     --display-name=harmony-weekly \
     --cron-schedule='TZ=Etc/UTC 0 3 * * MON' \
     --execution-display-name=harmony-regression \
     --project=PROJECT_ID --region=REGION \
     --notebook-runtime-template=harmony-t4 \
     --gcs-notebook-uri=gs://BUCKET/notebooks/run_harmony.ipynb \
     --gcs-output-uri=gs://BUCKET/harmony/executions/ \
     --service-account=SERVICE_ACCOUNT_EMAIL \
     --max-concurrent-runs=1
   ```

   [Guia de agendamento](https://docs.cloud.google.com/colab/docs/schedule-notebook-run) e [referência da CLI](https://docs.cloud.google.com/sdk/gcloud/reference/colab/schedules/create).
9. **Controlar custo e retenção.** Configurar timeout por execução, alertas de orçamento, política de retenção dos resultados e revisão periódica de execuções agendadas. A cobrança do Enterprise inclui runtime/GPU e discos; runtimes parados podem continuar cobrando armazenamento. [Preços](https://cloud.google.com/colab/pricing) e [modelo de runtimes](https://docs.cloud.google.com/colab/docs/runtimes).

O Colab Enterprise fornece o mecanismo para **provisionar e observar sem abrir o notebook**; os arquivos `bootstrap.sh`, teste numérico e notebook executor ainda precisam ser implementados e validados no projeto.

## 6. Critérios de conclusão e limites remanescentes

Uma rodada só deve ser marcada `pass` quando **todos** os itens seguintes forem verdadeiros: (a) GPU OpenCL real enumerada; (b) compilação do host e do kernel concluída; (c) comparação CPU/GPU dentro da tolerância definida; (d) comparação `np=1`/`np=2` sem divergência nas fronteiras; (e) execução assíncrona repetida termina sem timeout; (f) logs, versões, SHA e resultados foram enviados ao armazenamento; (g) todos os processos MPI retornaram código zero. Falhas de upload também tornam a rodada inconclusiva.

Mesmo após isso, **não** estarão validados: halo entre GPUs físicas distintas, MPI entre nós, topologia PCIe/NUMA de cluster, teto energético real, escalabilidade e ausência universal de deadlock ou race. Essas questões exigem um ambiente controlado com pelo menos duas GPUs físicas e, para comunicação de rede, dois nós. O Colab com uma T4 é uma excelente etapa funcional intermediária, não a etapa final de desempenho HPC.
