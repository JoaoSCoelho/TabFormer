### 1. Configurações Gerais

* **`--jid` (Job ID):** Útil se você estiver rodando o código em um cluster de computadores ou gerenciador de tarefas (como SLURM). É apenas um identificador numérico para a sua execução atual.
* **`--seed`:** A "semente" randômica que vimos no `main.py`. Fixar a seed garante que a aleatoriedade (como a inicialização dos pesos do modelo) seja sempre a mesma, permitindo reproduzir o mesmo experimento depois. O padrão é 9.

### 2. Configurações do Modelo e Arquitetura

* **`--lm_type` (Language Model Type):** Define se a arquitetura base será bidirecional (`bert`) ou causal/autorregressiva (`gpt2`).
* **`--flatten`:** Um dos conceitos mais importantes do TabFormer. Se você passar essa flag, o modelo pegará a tabela (linhas e colunas) e vai "achatá-la" em uma única sequência linear (como se fosse uma frase de texto comum). Se não passar, o modelo tentará manter a estrutura hierárquica das colunas.
* **`--field_ce` (Field-wise Cross Entropy):** Relacionado ao `flatten`. Se não estiver achatado, o modelo calcula a função de perda (Cross Entropy) de forma separada para cada campo/coluna (field) da tabela, em vez de calcular para a "frase" inteira.
* **`--mlm` e `--mlm_prob` (Masked Language Modeling):** Exclusivo para o BERT. O `--mlm` avisa que o treinamento usará máscaras. O `--mlm_prob` (padrão 0.15) diz que 15% dos dados da tabela serão escondidos (mascarados) para que o modelo tente adivinhar quais eram os valores originais, aprendendo assim as relações nos dados.
* **`--field_hs` (Field Hidden Size):** O tamanho do vetor de embeddings (representação interna) para cada coluna. O padrão é 768, que é exatamente a dimensão padrão do BERT-base.

### 3. Configurações de Dados

* **`--data_type`:** Qual dataset usar. `card` para transações de cartão de crédito (focado em detecção de fraude) ou `prsa` para dados climáticos de Pequim.
* **`--data_root`, `--data_fname`, `--data_extension`:** Dizem ao código exatamente onde procurar a pasta dos dados, qual o nome do arquivo e a extensão.
* **`--vocab_file`:** O arquivo onde o vocabulário será salvo. Modelos de IA não leem strings, eles leem IDs. Esse arquivo guarda o "dicionário" que mapeia os valores das colunas para números inteiros.
* **`--cached`:** Uma flag de performance. Se passada, o código tenta ler uma versão já processada e salva dos dados. Isso economiza muito tempo em execuções repetidas.
* **`--nrows`:** Número máximo de linhas para carregar. É excelente para **debug**. Se você quer só testar se o código roda sem erros, você passa `--nrows 1000` para não ter que esperar carregar milhões de transações.
* **`--user_ids`:** Permite treinar ou testar o modelo apenas com o histórico de usuários específicos.
* **`--skip_user`:** Se ativada, a coluna que identifica o "usuário" na tabela é ignorada, forçando o modelo a focar apenas no comportamento da transação, e não no ID de quem a fez.
* **`--stride` (Passo da Janela Deslizante):** Dados de transações são vistos como séries temporais. Se o modelo analisa, digamos, 10 transações por vez, o `stride` (padrão 5) dita que a próxima sequência a ser lida vai pular 5 posições para frente.

### 4. Configurações de Treinamento

* **`--output_dir` e `--log_dir`:** As pastas onde o modelo treinado (checkpoints) e os gráficos/métricas de treinamento serão salvos.
* **`--do_train` e `--do_eval`:** Chaves de liga/desliga. Dizem ao script se ele deve fazer o treinamento e/ou a avaliação do modelo.
* **`--num_train_epochs`:** Quantas vezes o modelo vai processar o dataset inteiro durante o treinamento (padrão 3).
* **`--save_steps`:** A cada 500 lotes (padrão), o modelo salva um *checkpoint* (um backup do estado atual de aprendizado).
* **`--checkpoint`:** Se o seu computador desligar no meio do treino, você passa o número do passo aqui para o modelo voltar a treinar de onde parou, em vez de começar do zero.