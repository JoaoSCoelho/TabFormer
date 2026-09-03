from os import makedirs
from os.path import join
import logging
import numpy as np
import torch
import random
from args import define_main_parser

from transformers import Trainer, TrainingArguments

from dataset.prsa import PRSADataset
from dataset.card import TransactionDataset
from models.modules import TabFormerBertLM, TabFormerGPT2
from misc.utils import random_split_dataset
from dataset.contabil import ContabilDataset

logger = logging.getLogger(__name__)
log = logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def main(args):
    # random seeds
    seed = args.seed
    random.seed(seed)  # python 
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda

    if args.data_type == 'card':
        dataset = TransactionDataset(root=args.data_root,
                                     fname=args.data_fname,
                                     fextension=args.data_extension,
                                     vocab_dir=args.output_dir,
                                     nrows=args.nrows,
                                     user_ids=args.user_ids,
                                     mlm=args.mlm,
                                     cached=args.cached,
                                     stride=args.stride,
                                     flatten=args.flatten,
                                     return_labels=False,
                                     skip_user=args.skip_user)
    elif args.data_type == 'prsa':
        dataset = PRSADataset(stride=args.stride,
                              mlm=args.mlm,
                              return_labels=False,
                              use_station=False,
                              flatten=args.flatten,
                              vocab_dir=args.output_dir)
    elif args.data_type == 'contabil':
        dataset = ContabilDataset(
            root=args.data_root,
            fname=args.data_fname,
            vocab_dir=args.output_dir,
            nrows=args.nrows,
            book_account_ids=args.user_ids, # Se aproveitar o argumento de IDs de conta
            mlm=args.mlm,
            cached=args.cached,
            stride=args.stride,
            flatten=args.flatten,
            group_by=args.group_by,
            skip_user=args.skip_user
        )

    else:
        raise Exception(f"data type '{args.data_type}' not defined")

    vocab = dataset.vocab
    custom_special_tokens = vocab.get_special_tokens()

    # split dataset into train, val, test [0.6. 0.2, 0.2]
    totalN = len(dataset)
    trainN = int(0.6 * totalN)

    valtestN = totalN - trainN
    valN = int(valtestN * 0.5)
    testN = valtestN - valN

    assert totalN == trainN + valN + testN

    lengths = [trainN, valN, testN]

    log.info(f"# lengths: train [{trainN}]  valid [{valN}]  test [{testN}]")
    log.info("# lengths: train [{:.2f}]  valid [{:.2f}]  test [{:.2f}]".format(trainN / totalN, valN / totalN,
                                                                               testN / totalN))

    train_dataset, eval_dataset, test_dataset = random_split_dataset(dataset, lengths)

    if args.lm_type == "bert":
        tab_net = TabFormerBertLM(custom_special_tokens,
                               vocab=vocab,
                               field_ce=args.field_ce,
                               flatten=args.flatten,
                               ncols=dataset.ncols,
                               field_hidden_size=args.field_hs
                               )
    else:
        tab_net = TabFormerGPT2(custom_special_tokens,
                             vocab=vocab,
                             field_ce=args.field_ce,
                             flatten=args.flatten,
                             ncols=dataset.ncols,                 
                             field_hidden_size=args.field_hs
                             )

    log.info(f"model initiated: {tab_net.model.__class__}")

    if args.flatten:
        collactor_cls = "DataCollatorForLanguageModeling"
    else:
        collactor_cls = "TransDataCollatorForLanguageModeling"

    log.info(f"collactor class: {collactor_cls}")
    data_collator = eval(collactor_cls)(
        tokenizer=tab_net.tokenizer, mlm=args.mlm, mlm_probability=args.mlm_prob
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,  # output directory
        num_train_epochs=args.num_train_epochs,  # total number of training epochs
        save_steps=args.save_steps,
        do_train=args.do_train,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=tab_net.model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # MUDANÇA: Ajustado para o padrão do Hugging Face v4.x
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = join(args.output_dir, f'checkpoint-{args.checkpoint}')

    if args.do_train:
        # 'resume_from_checkpoint' substitui o antigo 'model_path'
        trainer.train(resume_from_checkpoint=checkpoint_path)
        trainer.save_model(args.output_dir)

    if args.do_eval:
        log.info("Iniciando avaliação no conjunto de teste...")
        # Carrega os pesos salvos se formos apenas avaliar (sem treinar antes)
        if not args.do_train:
            import os
            model_file = os.path.join(args.output_dir, "pytorch_model.bin")
            if os.path.exists(model_file):
                tab_net.model.load_state_dict(torch.load(model_file))
                log.info("Pesos do modelo carregados com sucesso para avaliação.")
            else:
                log.warning("Arquivo pytorch_model.bin não encontrado. Avaliando com pesos aleatórios!")
                
        metrics = trainer.evaluate(eval_dataset=test_dataset)
        log.info(f"Métricas da Avaliação: {metrics}")


if __name__ == "__main__":

    parser = define_main_parser()
    opts = parser.parse_args()

    opts.log_dir = join(opts.output_dir, "logs")
    makedirs(opts.output_dir, exist_ok=True)
    makedirs(opts.log_dir, exist_ok=True)

    if opts.mlm and opts.lm_type == "gpt2":
        raise Exception("Error: GPT2 doesn't need '--mlm' option. Please re-run with this flag removed.")

    if not opts.mlm and opts.lm_type == "bert":
        raise Exception("Error: Bert needs '--mlm' option. Please re-run with this flag included.")

    main(opts)