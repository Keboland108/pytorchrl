"""Pretrain a GPT model."""
import os
import wandb
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import OpenAIGPTConfig, OpenAIGPTModel

import pytorchrl as prl
from pytorchrl.agent.env import VecEnv
from pytorchrl.utils import LoadFromFile, save_argparse, cleanup_log_dir
from pytorchrl.agent.actors import OnPolicyActor, get_feature_extractor, get_memory_network
from pytorchrl.envs.generative_chemistry.vocabulary import SMILESTokenizer, create_vocabulary
from pytorchrl.envs.generative_chemistry.generative_chemistry_env_factory import generative_chemistry_train_env_factory
from code_examples.train_genchem.ppo.train import get_args
from code_examples.train_genchem.ppo.pretrain_rnn_model import \
    is_valid_smile, filter_mol, read_and_filter_data, write_smiles_to_file, MolData, decrease_learning_rate


# testing
from pytorchrl.agent.actors.feature_extractors.gpt import GPT

# TODO: how to integrate huggingface models?
# TODO: create an env (or modify current) so each obs is the concatenation of all previous ones


if __name__ == "__main__":

    args = get_args()
    os.makedirs(args.log_dir, exist_ok=True)
    save_argparse(args, os.path.join(args.log_dir, "conf.yaml"), [])
    pretrained_ckpt = {}
    os.makedirs("data", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(f"{args.log_dir}/mols_filtered.smi"):
        if not os.path.exists(args.pretrainingset_path):
            raise ValueError(f"Missing training set: {args.pretrainingset_path}")
        print("\nReading smiles...")
        smiles_list = read_and_filter_data(
            args.pretrainingset_path,
            args.pretrain_max_heavy_atoms,
            args.pretrain_min_heavy_atoms,
            args.pretrain_element_list,
        )
        print("\nSaving filtered training data...")
        write_smiles_to_file(smiles_list, f"{args.log_dir}/mols_filtered.smi")
    else:
        fname = f"{args.log_dir}/mols_filtered.smi"
        smiles_list = []
        with open(fname, 'r') as f:
            for line in f:
                smiles_list.append(line.split()[0])

    if not os.path.exists(f"{args.log_dir}/pretrained_ckpt.prior"):
        print("\nConstructing vocabulary...")
        tokenizer = SMILESTokenizer()
        vocabulary = create_vocabulary(smiles_list, tokenizer=tokenizer)
        pretrained_ckpt["tokenizer"] = tokenizer
        pretrained_ckpt["vocabulary"] = vocabulary
        pretrained_ckpt["max_sequence_length"] = args.pretrain_max_smile_length
    else:
        pretrained_ckpt_dict = torch.load(f"{args.log_dir}/pretrained_ckpt.prior")
        tokenizer = pretrained_ckpt["tokenizer"] = pretrained_ckpt_dict["tokenizer"]
        vocabulary = pretrained_ckpt["vocabulary"] = pretrained_ckpt_dict["vocabulary"]
        pretrained_ckpt["max_sequence_length"] = args.pretrain_max_smile_length

    # Handle wandb init
    if args.wandb_key:
        mode = "online"
        wandb.login(key=str(args.wandb_key))
    else:
        mode = "disabled"

    with wandb.init(project=args.experiment_name, name=args.agent_name + "_pretrain", config=args, mode=mode):

        # Define Dataloader
        moldata = MolData(f"{args.log_dir}/mols_filtered.smi", vocabulary, tokenizer)
        data = DataLoader(
            moldata, batch_size=args.pretrain_batch_size, shuffle=True, drop_last=True, collate_fn=MolData.collate_fn)

        # Define env
        test_env, action_space, obs_space = VecEnv.create_factory(
            env_fn=generative_chemistry_train_env_factory,
            env_kwargs={
                "scoring_function": lambda a: {"reward": 1.0},
                "tokenizer": tokenizer, "vocabulary": vocabulary,
                "concatenate_obs": True, "smiles_max_length": args.pretrain_max_smile_length},
            vec_env_size=1)
        env = test_env(device)

        # Define model
        # Get original config
        model_config = OpenAIGPTConfig()
        # {
        #     "afn": "gelu",
        #     "attn_pdrop": 0.1,
        #     "embd_pdrop": 0.1,
        #     "initializer_range": 0.02,
        #     "layer_norm_epsilon": 1e-05,
        #     "model_type": "openai-gpt",
        #     "n_embd": 768,
        #     "n_head": 12,
        #     "n_layer": 12,
        #     "n_positions": 512,
        #     "predict_special_tokens": true,
        #     "resid_pdrop": 0.1,
        #     "summary_activation": null,
        #     "summary_first_dropout": 0.1,
        #     "summary_proj_to_labels": true,
        #     "summary_type": "cls_index",
        #     "summary_use_proj": true,
        #     "transformers_version": "4.21.2",
        #     "vocab_size": 40478
        # }

        # Adjust model size
        model_config.n_embd = 256
        model_config.n_head = 6
        model_config.n_layer = 2
        model_config.n_positions = 256
        model_config.vocab_size = len(vocabulary)

        feature_extractor_kwargs = {"transformers_config": model_config}
        pretrained_ckpt["feature_extractor_kwargs"] = feature_extractor_kwargs
        actor = OnPolicyActor.create_factory(
            obs_space, action_space, prl.PPO,
            feature_extractor_network=GPT,
            feature_extractor_kwargs={**feature_extractor_kwargs},
            recurrent_net=None,
            recurrent_net_kwargs=None)(device)

        # Define optimizer
        optimizer = torch.optim.Adam(actor.parameters(), lr=args.pretrain_lr)

        print("\nStarting pretraining...")
        for epoch in range(1, args.pretrain_epochs):

            with tqdm(enumerate(data), total=len(data)) as tepoch:

                tepoch.set_description(f"Epoch {epoch}")

                for step, batch in tepoch:

                    # Train mdoe
                    actor.train()

                    # Sample from DataLoader seqs = (batch_size, seq_length)
                    seqs = batch.long().to(device)

                    # Predict next token log likelihood. TODO: Ugly hack, abstract this forward pass
                    features = actor.policy_net.feature_extractor.feature_extractor(
                        input_ids=seqs[:, :-1].long(), attention_mask=torch.ones_like(seqs[:, :-1]).long()
                    ).last_hidden_state
                    logp_action, entropy_dist, dist = actor.policy_net.dist.evaluate_pred(features, seqs[:, 1:])

                    # Optimization step
                    loss = - logp_action.squeeze(-1).sum(1).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    info_dict = {}
                    total_steps = step + len(data) * (epoch - 1)
                    if (total_steps % args.pretrain_lr_decrease_period) == 0 and total_steps != 0:

                        # Eval mode
                        actor.eval()

                        # Decrease learning rate
                        decrease_learning_rate(optimizer, decrease_by=args.pretrain_lr_decrease_value)

                        # Generate a few molecules and check how many are valid
                        total_molecules = 100
                        valid_molecules = 0
                        list_molecules = []
                        list_tokens = []
                        list_entropy = []
                        for i in range(total_molecules):
                            next_obs, rhs, done = actor.actor_initial_states(env.reset())
                            molecule_length = 0
                            while not done:
                                obs = next_obs
                                with torch.no_grad():
                                    _, action, _, rhs, entropy_dist, dist = actor.get_action(
                                        obs, rhs=None, done=None, deterministic=False)
                                molecule_length += 1
                                next_obs, _, done, _ = env.step(action)
                            molecule = tokenizer.untokenize(vocabulary.decode(obs.cpu().numpy().squeeze(0)))
                            tokens = vocabulary.encode(tokenizer.tokenize(molecule))
                            if is_valid_smile(molecule):
                                valid_molecules += 1
                            list_molecules.append(molecule)
                            list_tokens.append(tokens)
                            list_entropy.append(entropy_dist.item())

                        # Check how many are repeated
                        ratio_repeated = len(set(list_molecules)) / len(list_molecules) if total_molecules > 0 else 0

                        # Add to info dict
                        info_dict.update({
                            "pretrain_avg_molecular_length": np.mean([len(s) for s in list_tokens]),
                            "pretrain_avg_entropy": np.mean(list_entropy),
                            "pretrain_valid_molecules": valid_molecules / total_molecules,
                            "pretrain_ratio_repeated": ratio_repeated
                        })

                        # Save model
                        pretrained_ckpt["network_weights"] = actor.state_dict()
                        torch.save(pretrained_ckpt, f"{args.log_dir}/pretrained_ckpt.prior")

                    tepoch.set_postfix(loss=loss.item())

                    # Wandb logging
                    info_dict.update({"pretrain_loss": loss.item()})
                    wandb.log(info_dict, step=total_steps)

    print("Finished!")

    # Pretrain model manually by using features = actor.policy_net.feature_extractor.feature_extractor(seqs[:, :-1]) - ok

    # Then make the forward pass of gpt so it only returns the last "non-masked" token - ok

    # Make the env accept again a single token as action - ok

    # Might need to do a different forward pass for data collection and for gradient compute, like for rnn's!

    # Then for training, obs will be fixed sized length 200 tensors, but actions will still be a single token

    # For evaluate actions, forward pass will returns a single prob distribution, and will work