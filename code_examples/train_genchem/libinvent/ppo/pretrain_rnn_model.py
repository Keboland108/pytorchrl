"""
Pretrain a GRU or LSTM model.
Code Adapted from https://github.com/MolecularAI/Lib-INVENT to work on PyTorchRL.
Requires preprocessed data as explained in https://github.com/MolecularAI/Lib-INVENT-dataset.
"""

import os
import copy
import glob
import wandb
import argparse
import numpy as np
from tqdm import tqdm
from rdkit import Chem
import itertools as it
import torch
import torch.nn.utils.rnn as tnnur
from torch.utils.data import Dataset, DataLoader
from reinvent_chemistry.file_reader import FileReader
from reinvent_chemistry.library_design import BondMaker, AttachmentPoints
from reinvent_chemistry import Conversions

import pytorchrl as prl
from pytorchrl.agent.env import VecEnv
from pytorchrl.utils import LoadFromFile, save_argparse, cleanup_log_dir
from pytorchrl.envs.generative_chemistry.vocabulary import LibinventVocabulary
from pytorchrl.agent.actors import OnPolicyActor, get_feature_extractor, get_memory_network
from pytorchrl.envs.generative_chemistry.libinvent.generative_chemistry_env_factory import libinvent_train_env_factory
from code_examples.train_genchem.libinvent.ppo.train_rnn_model import get_args


def decrease_learning_rate(optimizer, decrease_by=0.01):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)


def load_dataset(path):
    reader = FileReader([], None)
    return list(reader.read_library_design_data_file(path, num_fields=2))


class DecoratorDataset(Dataset):
    """Dataset that takes a list of (scaffold, decoration) pairs."""

    def __init__(self, scaffold_decoration_smi_list, vocabulary):
        self.vocabulary = vocabulary

        self._encoded_list = []
        for scaffold, dec in tqdm(scaffold_decoration_smi_list):
            en_scaff = self.vocabulary.scaffold_vocabulary.encode(self.vocabulary.scaffold_tokenizer.tokenize(scaffold))
            en_dec = self.vocabulary.decoration_vocabulary.encode(self.vocabulary.decoration_tokenizer.tokenize(dec))
            if en_scaff is not None and en_dec is not None:
                self._encoded_list.append((en_scaff, en_dec))

    def __getitem__(self, i):
        scaff, dec = self._encoded_list[i]
        return torch.tensor(scaff, dtype=torch.long), torch.tensor(dec, dtype=torch.long)

    def __len__(self):
        return len(self._encoded_list)

    @classmethod
    def collate_fn(cls, encoded_pairs):
        """
        Turns a list of encoded pairs (scaffold, decoration) of sequences and turns them into two batches.
        :param: A list of pairs of encoded sequences.
        :return: A tuple with two tensors, one for the scaffolds and one for the decorations in the same order as given.
        """
        encoded_scaffolds, encoded_decorations = list(zip(*encoded_pairs))
        return pad_batch(encoded_scaffolds), pad_batch(encoded_decorations)


def pad_batch(encoded_seqs):
    """
    Pads a batch.
    :param encoded_seqs: A list of encoded sequences.
    :return: A tensor with the sequences correctly padded.
    """
    seq_lengths = torch.tensor([len(seq) for seq in encoded_seqs], dtype=torch.int64)
    return tnnur.pad_sequence(encoded_seqs, batch_first=True), seq_lengths


if __name__ == "__main__":

    args = get_args()
    os.makedirs(args.log_dir, exist_ok=True)
    save_argparse(args, os.path.join(args.log_dir, "conf.yaml"), [])
    pretrained_ckpt = {}
    os.makedirs("data", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define classes required to merge scaffolds and decorations
    bond_maker = BondMaker()
    conversion = Conversions()
    attachment_points = AttachmentPoints()

    # Load training set
    print("\nLoading data...")
    if not os.path.exists(args.pretrainingset_path):
        raise ValueError(f"Missing training set: {args.pretrainingset_path}")
    training_set = load_dataset(args.pretrainingset_path)
    testing_scaffolds = ["[*]"]
    if args.pretestingset_path and os.path.exists(args.pretestingset_path):
        testing_set = load_dataset(args.pretestingset_path)
        testing_scaffolds = [i[0] for i in testing_set]

    # Create or load vocabularies
    if not os.path.exists(f"{args.log_dir}/pretrained_ckpt.prior"):
        print("\nConstructing vocabularies...")
        scaffold_list = [attachment_points.remove_attachment_point_numbers(i[0]) for i in training_set]
        decoration_list = [i[1] for i in training_set]
        vocabulary = LibinventVocabulary.from_lists(scaffold_list, decoration_list)  # Can take a long time!
        pretrained_ckpt["vocabulary"] = vocabulary
        pretrained_ckpt["max_sequence_length"] = args.pretrain_max_smile_length
        torch.save(pretrained_ckpt, f"{args.log_dir}/pretrained_ckpt.prior")
    else:
        print(f"\nCheckpoint {args.log_dir}/pretrained_ckpt.prior found. Loading...")
        pretrained_ckpt = torch.load(f"{args.log_dir}/pretrained_ckpt.prior")
        vocabulary = pretrained_ckpt["vocabulary"]
        pretrained_ckpt["max_sequence_length"] = args.pretrain_max_smile_length

    # Handle wandb init
    if args.wandb_key:
        mode = "online"
        wandb.login(key=str(args.wandb_key))
    else:
        mode = "disabled"

    with wandb.init(project=args.experiment_name, name=args.agent_name + "_pretrain", config=args, mode=mode):

        print("\nPreparing dataset and dataloader...")
        dataset = DecoratorDataset(training_set, vocabulary=vocabulary)  # Takes a long time!
        data = DataLoader(
            dataset,
            batch_size=args.pretrain_batch_size,
            shuffle=True, drop_last=True,
            collate_fn=dataset.collate_fn)

        # Define env
        test_env, action_space, obs_space = VecEnv.create_factory(
            env_fn=libinvent_train_env_factory,
            env_kwargs={
                "scoring_function": lambda a: {"reward": 1.0},
                "vocabulary": vocabulary, "smiles_max_length": args.pretrain_max_smile_length,
                "scaffolds": testing_scaffolds
            },
            vec_env_size=1)
        env = test_env(device)

        # Define model
        feature_extractor_kwargs = {}
        recurrent_net_kwargs = {
            "encoder_params": {"num_layers": 3, "num_dimensions": 512, "vocabulary_size": len(vocabulary.scaffold_vocabulary), "dropout": 0.1},
            "decoder_params": {"num_layers": 3, "num_dimensions": 512, "vocabulary_size": len(vocabulary.decoration_vocabulary), "dropout": 0.1}
        }
        actor = OnPolicyActor.create_factory(
            obs_space, action_space, prl.PPO,
            feature_extractor_network=torch.nn.Identity,
            feature_extractor_kwargs=feature_extractor_kwargs,
            recurrent_net=get_memory_network(args.recurrent_net),
            recurrent_net_kwargs={**recurrent_net_kwargs})(device)
        pretrained_ckpt["feature_extractor_kwargs"] = feature_extractor_kwargs
        pretrained_ckpt["recurrent_net_kwargs"] = recurrent_net_kwargs

        # Define optimizer
        optimizer = torch.optim.Adam(actor.parameters(), lr=args.pretrain_lr)

        print("\nStarting pretraining...")
        for epoch in range(1, args.pretrain_epochs):

            with tqdm(enumerate(data), total=len(data)) as tepoch:

                tepoch.set_description(f"Epoch {epoch}")

                for step, batch in tepoch:

                    # Separate batch data
                    ((scaffold_batch, scaffold_lengths), (decorator_batch, decorator_length)) = batch

                    # Prediction
                    encoded_seqs, encoder_rhs = actor.policy_net.memory_net._forward_encoder(
                        scaffold_batch.to(device), scaffold_lengths)
                    features, _, _ = actor.policy_net.memory_net._forward_decoder(
                        decorator_batch.to(device), decorator_length, encoded_seqs, encoder_rhs)
                    logp_action, entropy_dist, dist = actor.policy_net.dist.evaluate_pred(
                        features[:, :-1], decorator_batch.to(device)[:, 1:])

                    # Optimization step
                    loss = - logp_action.squeeze(-1).sum(-1).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    info_dict = {}
                    total_steps = step + len(data) * (epoch - 1)
                    if (total_steps % args.pretrain_lr_decrease_period) == 0 and total_steps != 0:

                        # Decrease learning rate
                        decrease_learning_rate(optimizer, decrease_by=args.pretrain_lr_decrease_value)

                        if args.pretestingset_path and os.path.exists(args.pretestingset_path):

                            # Generate a few molecules and check how many are valid
                            total_molecules = 100
                            valid_molecules = 0
                            list_decorations = []
                            list_num_tokens = []
                            list_entropy = []
                            for i in range(total_molecules):
                                num_tokens = 0
                                decoration = "^"
                                obs, rhs, done = actor.actor_initial_states(env.reset())
                                scaffold = vocabulary.decode_scaffold(obs["context"].cpu().numpy().squeeze(0))
                                with torch.no_grad():
                                    encoded_seqs, rhs = actor.policy_net.memory_net._forward_encoder(
                                        obs["context"].to(device), obs["context_length"].cpu().long())
                                    while not done:
                                        features, rhs, _ = actor.policy_net.memory_net._forward_decoder(
                                            obs["obs"].to(device), obs["obs_length"].cpu().long(), encoded_seqs, rhs)
                                        action, _, logp, entropy_dist, dist = actor.policy_net.dist(features.squeeze(0))
                                        obs, _, done, _ = env.step(action)
                                        decoration += vocabulary.decode_decoration_token(action)
                                        list_entropy.append(entropy_dist.item())
                                        num_tokens += 1

                                decoration = vocabulary.remove_start_and_end_tokens(decoration)
                                scaffold = attachment_points.add_attachment_point_numbers(scaffold, canonicalize=False)
                                molecule = bond_maker.join_scaffolds_and_decorations(scaffold, decoration)
                                smile = conversion.mol_to_smiles(molecule) if molecule else None
                                if smile:
                                    valid_molecules += 1
                                list_decorations.append(decoration)
                                list_num_tokens.append(num_tokens)

                            # Check how many are repeated
                            ratio_repeated = len(set(list_decorations)) / len(
                                list_decorations) if total_molecules > 0 else 0

                            # Add to info dict
                            info_dict.update({
                                "pretrain_avg_molecular_length": np.mean(list_num_tokens),
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
