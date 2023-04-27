import os.path
import torch
import random
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from vocsep.models.vocsep import HeteroVoiceLinkPredictionModel, VoiceLinkPredictionModel
from vocsep.data.datamodules.mix_vs import GraphMixVSDataModule
from pytorch_lightning.plugins import DDPPlugin
from vocsep.utils.visualization import show_voice_pr
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--collection', type=str, default="inventions",
                    choices=["inventions", "the_well-tempered_clavier_book_I", "the_well-tempered_clavier_book_II", "sinfonias", "haydn"],
                    help="Collections of pieces to use for the test set.")
parser.add_argument('--wandb_entity', type=str, default=None,
                    help="Your WandB user name. If not provided, it will be taken from the WANDB_ENTITY environment variable.")
parser.add_argument('--gpus', type=str, default="-1", help="GPU ids to use (you can use multiple by writting 0,1,2, etc). If -1, it will use CPU.")
parser.add_argument('--n_layers', type=int, default=2, help="Number of layers for the Graph Convolutional Network.")
parser.add_argument('--n_hidden', type=int, default=256, help="Number of hidden units for the Graph Convolutional Network.")
parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate of the Model.")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate of the Model.")
parser.add_argument('--weight_decay', type=float, default=5e-4, help="Weight decay of the Model.")
parser.add_argument("--pot_edges_max_dist", type=int, default=2, help="Maximum distance between nodes to consider them as potential edges in bars.")
parser.add_argument("--load_from_checkpoint", action="store_true", help="Load model from WANDB checkpoint")
parser.add_argument("--linear_assignment", action="store_true", help="Use linear assignment Hungarian algorithm for val and test predictions.")
parser.add_argument("--force_reload", action="store_true", help="Force reload of the data")
parser.add_argument("--model", type=str, default="ResConv", help="Block Convolution Model to use")
parser.add_argument("--reg_loss_weight", type=str, default="auto", help="Weight of the regularization loss. If 'auto', it augments every epoch end.", choices=["auto", "none", "fixed"])
parser.add_argument("--use_jk", action="store_true", help="Use Jumping Knowledge")
parser.add_argument("--tags", type=str, default="", help="Tags to add to the WandB run api")
parser.add_argument("--homogeneous", action="store_true", help="Use homogeneous graph convolution.")
parser.add_argument("--reg_loss_type", type=str, default="la", help="Use different regularization loss")



# for reproducibility
torch.manual_seed(0)
random.seed(0)
torch.use_deterministic_algorithms(True)


args = parser.parse_args()
if args.gpus == "-1":
    devices = None
else:
    devices = [eval(gpu) for gpu in args.gpus.split(",")]
rev_edges = "new_type"
collections = args.collection.split(",")
n_layers = args.n_layers
n_hidden = args.n_hidden
linear_assignment = args.linear_assignment
pot_edges_max_dist = args.pot_edges_max_dist
tags = args.tags.split(",")
force_reload = False
num_workers = 20


name = "{}GLAN-{}x{}-{}-lr={}-wd={}-dr={}-rl={}-jk={}".format(args.model,
    n_layers, n_hidden, "wLN" if args.linear_assignment else "woLN", args.lr,
    args.weight_decay, args.dropout, args.reg_loss_weight, args.use_jk)


wandb_logger = WandbLogger(
    log_model=True,
    entity=args.wandb_entity if args.wandb_entity is not None else os.getenv("WANDB_ENTITY"),
    project="Voice Separation",
    group=f"MixVS-{collections[0]}",
    job_type="Homogeneous-mkGLAN" if args.homogeneous else "Heterogeneous-mkGLAN",
    tags=tags,
    name=name)


datamodule = GraphMixVSDataModule(
    batch_size=1, num_workers=num_workers,
    force_reload=force_reload, test_collections=collections,
    pot_edges_max_dist=pot_edges_max_dist)
datamodule.setup()
if args.homogeneous:
    model = VoiceLinkPredictionModel(
        datamodule.features, n_hidden,
        n_layers=n_layers, dropout=args.dropout, lr=args.lr,
        weight_decay=args.weight_decay, reg_loss_weight=args.reg_loss_weight,
        jk=args.use_jk, model=args.model, linear_assignment=linear_assignment)
else:
    model = HeteroVoiceLinkPredictionModel(
        datamodule.features, n_hidden,
        n_layers=n_layers, lr=args.lr, dropout=args.dropout,
        weight_decay=args.weight_decay, linear_assignment=linear_assignment,
        model=args.model, jk=args.use_jk, reg_loss_weight=args.reg_loss_weight,
        reg_loss_type=args.reg_loss_type)

if args.load_from_checkpoint:
    # download checkpoint locally (if not already cached)
    import wandb
    run = wandb.init(project="Voice Separation", entity="vocsep", job_type="Heterogeneous-mkSAGE", group=f"MCMA-{collections[0]}", name=f"Sage-{n_layers}x{n_hidden}")
    artifact = run.use_artifact('vocsep/Voice Separation/model-1eii6k4w:v0', type='model')
    artifact_dir = artifact.download()

    print("Only monophonic:", model.linear_assignment)
    trainer = Trainer(
        max_epochs=50, accelerator="auto", devices=devices,
        num_sanity_val_steps=1,
        logger=wandb_logger,
    )
    trainer.test(model, datamodule, ckpt_path=os.path.join(os.path.normpath(artifact_dir), "model.ckpt"))

    prediction = trainer.predict(model, dataloaders=datamodule.test_dataloader())
    # Show target Voice
    pred_dict = prediction[0]
    show_voice_pr(
        pitches=pred_dict["note_array"][0], onsets=pred_dict["note_array"][3],
        durations=pred_dict["note_array"][4], voices=pred_dict["target_voices"])
    # Show predicted Voice
    show_voice_pr(
        pitches=pred_dict["note_array"][0], onsets=pred_dict["note_array"][3],
        durations=pred_dict["note_array"][4], voices=pred_dict["pred_voices"])
else:

    print("Only monophonic:", model.linear_assignment)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_fscore", mode="max")
    trainer = Trainer(
        max_epochs=50, accelerator="auto", devices=devices,
        num_sanity_val_steps=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        )

    #training
    trainer.fit(model, datamodule)

    # Testing with best model
    trainer.test(model, datamodule, ckpt_path=checkpoint_callback.best_model_path)
