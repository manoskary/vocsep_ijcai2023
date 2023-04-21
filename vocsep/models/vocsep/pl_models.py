from .lightning_base import VocSepLightningModule, to_dense_adj
from .VoicePred import LinkPredictionModel, HeteroLinkPredictionModel
from torch.nn import functional as F
import torch
from vocsep.utils import add_reverse_edges_from_edge_index
from scipy.sparse.csgraph import connected_components




class VoiceLinkPredictionModel(VocSepLightningModule):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_layers,
        activation=F.relu,
        dropout=0.5,
        lr=0.001,
        weight_decay=5e-4,
        linear_assignment=True,
        model="ResConv",
        jk=True,
        reg_loss_weight="auto"
    ):
        super(VoiceLinkPredictionModel, self).__init__(
            in_feats,
            n_hidden,
            n_layers,
            activation,
            dropout,
            lr,
            weight_decay,
            LinkPredictionModel,
            linear_assignment=linear_assignment,
            model_name=model,
            jk=jk,
            reg_loss_weight=reg_loss_weight
        )

    def training_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pos_edges = pot_edges[:, batch_labels.bool()]
        neg_labels = torch.where(~batch_labels.bool())[0]
        neg_edges = pot_edges[
            :, neg_labels[torch.randperm(len(neg_labels))][: pos_edges.shape[1]]
        ]
        h = self.module.embed(batch_inputs, edges)
        pos_pitch_score = self.pitch_score(pos_edges, na[:, 0])
        pos_onset_score = self.onset_score(pos_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        neg_pitch_score = self.pitch_score(neg_edges, na[:, 0])
        neg_onset_score = self.onset_score(neg_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        pos_out = self.module.predict(h, pos_edges, pos_pitch_score, pos_onset_score)
        neg_out = self.module.predict(h, neg_edges, neg_pitch_score, neg_onset_score)
        reg_loss = self.reg_loss(
            pot_edges, self.module.predict(h, pot_edges, pitch_score, onset_score), pos_edges, len(batch_inputs))
        batch_pred = torch.cat((pos_out, neg_out), dim=0)
        loss = self.train_loss(pos_out, neg_out)
        batch_pred = torch.cat((1 - batch_pred, batch_pred), dim=1).squeeze()
        targets = (
            torch.cat(
                (torch.ones(pos_out.shape[0]), torch.zeros(neg_out.shape[0])), dim=0
            )
            .long()
            .to(self.device)
        )
        self.log("train_regloss", reg_loss.item(), on_step=True, on_epoch=True, prog_bar=False, batch_size=1)
        self.log("regloss_weight", self.reg_loss_weight, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log("regloss_weighted", self.reg_loss_weight*reg_loss.item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.train_metric_logging_step(loss, batch_pred, targets)
        loss = loss + self.reg_loss_weight * reg_loss
        self.log("train_joinloss", loss.item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        batch_pred = self.module(pot_edges, batch_inputs, edges, pitch_score, onset_score)
        self.val_metric_logging_step(
            batch_pred, pot_edges, truth_edges, len(batch_inputs)
        )

    def test_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        batch_pred = self.module(pot_edges, batch_inputs, edges, pitch_score, onset_score)
        self.test_metric_logging_step(
            batch_pred, pot_edges, truth_edges, len(batch_inputs)
        )

    def compute_linkpred_loss(self, pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        )
        w_coef = pos_score.shape[0] / neg_score.shape[0]
        weight = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.ones(neg_score.shape[0]) * w_coef]
        )
        return F.binary_cross_entropy(scores.squeeze(), labels, weight=weight)



class HeteroVoiceLinkPredictionModel(VocSepLightningModule):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_layers,
        activation=F.relu,
        dropout=0.5,
        lr=0.001,
        weight_decay=5e-4,
        linear_assignment=True,
        model="ResConv",
        jk=True,
        reg_loss_weight="auto",
        reg_loss_type="la",
        tau=0.5
    ):
        super(HeteroVoiceLinkPredictionModel, self).__init__(
            in_feats,
            n_hidden,
            n_layers,
            activation,
            dropout,
            lr,
            weight_decay,
            HeteroLinkPredictionModel,
            linear_assignment=linear_assignment,
            model_name=model,
            jk=jk,
            reg_loss_weight=reg_loss_weight,
            reg_loss_type=reg_loss_type,
            tau=tau
        )

    def training_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pos_edges = pot_edges[:, batch_labels.bool()]
        neg_labels = torch.where(~batch_labels.bool())[0]
        neg_edges = pot_edges[
            :, neg_labels[torch.randperm(len(neg_labels))][: pos_edges.shape[1]]
        ]
        h = self.module.embed(batch_inputs, edges, edge_types)
        pos_pitch_score = self.pitch_score(pos_edges, na[:, 0])
        pos_onset_score = self.onset_score(pos_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        neg_pitch_score = self.pitch_score(neg_edges, na[:, 0])
        neg_onset_score = self.onset_score(neg_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        pos_out = self.module.predict(h, pos_edges, pos_pitch_score, pos_onset_score)
        neg_out = self.module.predict(h, neg_edges, neg_pitch_score, neg_onset_score)
        reg_loss = self.reg_loss(
            pot_edges, self.module.predict(h, pot_edges, pitch_score, onset_score), pos_edges, len(batch_inputs))
        batch_pred = torch.cat((pos_out, neg_out), dim=0)
        loss = self.train_loss(pos_out, neg_out)
        batch_pred = torch.cat((1 - batch_pred, batch_pred), dim=1).squeeze()
        targets = (
            torch.cat(
                (torch.ones(pos_out.shape[0]), torch.zeros(neg_out.shape[0])), dim=0
            )
            .long()
            .to(self.device)
        )
        self.log("train_regloss", reg_loss.item(), on_step=True, on_epoch=True, prog_bar=False, batch_size=1)
        self.log("regloss_weight", self.reg_loss_weight, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log("regloss_weighted", self.reg_loss_weight*reg_loss.item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.train_metric_logging_step(loss, batch_pred, targets)
        loss = loss + self.reg_loss_weight * reg_loss
        self.log("train_joinloss", loss.item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        batch_pred = self.module(pot_edges, batch_inputs, edges, edge_types, pitch_score, onset_score)
        self.val_metric_logging_step(
            batch_pred, pot_edges, truth_edges, len(batch_inputs)
        )

    def test_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        batch_pred = self.module(pot_edges, batch_inputs, edges, edge_types, pitch_score, onset_score)
        self.test_metric_logging_step(
            batch_pred, pot_edges, truth_edges, len(batch_inputs)
        )

    def predict_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        batch_pred = self.module(pot_edges, batch_inputs, edges, edge_types, pitch_score, onset_score)
        adj_pred, fscore = self.predict_metric_step(
            batch_pred, pot_edges, truth_edges, len(batch_inputs)
        )
        print(f"Piece {name} F-score: {fscore}")
        nov_pred, voices_pred = connected_components(csgraph=adj_pred, directed=False, return_labels=True)
        adj_target = to_dense_adj(truth_edges, max_num_nodes=len(batch_inputs)).squeeze().long().cpu()
        nov_target, voices_target = connected_components(csgraph=adj_target, directed=False, return_labels=True)
        return (
            name,
            voices_pred,
            voices_target,
            nov_pred,
            nov_target,
            na[:, 1],
            na[:, 2],
            na[:, 0],
        )

    def compute_linkpred_loss(self, pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        )
        w_coef = pos_score.shape[0] / neg_score.shape[0]
        weight = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.ones(neg_score.shape[0]) * w_coef]
        )
        return F.binary_cross_entropy(scores.squeeze(), labels, weight=weight)

