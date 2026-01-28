import numpy as np
import torch
from typing import List
from nltk.tree import Tree

from better_mistakes.trees import get_label
from better_mistakes.data.softmax_cascade import SoftmaxCascade


class HierarchicalLLLoss(torch.nn.Module):

    def __init__(self, hierarchy: Tree, classes: List[str], weights: Tree):
        super(HierarchicalLLLoss, self).__init__()

        assert hierarchy.treepositions() == weights.treepositions()

        positions_leaves = {get_label(hierarchy[p]): p for p in hierarchy.treepositions("leaves")}
        num_classes = len(positions_leaves)
        positions_leaves = [positions_leaves[c] for c in classes]
        positions_edges = hierarchy.treepositions()[1:]

        index_map_leaves = {positions_leaves[i]: i for i in range(len(positions_leaves))}
        index_map_edges = {positions_edges[i]: i for i in range(len(positions_edges))}

        edges_from_leaf = [[index_map_edges[position[:i]] for i in range(len(position), 0, -1)] for position in positions_leaves]

        num_edges = max([len(p) for p in edges_from_leaf])

        def get_leaf_positions(position):
            node = hierarchy[position]
            if isinstance(node, Tree):
                return node.treepositions("leaves")
            else:
                return [()]

        leaf_indices = [[index_map_leaves[position + leaf] for leaf in get_leaf_positions(position)] for position in positions_edges]

        self.onehot_den = torch.nn.Parameter(torch.zeros([num_classes, num_classes, num_edges]), requires_grad=False)
        self.onehot_num = torch.nn.Parameter(torch.zeros([num_classes, num_classes, num_edges]), requires_grad=False)
        self.weights = torch.nn.Parameter(torch.zeros([num_classes, num_edges]), requires_grad=False)

        for i in range(num_classes):
            for j, k in enumerate(edges_from_leaf[i]):
                self.onehot_num[i, leaf_indices[k], j] = 1.0
                self.weights[i, j] = get_label(weights[positions_edges[k]])
            for j, k in enumerate(edges_from_leaf[i][1:]):
                self.onehot_den[i, leaf_indices[k], j] = 1.0
            self.onehot_den[i, :, j + 1] = 1.0

    def forward(self, inputs, target):
        inputs = torch.unsqueeze(inputs, 1)
        num = torch.squeeze(torch.bmm(inputs, self.onehot_num[target]))
        den = torch.squeeze(torch.bmm(inputs, self.onehot_den[target]))
        idx = num != 0
        num[idx] = -torch.log(num[idx] / den[idx])
        num = torch.sum(torch.flip(self.weights[target] * num, dims=[1]), dim=1)
        return torch.mean(num)


class HierarchicalCrossEntropyLoss(HierarchicalLLLoss):

    def __init__(self, hierarchy: Tree, classes: List[str], weights: Tree):
        super(HierarchicalCrossEntropyLoss, self).__init__(hierarchy, classes, weights)

    def forward(self, inputs, index):
        return super(HierarchicalCrossEntropyLoss, self).forward(torch.nn.functional.softmax(inputs, 1), index)


class CosineLoss(torch.nn.Module):

    def __init__(self, embedding_layer):
        super(CosineLoss, self).__init__()
        self._embeddings = embedding_layer

    def forward(self, inputs, target):
        inputs = torch.nn.functional.normalize(inputs, p=2, dim=1)
        emb_target = self._embeddings(target)
        return 1 - torch.nn.functional.cosine_similarity(inputs, emb_target).mean()


class CosinePlusXentLoss(torch.nn.Module):

    def __init__(self, embedding_layer, xent_weight=0.1):
        super(CosinePlusXentLoss, self).__init__()
        self._embeddings = embedding_layer
        self.xent_weight = xent_weight

    def forward(self, inputs, target):
        inputs_cosine = torch.nn.functional.normalize(inputs, p=2, dim=1)
        emb_target = self._embeddings(target)
        loss_cosine = 1 - torch.nn.functional.cosine_similarity(inputs_cosine, emb_target).mean()
        loss_xent = torch.nn.functional.cross_entropy(inputs, target).mean()
        return loss_cosine + self.xent_weight * loss_xent


class RankingLoss(torch.nn.Module):

    def __init__(self, embedding_layer, batch_size, single_random_negative, margin=0.1):
        super(RankingLoss, self).__init__()
        self._embeddings = embedding_layer
        self._vocab_len = embedding_layer.weight.size()[0]
        self._margin = margin
        self._single_random_negative = single_random_negative
        self._batch_size = batch_size

    def forward(self, inputs, target):
        inputs = torch.nn.functional.normalize(inputs, p=2, dim=1)
        dot_product = torch.mm(inputs, self._embeddings.weight.t())
        true_embeddings = self._embeddings(target)
        negate_item = torch.sum(inputs * true_embeddings, dim=1, keepdim=True)
        if self._single_random_negative:
            dot_product_pruned = torch.zeros_like(negate_item)
            mask_margin_violating = self._margin + dot_product - negate_item > 0
            for i in range(self._batch_size):
                mask = mask_margin_violating[i, :] != 0
                num_valid = torch.sum(mask).item()
                margin_violating_samples_i = (mask_margin_violating[i, :] != 0).nonzero().squeeze()
                if num_valid > 1:
                    rnd_id = np.random.choice(num_valid, 1)
                    rnd_val = dot_product[i, margin_violating_samples_i[rnd_id]]
                else:
                    rnd_val = dot_product[i, margin_violating_samples_i]
                dot_product_pruned[i, 0] = rnd_val

            dot_product = dot_product_pruned

        full_rank_mat = self._margin + dot_product - negate_item
        relu_mat = torch.nn.ReLU()(full_rank_mat)
        summed_mat = torch.sum(relu_mat, dim=1)
        return summed_mat.mean()


class YOLOLoss(torch.nn.Module):

    def __init__(self, hierarchy: Tree, classes: List[str], weights: Tree):
        super(YOLOLoss, self).__init__()

        assert hierarchy.treepositions() == weights.treepositions()

        self.cascade = SoftmaxCascade(hierarchy, classes)

        weights_dict = {get_label(hierarchy[p]): get_label(weights[p]) for p in weights.treepositions()}
        self.weights = torch.nn.Parameter(torch.unsqueeze(torch.tensor([weights_dict[c] for c in classes], dtype=torch.float32), 0), requires_grad=False)

    def forward(self, inputs, target):
        return self.cascade.cross_entropy(inputs, target, self.weights)


class SupervisedContrastiveHierarchicalLoss(torch.nn.Module):
    def __init__(self, hierarchy: Tree, classes: List[str], weights: Tree, alpha=0.5):
        super(SupervisedContrastiveHierarchicalLoss, self).__init__()
        self.hierarchy = hierarchy
        self.classes = classes
        self.alpha = alpha
        self.ce = torch.nn.CrossEntropyLoss()
        pass

    def forward(self, inputs, target):
        logits_l1, logits_l2, logits_l3, features = inputs
        
        if isinstance(target, tuple) or isinstance(target, list):
            target_l1, target_l2, target_l3 = target
        else:
            target_l3 = target
            target_l1 = None
            target_l2 = None

        loss = 0
        if target_l3 is not None:
             mask_l3 = target_l3 != -1
             if mask_l3.sum() > 0:
                 loss += self.ce(logits_l3[mask_l3], target_l3[mask_l3])
        
        if target_l2 is not None:
            mask = target_l2 != -1
            if mask.sum() > 0:
                loss += self.ce(logits_l2[mask], target_l2[mask])

        if target_l1 is not None:
             loss += self.ce(logits_l1, target_l1)

        loss_reg = torch.mean(torch.norm(features, dim=1))
        
        return loss + self.alpha * loss_reg
