import torch, pdb, copy
import torch.nn as nn
import numpy as np
import itertools as it

class MultiLabel_BCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, y):
        loss = self.loss(x, y)
        return loss


class MultiLabel_BCE_mask(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, x, y, mask):
       loss = 0
       for i in range(x.size(1)):
           p_i = x[:, i]
           g_i = y[:, i]
           mask_i = mask[:, i]
           loss_i = self.bce(p_i, g_i) * mask_i
           loss += loss_i.mean()
       loss = loss / x.size(1)
       return loss


class pLoss_all_v3_joint(nn.Module):
    def __init__(self, hexG):
        super(pLoss_all_v3_joint, self).__init__()
        self.legal_state = hexG[0]
        self.label_state_idx = hexG[1]

    def forward(self, f, y, mask, auto_mode=False):
        pMargin = torch.zeros((f.shape[0], f.shape[1])).to(f.device)
        pJoint = torch.zeros((f.shape[0], 1)).to(f.device)

        S = self.legal_state.to(f.device)
        # f: bzxn S: (n+1)xn
        potential = torch.mm(S.double(), f.T.double())
        max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        J = torch.exp(potential - max_sf)  # (n+1)xbz
        z_ = torch.sum(J, dim=0)

        id_num = _check_abnornal(z_)

        for i in range(f.shape[1]):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        loss = []
        # optimize the joint probability over the observed labels
        for j in range(y.shape[0]): # y: [B,n]
            if j in id_num:
                continue

            string = ''.join(str(int(x)) for x in y[j])
            if string == '120110111101' or string == '120010111111': # special case on the unobserved labels, and processing it by marginalization
                loss_j = nn.BCELoss(reduction='none')(pMargin[j], y[j]) * mask[j]
                loss.append(loss_j.mean())

            else:
                # pJoint[j] = J[self.label_state_idx[str(int(y[j]))], j] / z_[j]
                try:
                    pJoint[j] = J[self.label_state_idx[string], j] / z_[j]
                except:
                    pdb.set_trace()
                loss_j = - torch.log(pJoint[j] + 1e-6)
                loss.append(loss_j)

        loss = sum(loss) / y.shape[0]

        if torch.isnan(loss) == 1:
            pdb.set_trace()

        if loss == np.inf:
            pdb.set_trace()

        return loss, pMargin

    def infer(self, f):
        pMargin = torch.zeros((f.shape[0], f.shape[1])).to(f.device)
        S = self.legal_state.to(f.device)
        potential = torch.mm(S.double(), f.T.double())

        max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        J = torch.exp(potential - max_sf)  # (n+1)xbz
        z_ = torch.sum(J, dim=0)

        # for Z_ is inf, ignore its pMargin loss
        id_num = _check_abnornal(z_)

        for i in range(f.shape[1]):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        return pMargin

def compute_one_level_loss(loss):
    total_loss = 0
    for loss_item in loss:
        total_loss += loss_item
    total_loss = total_loss / len(loss)
    return total_loss


def binary_cross_entropy_loss(predictions, labels):
    epsilon = 1e-7
    predictions = torch.clamp(predictions, epsilon, 1 - epsilon)
    # 计算正类别的损失
    pos_loss = -labels * torch.log(predictions)

    # 计算负类别的损失
    neg_loss = -(1 - labels) * torch.log(1 - predictions)

    # 计算平均损失
    loss_mean = torch.mean(pos_loss + neg_loss)
    loss = pos_loss + neg_loss

    return loss


class pLoss_all_v3(nn.Module):
    def __init__(self, hexG):
        super(pLoss_all_v3, self).__init__()
        self.legal_state = hexG[0]
        self.label_state_idx = hexG[1]

    def forward(self, f, y, auto_mode=False):
        y = y.float()
        pMargin = torch.zeros((f.shape[0], f.shape[1])).to(f.device)
        pJoint = torch.zeros((f.shape[0], 1)).to(f.device)

        S = self.legal_state.to(f.device)
        # f: bzxn S: (n+1)xn
        potential = torch.mm(S.double(), f.T.double())
        max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        J = torch.exp(potential - max_sf)  # (n+1)xbz
        # J = potential
        z_ = torch.sum(J, dim=0)

        id_num = _check_abnornal(z_)

        for i in range(f.shape[1]):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        # pMargin: [B, n], y: [B,n]
        loss = []
        # pdb.set_trace()
        for j in range(y.size(1)):
            pred_i = pMargin[:, j]
            target_i = y[:, j]
            # mask_i = mask[:, j]
            # loss_i = nn.BCELoss(reduction='none')(pred_i, target_i) #* mask_i
            # pdb.set_trace()
            loss_i = binary_cross_entropy_loss(pred_i, target_i)
            # pdb.set_trace()
            loss.append(loss_i.mean())
        # pdb.set_trace()
        if auto_mode:
            return loss, pMargin
        else:
            return sum(loss), pMargin

    def infer(self, f):
        pMargin = torch.zeros((f.shape[0], f.shape[1])).to(f.device)
        S = self.legal_state.to(f.device)
        potential = torch.mm(S.double(), f.T.double())

        max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        J = torch.exp(potential - max_sf)  # (n+1)xbz
        z_ = torch.sum(J, dim=0)

        # for Z_ is inf, ignore its pMargin loss
        id_num = _check_abnornal(z_)

        for i in range(f.shape[1]):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        return pMargin

    def forward_(self, f, y, mask=1, auto_mode=False):
        # pMargin = torch.zeros((f.shape[0], f.shape[1])).to(f.device)
        pMargin = torch.zeros((y.shape[0], y.shape[1])).to(f.device)
        pJoint = torch.zeros((f.shape[0], 1)).to(f.device)

        S = self.legal_state.to(f.device)
        # f: bzxn S: (n+1)xn
        # potential = torch.mm(S.double(), f.T.double())
        # max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        # J = torch.exp(potential - max_sf)  # (n+1)xbz
        J = f.T
        z_ = torch.sum(J, dim=0)
        # pdb.set_trace()

        id_num = _check_abnornal(z_)

        for i in range(y.shape[1]):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        # pMargin: [B, n], y: [B,n]
        # pdb.set_trace()
        loss = []
        for j in range(y.size(1)):
            pred_i = pMargin[:, j]
            target_i = y[:, j]
            # mask_i = mask[:, j]
            loss_i = nn.BCELoss(reduction='none')(pred_i, target_i) #* mask_i
            loss.append(loss_i.mean())
        if auto_mode:
            return loss, pMargin
        else:
            return sum(loss), pMargin

    def infer_(self, f):
        pMargin = torch.zeros((f.shape[0], 12)).to(f.device)
        S = self.legal_state.to(f.device)
        # potential = torch.mm(S.double(), f.T.double())
        #
        # max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        # J = torch.exp(potential - max_sf)  # (n+1)xbz
        J = f.T
        z_ = torch.sum(J, dim=0)

        # for Z_ is inf, ignore its pMargin loss
        id_num = _check_abnornal(z_)

        for i in range(12):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        return pMargin


from Loss import cal_loss_multilabel_fideltiy
class pLoss_all_fidelity(nn.Module):
    def __init__(self, hexG, dataset='ffsc'):
        super(pLoss_all_fidelity, self).__init__()
        self.legal_state = hexG[0]
        self.label_state_idx = hexG[1]
        self.fideltiy_loss = cal_loss_multilabel_fideltiy(dataset=dataset)
        self.dataset = dataset

    def transfer_logtis(self, dataset, pMargin):
        if dataset in ['ffsc']:
            logits_l1 = pMargin[:, 0]
            logits_l2 = pMargin[:, 1:6]
            logits_l3 = pMargin[:, 6:]
        else: # for ff++
            logits_l1 = pMargin[:, 0]
            logits_l2 = pMargin[:, 1:4]
            logits_l3 = pMargin[:, 4:]
        return logits_l1, logits_l2, logits_l3

    def forward(self, f, y, auto_mode=False):
        y = y.float()
        pMargin = torch.zeros((f.shape[0], f.shape[1])).to(f.device)
        pJoint = torch.zeros((f.shape[0], 1)).to(f.device)

        S = self.legal_state.to(f.device)
        # f: bzxn S: (n+1)xn
        potential = torch.mm(S.double(), f.T.double())
        max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        J = torch.exp(potential - max_sf)  # (n+1)xbz
        # J = potential
        z_ = torch.sum(J, dim=0)

        id_num = _check_abnornal(z_)

        for i in range(f.shape[1]):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        # 相当于在这个地方针对每个attribute node来获得其对应的fidelity loss
        # pdb.set_trace()
        logits_l1, logits_l2, logits_l3 = self.transfer_logtis(self.dataset, pMargin)
        total_loss, all_loss = self.fideltiy_loss(logits_l1, logits_l2, logits_l3, y)
        return all_loss, pMargin

        # # pMargin: [B, n], y: [B,n]
        # loss = []
        # # pdb.set_trace()
        # for j in range(y.size(1)):
        #     pred_i = pMargin[:, j]
        #     target_i = y[:, j]
        #     # mask_i = mask[:, j]
        #     # loss_i = nn.BCELoss(reduction='none')(pred_i, target_i) #* mask_i
        #     # pdb.set_trace()
        #     loss_i = binary_cross_entropy_loss(pred_i, target_i)
        #     # pdb.set_trace()
        #     loss.append(loss_i.mean())
        # # pdb.set_trace()
        # if auto_mode:
        #     return loss, pMargin
        # else:
        #     return sum(loss), pMargin

    def infer(self, f):
        pMargin = torch.zeros((f.shape[0], f.shape[1])).to(f.device)
        S = self.legal_state.to(f.device)
        potential = torch.mm(S.double(), f.T.double())

        max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        J = torch.exp(potential - max_sf)  # (n+1)xbz
        z_ = torch.sum(J, dim=0)

        # for Z_ is inf, ignore its pMargin loss
        id_num = _check_abnornal(z_)

        for i in range(f.shape[1]):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        return pMargin

    def forward_(self, f, y, mask=1, auto_mode=False):
        # pMargin = torch.zeros((f.shape[0], f.shape[1])).to(f.device)
        pMargin = torch.zeros((y.shape[0], y.shape[1])).to(f.device)
        pJoint = torch.zeros((f.shape[0], 1)).to(f.device)

        S = self.legal_state.to(f.device)
        # f: bzxn S: (n+1)xn
        # potential = torch.mm(S.double(), f.T.double())
        # max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        # J = torch.exp(potential - max_sf)  # (n+1)xbz
        J = f.T
        z_ = torch.sum(J, dim=0)
        # pdb.set_trace()

        id_num = _check_abnornal(z_)

        for i in range(y.shape[1]):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        # pMargin: [B, n], y: [B,n]
        # pdb.set_trace()
        loss = []
        for j in range(y.size(1)):
            pred_i = pMargin[:, j]
            target_i = y[:, j]
            # mask_i = mask[:, j]
            loss_i = nn.BCELoss(reduction='none')(pred_i, target_i) #* mask_i
            loss.append(loss_i.mean())
        if auto_mode:
            return loss, pMargin
        else:
            return sum(loss), pMargin

    def infer_(self, f):
        pMargin = torch.zeros((f.shape[0], 12)).to(f.device)
        S = self.legal_state.to(f.device)
        # potential = torch.mm(S.double(), f.T.double())
        #
        # max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        # J = torch.exp(potential - max_sf)  # (n+1)xbz
        J = f.T
        z_ = torch.sum(J, dim=0)

        # for Z_ is inf, ignore its pMargin loss
        id_num = _check_abnornal(z_)

        for i in range(12):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        return pMargin


class pLoss_all(nn.Module):
    def __init__(self, hexG, mode='joint'):
        super().__init__()
        self.legal_state = hexG[0]
        self.label_state_idx = hexG[1]
        self.mode = mode
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, f, y, y_name, mask):
        pMargin = torch.zeros((f.shape[0], f.shape[1])).to(f.device)
        pJoint = torch.zeros((f.shape[0], 1)).to(f.device)

        S = self.legal_state.to(f.device)
        # f: bzxn S: (n+1)xn
        potential = torch.mm(S.double(), f.T.double())
        max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        J = torch.exp(potential - max_sf)  # (n+1)xbz
        z_ = torch.sum(J, dim=0)

        id_num = _check_abnornal(z_)

        for i in range(f.shape[1]):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        loss = torch.zeros(y.shape[0], dtype=torch.float64).to(f.device)
        for j in range(y.shape[0]):
            if j in id_num:
                continue

            if self.mode == 'joint':
                if '2' in y_name[j]:
                    # multi_label_pMargin_loss = torch.log(pMargin[j] + 1e-8) * mask[j]
                    # loss[j] = -torch.mean(multi_label_pMargin_loss)
                    loss[j] = self.compute_loss_margin(pMargin[j], y[j].float(), mask[j].float())
                else:
                    y_idx = y_name[j]
                    pJoint[j] = J[self.label_state_idx[y_idx], j] / z_[j]
                    loss[j] = - torch.log(pJoint[j] + 1e-8)

            elif self.mode == 'margin':
                loss[j] = self.compute_loss_margin(pMargin[j], y[j].float(), mask[j].float())

                # multi_label_pMargin_loss = torch.log(pMargin[j]+1e-8) * mask[j]
                # loss[j] = -torch.mean(multi_label_pMargin_loss)
                # multi_label_pMargin = torch.index_select(pMargin[j], 0, torch.where(y[j]>0)[0])
                # loss[j] = - torch.log(multi_label_pMargin + 1e-6).sum()

        loss = torch.mean(loss)

        _check_loss(loss)

        return loss, pMargin

    def compute_loss_margin(self, pMargin, y, mask):
        loss_margin = self.bce(pMargin, y) * mask
        return loss_margin.mean()


    def infer(self, f):
        pMargin = torch.zeros((f.shape[0], f.shape[1])).to(f.device)
        S = self.legal_state.to(f.device)
        potential = torch.mm(S.double(), f.T.double())

        max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        J = torch.exp(potential - max_sf)  # (n+1)xbz
        z_ = torch.sum(J, dim=0)

        # for Z_ is inf, ignore its pMargin loss
        id_num = _check_abnornal(z_)

        for i in range(f.shape[1]):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        return pMargin

class pLoss_all_v2(nn.Module):
    def __init__(self, hexG, mode='joint'):
        super().__init__()
        self.legal_state = hexG[0]
        self.label_state_idx = hexG[1]
        self.mode = mode

    def forward(self, f, y, y_name, mask):
        pMargin = torch.zeros((f.shape[0], f.shape[1])).to(f.device)
        pJoint = torch.zeros((f.shape[0], 1)).to(f.device)

        S = self.legal_state.to(f.device)
        # f: bzxn S: (n+1)xn
        potential = torch.mm(S.double(), f.T.double())
        max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        J = torch.exp(potential - max_sf)  # (n+1)xbz
        z_ = torch.sum(J, dim=0)

        id_num = _check_abnornal(z_)

        for i in range(f.shape[1]):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        loss = torch.zeros(y.shape[0], dtype=torch.float64).to(f.device)
        for j in range(y.shape[0]):
            if j in id_num:
                continue

            if self.mode == 'joint':
                if '2' in y_name[j]:
                    multi_label_pMargin = torch.index_select(pMargin[j], 0, torch.where(y[j] == 1)[0])
                    loss[j] = - torch.log(multi_label_pMargin + 1e-8).sum()
                else:
                    y_idx = y_name[j]
                    pJoint[j] = J[self.label_state_idx[y_idx], j] / z_[j]
                    loss[j] = - torch.log(pJoint[j] + 1e-8)

            elif self.mode == 'margin':
                multi_label_pMargin = torch.index_select(pMargin[j], 0, torch.where(y[j]==1)[0])
                loss[j] = - torch.log(multi_label_pMargin + 1e-8).sum()

        loss = torch.mean(loss)

        _check_loss(loss)

        return loss, pMargin

    def compute_loss_margin(self, pMargin, y, mask):
        loss_margin = self.bce(pMargin, y) * mask
        return loss_margin.mean()


    def infer(self, f):
        pMargin = torch.zeros((f.shape[0], f.shape[1])).to(f.device)
        S = self.legal_state.to(f.device)
        potential = torch.mm(S.double(), f.T.double())

        max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        J = torch.exp(potential - max_sf)  # (n+1)xbz
        z_ = torch.sum(J, dim=0)

        # for Z_ is inf, ignore its pMargin loss
        id_num = _check_abnornal(z_)

        for i in range(f.shape[1]):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        return pMargin


# ================================================================================================
# build graph for FFpp
# utils function
def legal_state_ffso(Ee, Eh, all_state, parent_dict, children_dict):
    state = copy.deepcopy(all_state)
    state = state.numpy()
    for edge in Ee:
        num = 1
        idx = []
        for i in range(len(state)):
            if state[i, edge[0]] == 1 and state[i, edge[1]] == 1:
               idx.append(i)
               num += 1
        state = np.delete(state, idx, axis=0)

    for edge in Eh:
        num = 1
        idx = []
        for i in range(len(state)):
            # leaf node and its state
            leaf_node_label = edge[1]
            leaf_state = state[i, edge[1]]
            if leaf_node_label <= 3:
                if state[i, edge[0]] == 0 and state[i, edge[1]] == 1:
                    idx.append(i)
                    num += 1

                #
                else:
                    child_nodes = children_dict[str(leaf_node_label)]
                    child_nodes_state = []
                    for k in range(len(child_nodes)):
                        child_nodes_state.append(state[i, child_nodes[k]])
                    if sum(child_nodes_state) == 0 and leaf_state == 1:
                        idx.append(i)
                        num += 1

            else:
                parent_nodes = parent_dict[str(leaf_node_label)]
                parent_nodes_state = []
                for j in range(len(parent_nodes)):
                    parent_nodes_state.append(state[i, parent_nodes[j]])
                if sum(parent_nodes_state) == 0 and leaf_state == 1:
                    idx.append(i)
                    num += 1
        state = np.delete(state, idx, axis=0)
    return torch.tensor(state)


def graph_SA_ffso():
    Eh_edge = [
        (0,1), (0,2), (0,3),
        (1, 6), (1, 7),
        (2, 4), (2, 5), (2, 6), (2, 7), (2,8), (2,9),
        (3, 4), (3, 6), (3, 7), (3, 8)
    ]
    Ee_edge = []
    # prepare a dict/list that record the parent nodes for each node
    parent_dict = {
        "4": [2, 3],
        "5": [2],
        "6": [1, 2, 3],
        "7": [1, 2, 3],
        "8": [2, 3],
        "9": [2],
    }
    children_dict = {
        "1": [6, 7],
        "2": [4, 5, 6, 7, 8, 9],
        "3": [4, 6, 7, 8],
    }

    state = legal_state_ffso(Ee_edge, Eh_edge, bin_list(10), parent_dict,
                        children_dict)  # all the legal states under the proposed graph

    label_state = get_label_state_idx_ffso(state)  # correspond to the leaf node w.r.t the graph state

    return state, label_state

label_map_ffso = {
    '1010111111': [1, 0,1,0, 1,1,1,1,1,1], # id
    '1001000010': [1, 0,0,1, 0,0,0,0,1,0], # phy_nose
    '1001001100': [1, 0,0,1, 0,0,1,1,0,0], # phy_mouth
    '1100001100': [1, 1,0,0, 0,0,1,1,0,0], # expr_mouth
    '1001100000': [1, 0,0,1, 1,0,0,0,0,0], # phys_eyes
    '0000000000': [0, 0,0,0, 0,0,0,0,0,0], # real
}
def get_label_state_idx_ffso(state):
    length = state.shape[0]
    labels_id = {'1010111111': [], '1001000010': [], '1001001100': [], '1100001100': [],
                 '1001100000': [], '0000000000': []}
    for i in range(length):
        if (torch.tensor(label_map_ffso['1010111111']) == state[i]).all():
            labels_id['1010111111'] = i

        elif (torch.tensor(label_map_ffso['1001000010']) == state[i]).all():
            labels_id['1001000010'] = i

        elif (torch.tensor(label_map_ffso['1001001100']) == state[i]).all():
            labels_id['1001001100'] = i

        elif (torch.tensor(label_map_ffso['1100001100']) == state[i]).all():
            labels_id['1100001100'] = i

        elif (torch.tensor(label_map_ffso['1001100000']) == state[i]).all():
            labels_id['1001100000'] = i

        elif (torch.tensor(label_map_ffso['0000000000']) == state[i]).all():
            labels_id['0000000000'] = i

    return labels_id
# ================================================================================================


# build graph for FFSC
# utils function
def bin_list(nsize):
    s = list(it.product(range(2), repeat=nsize))
    out = torch.tensor(s)
    return out
def legal_state(Ee, Eh, all_state, parent_dict, children_dict):
    state = copy.deepcopy(all_state)
    state = state.numpy()
    for edge in Ee:
        num = 1
        idx = []
        for i in range(len(state)):
            if state[i, edge[0]] == 1 and state[i, edge[1]] == 1:
               idx.append(i)
               num += 1
        state = np.delete(state, idx, axis=0)

    for edge in Eh:
        num = 1
        idx = []
        for i in range(len(state)):
            # leaf node and its state
            leaf_node_label = edge[1]
            leaf_state = state[i, edge[1]]
            if leaf_node_label <= 5:
                if state[i, edge[0]] == 0 and state[i, edge[1]] == 1:
                    idx.append(i)
                    num += 1

                #
                else:
                    child_nodes = children_dict[str(leaf_node_label)]
                    child_nodes_state = []
                    for k in range(len(child_nodes)):
                        child_nodes_state.append(state[i, child_nodes[k]])
                    if sum(child_nodes_state) == 0 and leaf_state == 1:
                        idx.append(i)
                        num += 1

            else:
                parent_nodes = parent_dict[str(leaf_node_label)]
                parent_nodes_state = []
                for j in range(len(parent_nodes)):
                    parent_nodes_state.append(state[i, parent_nodes[j]])
                if sum(parent_nodes_state) == 0 and leaf_state == 1:
                    idx.append(i)
                    num += 1
        state = np.delete(state, idx, axis=0)
    return torch.tensor(state)


def graph_SA():
    Eh_edge = [
        (0,1), (0,2), (0,3), (0,4), (0,5),
        (1, 6), (1, 11),
        (2,6), (2,8), (2,9), (2, 10), (2, 11),
        (3,6), (3,7), (3,8), (3, 9), (3, 11),
        (4,6), (4,7), (4,8), (4,9), (4, 10), (4, 11),
        (5,6), (5,7), (5,9), (5,9), (5, 10), (5, 11),
    ]
    Ee_edge = []
    # prepare a dict/list that record the parent nodes for each node
    parent_dict = {
        "6": [1, 2, 3, 4, 5],
        "7": [3, 4, 5],
        "8": [2, 3, 4, 5],
        "9": [2, 3, 4, 5],
        "10": [2, 4, 5],
        "11": [1, 2, 3, 4, 5]
    }
    children_dict = {
        "1": [6, 11],
        "2": [6, 8, 9, 10, 11],
        "3": [6,7,8, 9, 11],
        "4": [6,7,8,9, 10, 11],
        "5": [6,7,8,9, 10, 11],
    }

    state = legal_state(Ee_edge, Eh_edge, bin_list(12), parent_dict,
                        children_dict)  # all the legal states under the proposed graph

    label_state = get_label_state_idx(state)  # correspond to the leaf node w.r.t the graph state

    return state, label_state

label_map = {
    '110000100001': [1,1,0,0,0,0,1,0,0,0,0,1], # age
    '101000001100': [1,0,1,0,0,0,0,0,1,1,0,0], # expr-smile
    '101000101111': [1,0,1,0,0,0,1,0,1,1,1,1], # expr-surprised
    '120110111101': [1,2,0,1,1,0,1,1,1,1,0,1], # gender
    '120010111111': [1,2,0,0,1,0,1,1,1,1,1,1], # ID
    '100001111111': [1,0,0,0,0,1,1,1,1,1,1,1], # pose
    '000000000000': [0,0,0,0,0,0,0,0,0,0,0,0], # real
}
def get_label_state_idx(state):
    length = state.shape[0]
    labels_id = {'110000100001': [], '101000001100': [], '101000101111': [], '120110111101': [],
                 '120010111111': [], '100001111111': [], '000000000000': []}
    for i in range(length):
        if (torch.tensor(label_map['110000100001']) == state[i]).all():
            labels_id['110000100001'] = i

        elif (torch.tensor(label_map['101000001100']) == state[i]).all():
            labels_id['101000001100'] = i

        elif (torch.tensor(label_map['101000101111']) == state[i]).all():
            labels_id['101000101111'] = i

        elif (torch.tensor(label_map['120110111101']) == state[i]).all():
            labels_id['120110111101'] = i

        elif (torch.tensor(label_map['120010111111']) == state[i]).all():
            labels_id['120010111111'] = i

        elif (torch.tensor(label_map['100001111111']) == state[i]).all():
            labels_id['100001111111'] = i

        elif (torch.tensor(label_map['000000000000']) == state[i]).all():
            labels_id['000000000000'] = i
    return labels_id


# global semantics only in the SLH
def graph_SA_global():
    Eh_edge = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)
    ]
    Ee_edge = []

    parent_dict = {}
    children_dict = {}

    state = legal_state(Ee_edge, Eh_edge, bin_list(6), parent_dict,
                        children_dict)

    label_state = get_label_state_idx_global(state)
    return state, label_state

label_map_G = {
    '110000': [1,1,0,0,0,0], # age
    '101001': [1,0,1,0,0,1], # expr
    '120110': [1,2,0,1,1,0], # gender
    '120010': [1,2,0,0,1,0], # ID
    '100001': [1,0,0,0,0,1], # pose
    '000000': [0,0,0,0,0,0], # real
}

def get_label_state_idx_global(state):
    length = state.shape[0]
    labels_id = {'110000': [], '101001': [], '120110': [],
                 '120010': [], '100001': [], '000000': []}
    for i in range(length):
        if (torch.tensor(label_map_G['110000']) == state[i]).all():
            labels_id['110000'] = i

        elif (torch.tensor(label_map_G['101001']) == state[i]).all():
            labels_id['101001'] = i

        elif (torch.tensor(label_map_G['120110']) == state[i]).all():
            labels_id['120110'] = i

        elif (torch.tensor(label_map_G['120010']) == state[i]).all():
            labels_id['120010'] = i

        elif (torch.tensor(label_map_G['100001']) == state[i]).all():
            labels_id['100001'] = i

        elif (torch.tensor(label_map_G['000000']) == state[i]).all():
            labels_id['000000'] = i
    return labels_id

# utils for pLoss
def _check_abnornal(z_):
    if np.inf in z_:
        pdb.set_trace()
        idx = z_ == np.inf
        id_num = [i for i, v in enumerate(list(idx)) if v == True]
    else:
        id_num = [-1]
    return id_num

def _check_loss(loss):
    if torch.isnan(loss) == 1:
        pdb.set_trace()

    if loss == np.inf:
        pdb.set_trace()


if __name__ == '__main__':
    g = graph_SA()
    print()