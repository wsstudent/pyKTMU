import os, sys
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from .evaluate_model import evaluate
from torch.autograd import Variable, grad
from .atkt import _l2_normalize_adv
from ..utils.utils import debug_print
from pykt.config import que_type_models
import pandas as pd

# 设置计算设备，优先使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cal_loss(model, ys, r, rshft, sm, preloss=[]):
    """
    根据模型名称和输出计算损失。

    Args:
        model: 模型对象。
        ys (list): 模型的预测输出列表。
        r (torch.Tensor): 当前时间步的作答情况 (response)。
        rshft (torch.Tensor): 下一时间步的作答情况 (shifted response)。
        sm (torch.Tensor): 用于在序列中选择有效时间步的掩码 (mask)。
        preloss (list, optional): 某些模型（如 AKT）产生的额外损失（如正则化项）。

    Returns:
        torch.Tensor: 计算得到的标量损失值。
    """
    model_name = model.model_name

    # ---- 不同模型的损失计算逻辑 ----

    # 1. 具有多种损失成分的模型 (如 SimpleKT, ATDKT 等)
    if model_name in [
        "atdkt",
        "simplekt",
        "stablekt",
        "bakt_time",
        "sparsekt",
        "cskt",
        "hcgkt",
    ]:
        y = torch.masked_select(ys[0], sm)  # 根据掩码选择有效的预测
        t = torch.masked_select(rshft, sm)  # 根据掩码选择有效的目标
        loss1 = binary_cross_entropy(y.double(), t.double())  # 主要的二元交叉熵损失

        # 根据 emb_type 添加不同的辅助损失
        if "predcurc" in model.emb_type:
            if "his" in model.emb_type:
                loss = model.l1 * loss1 + model.l2 * ys[1] + model.l3 * ys[2]
            else:
                loss = model.l1 * loss1 + model.l2 * ys[1]
        elif "predhis" in model.emb_type:
            loss = model.l1 * loss1 + model.l2 * ys[1]
        else:
            loss = loss1

    # 2. Rekt 模型
    elif model_name in ["rekt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double())

    # 3. UKT 模型
    elif model_name in ["ukt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss1 = binary_cross_entropy(y.double(), t.double())
        if model.use_CL:  # 如果使用对比学习 (Contrastive Learning)
            loss2 = ys[1]
            loss1 = loss1 + model.cl_weight * loss2
        loss = loss1

    # 4. 大多数标准模型的通用 BCE 损失
    elif model_name in [
        "rkt",
        "dimkt",
        "dkt",
        "dkt_forget",
        "dkvmn",
        "deep_irt",
        "kqn",
        "sakt",
        "saint",
        "atkt",
        "atktfix",
        "gkt",
        "skvmn",
        "hawkes",
    ]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double())

    # 5. DKT+ 的复合损失
    elif model_name == "dkt+":
        y_curr = torch.masked_select(ys[1], sm)  # 当前预测
        y_next = torch.masked_select(ys[0], sm)  # 下一时刻预测
        r_curr = torch.masked_select(r, sm)  # 当前作答
        r_next = torch.masked_select(rshft, sm)  # 下一时刻作答

        # 主要损失
        loss = binary_cross_entropy(y_next.double(), r_next.double())

        # 辅助损失项
        loss_r = binary_cross_entropy(y_curr.double(), r_curr.double())  # 重建损失
        loss_w1 = (
            torch.masked_select(
                torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=1, dim=-1), sm[:, 1:]
            ).mean()
            / model.num_c
        )  # L1 平滑项
        loss_w2 = (
            torch.masked_select(
                torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=2, dim=-1) ** 2, sm[:, 1:]
            ).mean()
            / model.num_c
        )  # L2 平滑项

        # 组合所有损失
        loss = (
            loss
            + model.lambda_r * loss_r
            + model.lambda_w1 * loss_w1
            + model.lambda_w2 * loss_w2
        )

    # 6. AKT 及其变体（包含正则化损失）
    elif model_name in [
        "akt",
        "extrakt",
        "folibikt",
        "robustkt",
        "akt_vector",
        "akt_norasch",
        "akt_mono",
        "akt_attn",
        "aktattn_pos",
        "aktmono_pos",
        "akt_raschx",
        "akt_raschy",
        "aktvec_raschx",
        "lefokt_akt",
        "dtransformer",
        "fluckt",
    ]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = (
            binary_cross_entropy(y.double(), t.double()) + preloss[0]
        )  # BCE + 正则化损失

    # 7. LPKT 的损失
    elif model_name == "lpkt":
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        criterion = nn.BCELoss(reduction="none")  # 不立即求和，以便后续处理
        loss = criterion(y, t).sum()

    return loss


def model_forward(model, data, rel=None):
    """
    执行模型的前向传播，并准备计算损失所需的输出。

    Args:
        model: 模型对象。
        data: 一个批次的数据，通常是一个字典或元组。
        rel (optional): RKT 模型需要的关系矩阵。

    Returns:
        torch.Tensor: 计算得到的损失值。对于某些模型，可能还会返回其他值。
    """
    model_name = model.model_name

    # --- 1. 解包数据 ---
    # 根据模型处理不同的数据格式
    if model_name in ["dkt_forget", "bakt_time"]:
        dcur, dgaps = data
    else:
        dcur = data

    # DIMKT 需要额外的难度信息
    if model_name in ["dimkt"]:
        q, c, r, t, sd, qd = (
            dcur["qseqs"].to(device),
            dcur["cseqs"].to(device),
            dcur["rseqs"].to(device),
            dcur["tseqs"].to(device),
            dcur["sdseqs"].to(device),
            dcur["qdseqs"].to(device),
        )
        qshft, cshft, rshft, tshft, sdshft, qdshft = (
            dcur["shft_qseqs"].to(device),
            dcur["shft_cseqs"].to(device),
            dcur["shft_rseqs"].to(device),
            dcur["shft_tseqs"].to(device),
            dcur["shft_sdseqs"].to(device),
            dcur["shft_qdseqs"].to(device),
        )
    else:
        q, c, r, t = (
            dcur["qseqs"].to(device),
            dcur["cseqs"].to(device),
            dcur["rseqs"].to(device),
            dcur["tseqs"].to(device),
        )
        qshft, cshft, rshft, tshft = (
            dcur["shft_qseqs"].to(device),
            dcur["shft_cseqs"].to(device),
            dcur["shft_rseqs"].to(device),
            dcur["shft_tseqs"].to(device),
        )

    m, sm = dcur["masks"].to(device), dcur["smasks"].to(device)

    # --- 2. 准备模型输入 ---
    ys, preloss = [], []  # ys 存储预测输出，preloss 存储额外损失
    # 构造交互序列，将初始状态与后续序列拼接，以输入 Transformer 类模型
    cq = torch.cat((q[:, 0:1], qshft), dim=1)
    cc = torch.cat((c[:, 0:1], cshft), dim=1)
    cr = torch.cat((r[:, 0:1], rshft), dim=1)

    # --- 3. 模型前向传播 ---
    # 这部分是模型的核心，根据 model_name 调用不同的模型前向传播逻辑

    # Hawkes 模型需要时间戳序列
    if model_name in ["hawkes"]:
        ct = torch.cat((t[:, 0:1], tshft), dim=1)

    # RKT 模型需要关系矩阵 rel
    elif model_name in ["rkt"]:
        y, attn = model(dcur, rel, train=True)
        ys.append(y[:, 1:])

    # ATDKT
    elif model_name in ["atdkt"]:
        y, y2, y3 = model(dcur, train=True)
        if model.emb_type.find("bkt") == -1 and model.emb_type.find("addcshft") == -1:
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys = [y, y2, y3]

    # SimpleKT 及其变体
    elif model_name in ["simplekt", "stablekt", "sparsekt", "cskt"]:
        y, y2, y3 = model(dcur, train=True)
        ys = [y[:, 1:], y2, y3]

    # Rekt
    elif model_name in ["rekt"]:
        y = model(dcur, train=True)
        ys = [y]

    # UKT
    elif model_name in ["ukt"]:
        if model.use_CL != 0:
            y, sim, y2, y3, temp = model(dcur, train=True)
            ys = [y[:, 1:], sim, y2, y3]
        else:
            y, y2, y3 = model(dcur, train=True)
            ys = [y[:, 1:], y2, y3]
    # HCGKT (对抗训练的变体)
    elif model_name in ["hcgkt"]:
        step_size = step_size
        step_m = step_m
        grad_clip = grad_clip
        mm = mm

        # the xxx.pt file of pre_load_gcn can be found in :
        # https://drive.google.com/drive/folders/1JWstsquI3TzbUlqB1EyCbjem4qPyRLCh?usp=drive_link
        matrix = None
        if dataset_name == "assist2009":
            pre_load_gcn = "../data/assist2009/ques_skill_gcn_adj.pt"
            matrix = torch.load(pre_load_gcn)
            if not matrix.is_sparse:
                matrix = matrix.to_sparse()
        elif dataset_name == "algebra2005":
            pre_load_gcn = "../data/algebra2005/ques_skill_gcn_adj.pt"
            matrix = torch.load(pre_load_gcn)
            if not matrix.is_sparse:
                matrix = matrix.to_sparse()
        elif dataset_name == "bridge2algebra2006":
            pre_load_gcn = "../data/bridge2algebra2006/ques_skill_gcn_adj.pt"
            matrix = torch.load(pre_load_gcn)
            if not matrix.is_sparse:
                matrix = matrix.to_sparse()
        elif dataset_name == "peiyou":
            pre_load_gcn = "../data/peiyou/ques_skill_gcn_adj.pt"
            matrix = torch.load(pre_load_gcn)
            if not matrix.is_sparse:
                matrix = matrix.to_sparse()
        elif dataset_name == "nips_task34":
            pre_load_gcn = "../data/nips_task34/ques_skill_gcn_adj.pt"
            matrix = torch.load(pre_load_gcn)
            if not matrix.is_sparse:
                matrix = matrix.to_sparse()
        perturb_shape = (matrix.shape[0], emb_size)
        perturb = (
            torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size).to(device)
        )
        perturb.requires_grad_()
        y, y2, y3, contrast_loss = model(dcur, train=True, perb=perturb)
        ys = [y[:, 1:], y2, y3]
        loss = cal_loss(model, ys, r, rshft, sm, preloss) + contrast_loss
        loss /= step_m
        opt.zero_grad()
        for _ in range(step_m - 1):
            loss.backward()
            perturb_data = perturb.detach() + step_size * torch.sign(
                perturb.grad.detach()
            )
            perturb.data = perturb_data.data
            perturb.grad[:] = 0
            y, y2, y3, contrast_loss = model(dcur, train=True, perb=perturb)
            ys = [y[:, 1:], y2, y3]
            loss = cal_loss(model, ys, r, rshft, sm, preloss) + contrast_loss
            loss /= step_m

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        model.sfm_cl.gcl.update_target_network(mm)
        return loss

    # DTransformer
    elif model_name in ["dtransformer"]:
        if model.emb_type == "qid_cl":
            y, loss = model.get_cl_loss(cc.long(), cr.long(), cq.long())  # with cl loss
        else:
            y, loss = model.get_loss(cc.long(), cr.long(), cq.long())
        ys.append(y[:, 1:])
        preloss.append(loss)

    # BAKT-Time
    elif model_name in ["bakt_time"]:
        y, y2, y3 = model(dcur, dgaps, train=True)
        ys = [y[:, 1:], y2, y3]

    # LPKT
    elif model_name in ["lpkt"]:
        cit = torch.cat((dcur["itseqs"][:, 0:1], dcur["shft_itseqs"]), dim=1)

    # DKT
    if model_name in ["dkt"]:
        y = model(c.long(), r.long())
        # one_hot 编码并选择与下一题目相关的预测
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y)

    # DKT+
    elif model_name == "dkt+":
        y = model(c.long(), r.long())
        y_next = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        y_curr = (y * one_hot(c.long(), model.num_c)).sum(-1)
        ys = [y_next, y_curr, y]

    # DKT-Forget
    elif model_name in ["dkt_forget"]:
        y = model(c.long(), r.long(), dgaps)
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y)

    # DKVNM, Deep-IRT, SKVMN
    elif model_name in ["dkvmn", "deep_irt", "skvmn"]:
        y = model(cc.long(), cr.long())
        ys.append(y[:, 1:])

    # KQN, SAKT
    elif model_name in ["kqn", "sakt"]:
        y = model(c.long(), r.long(), cshft.long())
        ys.append(y)

    # SAINT
    elif model_name in ["saint"]:
        y = model(cq.long(), cc.long(), r.long())
        ys.append(y[:, 1:])

    # AKT 及其变体
    elif model_name in [
        "akt",
        "extrakt",
        "folibikt",
        "robustkt",
        "akt_vector",
        "akt_norasch",
        "akt_mono",
        "akt_attn",
        "aktattn_pos",
        "aktmono_pos",
        "akt_raschx",
        "akt_raschy",
        "aktvec_raschx",
        "lefokt_akt",
        "fluckt",
    ]:
        y, reg_loss = model(cc.long(), cr.long(), cq.long())
        ys.append(y[:, 1:])
        preloss.append(reg_loss)

    # ATKT, ATKTFix (对抗训练)
    elif model_name in ["atkt", "atktfix"]:
        # 第一次前向传播计算正常 loss
        y, features = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        loss = cal_loss(model, [y], r, rshft, sm)

        # 计算对抗性扰动
        features_grad = grad(loss, features, retain_graph=True)
        p_adv = torch.FloatTensor(
            model.epsilon * _l2_normalize_adv(features_grad[0].data)
        )
        p_adv = Variable(p_adv).to(device)

        # 第二次前向传播计算对抗性 loss
        pred_res, _ = model(c.long(), r.long(), p_adv)
        pred_res = (pred_res * one_hot(cshft.long(), model.num_c)).sum(-1)
        adv_loss = cal_loss(model, [pred_res], r, rshft, sm)

        # 组合 loss
        loss = loss + model.beta * adv_loss

    # GKT
    elif model_name == "gkt":
        y = model(cc.long(), cr.long())
        ys.append(y)

    # LPKT
    elif model_name == "lpkt":
        y = model(cq.long(), cr.long(), cit.long())
        ys.append(y[:, 1:])

    # Hawkes
    elif model_name == "hawkes":
        y = model(cc.long(), cq.long(), ct.long(), cr.long())
        ys.append(y[:, 1:])

    # 其他 que_type 模型
    elif model_name in que_type_models and model_name not in ["lpkt", "rkt"]:
        y, loss = model.train_one_step(data)

    # DIMKT
    elif model_name == "dimkt":
        y = model(
            q.long(),
            c.long(),
            sd.long(),
            qd.long(),
            r.long(),
            qshft.long(),
            cshft.long(),
            sdshft.long(),
            qdshft.long(),
        )
        ys.append(y)

    # --- 4. 计算最终损失 ---
    # 对于非对抗训练等特殊情况，在此处统一调用 cal_loss
    if model_name not in ["atkt", "atktfix"] + que_type_models or model_name in [
        "lpkt",
        "rkt",
    ]:
        loss = cal_loss(model, ys, r, rshft, sm, preloss)

    # ukt 模型可能返回额外的温度参数
    if model_name in ["ukt"] and model.use_CL != 0:
        return loss, temp

    return loss


def train_model(
    model,
    train_loader,
    valid_loader,
    num_epochs,
    opt,
    ckpt_path,
    test_loader=None,
    test_window_loader=None,
    save_model=False,
    data_config=None,
    fold=None,
):
    """
    训练模型的主函数。

    Args:
        model: 待训练的模型对象。
        train_loader: 训练数据加载器。
        valid_loader: 验证数据加载器。
        num_epochs: 训练的总轮次。
        opt: 优化器。
        ckpt_path: 模型检查点和日志的保存路径。
        test_loader (optional): 测试数据加载器。
        test_window_loader (optional): 滑动窗口测试数据加载器。
        save_model (optional): 是否保存性能最佳的模型。
        data_config (optional): 数据集配置，主要用于 RKT 模型。
        fold (optional): 当前交叉验证折数，主要用于 RKT 模型。
    """
    # --- 1. 初始化跟踪变量 ---
    max_auc = 0  # 用于记录验证集上最好的 AUC 分数
    best_epoch = -1  # 记录最佳 AUC 对应的轮次
    train_step = 0  # 训练总步数计数器

    # 存储在最佳 epoch 上得到的各项指标
    best_valid_auc, best_valid_acc = -1, -1
    best_test_auc, best_test_acc = -1, -1
    best_window_auc, best_window_acc = -1, -1

    debug_print(
        text=f"开始训练模型: {model.model_name}，共 {num_epochs} 个 epochs。日志和模型将保存在: {ckpt_path}",
        fuc_name="train_model",
    )

    # --- 2. 模型特定设置 ---
    # RKT 模型需要一个预先计算好的关系矩阵 `rel`
    rel = None
    if model.model_name == "rkt":
        dpath = data_config["dpath"]
        dataset_name = dpath.split("/")[-1]
        tmp_folds = set(data_config["folds"]) - {fold}
        folds_str = "_" + "_".join([str(_) for _ in tmp_folds])

        fname = (
            "phi_array" + folds_str + ".pkl"
            if dataset_name not in ["algebra2005", "bridge2algebra2006"]
            else "phi_dict" + folds_str + ".pkl"
        )
        rel_path = os.path.join(dpath, fname)
        debug_print(
            text=f"正在为 RKT 模型加载关系矩阵: {rel_path}", fuc_name="train_model"
        )
        rel = pd.read_pickle(rel_path)

    # LPKT 模型使用学习率调度器 (learning rate scheduler)
    if model.model_name == "lpkt":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.5)
        debug_print(
            text="已为 LPKT 模型配置 StepLR 学习率调度器。", fuc_name="train_model"
        )

    # --- 3. 主训练循环 (Epoch Loop) ---
    for i in range(1, num_epochs + 1):
        loss_mean = []
        debug_print(
            text=f"======== Epoch {i}/{num_epochs} 开始 ========",
            fuc_name="train_model",
        )

        # --- 4. 批次循环 (Batch Loop) ---
        for data in train_loader:
            train_step += 1

            # a. 设置模型为训练模式
            model.train()

            # b. 前向传播和损失计算
            loss = model_forward(
                model, data, rel if model.model_name == "rkt" else None
            )

            # c. 反向传播和优化
            opt.zero_grad()
            loss.backward()

            # d. 梯度裁剪
            if model.model_name in ["rkt", "dtransformer"]:
                clip_grad_norm_(model.parameters(), 1.0)

            opt.step()

            # e. 记录和调试输出
            loss_mean.append(loss.detach().cpu().numpy())
            if train_step % 50 == 0:
                debug_print(
                    text=f"Epoch: {i}, Step: {train_step}, 当前 Batch Loss: {loss.item():.4f}",
                    fuc_name="train_model",
                )

        # --- 5. Epoch 结束后的操作 ---
        if model.model_name == "lpkt":
            scheduler.step()
            debug_print(
                text=f"Epoch {i}: LPKT 学习率已更新，当前 LR: {opt.param_groups[0]['lr']}",
                fuc_name="train_model",
            )

        loss_mean = np.mean(loss_mean)
        debug_print(
            text=f"Epoch {i}: 平均训练 Loss: {loss_mean:.4f}", fuc_name="train_model"
        )

        # --- 6. 评估模型 ---
        debug_print(text=f"Epoch {i}: 正在验证集上评估模型...", fuc_name="train_model")
        auc, acc = evaluate(
            model,
            valid_loader,
            model.model_name,
            rel if model.model_name == "rkt" else None,
        )
        debug_print(
            text=f"Epoch {i}: 验证集结果 - AUC: {auc:.4f}, ACC: {acc:.4f}",
            fuc_name="train_model",
        )

        # --- 7. 保存最佳模型 ---
        if auc > max_auc:
            debug_print(
                text=f"发现新的最佳模型！AUC 从 {max_auc:.4f} 提升至 {auc:.4f}。",
                fuc_name="train_model",
            )
            max_auc = auc
            best_epoch = i
            best_valid_auc, best_valid_acc = auc, acc

            if save_model:
                torch.save(
                    model.state_dict(),
                    os.path.join(ckpt_path, f"{model.emb_type}_model.ckpt"),
                )

            # 在找到最佳模型时，在测试集上进行评估
            if test_loader is not None:
                best_test_auc, best_test_acc = evaluate(
                    model,
                    test_loader,
                    model.model_name,
                    rel if model.model_name == "rkt" else None,
                )
            if test_window_loader is not None:
                best_window_auc, best_window_acc = evaluate(
                    model,
                    test_window_loader,
                    model.model_name,
                    rel if model.model_name == "rkt" else None,
                )

        # --- 8. 打印 Epoch 摘要信息 ---
        print(
            f"Epoch: {i}, validauc: {auc:.4f}, validacc: {acc:.4f}, best epoch: {best_epoch}, best auc: {max_auc:.4f}, train loss: {loss_mean:.4f}"
        )
        if i == best_epoch:
            print(
                f"         testauc: {round(best_test_auc, 4)}, testacc: {round(best_test_acc, 4)}, window_testauc: {round(best_window_auc, 4)}, window_testacc: {round(best_window_acc, 4)}"
            )

        # --- 9. 早停机制 (Early Stopping) ---
        if i - best_epoch >= 10:
            debug_print(
                text=f"早停触发：已连续 10 个 epochs 验证集 AUC 未提升。",
                fuc_name="train_model",
            )
            break

    # --- 10. 训练结束，返回最佳结果 ---
    debug_print(text="训练结束。", fuc_name="train_model")
    return (
        best_test_auc,
        best_test_acc,
        best_window_auc,
        best_window_acc,
        best_valid_auc,
        best_valid_acc,
        best_epoch,
    )

