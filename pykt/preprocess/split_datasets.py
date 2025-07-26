import os
import sys
import pandas as pd
import numpy as np
import json
import copy

# 尝试导入遗忘策略模块，如果不存在也不会报错
try:
    from ..utils.forget_strategy import ForgetStrategy
except ImportError:
    ForgetStrategy = None
    print("警告: 无法导入 ForgetStrategy 模块，遗忘功能将不可用")

# 定义数据中所有可能的字段（列名）
ALL_KEYS = [
    "fold",
    "uid",
    "questions",
    "concepts",
    "responses",
    "timestamps",
    "usetimes",
    "selectmasks",
    "is_repeat",
    "qidxs",
    "rest",
    "orirow",
    "cidxs",
]
# 定义那些每个用户只有一个值的字段（不需要在序列中切分的字段）
ONE_KEYS = ["fold", "uid"]


def read_data(fname, min_seq_len=3, response_set=[0, 1]):
    """
    从特定格式的文本文件中读取原始数据。

    文件格式为每个学生占用6行：
    1. 学生ID,序列长度
    2. 题目ID序列 (逗号分隔)
    3. 知识点ID序列 (逗号分隔)
    4. 作答结果序列 (逗号分隔)
    5. 时间戳序列 (逗号分隔)
    6. 答题用时序列 (逗号分隔)

    Args:
        fname (str): 输入数据文件的路径。
        min_seq_len (int): 允许的最短交互序列长度，低于此长度的学生数据将被丢弃。
        response_set (list): 合法的作答结果集合，用于数据校验。

    Returns:
        tuple: 包含一个Pandas DataFrame和一个包含有效字段名的集合 (df, effective_keys)。
    """
    effective_keys = set()  # 存储文件中实际存在的有效字段
    dres = dict()  # 用于构建DataFrame的字典
    delstu, delnum, badr = 0, 0, 0  # 统计被删除的学生数、交互数和含无效作答的记录数
    goodnum = 0  # 统计有效的总交互数
    
    with open(fname, "r", encoding="utf8") as fin:
        i = 0
        lines = fin.readlines()
        dcur = dict()  # 临时存储当前学生的数据
        
        while i < len(lines):
            line = lines[i].strip()
            # 每6行一个学生数据块
            if i % 6 == 0:  # 第1行: 学生ID和序列长度
                effective_keys.add("uid")
                tmps = line.split(",")
                # 兼容不同的ID格式
                if "(" in tmps[0]:
                    stuid, seq_len = tmps[0].replace("(", ""), int(tmps[2])
                else:
                    stuid, seq_len = tmps[0], int(tmps[1])
                # 如果序列长度小于最小要求，则跳过该学生
                if seq_len < min_seq_len:
                    i += 6
                    dcur = dict()
                    delstu += 1
                    delnum += seq_len
                    continue
                dcur["uid"] = stuid
                goodnum += seq_len
            elif i % 6 == 1:  # 第2行: 题目ID
                qs = []
                if line.find("NA") == -1:  # "NA"表示该字段无数据
                    effective_keys.add("questions")
                    qs = line.split(",")
                dcur["questions"] = qs
            elif i % 6 == 2:  # 第3行: 知识点ID
                cs = []
                if line.find("NA") == -1:
                    effective_keys.add("concepts")
                    cs = line.split(",")
                dcur["concepts"] = cs
            elif i % 6 == 3:  # 第4行: 作答结果
                effective_keys.add("responses")
                rs = []
                if line.find("NA") == -1:
                    flag = True
                    # 校验每个作答结果是否合法
                    for r in line.split(","):
                        try:
                            r = int(r)
                            if r not in response_set:  # 检查作答是否在预设集合内
                                print(f"错误: 在第 {i} 行发现无效的作答结果: {r}")
                                flag = False
                                break
                            rs.append(r)
                        except ValueError:
                            print(f"错误: 在第 {i} 行发现无法解析的作答结果: {r}")
                            flag = False
                            break
                    # 如果有不合法的作答，跳过该学生
                    if not flag:
                        i += 3  # 跳过该学生剩下的行
                        dcur = dict()
                        badr += 1
                        continue
                dcur["responses"] = rs
            elif i % 6 == 4:  # 第5行: 时间戳
                ts = []
                if line.find("NA") == -1:
                    effective_keys.add("timestamps")
                    ts = line.split(",")
                dcur["timestamps"] = ts
            elif i % 6 == 5:  # 第6行: 答题用时
                usets = []
                if line.find("NA") == -1:
                    effective_keys.add("usetimes")
                    usets = line.split(",")
                dcur["usetimes"] = usets

                # 一个学生的6行数据都处理完后，存入结果字典
                for key in effective_keys:
                    dres.setdefault(key, [])
                    if key != "uid":
                        # 将列表转换为逗号分隔的字符串
                        dres[key].append(",".join([str(k) for k in dcur[key]]))
                    else:
                        dres[key].append(dcur[key])
                dcur = dict()  # 重置临时字典
            i += 1
            
    df = pd.DataFrame(dres)
    # 打印数据读取和清洗的统计信息
    print(
        f"因序列过短删除学生数: {delstu}, 对应交互数: {delnum}, 因作答无效删除学生数: {badr}, 有效总交互数: {goodnum}"
    )
    return df, effective_keys


def extend_multi_concepts(df, effective_keys):
    """
    处理一个题目关联多个知识点的情况。
    如果一个知识点字段是 "c1_c2"，它会把这一行展开成两行，
    其它字段（如题目、作答）被复制，并添加'is_repeat'列来标记。

    Args:
        df (pd.DataFrame): 输入的DataFrame。
        effective_keys (set): 有效的字段名集合。

    Returns:
        tuple: 包含处理后的DataFrame和更新后的有效字段名集合 (df, effective_keys)。
    """
    if "questions" not in effective_keys or "concepts" not in effective_keys:
        print("数据中缺少'questions'或'concepts'字段，无法展开多知识点，返回原始数据。")
        return df, effective_keys

    # 需要展开的列是除了uid之外的所有列
    extend_keys = set(df.columns) - {"uid"}

    dres = {"uid": df["uid"]}
    for _, row in df.iterrows():
        dextend_infos = dict()
        # 将每个字段的字符串按逗号分割成列表
        for key in extend_keys:
            dextend_infos[key] = row[key].split(",")

        dextend_res = dict()
        for i in range(len(dextend_infos["questions"])):  # 遍历学生的每一次交互
            dextend_res.setdefault("is_repeat", [])
            # 检查知识点字段是否包含"_"，表示有多个知识点
            if dextend_infos["concepts"][i].find("_") != -1:
                ids = dextend_infos["concepts"][i].split("_")
                dextend_res.setdefault("concepts", [])
                dextend_res["concepts"].extend(ids)  # 添加所有拆分出的知识点
                # 复制其他字段
                for key in extend_keys:
                    if key != "concepts":
                        dextend_res.setdefault(key, [])
                        dextend_res[key].extend([dextend_infos[key][i]] * len(ids))
                # 标记第一个知识点为原始(0)，其余为重复(1)
                dextend_res["is_repeat"].extend(["0"] + ["1"] * (len(ids) - 1))
            else:
                # 如果只有一个知识点，直接添加
                for key in extend_keys:
                    dextend_res.setdefault(key, [])
                    dextend_res[key].append(dextend_infos[key][i])
                dextend_res["is_repeat"].append("0")  # 标记为原始

        # 将处理后的列表合并回字符串并存入最终结果
        for key in dextend_res:
            dres.setdefault(key, [])
            dres[key].append(",".join(dextend_res[key]))

    finaldf = pd.DataFrame(dres)
    effective_keys.add("is_repeat")  # 添加新列到有效字段集
    return finaldf, effective_keys


def id_mapping(df):
    """
    将数据集中的ID（如题目、知识点、用户ID）从原始字符串映射到从0开始的连续整数索引。

    Args:
        df (pd.DataFrame): 输入的DataFrame。

    Returns:
        tuple: 包含映射后的DataFrame和ID映射字典 (df, dkeyid2idx)。
    """
    id_keys = ["questions", "concepts", "uid"]  # 需要进行ID映射的列
    dres = dict()
    dkeyid2idx = dict()  # 存储映射关系: { 'questions': {'q1': 0, 'q2': 1}, ... }
    print(f"开始ID映射前的列名: {df.columns}")

    # 先把不需要映射的列直接复制过来
    for key in df.columns:
        if key not in id_keys:
            dres[key] = df[key]

    # 遍历每一行，对需要映射的列进行处理
    for i, row in df.iterrows():
        for key in id_keys:
            if key not in df.columns:
                continue
            dkeyid2idx.setdefault(key, dict())
            dres.setdefault(key, [])
            curids = []
            for id_val in row[key].split(","):
                # 如果ID未曾出现，分配一个新的整数索引
                if id_val not in dkeyid2idx[key]:
                    dkeyid2idx[key][id_val] = len(dkeyid2idx[key])
                curids.append(str(dkeyid2idx[key][id_val]))
            dres[key].append(",".join(curids))

    finaldf = pd.DataFrame(dres)
    return finaldf, dkeyid2idx


def train_test_split(df, test_ratio=0.2):
    """
    将数据集随机划分为训练集和测试集。

    Args:
        df (pd.DataFrame): 完整的DataFrame。
        test_ratio (float): 测试集所占的比例。

    Returns:
        tuple: (train_df, test_df)
    """
    # 设置随机种子以保证结果可复现，并打乱数据
    df = df.sample(frac=1.0, random_state=1024)
    datanum = df.shape[0]
    test_num = int(datanum * test_ratio)
    train_num = datanum - test_num
    train_df = df[0:train_num]
    test_df = df[train_num:]
    # 报告划分结果
    print(
        f"总样本数: {datanum}, 训练+验证集样本数: {train_num}, 测试集样本数: {test_num}"
    )
    return train_df, test_df


def KFold_split(df, k=5):
    """
    为数据集添加一个'fold'列，用于K折交叉验证。

    Args:
        df (pd.DataFrame): 输入的DataFrame（通常是训练+验证集）。
        k (int): 折数。

    Returns:
        pd.DataFrame: 增加了'fold'列的DataFrame。
    """
    # 打乱数据
    df = df.sample(frac=1.0, random_state=1024)
    datanum = df.shape[0]
    test_ratio = 1 / k
    test_num = int(datanum * test_ratio)
    rest = datanum % k  # 处理无法整除的余数

    start = 0
    folds = []
    # 均分数据到k个fold中
    for i in range(0, k):
        if rest > 0:
            end = start + test_num + 1
            rest -= 1
        else:
            end = start + test_num
        folds.extend([i] * (end - start))
        print(f"折-{i + 1}, 开始索引: {start}, 结束索引: {end}, 总数: {datanum}")
        start = end

    finaldf = copy.deepcopy(df)
    finaldf["fold"] = folds
    return finaldf


def apply_forget_strategy(
    df, strategy_name, forget_ratio=0.2, fold=None, **strategy_params
):
    """
    应用遗忘策略，将数据集划分为保留集和遗忘集。

    Args:
        df (pd.DataFrame): 原始数据DataFrame。
        strategy_name (str): 遗忘策略名称。
        forget_ratio (float): 遗忘比例。
        fold (int, optional): 当前fold编号（用于输出文件命名）。
        **strategy_params: 策略特定参数。

    Returns:
        dict: 包含retain_df和forget_df的字典。
    """
    if ForgetStrategy is None:
        raise ImportError(
            "遗忘策略模块(ForgetStrategy)不可用，请确保forget_strategy.py可访问。"
        )

    print(f"应用遗忘策略: {strategy_name}, 遗忘比例: {forget_ratio}")

    # 调用策略模块选择要遗忘的用户
    result = ForgetStrategy.select_forget_users(
        df, strategy=strategy_name, forget_ratio=forget_ratio, **strategy_params
    )

    forget_users = result["forget_users"]
    retain_users = result["retain_users"]
    strategy_info = result["strategy_info"]

    print(
        f"策略结果: 总用户数 {strategy_info['total_users']}, "
        f"遗忘 {strategy_info['forget_count']}, "
        f"保留 {strategy_info['retain_count']}, "
        f"实际遗忘比例 {strategy_info['actual_forget_ratio']:.3f}"
    )

    # 根据用户ID划分数据
    retain_df = df[df["uid"].isin(retain_users)].copy()
    forget_df = df[df["uid"].isin(forget_users)].copy()

    return {
        "retain_df": retain_df,
        "forget_df": forget_df,
        "strategy_info": strategy_info,
    }


def save_dcur(row, effective_keys):
    """
    一个辅助函数，将DataFrame的一行转换为一个字典，其中值为分割后的列表。
    """
    dcur = dict()
    for key in effective_keys:
        if key not in ONE_KEYS:
            dcur[key] = row[key].split(",")
        else:
            dcur[key] = row[key]
    return dcur


def generate_sequences(df, effective_keys, min_seq_len=3, maxlen=200, pad_val=-1):
    """
    将每个用户的长交互序列切分成固定长度（maxlen）的子序列。
    过长的序列被切块，过短的序列被填充。

    Args:
        df (pd.DataFrame): 输入的DataFrame。
        effective_keys (set): 有效的字段名。
        min_seq_len (int): 序列被丢弃的最小长度阈值。
        maxlen (int): 每个子序列的最大长度。
        pad_val (int): 用于填充的数值。

    Returns:
        pd.DataFrame: 处理成固定长度序列的DataFrame。
    """
    save_keys = list(effective_keys) + ["selectmasks"]
    dres = {"selectmasks": []}
    dropnum = 0  # 统计被丢弃的交互数
    
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)

        rest, lenrs = len(dcur["responses"]), len(dcur["responses"])
        j = 0
        # 如果序列长度超过maxlen，则进行切块
        while lenrs >= j + maxlen:
            rest = rest - maxlen
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    dres[key].append(",".join(dcur[key][j : j + maxlen]))
                else:  # ONE_KEYS字段直接复制
                    dres[key].append(dcur[key])
            # selectmasks全为1，表示都是有效数据
            dres["selectmasks"].append(",".join(["1"] * maxlen))
            j += maxlen

        # 剩下的部分如果太短，就丢弃
        if rest < min_seq_len:
            dropnum += rest
            continue

        # 对最后不足maxlen的部分进行填充
        pad_dim = maxlen - rest
        for key in effective_keys:
            dres.setdefault(key, [])
            if key not in ONE_KEYS:
                paded_info = np.concatenate(
                    [dcur[key][j:], np.array([pad_val] * pad_dim)]
                )
                dres[key].append(",".join([str(k) for k in paded_info]))
            else:
                dres[key].append(dcur[key])
        # selectmasks中，有效部分为1，填充部分为pad_val
        dres["selectmasks"].append(",".join(["1"] * rest + [str(pad_val)] * pad_dim))

    # 构建最终的DataFrame
    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    print(f"因序列切分后过短而丢弃的交互数: {dropnum}")
    return finaldf


def generate_window_sequences(df, effective_keys, maxlen=200, pad_val=-1):
    """
    使用滑动窗口的方式生成序列，主要用于测试。
    对于每个时间点t，生成一个以t结尾，长度为maxlen的序列。

    Args:
        df (pd.DataFrame): 输入的DataFrame。
        effective_keys (set): 有效字段名。
        maxlen (int): 窗口大小。
        pad_val (int): 填充值。

    Returns:
        pd.DataFrame: 滑动窗口生成的序列DataFrame。
    """
    save_keys = list(effective_keys) + ["selectmasks"]
    dres = {"selectmasks": []}
    
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)
        lenrs = len(dcur["responses"])
        
        if lenrs > maxlen:
            # 先处理第一个完整的窗口
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    dres[key].append(",".join(dcur[key][0:maxlen]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * maxlen))

            # 滑动窗口
            for j in range(maxlen + 1, lenrs + 1):
                for key in effective_keys:
                    dres.setdefault(key, [])
                    if key not in ONE_KEYS:
                        dres[key].append(
                            ",".join([str(k) for k in dcur[key][j - maxlen : j]])
                        )
                    else:
                        dres[key].append(dcur[key])
                # selectmasks只有最后一个位置是1，表示只预测当前时间点
                dres["selectmasks"].append(
                    ",".join([str(pad_val)] * (maxlen - 1) + ["1"])
                )
        else:
            # 如果序列长度不足maxlen，则直接填充
            pad_dim = maxlen - lenrs
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    paded_info = np.concatenate(
                        [dcur[key][0:], np.array([pad_val] * pad_dim)]
                    )
                    dres[key].append(",".join([str(k) for k in paded_info]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(
                ",".join(["1"] * lenrs + [str(pad_val)] * pad_dim)
            )

    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    return finaldf


def get_inter_qidx(df):
    """为数据集中的每一次交互分配一个全局唯一的ID。"""
    qidx_ids = []
    bias = 0  # 交互ID的偏移量
    inter_num = 0
    
    for _, row in df.iterrows():
        # 生成当前学生序列的交互ID
        ids_list = [str(x + bias) for x in range(len(row["responses"].split(",")))]
        inter_num += len(ids_list)
        ids = ",".join(ids_list)
        qidx_ids.append(ids)
        bias += len(ids_list)  # 更新偏移量
    
    # 断言检查，确保最后一个ID等于总交互数-1
    if ids_list:  # 确保列表不为空
        assert inter_num - 1 == int(ids_list[-1])

    return qidx_ids


def add_qidx(dcur, global_qidx):
    """为每次交互添加问题级别的全局ID (qidx) 和剩余重复次数 (rest)"""
    idxs, rests = [], []
    # 遍历is_repeat标志
    for r in dcur["is_repeat"]:
        # 只有原始问题（非重复）才会增加全局问题ID
        if str(r) == "0":
            global_qidx += 1
        idxs.append(global_qidx)

    # 计算每个问题在后续序列中还出现了多少次
    for i in range(0, len(idxs)):
        rests.append(idxs[i + 1 :].count(idxs[i]))
    return idxs, rests, global_qidx


def expand_question(dcur, global_qidx, pad_val=-1):
    """
    将用户的交互序列展开为以每个问题为单位的序列。
    即为每个问题生成一个包含其之前所有历史的序列。
    """
    dextend, dlast = dict(), dict()
    repeats = dcur["is_repeat"]
    last = -1
    # 添加问题ID和剩余次数
    dcur["qidxs"], dcur["rest"], global_qidx = add_qidx(dcur, global_qidx)
    
    for i in range(len(repeats)):
        # 如果是原始问题，记录下到此为止的历史
        if str(repeats[i]) == "0":
            for key in dcur.keys():
                if key in ONE_KEYS:
                    continue
                dlast[key] = dcur[key][0:i]

        # 处理第一个交互
        if i == 0:
            for key in dcur.keys():
                if key in ONE_KEYS:
                    continue
                dextend.setdefault(key, [])
                dextend[key].append([dcur[key][0]])
            dextend.setdefault("selectmasks", [])
            dextend["selectmasks"].append([pad_val])  # 第一个问题没有历史，mask为-1
        else:
            for key in dcur.keys():
                if key in ONE_KEYS:
                    continue
                dextend.setdefault(key, [])
                # 如果是连续的原始问题，则在上一序列基础上追加
                if last == "0" and str(repeats[i]) == "0":
                    dextend[key][-1] += [dcur[key][i]]
                else:  # 否则，新建一个序列，内容为历史+当前问题
                    dextend[key].append(dlast[key] + [dcur[key][i]])
            dextend.setdefault("selectmasks", [])
            # 相应地更新selectmasks
            if last == "0" and str(repeats[i]) == "0":
                dextend["selectmasks"][-1] += [1]
            elif len(dlast.get("responses", [])) == 0:  # 第一个问题
                dextend["selectmasks"].append([pad_val])
            else:
                dextend["selectmasks"].append(len(dlast["responses"]) * [pad_val] + [1])

        last = str(repeats[i])

    return dextend, global_qidx


def generate_question_sequences(
    df, effective_keys, window=True, min_seq_len=3, maxlen=200, pad_val=-1
):
    """生成以问题为单位的序列数据，用于问题级别的预测任务"""
    if "questions" not in effective_keys or "concepts" not in effective_keys:
        print(f"数据中缺少'questions'或'concepts'字段，无法生成问题序列！")
        return False, None

    save_keys = list(effective_keys) + ["selectmasks", "qidxs", "rest", "orirow"]
    dres = {}
    global_qidx = -1
    df["index"] = list(range(0, df.shape[0]))  # 记录原始行号
    
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)
        dcur["orirow"] = [row["index"]] * len(dcur["responses"])

        # 展开为问题序列
        dexpand, global_qidx = expand_question(dcur, global_qidx)
        seq_num = len(dexpand["responses"])
        
        for j in range(seq_num):
            curlen = len(dexpand["responses"][j])
            if curlen < 2:  # 不预测第一个题
                continue

            # 长度不足maxlen，进行填充
            if curlen < maxlen:
                for key in dexpand:
                    pad_dim = maxlen - curlen
                    paded_info = np.concatenate(
                        [dexpand[key][j], np.array([pad_val] * pad_dim)]
                    )
                    dres.setdefault(key, [])
                    dres[key].append(",".join([str(k) for k in paded_info]))
                for key in ONE_KEYS:
                    dres.setdefault(key, [])
                    dres[key].append(dcur[key])
            else:
                # 长度超过maxlen的处理逻辑
                if window:  # 使用滑动窗口
                    # 处理第一个完整窗口
                    if dexpand["selectmasks"][j][maxlen-1] == 1:
                        for key in dexpand:
                            dres.setdefault(key, [])
                            dres[key].append(",".join([str(k) for k in dexpand[key][j][0:maxlen]]))
                        for key in ONE_KEYS:
                            dres.setdefault(key, [])
                            dres[key].append(dcur[key])

                    # 滑动窗口处理剩余部分
                    for n in range(maxlen+1, curlen+1):
                        if dexpand["selectmasks"][j][n-1] == 1:
                            for key in dexpand:
                                dres.setdefault(key, [])
                                if key == "selectmasks":
                                    # 窗口模式：只有最后位置为1
                                    dres[key].append(",".join([str(pad_val)] * (maxlen - 1) + ["1"]))
                                else:
                                    # 提取窗口 [n-maxlen : n]
                                    dres[key].append(",".join([str(k) for k in dexpand[key][j][n-maxlen: n]]))
                            for key in ONE_KEYS:
                                dres.setdefault(key, [])
                                dres[key].append(dcur[key])
                else:  # 使用切块策略
                    k = 0
                    rest = curlen
                    while curlen >= k + maxlen:
                        rest = rest - maxlen
                        if dexpand["selectmasks"][j][k + maxlen - 1] == 1:
                            for key in dexpand:
                                dres.setdefault(key, [])
                                dres[key].append(",".join([str(s) for s in dexpand[key][j][k: k + maxlen]]))
                            for key in ONE_KEYS:
                                dres.setdefault(key, [])
                                dres[key].append(dcur[key])
                        k += maxlen
                    
                    # 处理剩余部分
                    if rest >= min_seq_len:  # 只有当剩余长度>=min_seq_len时才保留
                        # 自动跳过短序列
                        pad_dim = maxlen - rest
                        for key in dexpand:
                            dres.setdefault(key, [])
                            paded_info = np.concatenate(
                                [dexpand[key][j][k:], np.array([pad_val] * pad_dim)])
                            dres[key].append(",".join([str(s) for s in paded_info]))
                        for key in ONE_KEYS:
                            dres.setdefault(key, [])
                            dres[key].append(dcur[key])

    # 构建最终DataFrame
    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    return True, finaldf


def save_id2idx(dkeyid2idx, save_path):
    """将ID映射字典保存为JSON文件。"""
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w+", encoding="utf8") as fout:
        fout.write(json.dumps(dkeyid2idx, ensure_ascii=False, indent=4))


def write_config(
    dataset_name,
    dkeyid2idx,
    effective_keys,
    configf,
    dpath,
    k=5,
    min_seq_len=3,
    maxlen=200,
    flag=False,
    other_config={},
):
    """
    生成并写入数据集的配置文件（JSON格式）。
    """
    input_type, num_q, num_c = [], 0, 0
    if "questions" in effective_keys:
        input_type.append("questions")
        num_q = len(dkeyid2idx.get("questions", {}))
    if "concepts" in effective_keys:
        input_type.append("concepts")
        num_c = len(dkeyid2idx.get("concepts", {}))
    
    folds = list(range(0, k))
    dconfig = {
        "dpath": dpath,
        "num_q": num_q,
        "num_c": num_c,
        "input_type": input_type,
        "max_concepts": dkeyid2idx.get("max_concepts", -1),
        "min_seq_len": min_seq_len,
        "maxlen": maxlen,
        "emb_path": "",
        "train_valid_original_file": "train_valid.csv",
        "train_valid_file": "train_valid_sequences.csv",
        "folds": folds,
        "test_original_file": "test.csv",
        "test_file": "test_sequences.csv",
        "test_window_file": "test_window_sequences.csv",
    }
    dconfig.update(other_config)
    
    if flag:  # 如果生成了问题序列文件，也加入配置
        dconfig["test_question_file"] = "test_question_sequences.csv"
        dconfig["test_question_window_file"] = "test_question_window_sequences.csv"

    # 读取旧配置（如果存在），并更新或添加新数据集的配置
    try:
        with open(configf, "r", encoding="utf8") as fin:
            read_text = fin.read()
            if read_text.strip() == "":
                data_config = {dataset_name: dconfig}
            else:
                data_config = json.loads(read_text)
                if dataset_name in data_config:
                    data_config[dataset_name].update(dconfig)
                else:
                    data_config[dataset_name] = dconfig
    except FileNotFoundError:
        data_config = {dataset_name: dconfig}

    # 确保配置文件目录存在
    os.makedirs(os.path.dirname(configf), exist_ok=True)
    with open(configf, "w", encoding="utf8") as fout:
        data = json.dumps(data_config, ensure_ascii=False, indent=4)
        fout.write(data)


def calStatistics(df, stares, key):
    """计算DataFrame的统计信息。"""
    allin, allselect = 0, 0  # 总交互数，总选择数
    allqs, allcs = set(), set()  # 总问题数，总知识点数
    
    for i, row in df.iterrows():
        if not isinstance(row["responses"], str):
            continue
            
        rs = row["responses"].split(",")
        # 减去填充的部分
        curlen = len(rs) - rs.count("-1")
        allin += curlen
        
        if "selectmasks" in row and isinstance(row["selectmasks"], str):
            ss = row["selectmasks"].split(",")
            slen = ss.count("1")
            allselect += slen
            
        if "concepts" in row and isinstance(row["concepts"], str):
            cs = row["concepts"].split(",")
            fc = [c for c_list in [c.split("_") for c in cs] for c in c_list]
            curcs = set(fc) - {"-1"}
            allcs |= curcs
            
        if "questions" in row and isinstance(row["questions"], str):
            qs = row["questions"].split(",")
            curqs = set(qs) - {"-1"}
            allqs |= curqs
            
    stares.append(",".join([str(s) for s in [key, allin, df.shape[0], allselect]]))
    return allin, allselect, len(allqs), len(allcs), df.shape[0]


def get_max_concepts(df):
    """计算数据集中单个题目关联的最大知识点数量。"""
    max_concepts = 1
    for i, row in df.iterrows():
        if not isinstance(row["concepts"], str):
            continue
        cs = row["concepts"].split(",")
        num_concepts = max([len(c.split("_")) for c in cs if c != "-1"])
        if num_concepts > max_concepts:
            max_concepts = num_concepts
    return max_concepts


def main(
    dname,
    fname,
    dataset_name,
    configf,
    min_seq_len=3,
    maxlen=200,
    kfold=5,
    gen_forget_data=False,
    forget_ratio=0.2,
    **strategy_params,
):
    """
    数据预处理主函数。

    Args:
        dname (str): 输出数据文件夹路径。
        fname (str): 原始数据文件路径。
        dataset_name (str): 数据集名称。
        configf (str): 数据集配置文件路径。
        min_seq_len (int): 最小序列长度。
        maxlen (int): 最大序列长度。
        kfold (int): K折交叉验证的折数。
        gen_forget_data (bool): 是否为机器遗忘生成保留/遗忘数据集。
        forget_ratio (float): 遗忘比例。
        **strategy_params: 遗忘策略所需的额外参数。
    """
    stares = []  # 存储统计信息

    # 确保输出目录存在
    os.makedirs(dname, exist_ok=True)

    # 1. 读取数据
    print("步骤 1: 读取原始数据...")
    total_df, effective_keys = read_data(fname, min_seq_len, response_set=[0, 1])
    
    if "concepts" in effective_keys:
        max_concepts = get_max_concepts(total_df)
    else:
        max_concepts = -1

    oris, _, qs, cs, seqnum = calStatistics(total_df, stares, "原始数据")
    print(f"原始数据总交互数: {oris}, 问题数: {qs}, 知识点数: {cs}, 序列数: {seqnum}")

    # 2. 展开多知识点 & ID映射
    print("=" * 20)
    print("步骤 2: 展开多知识点并进行ID映射...")
    total_df, effective_keys = extend_multi_concepts(total_df, effective_keys)
    total_df, dkeyid2idx = id_mapping(total_df)
    dkeyid2idx["max_concepts"] = max_concepts

    extends, _, qs, cs, seqnum = calStatistics(total_df, stares, "展开多知识点后")
    print(
        f"展开多知识点后, 总交互数: {extends}, 问题数: {qs}, 知识点数: {cs}, 序列数: {seqnum}"
    )

    # 3. 保存ID映射
    print("=" * 20)
    print("步骤 3: 保存ID映射...")
    save_id2idx(dkeyid2idx, os.path.join(dname, "keyid2idx.json"))
    effective_keys.add("fold")
    config = [key for key in ALL_KEYS if key in effective_keys]

    # 4. 划分训练集和测试集
    print("=" * 20)
    print("步骤 4: 划分训练集和测试集...")
    train_df, test_df = train_test_split(total_df, 0.2)
    splitdf = KFold_split(train_df, kfold)

    # 5. 处理并保存训练验证集
    print("=" * 20)
    print("步骤 5: 生成训练验证集序列...")
    splitdf[config].to_csv(os.path.join(dname, "train_valid.csv"), index=None)
    ins, ss, qs, cs, seqnum = calStatistics(splitdf, stares, "原始训练验证集")
    print(
        f"原始训练验证集交互数: {ins}, 选择数: {ss}, 问题数: {qs}, 知识点数: {cs}, 序列数: {seqnum}"
    )

    split_seqs = generate_sequences(splitdf, effective_keys, min_seq_len, maxlen)
    ins, ss, qs, cs, seqnum = calStatistics(split_seqs, stares, "序列化训练验证集")
    print(
        f"序列化训练验证集交互数: {ins}, 选择数: {ss}, 问题数: {qs}, 知识点数: {cs}, 序列数: {seqnum}"
    )
    split_seqs.to_csv(os.path.join(dname, "train_valid_sequences.csv"), index=None)

    # 6. 处理并保存测试集
    print("=" * 20)
    print("步骤 6: 生成测试集序列...")
    test_df["fold"] = [-1] * test_df.shape[0]  # 测试集fold默认为-1
    test_df["cidxs"] = get_inter_qidx(test_df)  # 添加全局交互ID

    test_seqs = generate_sequences(
        test_df, list(effective_keys) + ["cidxs"], min_seq_len, maxlen
    )
    ins, ss, qs, cs, seqnum = calStatistics(test_df, stares, "原始测试集")
    print(
        f"原始测试集交互数: {ins}, 选择数: {ss}, 问题数: {qs}, 知识点数: {cs}, 序列数: {seqnum}"
    )

    ins, ss, qs, cs, seqnum = calStatistics(test_seqs, stares, "序列化测试集")
    print(
        f"序列化测试集交互数: {ins}, 选择数: {ss}, 问题数: {qs}, 知识点数: {cs}, 序列数: {seqnum}"
    )
    print("=" * 20)

    # 7. 生成测试集的窗口序列
    print("=" * 20)
    print("步骤 7: 生成测试集窗口序列...")
    test_window_seqs = generate_window_sequences(
        test_df, list(effective_keys) + ["cidxs"], maxlen
    )

    # 8. (可选) 生成问题级别的序列
    print("=" * 20)
    print("步骤 8: 生成问题级别序列...")
    flag, test_question_seqs = generate_question_sequences(
        test_df, effective_keys, False, min_seq_len, maxlen
    )
    flag, test_question_window_seqs = generate_question_sequences(
        test_df, effective_keys, True, min_seq_len, maxlen
    )

    # 9. 保存所有测试集文件
    print("=" * 20)
    print("步骤 9: 保存所有测试集文件...")
    test_df = test_df[config + ["cidxs"]]
    test_df.to_csv(os.path.join(dname, "test.csv"), index=None)
    test_seqs.to_csv(os.path.join(dname, "test_sequences.csv"), index=None)
    test_window_seqs.to_csv(
        os.path.join(dname, "test_window_sequences.csv"), index=None
    )

    ins, ss, qs, cs, seqnum = calStatistics(test_window_seqs, stares, "窗口化测试集")
    print(
        f"窗口化测试集交互数: {ins}, 选择数: {ss}, 问题数: {qs}, 知识点数: {cs}, 序列数: {seqnum}"
    )

    if flag:
        test_question_seqs.to_csv(
            os.path.join(dname, "test_question_sequences.csv"), index=None
        )
        test_question_window_seqs.to_csv(
            os.path.join(dname, "test_question_window_sequences.csv"), index=None
        )

        ins, ss, qs, cs, seqnum = calStatistics(
            test_question_seqs, stares, "问题序列测试集"
        )
        print(
            f"问题序列测试集交互数: {ins}, 选择数: {ss}, 问题数: {qs}, 知识点数: {cs}, 序列数: {seqnum}"
        )
        ins, ss, qs, cs, seqnum = calStatistics(
            test_question_window_seqs, stares, "问题窗口序列测试集"
        )
        print(
            f"问题窗口序列测试集交互数: {ins}, 选择数: {ss}, 问题数: {qs}, 知识点数: {cs}, 序列数: {seqnum}"
        )

    # 10. 写入最终配置文件
    print("=" * 20)
    print("步骤 10: 写入配置文件...")
    write_config(
        dataset_name=dataset_name,
        dkeyid2idx=dkeyid2idx,
        effective_keys=effective_keys,
        configf=configf,
        dpath=dname,
        k=kfold,
        min_seq_len=min_seq_len,
        maxlen=maxlen,
        flag=flag,
    )

    # 11. (可选) 生成机器遗忘数据集
    print("=" * 20)
    if gen_forget_data:
        print("=" * 20)
        print(f"步骤 11: 批量生成遗忘策略数据集，遗忘比例: {forget_ratio}")

        if ForgetStrategy is None:
            print("错误: ForgetStrategy 未导入，无法生成遗忘数据集。")
        else:
            all_strategies = [
                "random",
                "low_performance",
                "high_performance",
                "low_engagement",
                "unstable_performance",
            ]

            # 使用已划分好fold的训练验证集作为基础数据
            base_df = splitdf

            for strategy in all_strategies:
                print(f"\n--- 处理策略: {strategy} ---")
                try:
                    # 1. 应用遗忘策略
                    forget_result = apply_forget_strategy(
                        base_df, strategy, forget_ratio, **strategy_params
                    )

                    retain_df = forget_result["retain_df"]
                    forget_df = forget_result["forget_df"]

                    # 2. 为保留集和遗忘集分别生成序列
                    print("  正在为保留集生成序列...")
                    retain_seqs = generate_sequences(
                        retain_df, effective_keys, min_seq_len, maxlen
                    )

                    print("  正在为遗忘集生成序列...")
                    forget_seqs = generate_sequences(
                        forget_df, effective_keys, min_seq_len, maxlen
                    )

                    # 3. 构建文件名并保存
                    params_str = "_".join(
                        [f"{k}{v}" for k, v in sorted(strategy_params.items()) if v]
                    )
                    if params_str:
                        params_str = f"_{params_str}"

                    retain_filename = f"train_valid_sequences_retain_{strategy}_ratio{forget_ratio}{params_str}.csv"
                    forget_filename = f"train_valid_sequences_forget_{strategy}_ratio{forget_ratio}{params_str}.csv"

                    retain_path = os.path.join(dname, retain_filename)
                    forget_path = os.path.join(dname, forget_filename)

                    retain_seqs.to_csv(retain_path, index=False)
                    forget_seqs.to_csv(forget_path, index=False)

                    print(f"  已保存保留集: {retain_filename} ({len(retain_seqs)} 行)")
                    print(f"  已保存遗忘集: {forget_filename} ({len(forget_seqs)} 行)")

                except Exception as e:
                    print(f"  处理策略 {strategy} 时出错: {e}")
                    continue

            print("\n所有遗忘策略处理完成！")

    print("=" * 20)
    print("统计信息汇总:")
    print("\n".join(stares))
    print("数据预处理完成！")
    print("=" * 20)

