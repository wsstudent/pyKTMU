import os, sys
import argparse
from pykt.preprocess.split_datasets import main as split_concept
from pykt.preprocess.split_datasets_que import main as split_question
from pykt.preprocess import data_proprocess, process_raw_data

# 修改这一行，使用 split_concept.__module__ 来找到模块名，然后用 sys.modules 获取模块对象
print(f"正在加载的 split_datasets 模块位于: {sys.modules[split_concept.__module__].__file__}")


dname2paths = {
    "assist2009": "../data/assist2009/skill_builder_data_corrected_collapsed.csv",
    "assist2012": "../data/assist2012/2012-2013-data-with-predictions-4-final.csv",
    "assist2015": "../data/assist2015/2015_100_skill_builders_main_problems.csv",
    "algebra2005": "../data/algebra2005/algebra_2005_2006_train.txt",
    "bridge2algebra2006": "../data/bridge2algebra2006/bridge_to_algebra_2006_2007_train.txt",
    "statics2011": "../data/statics2011/AllData_student_step_2011F.csv",
    "nips_task34": "../data/nips_task34/train_task_3_4.csv",
    "poj": "../data/poj/poj_log.csv",
    "slepemapy": "../data/slepemapy/answer.csv",
    "assist2017": "../data/assist2017/anonymized_full_release_competition_dataset.csv",
    "junyi2015": "../data/junyi2015/junyi_ProblemLog_original.csv",
    "ednet": "../data/ednet/",
    "ednet5w": "../data/ednet/",
    "peiyou": "../data/peiyou/grade3_students_b_200.csv"
}
config = "../configs/data_config.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset_name", type=str, default="assist2015")
    parser.add_argument("-f","--file_path", type=str, default="../data/peiyou/grade3_students_b_200.csv")
    parser.add_argument("-m","--min_seq_len", type=int, default=3)
    parser.add_argument("-l","--maxlen", type=int, default=200)
    parser.add_argument("-k","--kfold", type=int, default=5)
    # 新增参数，用于触发遗忘数据集的生成
    parser.add_argument("--gen_forget_data", action="store_true", help="If set, generate retain/forget sets for all strategies.")
    parser.add_argument("--forget_ratio", type=float, default=0.2, help="Ratio of users to forget.")
    # parser.add_argument("--mode", type=str, default="concept",help="question or concept")
    args = parser.parse_args()

    print(args)

    # process raw data
    if args.dataset_name=="peiyou":
        dname2paths["peiyou"] = args.file_path
        print(f"fpath: {args.file_path}")
    dname, writef = process_raw_data(args.dataset_name, dname2paths)
    print("-"*50)
    print(f"dname: {dname}, writef: {writef}")
    # split
    os.system("rm " + dname + "/*.pkl")

    # for concept level model
    # split_concept(dname, writef, args.dataset_name, configf, args.min_seq_len,args.maxlen, args.kfold)
    # 传递新参数
    split_concept(
        dname, 
        writef, 
        args.dataset_name, 
        config, 
        args.min_seq_len,
        args.maxlen, 
        args.kfold,
        gen_forget_data=args.gen_forget_data,
        forget_ratio=args.forget_ratio
    )
    print("="*100)

    #for question level model
    split_question(dname, writef, args.dataset_name, config, args.min_seq_len,args.maxlen, args.kfold)

