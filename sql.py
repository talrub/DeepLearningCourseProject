import sqlite3 as sq
import secrets
import time
import statistics
from collections import defaultdict, OrderedDict
import random
from tabulate import tabulate
from fastargs.decorators import param, section


def update_model_stats_table_sql_script(model_id, data_seed, training_seed, num_training_samples, loss_bin_l,
                                        loss_bin_u, test_acc, perfect_models_percentage, train_time, perfect_model_count, tested_model_count,
                                        save_path, status):
    return f"""
    REPLACE INTO model_stats VALUES ( '{model_id}', {data_seed}, {training_seed}, {num_training_samples}, {loss_bin_l}, {loss_bin_u}, {test_acc}, {perfect_models_percentage}, {train_time}, {perfect_model_count}, {tested_model_count}, '{save_path}', '{status}' )"""


def create_model_stats_table_sql_script():
    return f"""CREATE TABLE IF NOT EXISTS model_stats (
	model_id TEXT PRIMARY KEY,
    data_seed INTEGER,
    training_seed INTEGER,
    num_train_samples INTEGER,
    loss_bin_l REAL,
    loss_bin_u REAL,
    test_acc REAL,
    perfect_models_percentage REAL,
    train_time REAL,
    perfect_model_count INTEGER,
    tested_model_count INTEGER,
    save_path TEXT,
    status TEXT);"""


def create_model_stats_table(db_path):
    con = sq.connect(db_path, isolation_level="EXCLUSIVE")
    con.execute(create_model_stats_table_sql_script())
    con.commit()
    con.close()


def update_model_stats_table(db_path, model_id, data_seed, training_seed, num_training_samples, loss_bin_l, loss_bin_u,
                             test_acc, perfect_models_percentage, train_time, perfect_model_count, tested_model_count, save_path, status):
    con = sq.connect(db_path)
    cur = con.cursor()
    cur.execute(
        update_model_stats_table_sql_script(model_id, data_seed, training_seed, num_training_samples,
            loss_bin_l, loss_bin_u, test_acc, perfect_models_percentage, train_time, perfect_model_count, tested_model_count, save_path, status)
    )
    # update model_stats table
    con.commit()
    con.close()


def delete_all_records_from_model_stats(db_path):
    con = sq.connect(db_path)
    cur = con.cursor()
    cur.execute("DELETE FROM model_stats")
    con.commit()
    print('All rows in model_stats were deleted.')
    con.close()


def set_all_combination_records_status_to_FAILED(db_path, num_train_samples, loss_bin_l, loss_bin_u):
    con = sq.connect(db_path)
    cur = con.cursor()
    update_query = """
    UPDATE model_stats
    SET status = 'FAILED', test_acc=0, perfect_models_percentage=0, perfect_model_count=0
    WHERE num_train_samples = ? AND loss_bin_l = ? AND loss_bin_u = ?
    """
    cur.execute(update_query, (num_train_samples, loss_bin_l, loss_bin_u))
    con.commit()
    con.close()
    print(f'The status of all records of combination:({num_train_samples},{loss_bin_l},{loss_bin_u}) was set to FALIED')


# def set_all_combination_records_status_to_FAILED(db_path,num_train_samples,loss_bin_l,loss_bin_u):
#   con = sq.connect(db_path)
#   cur = con.cursor()
#   update_query = """
#     UPDATE model_stats
#     SET status = 'FAILED', test_acc=0, perfect_model_count=0
#     WHERE num_train_samples = ? AND loss_bin_l = ? AND loss_bin_u = ?
#     """
#   cur.execute(update_query, (num_train_samples, loss_bin_l, loss_bin_u))
#   con.commit()
#   con.close()
#   print('The status of all records of combination:({num_train_samples},{loss_bin_l},{loss_bin_u}) was set to FALIED')


def get_model_stats_summary_sql_script():  # AVG(test_Acc) - Average test accuracy for each combination. We will use 'GROUP BY' only when we want to calculate 'SUM'/'AVG' etc.
    return """
    SELECT
        num_train_samples, loss_bin_l, loss_bin_u,
        SUM(perfect_model_count),
        AVG(test_acc),
        AVG(perfect_models_percentage),
        AVG(train_time)
    FROM
        model_stats
        model_stats
    WHERE 
        status = 'COMPLETE'
    GROUP BY
        num_train_samples, loss_bin_l, loss_bin_u
    ;"""


def get_model_FAILED_stats_summary_sql_script():
    return """
    SELECT
        num_train_samples, loss_bin_l, loss_bin_u
    FROM
        model_stats
    WHERE 
        status = 'FAILED'
    ;"""


def get_model_group_FAILED_stats_summary_sql_script():
    return """
    SELECT
        num_train_samples, loss_bin_l, loss_bin_u, SUM(perfect_model_count)
    FROM
        model_stats
    WHERE 
        status = 'FAILED'
    GROUP BY
        num_train_samples, loss_bin_l, loss_bin_u
    ;"""


def get_model_stats_summary(db_path, verbose=True, return_print=False):
    con = sq.connect(db_path)
    rows = con.execute(get_model_stats_summary_sql_script()).fetchall()
    if verbose and not return_print:
        print(tabulate(rows, headers=['num_train_samples', 'loss_bin_l', "loss_bin_u", "SUM(perfect_model_count)", "AVG(test_acc)", "AVG(perfect_models_percentage)", "AVG(train_time)"], tablefmt='psql'))
    elif verbose and return_print:
        return tabulate(rows, headers=['num_train_samples', 'loss_bin_l', "loss_bin_u", "SUM(perfect_model_count)", "AVG(test_acc)", "AVG(perfect_models_percentage)","AVG(train_time)"], tablefmt='psql')
    con.close()
    return rows


def get_model_FAILED_stats_summary(db_path, return_print=False):
    con = sq.connect(db_path)
    rows = con.execute(get_model_group_FAILED_stats_summary_sql_script()).fetchall()
    if not return_print:
        print(tabulate(rows, headers=['num_train_samples', 'loss_bin_l', "loss_bin_u", "SUM(perfect_model_count)"],
                       tablefmt='psql'))
    else:
        return tabulate(rows, headers=['num_train_samples', 'loss_bin_l', "loss_bin_u", "SUM(perfect_model_count)"],
                        tablefmt='psql')
    con.close()


def get_stds_of_avg_acuuracies(db_path, return_print=False):
    con = sq.connect(db_path)
    cur = con.cursor()
    rows = cur.execute("SELECT num_train_samples, loss_bin_l, loss_bin_u, test_acc FROM model_stats WHERE status = 'COMPLETE'").fetchall()
    std_dict = {}
    output_str = ""
    for row in rows:
        num_train_samples, loss_bin_l, loss_bin_u, test_acc = row  # test_acc is actually avg_test_acc over 'target_model_count_subrun' accuracies
        group_key = (num_train_samples, loss_bin_l, loss_bin_u)
        if group_key not in std_dict:
            std_dict[group_key] = []
        std_dict[group_key].append(test_acc)
    sorted_std_dict = OrderedDict(sorted(std_dict.items()))
    for group_key, test_acc_list in sorted_std_dict.items():
        if len(test_acc_list) > 1:
            print(f"DEBUG: get_stds_of_avg_acuuracies: len(test_acc_list)={len(test_acc_list)} std={statistics.stdev(test_acc_list)} normalize_std={statistics.stdev(test_acc_list) / len(test_acc_list)}")
            std = statistics.stdev(test_acc_list) / len(test_acc_list)
            curr_str = f"num_train_samples:{group_key[0]} Train Loss:({group_key[1]},{group_key[2]}) Test Accuracy STD:{std}"
        else:
            curr_str = f"num_train_samples:{group_key[0]} Train Loss:({group_key[1]},{group_key[2]}) can't calculate std for {len(test_acc_list)} test accuracies"
        print(curr_str)
        if return_print:
            output_str += curr_str + "\n"
    return output_str


def get_model_stats(db_path):
    con = sq.connect(db_path)
    rows = con.execute("SELECT * FROM model_stats").fetchall()

    con.close()
    return rows


@section('distributed')
@param('training_seed')
@param('data_seed')
def get_next_config(db_path, loss_bins, num_samples, training_seed=None, data_seed=None):
    # this function has to figure out the loss bins, num_samples, and data_seed it need to select
    # loss bins and num samples will depend on two things
    # 1. the number of models within the combination of loss bins and num samples with status complete
    # 2. the latest tested data_seed
    # we will focus on the first one
    # we will first query a summary table from the model stats table
    # we will then find the combination with the lowest number of models
    con = sq.connect(db_path)
    con.execute("BEGIN EXCLUSIVE")
    failed_rows = con.execute(get_model_FAILED_stats_summary_sql_script()).fetchall()
    failed_combinations_dict = {}
    for failed_row in failed_rows:
        failed_combinations_dict[(failed_row[0], failed_row[1], failed_row[2])] = True
    rows = con.execute(get_model_stats_summary_sql_script()).fetchall()
    model_cnt_dict = defaultdict(int)
    for row in rows:
        model_cnt_dict[(row[0], row[1], row[2])] = row[3]
    min_cnt = float('inf')
    for loss_bin in loss_bins:
        for num_sample in num_samples:
            if (num_sample, loss_bin[0], loss_bin[1]) in failed_combinations_dict:
                continue
            model_cnt = model_cnt_dict[(num_sample, loss_bin[0], loss_bin[1])]
            if model_cnt < min_cnt:
                min_cnt = model_cnt
                next_loss_bin = loss_bin
                next_num_sample = num_sample
    rows = con.execute("""
    SELECT
        MAX(data_seed),
        MAX(training_seed)
    FROM
        model_stats
    ;""")
    data_seed_next, training_seed_next = rows.fetchone()

    loss_bin_l, loss_bin_u = next_loss_bin
    model_id = secrets.token_hex(8)  # generating 8-bytes of random text string in hexadecimal as model_id
    data_seed_next = 100 if data_seed_next is None else data_seed_next + 1
    training_seed_next = 200 if training_seed_next is None else training_seed_next + 1
    num_train_samples = next_num_sample
    test_acc = -999
    perfect_models_percentage = -999
    train_time = -999
    tested_model_count = -999
    perfect_model_count = -999

    save_path = ""
    status = "PENDING"

    con.execute(
        update_model_stats_table_sql_script(model_id, data_seed_next, training_seed_next, num_train_samples, loss_bin_l,
                                            loss_bin_u, test_acc, perfect_models_percentage, train_time, perfect_model_count, tested_model_count,
                                            save_path, status))
    con.commit()
    con.close()
    return model_id, next_loss_bin, next_num_sample, data_seed_next, training_seed_next, min_cnt


def get_failed_combinations_dict(db_path):
    con = sq.connect(db_path)
    con.execute("BEGIN EXCLUSIVE")
    failed_rows = con.execute(get_model_FAILED_stats_summary_sql_script()).fetchall()
    failed_combinations_dict = {}
    for failed_row in failed_rows:
        failed_combinations_dict[(failed_row[0], failed_row[1], failed_row[2])] = True
    con.commit()
    con.close()
    return failed_combinations_dict


def get_combination_model_count(db_path, num_sample, loss_bin):
    con = sq.connect(db_path)
    con.execute("BEGIN EXCLUSIVE")
    rows = con.execute(get_model_stats_summary_sql_script()).fetchall()
    con.commit()
    con.close()
    for row in rows:
        if row[0] == num_sample and row[1] == loss_bin[0] and row[2] == loss_bin[1]:
            print(f"row[3]={row[3]}")
            return row[3]
    return 0


def get_next_config_same_seeds(db_path, loss_bin, num_samples, data_seed, training_seed):
    combination_model_count = get_combination_model_count(db_path, num_samples, loss_bin)
    con = sq.connect(db_path)
    con.execute("BEGIN EXCLUSIVE")
    loss_bin_l, loss_bin_u = loss_bin
    model_id = secrets.token_hex(8)  # generating 8-bytes of random text string in hexadecimal as model_id
    test_acc = -999
    perfect_models_percentage = -999
    train_time = -999
    tested_model_count = -999
    perfect_model_count = -999
    save_path = ""
    status = "PENDING"
    con.execute(update_model_stats_table_sql_script(model_id, data_seed, training_seed, num_samples, loss_bin_l, loss_bin_u,
                                            test_acc, perfect_models_percentage, train_time, perfect_model_count, tested_model_count, save_path,
                                            status))
    con.commit()
    con.close()
    return model_id, combination_model_count


if __name__ == "__main__":
    from fastargs import Section, Param

    Section("distributed").params(
        loss_thres=Param(str, default="0.3,0.4,0.5"),
        num_samples=Param(str, default="2,4,8"),
        excluded_cells=Param(str, default="", desc='ex: 32_(0.3, 0.35)/16_(0.3, 0.35)'),
        target_model_count_subrun=Param(int, default=1),
        training_seed=Param(int, default=None,
                            desc='If there is no training seed, then the training seed increment with every new runs'),
        data_seed=Param(int, default=None, desc='If there is no data seed, then the training seed increment with every new runs, otherwise, it is fix')
    )
    db_path = "tutorial.db"
    create_model_stats_table(db_path)

    loss_bins = [(0.1, 0.2), (0.2, 0.3)]
    num_samples = [16, 32]
    for i in range(10):
        print(i)
        next_config = get_next_config(db_path, [(0.1, 0.2), (0.2, 0.3)], [16, 32])
        model_id, (loss_bin_l, loss_bin_u), num_training_samples, data_seed, training_seed, _ = next_config
        print("next config: ", next_config)

        # simulated training steps
        test_acc = random.random()
        train_time = random.random() * 1000
        tested_model_count = int(random.random() * 200)
        save_path = f"path/to/model_{data_seed}_{training_seed}_{loss_bin_l},{loss_bin_u}_{num_training_samples}"
        status = "COMPLETE"
        update_model_stats_table(db_path, model_id, data_seed, training_seed, num_training_samples, 1,
                                 loss_bin_l, loss_bin_u, test_acc, train_time, tested_model_count, save_path, status)

    rows = get_model_stats(db_path)
    unique_data_seed = set()
    for row in rows:
        if row[1] in unique_data_seed:
            print("duplicate data seed!", row[1])
        else:
            unique_data_seed.add(row[1])
    print("program finished")
    time.sleep(10)
