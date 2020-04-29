"""
file manager utility functions
"""

import pickle, tqdm
from src import Shaper, Timer


def _prepare_export_val(sales, id, iteration, start, finish):
    delim_c = ","

    row_pr = delim_c.join([str(i) for i in sales[start:finish]])
    row = delim_c.join([id, row_pr])

    if iteration == 0:
        HEADER = "id,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18,F19,F20,F21,F22,F23,F24,F25,F26,F27,F28"
        delim_n = "\n"
        row = delim_n.join([HEADER, row])

    return row


def _export(pred_sales_l, ids, csv_fp_out):

    print("\nstarting to export validation...")
    row_str = ""
    for i, each_item in tqdm.tqdm(enumerate(pred_sales_l)):
        row_str += _prepare_export_val(each_item, ids[i], i, 0, 28)
        row_str += "\n"

    print("\nwriting out csv file...")
    with open(csv_fp_out , "w") as pred_f:
        pred_f.writelines(row_str)

    print("\nstarting to export evaluation...")

    row_str = ""
    for i, each_item in tqdm.tqdm(enumerate(pred_sales_l)):
        row_str += _prepare_export_val(each_item, ids[i].replace('validation', 'evaluation'), 1,
                                                 28, 57)
        row_str += "\n"
        i += 1

    with open(csv_fp_out, "a") as pred_f:
        pred_f.writelines(row_str)

    print("done")
    print("csv file at: %s" %(csv_fp_out))


def export():
    date_str = Timer.get_timestamp_str()
    csv_fp_out = "predictions/sales_predictions_%s.csv" %(date_str)

    val = pickle.load(open("../objs/eda_objs/1", "rb"))
    ids = val['id']
    fb_predictions_l = Shaper.load("eda_objs/s_predt_prophet_no_holiday_20_4_19.pkl")

    _export(fb_predictions_l, ids, csv_fp_out)

export()