from pathlib import Path
import re

import pandas as pd


# ============================================================
# Purpose
# ============================================================
# This script merges all CRMLSSold*.csv files from the raw folder.
#
# The raw files do not all have exactly the same columns:
# - Some months have latfilled and lonfilled.
# - Some months have ListAgentAOR and BuyerAgentAOR.
# - Some old compensation columns disappear later.
#
# To avoid losing any useful field, this script first collects the full list of column names from all files. Then each CSV is re-aligned to that full
# column list before merging. If a file does not have a column, pandas fills that column with missing values for that file's rows.
#
# One output file is created:
# data/merged_crmls_sold.csv


PROJECT_FOLDER = Path(__file__).resolve().parent.parent

# All original CSV files are stored here.
RAW_FOLDER = PROJECT_FOLDER / "raw"

# The merged output file will be stored in this folder.
DATA_FOLDER = PROJECT_FOLDER / "data"

# The final merged CSV file.
MERGED_FILE = DATA_FOLDER / "merged_crmls_sold.csv"


def make_month_list(start_date, end_date):
    """
    Make a list of months between two dates.

    Example:
    start_date = "20220101"
    end_date = "20220331"
    result = ["2022-01", "2022-02", "2022-03"]

    This is needed because one raw file covers 2022-01 to 2023-12.
    """
    months = []

    year = int(start_date[:4])
    month = int(start_date[4:6])
    end_year = int(end_date[:4])
    end_month = int(end_date[4:6])

    while year < end_year or (year == end_year and month <= end_month):
        month_text = str(year) + "-" + str(month).zfill(2)
        months.append(month_text)

        month = month + 1

        if month == 13:
            month = 1
            year = year + 1

    return months


def get_periods_from_file_name(file_name):
    """
    Get the month or month range from a raw file name.

    Examples:
    CRMLSSold202401_filled.csv -> ["2024-01"]
    CRMLSSold20220101_20231231_filled.csv -> ["2022-01", ..., "2023-12"]
    """
    range_match = re.search(r"CRMLSSold(\d{8})_(\d{8})", file_name)

    if range_match:
        start_date = range_match.group(1)
        end_date = range_match.group(2)
        return make_month_list(start_date, end_date)

    month_match = re.search(r"CRMLSSold(\d{6})", file_name)

    if month_match:
        month_text = month_match.group(1)
        return [month_text[:4] + "-" + month_text[4:6]]

    return [file_name]


def get_period_label(periods):
    """
    Turn a period list into one label.

    If a file covers only one month, return that month.
    If a file covers many months, return a range label.
    """
    if len(periods) == 1:
        return periods[0]

    return periods[0] + "_to_" + periods[-1]


def get_sort_value(file_name):
    """
    Create a sortable value from a file name.

    This keeps the files in time order before merging.
    """
    range_match = re.search(r"CRMLSSold(\d{8})_(\d{8})", file_name)

    if range_match:
        return range_match.group(1)

    month_match = re.search(r"CRMLSSold(\d{6})", file_name)

    if month_match:
        return month_match.group(1) + "01"

    return file_name


def get_raw_files():
    """
    Find all CRMLSSold*.csv files in the raw folder.
    """
    raw_files = list(RAW_FOLDER.glob("CRMLSSold*.csv"))
    raw_files = sorted(raw_files, key=lambda file_path: get_sort_value(file_path.name))
    return raw_files


def read_header(file_path):
    """
    Read only the header row of one CSV file.

    nrows=0 means pandas only reads the column names.
    This is much faster than reading the whole file.
    """
    sample_data = pd.read_csv(file_path, nrows=0, encoding="utf-8-sig")
    return list(sample_data.columns)


def build_file_info(raw_files):
    """
    Build a simple information list for all raw files.

    Each item stores:
    - file path
    - file name
    - month information
    - column names
    """
    file_info = []

    for file_path in raw_files:
        columns = read_header(file_path)
        periods = get_periods_from_file_name(file_path.name)

        one_file = {
            "file_path": file_path,
            "file_name": file_path.name,
            "periods": periods,
            "period_label": get_period_label(periods),
            "columns": columns,
        }

        file_info.append(one_file)

    return file_info


def build_month_info(file_info):
    """
    Build monthly schema information.

    The 2022-2023 file is one CSV file, but it covers many months.
    For schema checking, this function expands it into one row per month.
    """
    month_info = []

    for one_file in file_info:
        for period in one_file["periods"]:
            one_month = {
                "period": period,
                "file_name": one_file["file_name"],
                "columns": one_file["columns"],
            }

            month_info.append(one_month)

    return month_info


def build_all_columns(file_info):
    """
    Build the full column list from all raw files.

    The order is simple:
    - Start with the columns from the first file.
    - When a new column appears in a later file, add it to the end.
    """
    all_columns = []

    for one_file in file_info:
        for column in one_file["columns"]:
            if column not in all_columns:
                all_columns.append(column)

    return all_columns


def print_field_changes(month_info):
    """
    Print monthly added and deleted fields.

    This keeps the useful checking step from the earlier script, but it does
    not create any extra report files. The only saved output is the merged CSV.
    """
    previous_columns = None

    print("")
    print("Monthly field changes:")

    for one_month in month_info:
        current_columns = set(one_month["columns"])

        if previous_columns is None:
            print(one_month["period"] + ": baseline columns = " + str(len(current_columns)))
        else:
            added_fields = sorted(current_columns - previous_columns)
            deleted_fields = sorted(previous_columns - current_columns)

            if len(added_fields) > 0 or len(deleted_fields) > 0:
                print(one_month["period"] + " | " + one_month["file_name"])

                if len(added_fields) > 0:
                    print("  added: " + ", ".join(added_fields))

                if len(deleted_fields) > 0:
                    print("  deleted: " + ", ".join(deleted_fields))

        previous_columns = current_columns


def get_source_period(one_file, data):
    """
    Create the source_period column for the merged data.

    For normal monthly files, the month comes from the file name.

    For the big 2022-2023 file, the file name only gives a date range.
    In that case, use CloseDate to assign each row to its actual month.
    If CloseDate is missing or cannot be parsed, use the range label instead.
    """
    if len(one_file["periods"]) == 1:
        return one_file["period_label"]

    if "CloseDate" in data.columns:
        close_dates = pd.to_datetime(data["CloseDate"], errors="coerce")
        source_period = close_dates.dt.strftime("%Y-%m")
        source_period = source_period.fillna(one_file["period_label"])
        return source_period

    return one_file["period_label"]


def merge_datasets(file_info, all_columns):
    """
    Read all CSV files, align columns, and merge them.
    """
    dataframes = []

    for one_file in file_info:
        print("Reading " + one_file["file_name"])

        data = pd.read_csv(one_file["file_path"], encoding="utf-8-sig", low_memory=False)

        # Reindex aligns the dataframe to the full column list.
        # Columns missing from this file become empty values.
        data = data.reindex(columns=all_columns)

        # These two extra columns help us know where each row came from.
        data.insert(0, "source_period", get_source_period(one_file, data))
        data.insert(1, "source_file", one_file["file_name"])

        dataframes.append(data)

    merged_data = pd.concat(dataframes, ignore_index=True, sort=False)
    return merged_data


def main():
    """
    Main workflow:
    1. Create the data folder if needed.
    2. Read all raw file names and headers.
    3. Print monthly field changes.
    4. Merge all data by column name.
    5. Save only data/merged_crmls_sold.csv.
    """
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    raw_files = get_raw_files()

    if len(raw_files) == 0:
        print("No CRMLSSold*.csv files found in " + str(RAW_FOLDER))
        return

    print("Found " + str(len(raw_files)) + " raw files.")

    file_info = build_file_info(raw_files)
    month_info = build_month_info(file_info)
    all_columns = build_all_columns(file_info)

    print_field_changes(month_info)

    print("")
    print("Merging files...")
    merged_data = merge_datasets(file_info, all_columns)

    merged_data.to_csv(MERGED_FILE, index=False, encoding="utf-8-sig")

    print("")
    print("Done.")
    print("Merged rows: " + str(len(merged_data)))
    print("Original fields kept: " + str(len(all_columns)))
    print("Extra source columns added: source_period, source_file")
    print("Output file: " + str(MERGED_FILE))


if __name__ == "__main__":
    main()
