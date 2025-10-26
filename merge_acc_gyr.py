import os
import re
import glob
import pandas as pd


def find_time_col(df):
    # prefer a column named 'time' (case-insensitive), else first column
    for c in df.columns:
        if c.lower() == 'time':
            return c
    return df.columns[0]


def prefix_columns(df, prefix, exclude=None):
    if exclude is None:
        exclude = []
    new_cols = {}
    for c in df.columns:
        if c in exclude:
            new_cols[c] = c
        else:
            new_cols[c] = f"{prefix}_{c}"
    return df.rename(columns=new_cols)


def split_metadata(prefix):
    # prefix e.g. 'Jean_walking11' or 'Christine_jumping1'
    parts = prefix.split('_')
    subject = parts[0] if len(parts) > 0 else ''
    activity = ''
    session = ''
    if len(parts) > 1:
        # separate trailing digits from activity
        m = re.match(r"^([A-Za-z]+)(\d*)$", parts[1])
        if m:
            activity = m.group(1).lower()
            session = m.group(2)
        else:
            activity = parts[1].lower()
    return subject, activity, session


def merge_pair(acc_file, gyr_file):
    # Read files
    acc = pd.read_csv(acc_file, low_memory=False)
    gyr = pd.read_csv(gyr_file, low_memory=False)

    acc_time_col = find_time_col(acc)
    gyr_time_col = find_time_col(gyr)

    # normalize time column name to 'time'
    acc = acc.rename(columns={acc_time_col: 'time'})
    gyr = gyr.rename(columns={gyr_time_col: 'time'})

    # ensure integer timestamps if possible
    try:
        acc['time'] = acc['time'].astype('int64')
    except Exception:
        pass
    try:
        gyr['time'] = gyr['time'].astype('int64')
    except Exception:
        pass

    # Try exact merge on time
    merged = pd.merge(acc, gyr, on='time', how='inner', suffixes=('_acc', '_gyr'))

    if merged.shape[0] == 0:
        # fallback to nearest-time merge using merge_asof
        acc_sorted = acc.sort_values('time').reset_index(drop=True)
        gyr_sorted = gyr.sort_values('time').reset_index(drop=True)
        # choose a reasonable tolerance: 5 ms in nanoseconds (5e6) if integers
        tol = None
        if pd.api.types.is_integer_dtype(acc_sorted['time'].dtype):
            tol = int(5e6)
        try:
            merged = pd.merge_asof(acc_sorted, gyr_sorted, on='time', direction='nearest', tolerance=tol, suffixes=('_acc', '_gyr'))
        except Exception:
            # last-resort outer join then drop rows without either side
            merged = pd.merge(acc, gyr, on='time', how='outer', suffixes=('_acc', '_gyr'))
            merged = merged.dropna(subset=[c for c in merged.columns if c.endswith('_acc')][:1] + [c for c in merged.columns if c.endswith('_gyr')][:1])

    return merged


def process_folder(folder):
    pattern_acc = os.path.join(folder, '*_Accelerometer.csv')
    pattern_gyr = os.path.join(folder, '*_Gyroscope.csv')
    acc_files = glob.glob(pattern_acc)
    gyr_files = glob.glob(pattern_gyr)

    # map prefix -> filepath
    def prefix_of(path):
        name = os.path.basename(path)
        if name.lower().endswith('_accelerometer.csv'):
            return name[:-len('_Accelerometer.csv')]
        if name.lower().endswith('_gyroscope.csv'):
            return name[:-len('_Gyroscope.csv')]
        return os.path.splitext(name)[0]

    acc_map = {prefix_of(f): f for f in acc_files}
    gyr_map = {prefix_of(f): f for f in gyr_files}

    common_prefixes = sorted(set(acc_map.keys()) & set(gyr_map.keys()))
    merged_list = []
    for p in common_prefixes:
        acc_file = acc_map[p]
        gyr_file = gyr_map[p]
        m = merge_pair(acc_file, gyr_file)

        # prefix acc/gyr columns to make names explicit
        # first ensure 'time' present
        if 'time' not in m.columns:
            # try to find a time-like column
            candidates = [c for c in m.columns if 'time' in c.lower()]
            if candidates:
                m = m.rename(columns={candidates[0]: 'time'})

        # prefix existing acc/gyr columns (detect by suffixes or original names)
        acc_cols = [c for c in m.columns if c.endswith('_acc')]
        gyr_cols = [c for c in m.columns if c.endswith('_gyr')]
        other_time = ['time']

        # For clarity, remove suffix from acc/gyr column names and prefix with acc_/gyr_
        rename_map = {}
        for c in acc_cols:
            base = c[:-4]
            rename_map[c] = f'acc_{base}'
        for c in gyr_cols:
            base = c[:-4]
            rename_map[c] = f'gyr_{base}'
        m = m.rename(columns=rename_map)

        # add metadata columns
        subject, activity, session = split_metadata(p)
        m['subject'] = subject
        m['activity'] = activity
        m['session'] = session
        m['prefix'] = p

        merged_list.append(m)

    if merged_list:
        combined = pd.concat(merged_list, ignore_index=True, sort=False)
    else:
        combined = pd.DataFrame()
    return combined


def main():
    root = os.getcwd()
    train_folder = os.path.join(root, 'Train_unproccessed_data')
    test_folder = os.path.join(root, 'Test_unproccessed_data')

    print('Processing Train folder...')
    train_combined = process_folder(train_folder) if os.path.isdir(train_folder) else pd.DataFrame()
    print(f'Train merged rows: {0 if train_combined.empty else train_combined.shape[0]}')
    if not train_combined.empty:
        train_combined.to_csv('combined_train_merged.csv', index=False)

    print('Processing Test folder...')
    test_combined = process_folder(test_folder) if os.path.isdir(test_folder) else pd.DataFrame()
    print(f'Test merged rows: {0 if test_combined.empty else test_combined.shape[0]}')
    if not test_combined.empty:
        test_combined.to_csv('combined_test_merged.csv', index=False)

    # write an all-in-one file
    if not train_combined.empty and not test_combined.empty:
        all_combined = pd.concat([train_combined, test_combined], ignore_index=True, sort=False)
    else:
        all_combined = train_combined if not train_combined.empty else test_combined

    if not all_combined.empty:
        all_combined.to_csv('combined_all_merged.csv', index=False)
        print(f'All merged rows: {all_combined.shape[0]}')

    print('Done. Files written: combined_train_merged.csv, combined_test_merged.csv, combined_all_merged.csv (when applicable)')


if __name__ == '__main__':
    main()
