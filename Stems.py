import pandas as pd


def find_stems(stem, raw_csv, stem_csv, row_count):
    df_find = pd.read_csv(raw_csv)
    stem_list_raw = []
    for row in df_find['Review'][:row_count]:
        words_list = row.split()
        for word in words_list:
            if stem in word:
                stem_list_raw.append(word)

    df_find_stem = pd.read_csv(stem_csv)
    stem_list = []
    for row in df_find_stem['Reviews'][:row_count]:
        words_list2 = row.split()
        for word in words_list2:
            if stem in word:
                stem_list.append(word)

    raw_set = set(stem_list_raw)
    proc_set = set(stem_list)

    return print(
        f'Unique occurrences of words containing \'{stem}\' in the first {row_count} rows of \'{raw_csv}\': {raw_set} \n'
        f'Unique occurrences of words containing \'{stem}\' in the first {row_count} rows of \'{stem_csv}\': {proc_set}')


if __name__ == "__main__":
    print(find_stems('purchas', 'car-reviews.csv', 'stemmed_reviews.csv', 20))
