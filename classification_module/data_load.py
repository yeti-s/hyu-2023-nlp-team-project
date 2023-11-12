import os
import time
import pandas as pd

'''
load_one_labeled_data:
    *L.json(labeled dats) 파일 하나를 읽어서 데이터프레임으로 리턴하는 함수입니다.
 '''
def load_one_labeled_data(filepath: str) -> pd.DataFrame:
    df = pd.read_json(filepath)
    df_src = df['sourceDataInfo'] # source data만 읽어오기
    df_lb = df['labeledDataInfo']

    sentence_info = df_src['sentenceInfo']
    sentence_info_list = list()
    for sentence_item in sentence_info:
        sentence_info_list.append(sentence_item)

    df_src['sentenceInfo'] = pd.DataFrame(sentence_info_list) # sentenceInfo를 dataframe으로 바꾸기

    # label 값을 채우기
    df_src['clickbaitClass'] = df_lb['clickbaitClass']

    # newTitle, referSentenceInfo drop
    # df_src.drop(['newTitle', 'referSentenceInfo'], axis=1, inplace = True)

    return df_src


'''
load_subdir_labeled_data:
    labeled data가 카테고리와 생성 방법 별로(ex. EC, ET, Auto, Direct) 디렉토리가 나뉘어져 있습니다.
    그 하나의 디렉토리 안에 있는 데이터들을 데이터프레임으로 읽어서 list에 넣은 뒤 반환합니다.
 '''
def load_subdir_labeled_data(dirpath: str) -> list:
    sub_labeled_data = list()
    filenames = os.listdir(dirpath)
    # print(filenames)

    for i, json_file in enumerate(filenames):
        full_json_filepath = os.path.join(dirpath, json_file)
        # json 파일 읽기
        df = load_one_labeled_data(full_json_filepath)
        sub_labeled_data.append(df)

    return sub_labeled_data

'''
unzip_subdir_labeled_data:
    디렉토리 밑에 있는 .zip 파일을 모두 압축풀기한 후,
    압축이 풀린 디렉토리의 이름 리스트를 반환합니다.
'''
def unzip_subdir_labeled_data(dirpath: str) -> list:
    filenames = os.listdir(dirpath)

    print('[INFO] unzipping files of ' + dirpath)

    unzipped_dirnames = list()
    for i, zip_file in enumerate(filenames):
        full_zip_filepath = os.path.join(dirpath, zip_file)
        output_dirpath = full_zip_filepath.strip('.zip')
        os.system('mkdir ' + output_dirpath)

        # zip file이 아니라면 처리하지 않습니다.
        if full_zip_filepath.split('.')[-1] != 'zip':
            continue
        
        os.system('unzip \\' + full_zip_filepath + ' -x /' + ' -d '+ output_dirpath)

        unzipped_dirnames.append(output_dirpath)
    
    print('[INFO] unzip done')

    return unzipped_dirnames

    

'''
load_entire_labeled_data:
    전체 데이터를 읽습니다.
    do_unzip 값이 true라면, 하위 디렉토리의 압축을 푼 뒤 데이터를 불러오고,
    false 값이라면 압축을 푸는 작업을 생략합니다.
'''
def load_entire_labeled_data(dirpath: str, do_unzip: bool) -> pd.DataFrame:
    print("[INFO] loading labeled data...")
    start_time = time.time()
    entire_labeled_data = list()

    # 먼저 디렉토리 안에 있는 zip file의 압축을 모두 풀어줍니다.
    if do_unzip:
        unzipped_dirnames = unzip_subdir_labeled_data(dirpath)

    # 현재 디렉토리 밑에 있는 모든 하위 디렉토리를 읽어옵니다.
    sub_directories = os.listdir(dirpath)
    for i, dirname in enumerate(sub_directories):
        full_dirpath = os.path.join(dirpath, dirname)
        # print('[DEBUG line 48]: ', full_dirpath)

        # 파일은 처리하지 않습니다.
        if not os.path.isdir(full_dirpath):
            continue

        # 디렉토리라면, 그 디렉토리 안에 있는 json 파일 내용을 읽어옵니다.
        sub_labeled_data_list = load_subdir_labeled_data(full_dirpath)
        entire_labeled_data += sub_labeled_data_list
    
    entire_labeled_data_df = pd.DataFrame(entire_labeled_data)

    end_time = time.time()
    print('[INFO] program was done.')
    print('[INFO] execution time: ', end_time-start_time)

    return entire_labeled_data_df

if __name__ == '__main__':
    # 테스트를 위한 메인 함수
    current_dir = os.getcwd()
    print(current_dir)
    df = load_entire_labeled_data(current_dir+'/data/training_set', 0)