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
    df_src['sentenceInfo'] = pd.DataFrame(df_src['sentenceInfo']) # sentenceInfo를 dataframe으로 바꾸기

    # label 값을 채우기
    df_src['newTitle'] = df_lb['newTitle']
    df_src['clickbaitClass'] = df_lb['clickbaitClass']
    df_src['referSentenceInfo'] = df_lb['referSentenceInfo']

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
        # print('[DEBUG line 33]: ', full_json_filepath)
        # json 파일 읽기
        df = load_one_labeled_data(full_json_filepath)
        sub_labeled_data.append(df)

    return sub_labeled_data

'''
unzip_subdir_labeled_data:
    디렉토리 밑에 있는 .zip 파일을 모두 해제하는 함수
'''
def unzip_subdir_labeled_data(dirpath: str) -> list:
    filenames = os.listdir(dirpath)

    print('[INFO] unzipping files of ' + dirpath)

    for i, zip_file in enumerate(filenames):
        full_zip_filepath = os.path.join(dirpath, zip_file)
        output_dirpath = '.' + full_zip_filepath.strip('.zip')
        os.system('mkdir ' + output_dirpath)

        # zip file이 아니라면 처리하지 않습니다.
        if full_zip_filepath.split('.')[-1] != 'zip':
            continue
        
        print(full_zip_filepath, output_dirpath)

        # os.system('unzip ' + full_zip_filepath + ' -x /' + ' -d '+ output_dirpath)
    
    print('[INFO] unzip done')

    

'''
load_entire_labeled_data:
    전체 데이터를 읽습니다.
'''
def load_entire_labeled_data(dirpath: str) -> pd.DataFrame:
    entire_labeled_data = list()

    # 먼저 디렉토리 안에 있는 zip file의 압축을 모두 풀어줍니다.
    unzip_subdir_labeled_data(dirpath)

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
    return entire_labeled_data_df

if __name__ == '__main__':
    start_time = time.time()
    # 테스트를 위한 메인 함수
    df = load_entire_labeled_data('./data/training_test')

    print('[INFO] program was done.')
    print(len(df))

    end_time = time.time()
    print('[INFO] execution time: ', end_time-start_time)