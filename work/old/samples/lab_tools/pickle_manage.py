import pickle

PICKLE_DIR = "./pickles"

def output_object(obj, pickle_name="pickle.pickle", pickle_dir=PICKLE_DIR):
    '''pickle形式でPythonオブジェクトを保存する
    引数:
        - obj (Object)         : 保存するPythonオブジェクト
        - pickle_name (string) : pickleファイルの名前
        - pickle_dir (string)  : pickleファイルを保存するディレクトリ
    '''
    with open(pickle_name, "wb") as p:
        pickle.dump(obj, p)

def load_pickle(pickle_file, is_load_detail=False):
    '''pickle形式のファイルを読み込む
    引数:
        - pickle_name (string) : pickleファイルのパス

    戻値:
        - obj (Object) : pickleファイルを読み込んだオブジェクト
    '''
    with open(pickle_file, "rb") as p:
        obj = pickle.load(p)

    return obj