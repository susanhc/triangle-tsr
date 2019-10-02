import argparse

def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('filename',nargs=2,type=int)
    args = vars(ap.parse_args())
    return args

if __name__=='__main__':
    args = args_parse()