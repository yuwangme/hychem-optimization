def fix_dir(dir):
    ''' append backslash to dir if necessary '''
    if dir[-1] != '/':
        dir += '/'
    return dir
