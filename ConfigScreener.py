import os
import argparse
import time
import ilock


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('script', type=str, help="Script to execute")
    parser.add_argument('folder', type=str, help="Folder with config files")
    parser.add_argument('device', type=int, help="Cuda device index")
    opt = parser.parse_args()

    assert opt.script.endswith('.py')
    assert os.path.exists(opt.script)

    print("[INFO] Launched ConfigScreener.py for training with script <{}>".format(opt.script))

    parent_folder = os.path.abspath(opt.folder)

    print(
        "[INFO] Screening parent folder <{}> with sub-folders <open>, <running>, <failed>, <done>".format(parent_folder)
    )

    sub_folder_names = {'open', 'running', 'done', 'failed'}
    sub_folders = {
        folder_name: os.path.join(os.path.abspath(opt.folder), folder_name)
        for folder_name in sub_folder_names
    }

    for folder in sub_folders.values():
        if not os.path.isdir(folder):
            print("[INFO] Creating folder <{}>".format(folder))
            os.makedirs(folder)

    device = opt.device

    print("[INFO] Running processes on CUDA device <{}>".format(device))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    cwd = os.getcwd()
    print("[INFO] Working directory: {}".format(cwd))

    repetitions = 3

    def getScripts(folder):
        scripts = [
            f for f in sorted(os.listdir(folder))
            if (os.path.isfile(os.path.join(folder, f)) and f.endswith(".json"))
        ]
        return scripts

    def renameFile(file_name, current_folder, new_folder):
        current_path =  os.path.join(current_folder, file_name)
        assert os.path.exists(current_path)
        assert os.path.isdir(new_folder)
        new_path = os.path.join(new_folder, file_name)
        if os.path.exists(new_path):
            old_file_name, suffix = file_name.split('.')
            i = 0
            while os.path.exists(os.path.join(new_folder, old_file_name + "_{}.".format(i) + suffix)):
                i += 1
            file_name = old_file_name + "_{}.".format(i) + suffix
            print("[INFO] New file name: {}".format(file_name))
            new_path = os.path.join(new_folder, file_name)
        os.rename(current_path, new_path)
        return file_name

    while True:
        print("[INFO] Screening folder <open>")
        # check for a new script file
        file = None
        with ilock.ILock("KevinConfigScreener"):
            scripts = getScripts(sub_folders['open'])
            if len(scripts) == 0:
                print("[INFO] No config files found")
                time.sleep(10)
                continue
            file = scripts[0]
            file = renameFile(file, sub_folders['open'], sub_folders['running'])
        print("[INFO] Processing file <{}>".format(file))

        # launch that file
        for _ in range(repetitions):
            time_start = time.time()
            os.system("python {} {} {}".format(opt.script, os.path.join(sub_folders['running'], file), device))
            time_end = time.time()
            time_min = (time_end - time_start) // 60
            print("[INFO] File <{}> finished after {} minutes.".format(file, time_min))

        # copy to either done or failed
        if time_min < 2:
            print("[INFO] Script took less than 2 min, probably failed")
            renameFile(file, sub_folders['running'], sub_folders['failed'])
        else:
            renameFile(file, sub_folders['running'], sub_folders['done'])

        time.sleep(5)
