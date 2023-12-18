import os

raxmlng_path = ""
predictor_path = ""


def get_difficulty(prefix):
    if not os.path.isfile(prefix):
        return float("nan")
    with open(prefix, "r") as outfile:
        lines = outfile.readlines()
        if len(lines) == 0:
            return float("nan")
        return float(lines[0])

def write_padded_msa(msa_path, outpath):
    with open(msa_path, "r") as msa_file:
        msa_string = msa_file.read()
    parts = msa_string.split("\n\n")
    lines = parts[-1].split("\n")
    block_size = len(lines[1].split(" ")[-1])
    if block_size == 10:
        padding_size = 10
        append_string = " ----------"
    else:
        padding_size = 10 - block_size
        append_string = "-" * padding_size
    if len(parts) != 1:
        msa_string = "\n\n".join(parts[:-1] + ["\n".join([line + append_string for line in lines[:-1]] + [lines[-1]])])
    else:
        msa_string = "\n".join([lines[0]] + [line + append_string for line in lines[1:-1]] + [lines[-1]])

    parts = msa_string.split("\n")
    sub_parts = parts[0].split(" ")

    msa_string = "\n".join([" ".join(sub_parts[:-1] + [str(int(sub_parts[-1]) + padding_size)])] + parts[1:])

    with open(outpath, "w+") as new_msa_file:
        new_msa_file.write(msa_string)

def run(msa_path, prefix):
    if os.path.isfile(prefix):
        print("Files with prefix " + prefix + " already exist")
        return
    d = "/".join(prefix.split("/")[:-1])
    if not os.path.isdir(d):
        os.makedirs(d)
    command = "pythia -m " + msa_path + " -o " + prefix + " -r " + raxmlng_path + " -p " + predictor_path + " --removeDuplicates -v"
    print(command)
    os.system(command)


def run_with_padding(msa_path, prefix):
    run(msa_path, prefix)
    d = get_difficulty(prefix)
    if d != d:
        if not os.path.isfile(msa_path):
            print("MSA " + msa_path + " does not exist")
            return
        os.remove(prefix)
        write_padded_msa(msa_path, "temp.phy")
        run("temp.phy", prefix)
        os.remove("temp.phy")



