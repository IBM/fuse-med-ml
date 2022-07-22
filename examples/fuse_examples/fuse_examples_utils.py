import os


def ask_user(yes_no_question) -> bool:
    res = ""
    while res not in ["y", "n"]:
        res = input(f"{yes_no_question}? [y/n]")
    return res == "y"


def get_fuse_examples_user_dir() -> str:
    if "USER_HOME_PATH" in os.environ:
        fuse_examples_dir = os.path.join(os.environ["USER_HOME_PATH"], "fuse_examples")
    else:
        fuse_examples_dir = "./fuse_examples"
    if not os.path.exists(fuse_examples_dir):
        os.mkdir(fuse_examples_dir)
    return fuse_examples_dir
