import subprocess


def get_commit_id():
    try:
        return (
            subprocess.check_output("git rev-parse HEAD", shell=True).decode().strip()
        )
    except:
        return None


def get_branch_name():
    try:
        return (
            subprocess.check_output("git rev-parse --abbrev-ref HEAD", shell=True)
            .decode()
            .strip()
        )
    except:
        return None
