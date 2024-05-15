def main():
    print("hello")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    else:
        import subprocess

        # import getpass

        # Your program logic here

        # Prompt the user for their sudo password
        sudo_password = "nope"

        # Create the command to shut down the computer
        shutdown_command = f"nvidia-smi"

        # Execute the command
        print("Running shutdown command")
        subprocess.run(shutdown_command, shell=True, check=True)
        print("Shutdown command run, can still be canceled")
