import shutil

cfg_path = 'training1.yaml'
save_path = '/dev/null/'  # Corrected path without the trailing slash

# Correct usage to copy and discard the file content
shutil.copyfile(cfg_path, save_path)

