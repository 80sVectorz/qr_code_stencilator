import os
import sys
import PyInstaller.__main__

VERSION = sys.argv[-1]
DIST_PATH = f".\\dist\\qr_code_stencilator_{VERSION}"
VERSION_FILE = f'{DIST_PATH}\\VERSION.txt'

os.system(f'mkdir "{DIST_PATH}"')

try:
    with open(VERSION_FILE,'w') as f:
        f.write(VERSION)
except:
    pass

os.system(f'attrib +h "{VERSION_FILE}"')

PyInstaller.__main__.run([
    './src/qr_code_stencilator/main.py',
    '--onefile',
    '--noconfirm',
    '--nowindow',
    '--name=qr_code_stencilator',
    f'--distpath={DIST_PATH.replace('\\','/')}'
])

os.system(f'copy paper_formats.toml {DIST_PATH}')
os.system(f'mkdir {DIST_PATH}\\presets')
os.system(f'copy presets\\default.toml {DIST_PATH}\\presets')
os.system(f'copy presets\\example_preset.toml {DIST_PATH}\\presets')
os.system(f'mkdir {DIST_PATH}\\outputs')
