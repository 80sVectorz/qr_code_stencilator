import os
import sys
import PyInstaller.__main__
from pathlib import Path
import shutil

def build(program_path:Path, dist_path:Path, version:str):
    on_windows = (os.name == 'nt')

    version_file = dist_path/'VERSION.txt'
    if version_file.exists():
        os.remove(version_file)
        
    with open(version_file,'w') as f:
        f.write(version)

    PyInstaller.__main__.run([
        './src/qr_code_stencilator/main.py',
        '--onefile',
        '--noconfirm',
        '--nowindow',
        '--name=qr_code_stencilator',
        f'--distpath={dist_path.as_posix()}'
    ])

    shutil.copy2(program_path/'paper_formats.toml', dist_path)

    presets_path = program_path/'presets/'
    dist_presets_path = dist_path/'presets/'
    dist_presets_path.mkdir(exist_ok=True)

    shutil.copy2(presets_path/'default.toml', dist_presets_path)
    shutil.copy2(presets_path/'example_preset.toml', dist_presets_path)

    dist_outputs_path = dist_path/'outputs/'
    dist_outputs_path.mkdir(exist_ok=True)

    shutil.copy2(program_path/'README.md', dist_path)

    documentation_path = program_path/'documentation/'
    dist_documentation_path = dist_path/'documentation/'

    shutil.copytree(documentation_path,dist_documentation_path, dirs_exist_ok=True)
    shutil.rmtree(dist_documentation_path/'example_gif_frames/')

    if on_windows:
        os.system(f'attrib +h "{version_file}"')
        os.system(f'attrib +h "{dist_presets_path/"default.toml"}"')

if __name__ == "__main__":
    version = sys.argv[-1]

    dist_path = Path(f'./dist/qr_code_stencilator_{version}')
    dist_path.mkdir(parents=True,exist_ok=True)

    program_path = Path(os.getcwd())

    build(program_path, dist_path, version)