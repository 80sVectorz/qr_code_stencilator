from rich import print
import traceback
import argparse
from pathlib import Path
import janky_verbosity
from qr_code_stencilator import read_qr_code, generate_raw_stencils
from paper_stencils import PaperStencilGenerator, PaperStencilGeneratorPreset
from inspect import getsourcefile
import os
from os.path import abspath
import toml
import cv2

def main():
    parser = argparse.ArgumentParser(
        prog='QR Code Stencilator',
        description='A program that computes the stencils needed to paint a QR code with two colors',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('imgpath', help=
'''The path to your QR code image.
The image should be of a standard clear blocky QR code.
Photos and stylized QR codes are not supported.
''')
    parser.add_argument('-o','--outputpath', help=
'''The dir the generated stencils will be exported to.
By default a new sub directory is created in the default outputs directory.
''')
    parser.add_argument('-p', '--preset', help=
'''Name or path of one or more preset files. Presets are stored in the presets directory.
See presets/example_preset.toml for more info.
''')
#     parser.add_argument('--instructions', help=
# '''Advanced instructions allowing for multiple outputs using different presets and preset merging.
# The instructions encapsulated in quotes. Read help/.batching_and_preset_merging.md for more info.
# ''') Will be added in v1.1
    parser.add_argument('-v','--verbosity', default='SIMPLE',choices=('ERROR','WARNING','DEBUG','SIMPLE','SILENT'), help=
'''What verbosity to run with. By default it's SIMPLE.
    ERROR   : Only show errors.
    WARNING : Show errors and warnings.
    DEBUG   : Show all messages including debug info.
    SIMPLE  : Only show messages interesting to the average user.
    SILENT  : Stay silent.
''')

    args = parser.parse_args()

    verbosity = args.verbosity
    print_error,print_warning,print_debug,print = janky_verbosity.create_log_functions(verbosity)

    target_image_file = Path(args.imgpath)
    if not target_image_file.is_file():
        print_error("ERROR: Invalid file path")
        return
        
    program_path = Path(abspath(getsourcefile(lambda:0)))
    if os.path.basename(program_path).endswith('main.py'):
        top_level_path = program_path.parent/'../..'
    elif os.path.basename(program_path).endswith('.exe'):
        top_level_path = program_path.parent

    paper_formats_file_path = top_level_path/'paper_formats.toml'
    if not paper_formats_file_path.is_file():
        print_error("ERROR: Couldn't find paper_formats.toml")
        return
    
    with open(paper_formats_file_path,'r') as f:
        paper_formats_toml = '\n'.join(f.readlines())
    
    presets_path = Path(top_level_path)/'presets'

    default_preset_file = presets_path/'default.toml'
    if not Path(default_preset_file).is_file():
        print_error("ERROR: default.toml missing")

    preset_file_paths = [default_preset_file]
    p = args.preset
    if args.preset is not None:
        target_preset_path = Path(p)
        if not target_preset_path.is_file():
            target_preset_path = presets_path/f"{p}{'.toml' if not p.endswith('.toml') else ''}"
            if not target_preset_path.is_file():
                print_error(f"ERROR: Couldn't find the requested preset '{p}'")
                return
        preset_file_paths.append(target_preset_path)
        # for p in [str(p) for p in args.presets]:
        #     target_preset_path = Path(p)
        #     if not target_preset_path.is_file():
        #         target_preset_path = presets_path/f"{p}{'.toml' if not p.endswith('.toml') else ''}"
        #         if not target_preset_path.is_file():
        #             print_error(f"ERROR: Couldn't find the requested preset '{p}'")
        #             return
        #     preset_file_paths.append(target_preset_path) For v1.1

    if args.outputpath is not None:
        output_path = Path(args.outputpath)
        if not output_path.is_dir():
            print_error(f"ERROR: Invalid output path")
            return           
    else:
        output_path = top_level_path/f"outputs/{target_image_file.name.split('.')[0]}_stencils"

    presets_toml = []
    for file_path in preset_file_paths:
        try:
            with open(file_path,'r') as f:
                presets_toml.append('\n'.join(f.readlines()))
        except:
            print_error("ERROR: Failed to load preset files")
            return

    try:
        print_debug("INFO: Reading file and parsing QR code")
        raw_qr_code = read_qr_code(target_image_file.absolute().resolve().as_posix(),verbosity=verbosity)
        assert raw_qr_code is not None
    except:
        print_error(f"ERROR: Failed to read QR code")
        return
    
    try:
        print_debug("INFO: Creating raw stencils")
        raw_stencils = generate_raw_stencils(raw_qr_code)
    except:
        print_error(f"ERROR: Failed to generate initial raw stencils")
        return
    
    psg_preset = PaperStencilGeneratorPreset.load_from_toml_multi(presets_toml)
    paper_formats = toml.loads(paper_formats_toml)
    try:
        print_debug("INFO: Creating PSG object")
        psg = PaperStencilGenerator(psg_preset,paper_formats)
    except Exception as e:
        print_error(f"ERROR: Failed to create PSG object")
        print_debug(traceback.format_exc())
        return

    paper_stencils = []
    for i,s in enumerate(raw_stencils):
        try:
            print_debug(f"INFO: Generating paper stencil #{i+1}")
            paper_stencils.append(psg.generate_stencil(s,i))
        except Exception as e:
            print_error(f"ERROR: Failed to generate paper stencil #{i+1}")
            print_debug(traceback.format_exc())
            return

    if args.outputpath is None:
        output_path.mkdir(parents=True,exist_ok=True)
    for i,stencil in enumerate(paper_stencils):
        try:
            stencil_file_path = output_path/f"stencil{i}.png"
            cv2.imwrite(stencil_file_path.resolve(),stencil*255)
        except Exception as e:
            print_error(f"ERROR: Failed to export paper stencil #{i+1}")
            print_debug(traceback.format_exc())
            return
    
    print("INFO: Stencil generation completed successfully:")
    print(f"INFO: #Stencils required for perfect recreation : {len(paper_stencils)}")
    print(f"INFO: Stencils have been exported to {output_path}")

    try:
        os.startfile(output_path)
    except:
        pass
    
if __name__ == "__main__":
    main()
