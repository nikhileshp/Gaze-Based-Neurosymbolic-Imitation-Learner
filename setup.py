from setuptools import setup, find_packages
import os


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths


# Collect lark grammar files bundled with nsfr
nsfr_lark_files = package_files('core/nsfr/lark')

setup(
    name='grail',
    version='0.1.0',
    description='Gaze-Based Neurosymbolic Imitation Learner',
    author='Nikhilesh Prabhakar',
    python_requires='>=3.9',

    # Robust programmatic package discovery
    # We want nsfr, ocatari, and envs to be top-level aliases of their core/ counterparts.
    # We also want core and scripts to be available as top-level packages.
    packages=(
        ['nsfr', 'ocatari', 'envs', 'core', 'scripts'] +
        ['nsfr.' + p for p in find_packages(where='core/nsfr')] +
        ['ocatari.' + p for p in find_packages(where='core/ocatari')] +
        ['envs.' + p for p in find_packages(where='core/envs')] +
        ['core.' + p for p in find_packages(where='core')] +
        ['scripts.' + p for p in find_packages(where='scripts')]
    ),
    package_dir={
        'nsfr': 'core/nsfr',
        'ocatari': 'core/ocatari',
        'envs': 'core/envs',
        'core': 'core',
        'scripts': 'scripts',
    },

    # Ship the NSFR lark grammar files as package data
    package_data={
        'nsfr': ['lark/*'],
    },
    include_package_data=True,

    # ---------------------------------------------------------------
    # Console entry points:  `gbnil-<cmd>` available after `pip install -e .`
    # ---------------------------------------------------------------
    entry_points={
        'console_scripts': [
            # Training
            'grail-train          = scripts.training.train_il:main',
            'grail-train-bc       = scripts.training.train_bc:main',
            'grail-train-bc-pt    = scripts.training.train_bc_pt:main',
            'grail-train-gaze     = scripts.training.train_gaze:main',
            # Evaluation
            'grail-eval           = scripts.evaluation.evaluate_model:main',
            'grail-eval-bc        = scripts.evaluation.evaluate_bc_model:main',
            # Preprocessing
            'grail-convert        = scripts.preprocess.convert_trajectories_to_pt:main',
            'grail-precompute     = scripts.preprocess.precompute_valuations:main',
            'grail-gen-atoms      = scripts.preprocess.generate_valuation_atoms:main',
            # Play / visualisation
            'grail-play           = scripts.play.play_il_gui:main',
            'grail-visualize      = scripts.visualization.visualize_trajectory:main',
            'grail-visualize-pt   = scripts.visualization.visualize_pt:main',
        ],
    },

    install_requires=[
        'torch',
        'numpy<2.0.0',
        'pandas',
        'tqdm',
        'pillow',
        'opencv-python',
        'matplotlib',
        'scipy',
        'seaborn',
        'scikit-learn',
        'gymnasium[atari,accept-rom-license]',
        'lark',
        'PyYAML',
        'wandb',
        'rtpt',
        'termcolor',
        'scikit-image',
        'keyboard',
        'pygame',
        'pyfiglet',
    ],
)
