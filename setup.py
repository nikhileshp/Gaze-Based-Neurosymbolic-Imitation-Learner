from setuptools import setup, find_packages
import os

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths

# Collect data files from core components
nsfr_lark_files = package_files('core/nsfr/lark')

setup(
    name='gbnil',
    version='0.1.0',
    description='Gaze-Based Neurosymbolic Imitation Learner',
    author='Nikhilesh Prabhakar',
    python_requires='>=3.9',
    packages=find_packages(where='.', include=['scripts*', 'core*']),
    package_data={
        'core.nsfr': ['lark/*'],
    },
    include_package_data=True,
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
