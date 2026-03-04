from setuptools import setup, find_packages

setup(
    name='gbnil',
    version='0.1.0',
    description='Gaze-Based Neurosymbolic Imitation Learner',
    python_requires='>=3.9',
    packages=find_packages(where='.', include=['scripts*']),
    install_requires=[
        'torch',
        'numpy',
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
    ],
)
