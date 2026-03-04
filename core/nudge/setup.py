from setuptools import setup, find_packages

setup(
    name='nudge',
    version='0.5.0',
    author='Hikaru Shindo',
    author_email='hikisan.gouv',
    packages=find_packages(),
    include_package_data=True,
    url='tba',
    description='Neurally gUided Differentiable loGic policiEs (NUDGE)',
    install_requires=['torch', 'numpy'],
)
